import asyncio
import json
import uuid
from copy import copy
from typing import Literal, Optional, OrderedDict, Union

from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseOutputItem, ResponseOutputMessage,
                                    ResponseOutputText, ResponseReasoningItem)
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.tool import Tool
from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort, Role,
                            SystemContent, TextContent, ToolDescription,
                            load_harmony_encoding)

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (ResponseInputOutputItem,
                                                ResponsesRequest,
                                                ResponsesResponse)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None


def random_uuid():
    return str(uuid.uuid4().hex)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def decode_tokens(tokens):
    return get_encoding().decode(tokens)


def parse_response_input(
    input_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]]
) -> Message:
    if not isinstance(input_msg, dict):
        input_msg = input_msg.model_dump()

    logger.debug(f"------- Parsing input -----------")
    logger.debug(input_msg)
    logger.debug("")

    if "type" not in input_msg or input_msg["type"] == "message":
        role = input_msg["role"]
        content = input_msg["content"]
        if role == "system":
            # User is trying to set a system message. Change it to:
            # <|start|>developer<|message|># Instructions
            # {instructions}<|end|>
            role = "developer"
            text_prefix = "Instructions:\n"
        else:
            text_prefix = ""
        if isinstance(content, str):
            msg = Message.from_role_and_content(role, text_prefix + content)
        elif isinstance(content, list):
            contents = [
                TextContent(text=text_prefix + c["text"]) for c in content
            ]
            msg = Message.from_role_and_contents(role, contents)
        else:
            logger.warning("Invalid input message type")
            msg = None
    elif input_msg["type"] == "function_call_output":
        call_id = input_msg["call_id"]
        call_response: Optional[ResponseFunctionToolCall] = None
        for prev_response in reversed(prev_responses):
            if isinstance(prev_response, ResponseFunctionToolCall
                          ) and prev_response.call_id == call_id:
                call_response = prev_response
                break
        if call_response is None:
            raise ValueError(f"No call message found for {call_id}")
        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{call_response.name}"),
            input_msg["output"])
    elif input_msg["type"] == "reasoning":
        content = input_msg["content"]
        assert len(content) == 1
        msg = Message.from_role_and_content(Role.ASSISTANT, content[0]["text"])
    elif input_msg["type"] == "function_call":
        msg = Message.from_role_and_content(Role.ASSISTANT,
                                            input_msg["arguments"])
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{input_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {input_msg['type']}")
    return msg


class ConversationHistoryStore:

    def __init__(self, capacity: int = 16, max_conversations=32):
        self.response_capacity = capacity
        self.conversation_capacity = capacity * 4
        self.max_conversations = max_conversations

        self.responses_lock = asyncio.Lock()
        self.responses: OrderedDict[str, ResponsesResponse] = OrderedDict()

        self.conversations_lock = asyncio.Lock()
        self.conversations: OrderedDict[str, list[Message]] = OrderedDict()
        self.response_to_conversation: dict[str, str] = {}

    async def load_response(self, resp_id: str) -> ResponsesResponse:
        logger.debug(f"ConversationHistoryStore loading resp: {resp_id}")
        async with self.responses_lock:
            return self.responses.get(resp_id)

    async def store_response(self,
                             resp: ResponsesResponse,
                             resp_msgs: Optional[list[Message]] = [],
                             prev_resp_id: Optional[str] = None) -> None:
        resp_id = resp.id
        logger.debug(f"ConversationHistoryStore storing resp: {resp_id}")
        async with self.responses_lock:
            self.responses[resp_id] = resp
            if len(self.responses) > self.response_capacity:
                self._pop_response()

        async with self.conversations_lock:
            conversation_id: str
            if resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[resp_id]
                self.conversations[conversation_id].extend(resp_msgs)
            elif prev_resp_id is not None:
                conversation_id = self.response_to_conversation[prev_resp_id]
                self.conversations[conversation_id].extend(resp_msgs)
                while len(self.conversations[conversation_id]
                          ) > self.conversation_capacity:
                    self._pop_conversation()
            else:
                conversation_id = random_uuid()
                self.conversations[conversation_id] = resp_msgs

            logger.debug(f" * storing at conversation id: {conversation_id}")

            self.response_to_conversation[resp_id] = conversation_id
            self._update_visited_conversation(conversation_id)

    async def store_messages(self, resp_id: str, msgs: list[Message],
                             prev_resp_id: Optional[str]):
        logger.debug(f"ConversationHistoryStore storing msg:")
        for msg in msgs:
            logger.debug(f" -> {msg.to_json()}")

        async with self.conversations_lock:
            conversation_id: str
            if prev_resp_id is not None:
                conversation_id = self.response_to_conversation[prev_resp_id]
            else:
                conversation_id = random_uuid()

            logger.debug(f" * storing at conversation: {conversation_id}")
            self.conversations[conversation_id] = msgs
            if len(self.conversations[conversation_id]
                   ) > self.conversation_capacity:
                self._pop_conversation()

            self.response_to_conversation[resp_id] = conversation_id
            self._update_visited_conversation(conversation_id)

    async def append_messages(self, resp_id: str, msgs: list[Message]):
        logger.debug(f"ConversationHistoryStore appending msgs:")
        for msg in msgs:
            logger.debug(f" -> {msg.to_json()}")

        async with self.conversations_lock:
            assert resp_id in self.response_to_conversation
            conversation_id = self.response_to_conversation[resp_id]

            logger.debug(f" * appending at conversation: {conversation_id}")
            self.conversations[conversation_id].extend(msgs)
            if len(self.conversations[conversation_id]
                   ) > self.conversation_capacity:
                self._pop_conversation()
            self._update_visited_conversation(conversation_id)

    async def get_conversation_history(self, resp_id: str) -> list[Message]:
        logger.debug(f"ConversationHistoryStore getting prev_msgs:")
        logger.debug(f" -> prev_resp_id: {resp_id}")
        async with self.conversations_lock:
            if resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[resp_id]
                self._update_visited_conversation(conversation_id)
                return self.conversations.get(conversation_id, [])

            return []

    def _update_visited_conversation(self, conversation_id) -> None:
        if conversation_id not in self.conversations:
            return

        self.conversations.move_to_end(conversation_id)
        if len(self.conversations) > self.max_conversations:
            removed_id, _ = self.conversations.popitem(last=False)
            logger.debug(
                f"ConversationHistoryStore Removing conversation {removed_id}")

    def _pop_conversation(self, resp_id) -> None:
        conversation_id = self.response_to_conversation[resp_id]
        conversation = self.conversations[conversation_id]
        first_conversation_range = []
        for i, msg in enumerate(conversation):
            if msg.author.role == Role.USER:
                first_conversation_range.append(i)
            elif msg.channel == "final":
                first_conversation_range.append(i)
                break
        del conversation[
            first_conversation_range[0]:first_conversation_range[1] + 1]

    def _pop_response(self) -> None:
        logger.debug(f"responses type: {type(self.responses)}")
        resp_id, _ = self.responses.popitem(last=False)
        self.response_to_conversation.pop(resp_id)


def get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort])
    if start_date:
        sys_msg_content = sys_msg_content.with_conversation_start_date(
            start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(instructions: Optional[str] = None,
                          tools: Optional[list[Tool]] = None) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if tool.type in ("web_search_preview", "code_interpreter"):
                # These are built-in tools that are added to the system message.
                pass
            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions)
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def construct_harmony_messages(
    request: ResponsesRequest,
    prev_response: Optional[ResponsesResponse],
    prev_msgs: list[Message] = [],
) -> list[Message]:
    """
        Construct messages from request input, includes conversation history messages if exists.
        """
    messages: list[Message] = []
    if prev_response is None:
        # New conversation.
        reasoning_effort = (request.reasoning.effort
                            if request.reasoning else None)
        sys_msg = get_system_message(reasoning_effort=reasoning_effort, )
        messages.append(sys_msg)
        dev_msg = get_developer_message(request.instructions, request.tools)
        messages.append(dev_msg)
    else:
        ## These codes remove the reasoning message of prev turn, why?
        # if len(prev_msgs) > 0:
        #     last_msg = prev_msgs[-1]
        #     assert isinstance(last_msg, Message)
        #     if last_msg.channel == "final":
        #         prev_final_msg_idx = -1
        #         for i in range(len(prev_msgs) - 2, -1, -1):
        #             prev_msg_i = prev_msgs[i]
        #             assert isinstance(prev_msg_i, Message)
        #             if prev_msg_i.channel == "final":
        #                 prev_final_msg_idx = i
        #                 break
        #         recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1:]
        #         del prev_msgs[prev_final_msg_idx + 1:]
        #         for msg in recent_turn_msgs:
        #             assert isinstance(msg, Message)
        #             if msg.channel != "analysis":
        #                 prev_msgs.append(msg)
        messages.extend(prev_msgs)
    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append(get_user_message(request.input))
    else:
        if prev_response is not None:
            prev_outputs = copy(prev_response.output)
        else:
            prev_outputs = []
        for input_msg in request.input:
            msg = parse_response_input(input_msg, prev_outputs)
            if msg is not None:
                messages.append(msg)
            # User passes in a a tool call request and its output. We need
            # to add the tool call request to prev_outputs so that the
            # parse_response_input can find the tool call request when
            # parsing the tool call output.
            if isinstance(input_msg, ResponseFunctionToolCall):
                prev_outputs.append(input_msg)
    return messages


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    logger.debug("Rendering conversation:")
    logger.debug(conversation.to_json())
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


def parse_output_tokens(tokens: list[int]) -> list[Message]:
    ## WAR gpt-oss-20b issue: https://github.com/vllm-project/vllm/issues/22519
    call_token = 200012
    start_token = 200006
    call_commentay = [call_token, 12606, 815]
    start_idx = 0
    while call_token in tokens[start_idx:]:
        call_idx = start_idx + tokens[start_idx:].index(call_token)
        next_tokens = tokens[call_idx:call_idx + len(call_commentay)]
        if next_tokens == call_commentay:
            # need fix
            # fix_token_start = call_idx + len(call_commentay)
            tokens[call_idx + 1] = start_token
            tokens[call_idx + 1] = 173781  # assistant

        start_idx = call_idx + 1

    # if stop_idx != -1:
    # tokens = tokens[:stop_idx+1]
    return get_encoding().parse_messages_from_completion_tokens(
        tokens, role=Role.ASSISTANT)


def parse_output_message(message: Message) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.
    """
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        # TODO: translate to url properly!
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search")
        elif recipient == "browser.open":
            action = ActionOpenPage(url=f"cursor:{browser_call.get('url', '')}",
                                    type="open_page")
        elif recipient == "browser.find":
            action = ActionFind(pattern=browser_call["pattern"],
                                url=f"cursor:{browser_call.get('url', '')}",
                                type="find")
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[Content(text=content.text, type="reasoning_text")],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        if message.recipient.startswith("functions."):
            function_name = message.recipient.split(".")[-1]
            for content in message.content:
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=function_name,
                    id=f"ft_{random_id}",
                )
                output_items.append(response_item)
        elif message.recipient.startswith(
                "python") or message.recipient.startswith("browser"):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[Content(text=content.text, type="reasoning_text")],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            contents.append(output_text)
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=contents,
            role=message.author.role,
            status="completed",
            type="message",
        )
        output_items.append(text_item)
    else:
        raise ValueError(f"Unknown channel: {message.channel}")
    return output_items
