"""Tests for content parts that are not text and not media.

Both shapes here were taken from traffic that reached a GLM-5.2 deployment,
not invented: a `reasoning` part replayed inside an assistant turn, and an
image sent to a model with no vision. Each raised, and between them they
accounted for 81,853 exceptions in one run.
"""

import unittest
from unittest import mock

from tensorrt_llm.inputs.utils import MultimodalDataTracker
from tensorrt_llm.serve.chat_utils import (parse_chat_message_content,
                                           parse_chat_message_content_part)


def tracker(model_type="glm_moe_dsa"):
    return MultimodalDataTracker(model_type)


class ReasoningParts(unittest.TestCase):
    """A client that replays an assistant turn sends back its reasoning.

    The OpenAI schema has no reasoning content part, so it arrives as one only
    because the client put it there. It still has to be understood: raising
    ends the conversation, and a conversation ends the moment the model thinks
    once, since every later turn carries the same history.
    """

    def test_reasoning_part_is_lifted_onto_the_message(self):
        message = {
            "role": "assistant",
            "content": [{"type": "reasoning", "text": "Let me think about the kernel."}],
        }
        result = parse_chat_message_content(message, tracker())
        self.assertEqual("Let me think about the kernel.", result["reasoning_content"])

    def test_reasoning_is_kept_out_of_the_prompt(self):
        """The template rebuilds the <think> block; leaving it in `content` too
        would show the model its own reasoning twice."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "text": "thinking out loud"},
                {"type": "text", "text": "the answer is 4"},
            ],
        }
        result = parse_chat_message_content(message, tracker())
        self.assertEqual("the answer is 4", result["content"])
        self.assertEqual("thinking out loud", result["reasoning_content"])

    def test_several_reasoning_parts_join(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "text": "first"},
                {"type": "reasoning", "text": "second"},
            ],
        }
        result = parse_chat_message_content(message, tracker())
        self.assertEqual("first\nsecond", result["reasoning_content"])

    def test_thinking_is_accepted_under_its_other_names(self):
        for part_type in ("reasoning", "thinking", "reasoning_content"):
            with self.subTest(part_type=part_type):
                message = {"role": "assistant",
                           "content": [{"type": part_type, "text": "x"}]}
                result = parse_chat_message_content(message, tracker())
                self.assertEqual("x", result["reasoning_content"])

    def test_a_message_without_reasoning_does_not_gain_the_field(self):
        message = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
        self.assertNotIn("reasoning_content",
                         parse_chat_message_content(message, tracker()))

    def test_an_empty_reasoning_part_is_not_carried(self):
        message = {"role": "assistant", "content": [{"type": "reasoning", "text": ""}]}
        self.assertNotIn("reasoning_content",
                         parse_chat_message_content(message, tracker()))

    def test_a_long_replayed_turn_is_handled(self):
        """The captured part was 99,320 characters; nothing here is length-bound."""
        text = "step\n" * 20000
        message = {"role": "assistant", "content": [{"type": "reasoning", "text": text}]}
        result = parse_chat_message_content(message, tracker())
        self.assertEqual(text, result["reasoning_content"])


class MediaOnATextOnlyModel(unittest.TestCase):
    """GLM has no vision. An image used to be accepted here and then fail in
    the placeholder registry with `Unknown modality: image`, which names the
    modality but not the model."""

    IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}

    def test_an_image_becomes_a_note_rather_than_an_error(self):
        result = parse_chat_message_content_part(self.IMAGE, tracker())
        self.assertIsInstance(result, str)
        self.assertIn("image", result)
        self.assertIn("text only", result)

    def test_the_rest_of_the_message_survives(self):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this picture?"},
                self.IMAGE,
            ],
        }
        result = parse_chat_message_content(message, tracker())
        self.assertIn("what is in this picture?", result["content"])
        self.assertIn("omitted", result["content"])
        self.assertEqual([], result["media"])

    def test_every_media_type_degrades(self):
        parts = [
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
            {"type": "video_url", "video_url": {"url": "http://x/a.mp4"}},
            {"type": "audio_url", "audio_url": {"url": "http://x/a.wav"}},
            {"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}},
        ]
        for part in parts:
            with self.subTest(part_type=part["type"]):
                result = parse_chat_message_content_part(part, tracker())
                self.assertIsInstance(result, str)
                self.assertIn("omitted", result)

    def test_a_model_that_does_have_vision_still_loads_the_image(self):
        """The degradation must key on the model, not on the part."""
        with mock.patch(
                "tensorrt_llm.serve.chat_utils.MULTIMODAL_PLACEHOLDER_REGISTRY"
        ) as registry:
            registry.is_valid.return_value = True
            result = parse_chat_message_content_part(self.IMAGE, tracker("qwen2_vl"))
        self.assertNotIsInstance(result, str)
        self.assertEqual("image", result["modality"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
