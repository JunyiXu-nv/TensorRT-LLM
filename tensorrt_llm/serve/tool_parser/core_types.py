# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/qwen25_detector.py
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string
    # What the parser changed or recovered to deliver this call, one dict per
    # event with a "kind" (see kimi_k3_tool_parser). None when nothing was.
    normalizations: Optional[List[Dict[str, Any]]] = None


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""
_GetInfoFunc = Callable[[str], StructureInfo]
