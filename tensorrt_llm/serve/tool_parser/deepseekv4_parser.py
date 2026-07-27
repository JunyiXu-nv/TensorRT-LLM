# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .deepseekv32_parser import DeepSeekV32Parser


class DeepSeekV4Parser(DeepSeekV32Parser):
    """Tool parser for the DeepSeek V4 DSML tool call format."""

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"  # nosec B105
        self.eot_token = "</｜DSML｜tool_calls>"  # nosec B105
        self._markdown_code_delimiter: int | None = None
        self._pending_backticks = 0

    @staticmethod
    def _resolve_backtick_run(
        delimiter: int | None,
        pending_backticks: int,
    ) -> tuple[int | None, int]:
        if pending_backticks:
            if delimiter is None:
                delimiter = pending_backticks
            elif delimiter == pending_backticks:
                delimiter = None
        return delimiter, 0

    @classmethod
    def _advance_markdown_state(
        cls,
        text: str,
        delimiter: int | None,
        pending_backticks: int,
    ) -> tuple[int | None, int]:
        for char in text:
            if char == "`":
                pending_backticks += 1
                continue
            delimiter, pending_backticks = cls._resolve_backtick_run(
                delimiter,
                pending_backticks,
            )
        return delimiter, pending_backticks

    def _consume_normal_text(self, text: str) -> None:
        self._markdown_code_delimiter, self._pending_backticks = (
            self._advance_markdown_state(
                text,
                self._markdown_code_delimiter,
                self._pending_backticks,
            )
        )

    def _find_first_control(self) -> tuple[int, str] | None:
        """Find the first control token outside Markdown code spans."""
        matches: list[tuple[int, str]] = []
        for token in self._control_tokens():
            position = self._buffer.find(token)
            while position != -1:
                matches.append((position, token))
                position = self._buffer.find(token, position + 1)
        matches.sort(key=lambda match: match[0])

        delimiter = self._markdown_code_delimiter
        pending_backticks = self._pending_backticks
        scanned_until = 0
        for position, token in matches:
            delimiter, pending_backticks = self._advance_markdown_state(
                self._buffer[scanned_until:position],
                delimiter,
                pending_backticks,
            )
            delimiter, pending_backticks = self._resolve_backtick_run(
                delimiter,
                pending_backticks,
            )
            scanned_until = position

            # EOS is a generation boundary even when a Markdown span is
            # unclosed. DSML protocol markers inside Markdown are prose.
            if token == self._eos_token or delimiter is None:
                return position, token
        return None
