# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone unit tests for the `_serve_stability` helpers.

These tests exercise the helpers in isolation (no full `tensorrt_llm` import)
so they run quickly and stay green even on CPU-only CI shards.
"""

from __future__ import annotations

import click
import pytest

from tensorrt_llm.commands._serve_stability import (
    ALLOWED_TAGS,
    collect_command_options,
    get_option_stability,
    help_info_with_stability_tag,
    stability_option,
)


def test_help_info_includes_tag():
    out = help_info_with_stability_tag("Port of the server.", "beta")
    assert out.startswith(":tag:`beta`")
    assert "Port of the server." in out


def test_help_info_rejects_invalid_tag():
    with pytest.raises(ValueError, match="Invalid stability tag"):
        help_info_with_stability_tag("x", "experimental")  # type: ignore[arg-type]


def test_stability_option_stamps_attribute():
    @click.command("toy")
    @stability_option(
        "--threshold", type=int, default=5, status="prototype", help="A tunable knob."
    )
    def toy(threshold): ...

    option = next(p for p in toy.params if p.name == "threshold")
    assert get_option_stability(option) == "prototype"
    assert option.help.startswith(":tag:`prototype`")


def test_stability_option_rejects_invalid_status():
    with pytest.raises(ValueError, match="Invalid stability tag"):

        @click.command("toy")
        @stability_option(
            "--threshold",
            status="experimental",  # type: ignore[arg-type]
            help="x",
        )
        def toy(threshold): ...


def test_get_option_stability_reads_legacy_help_tag():
    """Cover legacy options tagged via help_info_with_stability_tag.

    Existing options pass the rendered help string directly to ``@click.option``,
    so the checker must still be able to extract the tag from that text.
    """

    @click.command("toy")
    @click.option(
        "--legacy", default=None, help=help_info_with_stability_tag("an old option", "stable")
    )
    def toy(legacy): ...

    option = next(p for p in toy.params if p.name == "legacy")
    assert get_option_stability(option) == "stable"


def test_get_option_stability_returns_none_for_untagged():
    @click.command("toy")
    @click.option("--bare", default=None, help="No tag at all.")
    def toy(bare): ...

    option = next(p for p in toy.params if p.name == "bare")
    assert get_option_stability(option) is None


def test_collect_command_options_skips_arguments():
    @click.command("toy")
    @click.argument("model", type=str)
    @stability_option("--port", type=int, default=8000, status="beta", help="port")
    def toy(model, port): ...

    out = collect_command_options(toy)
    # `model` is a positional arg, not an option, so should be excluded.
    assert "model" not in out
    assert "port" in out
    assert out["port"]["status"] == "beta"
    assert out["port"]["type"] == "int"
    assert out["port"]["default"] == 8000


def test_allowed_tags_match_expected_set():
    assert set(ALLOWED_TAGS) == {"stable", "beta", "prototype", "deprecated"}
