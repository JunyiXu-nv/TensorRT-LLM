# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stability-tag plumbing for the `trtllm-serve` CLI.

This module gives every Click option on `trtllm-serve` an explicit, machine-readable
stability status (`stable | beta | prototype | deprecated`). The status is:

  1. rendered in `--help` (via `help_info_with_stability_tag`), and
  2. attached to the option as metadata so a CI test can diff the live CLI surface
     against a checked-in reference YAML (`tests/unittest/api_stability/references/
     trtllm_serve_cli.yaml`).

Use `stability_option(...)` as a *required* drop-in replacement for `@click.option`
when adding new options to any `trtllm-serve` subcommand. Plain `@click.option` is
deliberately not banned in this PR, but the stability checker will fail if a CLI
option is missing from the reference YAML — which forces the contributor to make
an explicit status choice.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

import click

StabilityTag = Literal["stable", "beta", "prototype", "deprecated"]

ALLOWED_TAGS: tuple[StabilityTag, ...] = ("stable", "beta", "prototype", "deprecated")

# Attribute name we stamp on every Click option for the stability checker to read.
# Using a private attribute keeps it invisible to Click's own machinery.
_STABILITY_ATTR = "_trtllm_stability"


def help_info_with_stability_tag(help_str: str, tag: StabilityTag) -> str:
    """Append stability info to the help string.

    Kept here (rather than in `serve.py`) so any tooling that needs to read the
    raw tag can import it from a single place.
    """
    if tag not in ALLOWED_TAGS:
        raise ValueError(f"Invalid stability tag {tag!r}; must be one of {ALLOWED_TAGS}.")
    return f":tag:`{tag}` {help_str}"


def stability_option(*param_decls: str, status: StabilityTag, help: str, **kwargs: Any) -> Callable:
    """A `@click.option` wrapper that **requires** an explicit stability status.

    Differences from bare `@click.option`:
      * `status=` is a mandatory keyword argument.
      * `help=` is also required (so the rendered help carries the stability tag).
      * The chosen status is stamped onto the resulting `click.Option` instance
        as `option._trtllm_stability`, so the stability checker can read it back.

    Example::

        @stability_option("--max_batch_size",
                           type=int,
                           default=None,
                           status="beta",
                           help="Maximum number of requests per batch.")
        def serve(..., max_batch_size: int | None, ...): ...
    """
    if status not in ALLOWED_TAGS:
        raise ValueError(f"Invalid stability tag {status!r}; must be one of {ALLOWED_TAGS}.")

    tagged_help = help_info_with_stability_tag(help, status)
    click_decorator = click.option(*param_decls, help=tagged_help, **kwargs)

    def decorator(f: Callable) -> Callable:
        wrapped = click_decorator(f)
        # Click stores the new param at the end of __click_params__; stamp it.
        params = getattr(wrapped, "__click_params__", None)
        if params:
            setattr(params[-1], _STABILITY_ATTR, status)
        return wrapped

    return decorator


_HELP_TAG_RE = re.compile(r":tag:`(stable|beta|prototype|deprecated)`")


def get_option_stability(option: click.Option) -> StabilityTag | None:
    """Return the stability tag attached to a Click option, or None if absent.

    Two sources, in order of preference:

    1. The `_trtllm_stability` attribute, stamped by `stability_option(...)`.
    2. A `:tag:`<status>`` prefix in the rendered help string, produced by the
       legacy `help_info_with_stability_tag` helper. This lets us cover the
       existing options that were tagged before `stability_option` existed,
       without forcing a single-PR refactor of every option site.

    Returns None for options created via bare `@click.option` (no tag, no
    legacy helper). The stability checker treats `None` as a violation.
    """
    explicit = getattr(option, _STABILITY_ATTR, None)
    if explicit is not None:
        return explicit
    help_text = option.help or ""
    match = _HELP_TAG_RE.search(help_text)
    if match:
        return match.group(1)  # type: ignore[return-value]
    return None


def collect_command_options(command: click.Command) -> dict[str, dict[str, Any]]:
    """Introspect a Click command and return a mapping of option-name -> spec.

    The returned spec is the on-disk YAML representation used by the stability
    checker. Each entry contains: ``status``, ``type``, ``default``,
    ``required``, ``multiple``, ``is_flag``.

    Positional arguments (``click.Argument``) are skipped — they're part of the
    invocation contract and don't carry a stability tag in the same way.
    """
    out: dict[str, dict[str, Any]] = {}
    for param in command.params:
        if not isinstance(param, click.Option):
            continue
        name = param.name
        out[name] = {
            "status": get_option_stability(param),
            "type": _format_type(param.type),
            "default": _format_default(param.default),
            "required": bool(param.required),
            "multiple": bool(param.multiple),
            "is_flag": bool(param.is_flag),
        }
    return out


def _format_type(t: Any) -> str:
    """Render a Click ParamType to a short, stable string for the YAML."""
    if isinstance(t, click.types.IntParamType):
        return "int"
    if isinstance(t, click.types.FloatParamType):
        return "float"
    if isinstance(t, click.types.BoolParamType):
        return "bool"
    if isinstance(t, click.types.StringParamType):
        return "str"
    if isinstance(t, click.Choice):
        return f"Choice({sorted(t.choices)!r})"
    if isinstance(t, click.Path):
        return "Path"
    # Fall back to the type's name attribute.
    return getattr(t, "name", t.__class__.__name__)


def _format_default(value: Any) -> Any:
    """Render a Click default to a YAML-serializable form."""
    if callable(value):
        # ``default_factory``-style callables: record their fully-qualified name
        # rather than the live object, so the YAML stays stable across runs.
        mod = getattr(value, "__module__", "")
        qual = getattr(value, "__qualname__", repr(value))
        return f"<callable:{mod}.{qual}>" if mod else f"<callable:{qual}>"
    if isinstance(value, (tuple, list)):
        return list(value)
    return value
