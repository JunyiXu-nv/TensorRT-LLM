#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Python protobuf bindings for ``trtllm.serve_engine.v1``."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_DIR = REPO_ROOT / "tensorrt_llm" / "contracts" / "serve_engine" / "v1"
PROTO_ROOT = CONTRACT_DIR / "proto"
PROTO_PKG_DIR = PROTO_ROOT / "trtllm" / "serve_engine" / "v1"
OUT_DIR = CONTRACT_DIR / "_generated"

SPDX_HEADER = (
    "# -*- coding: utf-8 -*-\n"
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# ruff: noqa\n"
    "# isort: skip_file\n"
    "# yapf: disable\n"
)

_IMPORT_RE = re.compile(r"^from trtllm\.serve_engine\.v1 import ", flags=re.M)


def _generate_to(target_dir: Path) -> dict[str, str]:
    from grpc_tools import protoc

    proto_files = sorted(PROTO_PKG_DIR.glob("*.proto"))
    if not proto_files:
        raise RuntimeError(f"No .proto files found under {PROTO_PKG_DIR}")

    args = [
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"--python_out={target_dir}",
        *[str(p) for p in proto_files],
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise RuntimeError(f"protoc failed with exit code {rc}")

    gen_subdir = target_dir / "trtllm" / "serve_engine" / "v1"
    outputs: dict[str, str] = {}
    for pb2_path in sorted(gen_subdir.glob("*_pb2.py")):
        raw = pb2_path.read_text(encoding="utf-8")
        raw = re.sub(r"^# -\*- coding: utf-8 -\*-\n", "", raw)
        rewritten = _IMPORT_RE.sub(
            "from tensorrt_llm.contracts.serve_engine.v1._generated import ",
            raw,
        )
        outputs[pb2_path.name] = SPDX_HEADER + rewritten
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in generated files match .proto sources without writing.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        generated = _generate_to(Path(tmpdir))

    if args.check:
        mismatches = []
        for name, expected in generated.items():
            actual_path = OUT_DIR / name
            if not actual_path.exists() or actual_path.read_text(encoding="utf-8") != expected:
                mismatches.append(name)
        extra = [p.name for p in OUT_DIR.glob("*_pb2.py") if p.name not in generated]
        if mismatches or extra:
            print(
                f"Serve-engine protobuf bindings out of sync: mismatches={mismatches}, extra={extra}",
                file=sys.stderr,
            )
            return 1
        print(f"Serve-engine protobuf bindings up to date ({len(generated)} files).")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (OUT_DIR / name).write_text(content, encoding="utf-8")
    print(f"Generated {len(generated)} protobuf modules under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
