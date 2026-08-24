#!/usr/bin/env python3
"""Report deployment dependencies without importing heavyweight packages."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = PIPELINE_ROOT / "src"
if str(RUNTIME_SOURCE) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SOURCE))

from voice_embedding_onnx.runtime_onnx import (  # noqa: E402
    OrtPipelineError,
    select_ort_assets,
)


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _native_capabilities(
    binary: Path, expected_key: str, expected_value: object,
) -> tuple[bool, dict | None, str | None]:
    if not binary.is_file():
        return False, None, f"missing: {binary}"
    completed = subprocess.run(
        [str(binary), "--version"],
        cwd=binary.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False, None, completed.stderr.strip()
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, None, "--version output is not JSON"
    if not isinstance(value, dict):
        return False, None, "--version JSON root is not an object"
    return value.get(expected_key) == expected_value, value, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--c", action="store_true")
    backend.add_argument("--ort", action="store_true")
    parser.add_argument(
        "--fbank",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--speaker-verify",
        type=Path,
        default=PIPELINE_ROOT / "runtime/campp_speaker_verify",
    )
    parser.add_argument(
        "--ort-asset-manifest",
        type=Path,
        default=PIPELINE_ROOT / "runtime_onnx/assets.json",
    )
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    imports: dict[str, list[str]] = {"torch": [], "torchaudio": []}
    for path in sorted(RUNTIME_SOURCE.rglob("*.py")):
        modules = _imported_modules(path)
        for module in imports:
            if module in modules:
                imports[module].append(path.relative_to(PIPELINE_ROOT).as_posix())
    fbank = args.fbank or PIPELINE_ROOT / (
        "runtime_onnx/campp_fbank" if args.ort else "runtime/campp_fbank"
    )
    native_ready, native_capabilities, native_error = _native_capabilities(
        fbank.resolve(), "frontend", "campp-kaldi-native-fbank",
    )
    verifier_ready = False
    verifier_capabilities = None
    verifier_error = None
    if not args.ort:
        verifier_ready, verifier_capabilities, verifier_error = _native_capabilities(
            args.speaker_verify.resolve(), "application", "campp_speaker_verify",
        )
    installed = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "onnxruntime", "torch", "torchaudio")
    }
    ort_assets_ready = False
    ort_assets_error = None
    if args.ort:
        try:
            select_ort_assets(args.ort_asset_manifest.resolve(), 98)
            ort_assets_ready = True
        except (OrtPipelineError, OSError) as exc:
            ort_assets_error = str(exc)
    payload = {
        "selected_backend": "onnxruntime-cpu" if args.ort else "campp-c-runtime",
        "runtime_torch_required": any(imports.values()),
        "runtime_import_locations": imports,
        "installed_without_importing": installed,
        "native_fbank": {
            "path": str(fbank.resolve()),
            "ready": native_ready,
            "capabilities": native_capabilities,
            "error": native_error,
        },
        "native_speaker_verify": {
            "path": str(args.speaker_verify.resolve()),
            "required": not args.ort,
            "ready": verifier_ready,
            "capabilities": verifier_capabilities,
            "error": verifier_error,
        },
        "onnx_runtime_assets": {
            "manifest": str(args.ort_asset_manifest.resolve()),
            "required": bool(args.ort),
            "ready": ort_assets_ready,
            "error": ort_assets_error,
        },
        "deployment_contract": {
            "verification_python": bool(args.ort),
            "verification_numpy": bool(args.ort),
            "enrollment_python": True,
            "enrollment_numpy": True,
            "torch": False,
            "torchaudio": False,
            "native_fbank": True,
            "onnxruntime": bool(args.ort),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ready = not payload["runtime_torch_required"] and native_ready
    if args.ort:
        ready = ready and installed["numpy"] and installed["onnxruntime"]
        ready = ready and ort_assets_ready
    else:
        ready = ready and verifier_ready
    return 0 if ready or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
