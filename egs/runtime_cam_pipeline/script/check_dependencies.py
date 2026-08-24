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


def _native_capabilities(binary: Path) -> tuple[bool, dict | None, str | None]:
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
    return value.get("frontend") == "campp-kaldi-native-fbank", value, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fbank",
        type=Path,
        default=PIPELINE_ROOT / "runtime/campp_fbank",
    )
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    imports: dict[str, list[str]] = {"torch": [], "torchaudio": []}
    for path in sorted(RUNTIME_SOURCE.rglob("*.py")):
        modules = _imported_modules(path)
        for module in imports:
            if module in modules:
                imports[module].append(path.relative_to(PIPELINE_ROOT).as_posix())
    native_ready, native_capabilities, native_error = _native_capabilities(
        args.fbank.resolve()
    )
    payload = {
        "runtime_torch_required": any(imports.values()),
        "runtime_import_locations": imports,
        "installed_without_importing": {
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "torch", "torchaudio")
        },
        "native_fbank": {
            "path": str(args.fbank.resolve()),
            "ready": native_ready,
            "capabilities": native_capabilities,
            "error": native_error,
        },
        "deployment_contract": {
            "python": True,
            "numpy": True,
            "torch": False,
            "torchaudio": False,
            "native_fbank": True,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ready = not payload["runtime_torch_required"] and native_ready
    return 0 if ready or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
