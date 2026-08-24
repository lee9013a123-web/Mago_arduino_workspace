#!/usr/bin/env python3
"""Serve the pipeline command box and streaming terminal UI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from similarity_detect.web_terminal import (  # noqa: E402
    WebTerminalError,
    serve_web_terminal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--access-log", action="store_true")
    args = parser.parse_args()
    try:
        print(f"pipeline root: {PIPELINE_ROOT}")
        print(f"web terminal: http://{args.host}:{args.port}")
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            print("LAN mode: connect with http://<arduino-ip>:" + str(args.port))
        print("allowed scripts: enroll_speaker, verify_speaker, list_microphones")
        serve_web_terminal(
            pipeline_root=PIPELINE_ROOT,
            host=args.host,
            port=args.port,
            access_log=args.access_log,
        )
        return 0
    except (WebTerminalError, OSError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nweb terminal stopped")
            return 0
        print(f"web terminal failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
