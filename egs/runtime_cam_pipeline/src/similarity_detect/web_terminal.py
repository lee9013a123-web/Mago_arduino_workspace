"""Dependency-free HTTP terminal for the microphone pipeline CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import threading
from typing import Iterator
from urllib.parse import urlsplit


ALLOWED_SCRIPTS = frozenset({
    "enroll_speaker.py",
    "verify_speaker.py",
    "list_microphones.py",
})
NATIVE_VERIFY_BINARY = "campp_speaker_verify"
MAX_COMMAND_BYTES = 8192
SHELL_CONTROL_TOKENS = frozenset({
    "|", "||", "&&", ";", "<", ">", ">>", "2>", "2>>", "&",
})


class WebTerminalError(RuntimeError):
    """Raised when a browser command violates the terminal contract."""


class CommandBusyError(WebTerminalError):
    """Raised when another microphone command is already running."""


@dataclass(frozen=True)
class ParsedCommand:
    display: str
    argv: list[str]


@dataclass(frozen=True)
class RunningCommand:
    parsed: ParsedCommand
    process: subprocess.Popen[str]


def parse_pipeline_command(
    command: str, *, pipeline_root: Path,
    python_executable: str | None = None,
) -> ParsedCommand:
    if not isinstance(command, str) or not command.strip():
        raise WebTerminalError("command is empty")
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise WebTerminalError("command is too long")
    if "\0" in command or "\n" in command or "\r" in command:
        raise WebTerminalError("command must be a single line")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise WebTerminalError(f"cannot parse command: {exc}") from exc
    if not tokens:
        raise WebTerminalError("command has no executable")
    requested_executable = tokens.pop(0)
    executable_name = Path(requested_executable).name.lower()
    for token in tokens:
        if token in SHELL_CONTROL_TOKENS:
            raise WebTerminalError("shell pipes, redirects, and chaining are blocked")
        if "`" in token or "$(" in token:
            raise WebTerminalError("shell substitutions are blocked")

    if executable_name == NATIVE_VERIFY_BINARY:
        binary = (pipeline_root / "runtime" / NATIVE_VERIFY_BINARY).resolve()
        requested = Path(requested_executable)
        requested = (
            requested.resolve()
            if requested.is_absolute()
            else (pipeline_root / requested).resolve()
        )
        if requested != binary:
            raise WebTerminalError(
                f"native verifier must be invoked as ./runtime/{NATIVE_VERIFY_BINARY}"
            )
        if not binary.is_file() or not binary.is_relative_to(
            pipeline_root.resolve()
        ):
            raise WebTerminalError(f"native verifier is missing: {binary}")
        return ParsedCommand(
            display=command.strip(),
            argv=[str(binary), *tokens],
        )

    if not re.fullmatch(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", executable_name):
        raise WebTerminalError(
            "only the native verifier or approved Python utilities are allowed"
        )
    if tokens and tokens[0] == "-u":
        tokens.pop(0)
    if not tokens or tokens[0].startswith("-"):
        raise WebTerminalError("Python interpreter options are not allowed")
    script_name = Path(tokens.pop(0)).name
    if script_name not in ALLOWED_SCRIPTS:
        raise WebTerminalError(
            f"script {script_name!r} is not allowed; expected "
            f"{sorted(ALLOWED_SCRIPTS)}"
        )
    script_path = (pipeline_root / "script" / script_name).resolve()
    if not script_path.is_file() or not script_path.is_relative_to(
        pipeline_root.resolve()
    ):
        raise WebTerminalError(f"pipeline script is missing: {script_path}")
    interpreter = python_executable or sys.executable
    return ParsedCommand(
        display=command.strip(),
        argv=[interpreter, "-u", str(script_path), *tokens],
    )


class CommandController:
    """Allows one recording/inference command and streams merged output."""

    def __init__(self, pipeline_root: Path) -> None:
        self.pipeline_root = pipeline_root.resolve()
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running: RunningCommand | None = None

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._running is not None

    def start(self, command: str) -> RunningCommand:
        parsed = parse_pipeline_command(
            command,
            pipeline_root=self.pipeline_root,
        )
        if not self._run_lock.acquire(blocking=False):
            raise CommandBusyError("another pipeline command is already running")
        try:
            environment = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "ORT_NUM_THREADS": "1",
            }
            process = subprocess.Popen(
                parsed.argv,
                cwd=self.pipeline_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=(os.name == "posix"),
            )
            running = RunningCommand(parsed=parsed, process=process)
            with self._state_lock:
                self._running = running
            return running
        except Exception:
            self._run_lock.release()
            raise

    def stream(self, running: RunningCommand) -> Iterator[str]:
        try:
            yield f"$ {running.parsed.display}\n\n"
            if running.process.stdout is None:
                raise WebTerminalError("command stdout pipe is unavailable")
            for line in iter(running.process.stdout.readline, ""):
                yield line
            return_code = running.process.wait()
            yield f"\n[process exited with code {return_code}]\n"
        finally:
            if running.process.stdout is not None:
                running.process.stdout.close()
            if running.process.poll() is None:
                self._terminate(running.process)
                try:
                    running.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._kill(running.process)
                    running.process.wait()
            with self._state_lock:
                if self._running is running:
                    self._running = None
            self._run_lock.release()

    def stop(self) -> bool:
        with self._state_lock:
            running = self._running
        if running is None or running.process.poll() is not None:
            return False
        self._terminate(running.process)
        return True

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()

    @staticmethod
    def _kill(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()


class PipelineWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, server_address: tuple[str, int], *, pipeline_root: Path,
        index_html: bytes, access_log: bool,
    ) -> None:
        self.pipeline_root = pipeline_root.resolve()
        self.index_html = index_html
        self.controller = CommandController(self.pipeline_root)
        self.access_log = access_log
        super().__init__(server_address, PipelineRequestHandler)


class PipelineRequestHandler(BaseHTTPRequestHandler):
    server: PipelineWebServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: object) -> None:
        if self.server.access_log:
            super().log_message(format_string, *args)

    def _send_bytes(
        self, status: HTTPStatus, body: bytes, content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, value: object) -> None:
        body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise WebTerminalError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_COMMAND_BYTES * 2:
            raise WebTerminalError("request body size is invalid")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebTerminalError("request body must be UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise WebTerminalError("request JSON root must be an object")
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                self.server.index_html,
                "text/html; charset=utf-8",
            )
        elif path == "/api/health":
            self._send_json(HTTPStatus.OK, {
                "ready": True,
                "busy": self.server.controller.busy,
                "allowed_scripts": sorted(ALLOWED_SCRIPTS),
            })
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/stop":
            stopped = self.server.controller.stop()
            self._send_json(HTTPStatus.OK, {"stopped": stopped})
            return
        if path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            request = self._read_json()
            command = request.get("command")
            if not isinstance(command, str):
                raise WebTerminalError("request has no string command")
            running = self.server.controller.start(command)
        except CommandBusyError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return
        except (WebTerminalError, OSError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        stream = self.server.controller.stream(running)
        try:
            for text in stream:
                self.wfile.write(text.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.server.controller.stop()
        finally:
            stream.close()


def serve_web_terminal(
    *, pipeline_root: Path, host: str, port: int,
    access_log: bool = False,
) -> None:
    if not 0 < port < 65536:
        raise WebTerminalError(f"invalid TCP port: {port}")
    index_path = pipeline_root / "web/index.html"
    if not index_path.is_file():
        raise WebTerminalError(f"web UI is missing: {index_path}")
    server = PipelineWebServer(
        (host, port),
        pipeline_root=pipeline_root,
        index_html=index_path.read_bytes(),
        access_log=access_log,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.controller.stop()
        server.server_close()
