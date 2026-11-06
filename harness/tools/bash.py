import asyncio
import os
import signal
import subprocess
import sys

from harness.tools.registry import ToolRegistry
from harness.tools.workspace import Workspace

MAX_OUTPUT_CHARS = 8_000
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 1_800   # builds (xcodebuild, cmake) need minutes, not seconds


async def run_shell(
    command: str,
    timeout: float,
    cwd: str | None = None,
) -> tuple[int | None, str]:
    """
    Run a shell command in its own process group.

    Returns (exit_code, combined_output). exit_code is None on timeout.
    On timeout the whole group is killed — `proc.kill()` alone only kills the
    shell and leaves compilers/test runners it started running in the background.
    """
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32" else {"start_new_session": True}
    )
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **group,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_group(proc)
        await _reap(proc)
        return None, f"[error] command timed out after {timeout:g}s"
    except asyncio.CancelledError:
        # The run was aborted (budget hit, sibling failed): don't leave orphans.
        _kill_group(proc)
        await _reap(proc)
        raise
    return proc.returncode, stdout.decode(errors="replace")


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if sys.platform == "win32":
        # No process groups/killpg on Windows: taskkill /T kills the whole tree.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _reap(proc: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def _truncate(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    # Keep head and tail: errors are usually at the end of build logs.
    half = MAX_OUTPUT_CHARS // 2
    return (
        output[:half]
        + f"\n[... {len(output) - MAX_OUTPUT_CHARS:,} chars truncated — {len(output):,} total ...]\n"
        + output[-half:]
    )


def register(registry: ToolRegistry, workspace: Workspace | None = None) -> None:
    ws = workspace or Workspace()

    async def _bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        timeout = max(1, min(int(timeout), MAX_TIMEOUT))
        code, output = await run_shell(command, timeout, cwd=str(ws.cwd))
        if code is None:
            return output
        exit_note = "" if code == 0 else f"\n[exit code {code}]"
        return _truncate(output) + exit_note

    registry.register(
        name="bash",
        description=(
            "Run a shell command and return its output (stdout + stderr combined). "
            "Use for running scripts, compiling, testing, searching, listing files. "
            f"Output over {MAX_OUTPUT_CHARS} chars keeps the start and end. "
            f"Default timeout {DEFAULT_TIMEOUT}s; pass a larger timeout (max {MAX_TIMEOUT}s) for builds."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds. Default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT}.",
                },
            },
            "required": ["command"],
        },
    )(_bash)
