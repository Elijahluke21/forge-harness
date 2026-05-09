import asyncio

from harness.tools.registry import ToolRegistry

MAX_OUTPUT_CHARS = 8_000
DEFAULT_TIMEOUT = 30


async def _bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"[error] command timed out after {timeout}s"

    output = stdout.decode(errors="replace")
    exit_note = "" if proc.returncode == 0 else f"\n[exit code {proc.returncode}]"

    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n[... truncated — {len(output)} chars total]"

    return output + exit_note


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="bash",
        description=(
            "Run a shell command and return its output (stdout + stderr combined). "
            "Use for running scripts, compiling, searching, listing files, etc. "
            f"Output is capped at {MAX_OUTPUT_CHARS} chars. Default timeout is {DEFAULT_TIMEOUT}s."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds. Default {DEFAULT_TIMEOUT}.",
                },
            },
            "required": ["command"],
        },
    )(_bash)
