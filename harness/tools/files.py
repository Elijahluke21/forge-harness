import os

from harness.tools.registry import ToolRegistry

MAX_READ_CHARS = 8_000


async def _read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"[error] file not found: {path}"
    except OSError as e:
        return f"[error] could not read {path}: {e}"

    start = max(0, offset - 1)
    end = start + limit if limit is not None else len(lines)
    selected = lines[start:end]

    # Format with line numbers (cat -n style)
    numbered = "".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))

    if len(numbered) > MAX_READ_CHARS:
        numbered = numbered[:MAX_READ_CHARS] + f"\n[... truncated — use offset/limit to read more]"

    return numbered or "[empty file]"


async def _write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"wrote {len(content)} chars to {path}"
    except OSError as e:
        return f"[error] could not write {path}: {e}"


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="read_file",
        description=(
            "Read a file and return its contents with line numbers. "
            "Use offset and limit to read specific sections of large files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                },
                "offset": {
                    "type": "integer",
                    "description": "First line to read (1-indexed). Default 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Omit for full file.",
                },
            },
            "required": ["path"],
        },
    )(_read_file)

    registry.register(
        name="write_file",
        description="Write content to a file, creating it and any parent directories if needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    )(_write_file)
