import os
import tempfile

from harness.tools.registry import ToolRegistry
from harness.tools.workspace import OutsideWorkspace, Workspace

MAX_READ_CHARS = 8_000


def _atomic_write(path: str, content: str) -> None:
    """Write via temp file + rename so a crash never leaves a half-written file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".forge-tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if os.path.exists(path):
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def register(registry: ToolRegistry, workspace: Workspace | None = None) -> None:
    ws = workspace or Workspace()

    async def _read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
        try:
            resolved = ws.resolve(path)
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OutsideWorkspace as e:
            return f"[error] {e}"
        except FileNotFoundError:
            return f"[error] file not found: {path}"
        except IsADirectoryError:
            return f"[error] {path} is a directory — use bash `ls` to list it"
        except OSError as e:
            return f"[error] could not read {path}: {e}"

        start = max(0, int(offset) - 1)
        end = start + int(limit) if limit is not None else len(lines)
        selected = lines[start:end]

        numbered = "".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        if len(numbered) > MAX_READ_CHARS:
            cut = numbered[:MAX_READ_CHARS]
            shown = cut.count("\n")
            numbered = (
                cut + f"\n[... truncated at line {start + shown} of {len(lines)} — "
                f"use offset={start + shown + 1} to read more]"
            )
        return numbered or "[empty file]"

    async def _write_file(path: str, content: str) -> str:
        try:
            resolved = ws.resolve(path)
            _atomic_write(str(resolved), content)
            return f"wrote {len(content)} chars to {resolved}"
        except OutsideWorkspace as e:
            return f"[error] {e}"
        except OSError as e:
            return f"[error] could not write {path}: {e}"

    async def _edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        try:
            resolved = ws.resolve(path)
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()
        except OutsideWorkspace as e:
            return f"[error] {e}"
        except FileNotFoundError:
            return f"[error] file not found: {path} — use write_file to create it"
        except OSError as e:
            return f"[error] could not read {path}: {e}"

        if not old_text:
            return "[error] old_text must not be empty"
        count = content.count(old_text)
        if count == 0:
            return "[error] old_text not found — read_file the exact current text and try again"
        if count > 1 and not replace_all:
            return (
                f"[error] old_text appears {count} times — include more surrounding "
                "lines to make it unique, or set replace_all=true"
            )
        updated = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
        try:
            _atomic_write(str(resolved), updated)
        except OSError as e:
            return f"[error] could not write {path}: {e}"
        return f"edited {resolved} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''})"

    registry.register(
        name="read_file",
        description=(
            "Read a file and return its contents with line numbers. "
            "Use offset and limit to read specific sections of large files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "Path to the file (relative to the workspace)."},
                "offset": {"type": "integer", "description": "First line to read (1-indexed). Default 1."},
                "limit":  {"type": "integer", "description": "Maximum number of lines to read. Omit for full file."},
            },
            "required": ["path"],
        },
    )(_read_file)

    registry.register(
        name="write_file",
        description=(
            "Create a new file, or completely replace one, with the given content. "
            "For changing part of an existing file, prefer edit_file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Path to write (relative to the workspace)."},
                "content": {"type": "string", "description": "Full content to write to the file."},
            },
            "required": ["path", "content"],
        },
    )(_write_file)

    registry.register(
        name="edit_file",
        description=(
            "Replace an exact snippet of text in an existing file. old_text must match the "
            "file exactly (including indentation) and be unique unless replace_all is true. "
            "Safer and cheaper than rewriting the whole file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path":        {"type": "string"},
                "old_text":    {"type": "string", "description": "Exact text currently in the file."},
                "new_text":    {"type": "string", "description": "Text to put in its place."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
            },
            "required": ["path", "old_text", "new_text"],
        },
    )(_edit_file)
