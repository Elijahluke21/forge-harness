"""
One shell for every command Forge runs: the bash tool, checks, hooks, on_finish.

Models, checks.yaml files and hooks are written in POSIX sh. On Windows the
default shell is cmd.exe, where `test`, `sleep`, `$VAR` and `;` don't work, so
on Windows we run commands through Git Bash when it is installed.

Resolution order on Windows:
  1. FORGE_SHELL env var (path to a bash/sh executable)
  2. Git for Windows' bash in the usual install locations
  3. `bash` on PATH, unless it is System32\\bash.exe (that's WSL, a different
     filesystem and environment)
  4. fall back to cmd.exe (the old behaviour)
"""
from __future__ import annotations

import asyncio
import functools
import os
import shutil
import sys
from pathlib import Path


def _git_bash_candidates() -> list[Path]:
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
    ]
    return [Path(r) / "Git" / "bin" / "bash.exe" for r in roots if r]


@functools.lru_cache(maxsize=1)
def posix_shell() -> str | None:
    """Path to a POSIX shell to use on Windows, or None (use the platform default)."""
    if sys.platform != "win32":
        return None
    override = os.environ.get("FORGE_SHELL")
    if override:
        return override
    for p in _git_bash_candidates():
        if p.is_file():
            return str(p)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


async def spawn_shell(command: str, **kwargs) -> asyncio.subprocess.Process:
    """Start `command` in a POSIX shell (Git Bash on Windows when available)."""
    sh = posix_shell()
    if sh:
        return await asyncio.create_subprocess_exec(sh, "-c", command, **kwargs)
    return await asyncio.create_subprocess_shell(command, **kwargs)


__all__ = ["posix_shell", "spawn_shell"]
