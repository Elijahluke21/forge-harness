"""
Workspace confinement for file and shell tools.

With `root` set, file tools refuse any path that resolves outside it (including
through `..` or symlinks) and bash runs with `root` as its working directory.
With `root` unset the tools behave as before (current directory, no limits).

This is a guard against an agent wandering, not a security sandbox: bash can
still `cd /` and do anything your user account can. Real isolation needs a
container or a separate OS user (see PLAN.md, 7.6).
"""

from pathlib import Path


class OutsideWorkspace(ValueError):
    pass


class Workspace:
    def __init__(self, root: str | Path | None = None):
        self.root: Path | None = Path(root).expanduser().resolve() if root else None

    @property
    def cwd(self) -> Path:
        return self.root or Path.cwd()

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        p = p.resolve()
        if self.root and p != self.root and self.root not in p.parents:
            raise OutsideWorkspace(f"path {path!r} is outside the workspace {self.root}")
        return p
