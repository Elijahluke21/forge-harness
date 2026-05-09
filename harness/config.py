"""
Forge configuration.

Load order (later overrides earlier):
  1. ~/.forge/config.yaml   — user global config
  2. .forge.yaml            — project-level config (current directory)
  3. env var FORGE_CONFIG   — explicit path override

All fields have sensible defaults so a missing config file works fine.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GLOBAL_CONFIG = Path.home() / ".forge" / "config.yaml"
LOCAL_CONFIG  = Path(".forge.yaml")


@dataclass
class ForgeConfig:
    # Model / backend
    backend:          str   = "ollama"
    model:            str   = "qwen3.6:27b"
    max_tokens:       int   = 8096
    context_window:   int   = 32_768
    compact_at:       float = 0.80
    budget_fraction:  float = 0.90
    stream:           bool  = True

    # Agent behaviour
    max_depth:        int   = 4
    max_iterations:   int   = 20

    # Paths
    memory_path:      str   = str(Path.home() / ".forge" / "memory.json")

    # Permissions — tools that require user confirmation before running
    require_confirm:  list[str] = field(default_factory=list)

    # Hooks — shell commands run before/after every tool call
    # Each entry: {"on": "pre"|"post", "tool": "*"|"<name>", "cmd": "..."}
    hooks:            list[dict] = field(default_factory=list)


def load_config() -> ForgeConfig:
    cfg: dict = {}

    def _merge(path: Path) -> None:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            cfg.update(data)

    _merge(GLOBAL_CONFIG)
    _merge(LOCAL_CONFIG)

    env_path = os.environ.get("FORGE_CONFIG")
    if env_path:
        _merge(Path(env_path))

    # Flatten nested keys (e.g. permissions.require_confirm → require_confirm)
    if "permissions" in cfg:
        perm = cfg.pop("permissions")
        cfg.setdefault("require_confirm", perm.get("require_confirm", []))

    valid = {f.name for f in ForgeConfig.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in cfg.items() if k in valid}
    return ForgeConfig(**filtered)
