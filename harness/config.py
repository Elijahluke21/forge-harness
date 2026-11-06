"""
Forge configuration.

Load order (later overrides earlier):
  1. ~/.forge/config.yaml   — user global config
  2. .forge.yaml            — project-level config (current directory)
  3. env var FORGE_CONFIG   — explicit path override

All fields have sensible defaults so a missing config file works fine.
Unknown keys are reported instead of silently ignored, so a typo like
`max_iteration:` doesn't quietly leave the default in place.
"""

import os
import sys
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
    base_url:         str   = ""          # Ollama/OpenAI-compatible endpoint override
    max_tokens:       int   = 8096
    context_window:   int   = 32_768
    compact_at:       float = 0.80
    budget_fraction:  float = 0.90
    stream:           bool  = True

    # Optional stronger model the autopilot escalates planning to when stuck
    escalation_backend: str = ""
    escalation_model:   str = ""

    # Agent behaviour
    max_depth:        int   = 4
    max_iterations:   int   = 20

    # Workspace confinement for file tools (and bash working directory).
    # Empty = current directory, unconfined.
    workspace:        str   = ""

    # Run budget — shared by the whole agent tree. 0 = unlimited.
    budget_max_tokens:     int   = 0
    budget_max_spawns:     int   = 0
    budget_max_tool_calls: int   = 0
    budget_max_minutes:    float = 0

    # Autopilot
    max_rounds:       int   = 8
    stall_rounds:     int   = 3
    max_parallel:     int   = 3
    on_finish:        str   = ""    # shell command; gets FORGE_STATUS, FORGE_RUN_DIR, FORGE_SUMMARY

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
            if not isinstance(data, dict):
                raise ValueError(f"{path}: config must be a YAML mapping")
            cfg.update(data)

    _merge(GLOBAL_CONFIG)
    _merge(LOCAL_CONFIG)

    env_path = os.environ.get("FORGE_CONFIG")
    if env_path:
        _merge(Path(env_path))

    # Flatten nested sections (e.g. permissions.require_confirm → require_confirm)
    if "permissions" in cfg:
        perm = cfg.pop("permissions") or {}
        cfg.setdefault("require_confirm", perm.get("require_confirm", []))
    if "budget" in cfg:
        budget = cfg.pop("budget") or {}
        for key in ("max_tokens", "max_spawns", "max_tool_calls", "max_minutes"):
            if key in budget:
                cfg.setdefault(f"budget_{key}", budget[key])
    if "autopilot" in cfg:
        ap = cfg.pop("autopilot") or {}
        for key in ("max_rounds", "stall_rounds", "max_parallel", "on_finish"):
            if key in ap:
                cfg.setdefault(key, ap[key])
    if "escalation" in cfg:
        esc = cfg.pop("escalation") or {}
        cfg.setdefault("escalation_backend", esc.get("backend", ""))
        cfg.setdefault("escalation_model", esc.get("model", ""))

    valid = set(ForgeConfig.__dataclass_fields__)
    unknown = sorted(set(cfg) - valid)
    if unknown:
        print(f"[forge] ignoring unknown config keys: {', '.join(unknown)}", file=sys.stderr)
    return ForgeConfig(**{k: v for k, v in cfg.items() if k in valid})
