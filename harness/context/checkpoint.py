"""Conversation checkpointing — save and restore message history to/from disk."""
import json
from datetime import datetime
from pathlib import Path


def save(messages: list[dict], path: str, system: str = "") -> None:
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": system,
        "messages": messages,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(path: str) -> tuple[list[dict], str]:
    """Returns (messages, system)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["messages"], data.get("system", "")
