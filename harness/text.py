"""Small text helpers shared across the harness."""

import json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """
    Remove Qwen-style <think>...</think> reasoning blocks.

    Reasoning models served through Ollama often put their thinking inline in
    the content. Anything that parses model output (plans, distillation,
    summaries) must strip it first or JSON parsing fails.
    An unterminated <think> (output cut off mid-thought) drops everything after it.
    """
    text = _THINK_RE.sub("", text)
    lower = text.lower()
    if "<think>" in lower:
        text = text[: lower.index("<think>")]
    return text.strip()


def extract_json(text: str):
    """
    Parse JSON from model output, tolerating think blocks, markdown fences,
    and leading/trailing prose. Raises ValueError if nothing parses.
    """
    text = strip_think(text)

    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost [...] or {...} span.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")


def tail(text: str, max_chars: int) -> str:
    """Keep the end of `text` — build/test failures are usually at the bottom."""
    if len(text) <= max_chars:
        return text
    return f"[... {len(text) - max_chars:,} chars cut ...]\n" + text[-max_chars:]
