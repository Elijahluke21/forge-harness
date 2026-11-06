import hashlib
import json
import tempfile
from pathlib import Path

from harness.client import BaseClient, Response
from harness.text import strip_think

# Default context windows
# These are what the harness budgets for, not each model's maximum: a local
# 27B model's usable window is limited by VRAM for the KV cache.
CONTEXT_WINDOWS = {
    "qwen3.6:27b": 32_768,
    "qwen3.8:27b": 32_768,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-haiku-4-5": 200_000,
    "default": 32_768,
}

COMPACT_AT   = 0.85   # summarize when context hits this fraction
WARN_AT      = 0.70   # log a warning at this fraction
OFFLOAD_CHARS = 3_000  # tool results larger than this go to a temp file


class ContextManager:
    """
    Tracks token usage, offloads large tool results to disk, and compacts
    the message history via summarization when the context budget fills up.

    Uses actual token counts from API responses as the primary signal.
    No external tokenizer library needed.
    """

    def __init__(
        self,
        model: str = "default",
        context_window: int | None = None,
        compact_at: float = COMPACT_AT,
        offload_chars: int = OFFLOAD_CHARS,
        budget_fraction: float = 1.0,
        offload_dir: str | Path | None = None,
    ):
        base_window = context_window or CONTEXT_WINDOWS.get(model, CONTEXT_WINDOWS["default"])
        self.base_window = base_window
        self.context_window = int(base_window * budget_fraction)
        self.compact_at = compact_at
        self.offload_chars = offload_chars

        self.last_input_tokens: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.compactions: int = 0

        # Offloaded results must live somewhere read_file is allowed to reach;
        # with a confined workspace, pass a directory inside it.
        if offload_dir:
            self._offload_dir = Path(offload_dir)
            self._offload_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._offload_dir = Path(tempfile.mkdtemp(prefix="forge_ctx_"))
        self.offload_dir = self._offload_dir

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def record(self, response: Response) -> None:
        """Call after every API response to keep token counts current."""
        self.last_input_tokens = response.input_tokens
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

    @property
    def usage_fraction(self) -> float:
        """Fraction of context window used (based on last API call)."""
        if self.context_window == 0:
            return 0.0
        return self.last_input_tokens / self.context_window

    @property
    def usage_summary(self) -> str:
        pct = int(self.usage_fraction * 100)
        k = self.last_input_tokens / 1000
        cap_k = self.context_window / 1000
        return f"{k:.1f}k/{cap_k:.0f}k ({pct}%)"

    # ------------------------------------------------------------------
    # Large result offloading  (3.3)
    # ------------------------------------------------------------------

    def offload_if_large(self, content: str) -> str:
        """
        If content exceeds the threshold, write it to a temp file and return
        a compact pointer the model can act on with read_file.
        """
        if len(content) <= self.offload_chars:
            return content

        fname = f"result_{hashlib.md5(content.encode()).hexdigest()[:12]}.txt"
        path = self._offload_dir / fname
        path.write_text(content, encoding="utf-8")

        preview = content[:400].rstrip()
        return (
            f"[Output offloaded to conserve context — {len(content):,} chars total]\n"
            f"File: {path}\n"
            f"Preview:\n{preview}\n"
            f"...\n"
            f"Use read_file with offset/limit to retrieve specific sections."
        )

    # ------------------------------------------------------------------
    # Compaction  (3.2 + 3.4)
    # ------------------------------------------------------------------

    async def maybe_compact(
        self,
        messages: list[dict],
        client: BaseClient,
        system: str = "",
    ) -> str | None:
        """
        Check budget; compact if over threshold.
        Returns a status string if compaction happened, None otherwise.
        """
        if self.usage_fraction < self.compact_at:
            return None
        return await self._compact(messages, client)

    async def _compact(self, messages: list[dict], client: BaseClient) -> str:
        """
        Summarize the middle of the message history.

        Keeps the first message (the task) and a tail of recent turns, and
        folds a summary of everything between into the first message.

        The cut points must respect tool pairing: every assistant `tool_use`
        needs its `tool_result` in the very next user message, or the API
        rejects the history. So the tail always starts on an assistant turn,
        and the summary is merged into the first user message rather than
        inserted as a second consecutive user message.
        """
        if len(messages) <= 8:
            return "context high but history too short to compact"

        first = messages[0]
        tail_start = max(1, len(messages) - 6)
        while tail_start < len(messages) and messages[tail_start]["role"] != "assistant":
            tail_start += 1
        if tail_start >= len(messages) - 1 or tail_start <= 1:
            return "context high but no safe point to compact"

        mid = messages[1:tail_start]
        tail = messages[tail_start:]

        resp = await client.send(
            messages=[{"role": "user", "content": self._messages_to_transcript(mid)}],
            system=(
                "Summarize this conversation segment precisely. "
                "Preserve: all files created/modified (with paths), commands run, "
                "command outputs that matter, decisions made, and any facts established. "
                "Be concise. Output the summary only — no preamble."
            ),
            max_tokens=700,
        )
        summary = strip_think(resp.text) or "(summary unavailable)"

        note = f"[Context compacted — {len(mid)} earlier messages summarized]\n\n{summary}"
        if isinstance(first.get("content"), str):
            new_first = {"role": "user", "content": f"{first['content']}\n\n{note}"}
        else:
            new_first = {
                "role": "user",
                "content": list(first["content"]) + [{"type": "text", "text": note}],
            }

        messages.clear()
        messages.extend([new_first] + tail)
        self.compactions += 1
        return f"compacted {len(mid)} messages into summary (compaction #{self.compactions})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_transcript(messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content[:800]}")
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        lines.append(f"{role}: {block['text'][:800]}")
                    elif btype == "tool_use":
                        args = json.dumps(block.get("input", {}))[:300]
                        lines.append(f"{role} [tool: {block['name']}] {args}")
                    elif btype == "tool_result":
                        val = str(block.get("content", ""))[:400]
                        lines.append(f"TOOL_RESULT: {val}")
        return "\n".join(lines)
