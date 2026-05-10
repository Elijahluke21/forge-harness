import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from harness.agent import Agent, SpawnEvent
from harness.context import checkpoint
from harness.memory.distill import distill
from harness.prompt import build_system_prompt

console = Console()

COMMANDS = {
    "/exit":              "Quit Forge (auto-distills memory on exit).",
    "/quit":              "Quit Forge.",
    "/clear":             "Clear conversation history.",
    "/history":           "Show message count.",
    "/save [path]":       "Save conversation checkpoint.",
    "/load [path]":       "Load conversation checkpoint.",
    "/distill":           "Extract memorable facts from this conversation.",
    "/memory":            "Show all stored memories.",
    "/forget <key>":      "Delete a memory by key.",
    "/stats":             "Show session stats.",
    "/export [md|json]":  "Export conversation to file.",
    "/help":              "Show this help.",
}

BANNER = """
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

_BRANCH = "├─"
_LAST   = "└─"
_PIPE   = "│ "
_SPACE  = "  "

_SESSION_START = time.monotonic()


def _tree_prefix(depth: int) -> str:
    if depth == 0:
        return ""
    return (_PIPE + _SPACE) * (depth - 1) + _BRANCH + " "


def _print_banner(agent: Agent) -> None:
    console.print(BANNER, style="bold cyan", highlight=False)
    console.print(f"  model   : [bold]{agent.client.model}[/bold]")
    console.print(f"  tools   : [bold]{', '.join(agent.registry.names())}[/bold]")
    console.print(f"  type [bold]/help[/bold] for commands, [bold]Ctrl+D[/bold] to exit\n")


# ---------------------------------------------------------------------------
# Input box layout
#
#   [status info line]
#   ─────────────────────────────────   ← top rule
#   forge> user types here              ← prompt (cursor here while typing)
#   ─────────────────────────────────   ← bottom rule (visible while typing)
#   ◆ Thinking… 4s → ◆ Thought for 4s  ← persists above the response
#   response streams here…
#   [footer: elapsed · tokens · ctx]
# ---------------------------------------------------------------------------

def _uptime_str() -> str:
    total = int(time.monotonic() - _SESSION_START)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _build_status_line(agent: Agent) -> str:
    """Return the status info markup string (no newline)."""
    seg: list[str] = []
    seg.append(f"[bold cyan]{agent.client.model}[/bold cyan]")

    if agent.context_manager:
        cm = agent.context_manager
        pct = cm.usage_fraction
        col = "green" if pct < 0.70 else ("yellow" if pct < 0.85 else "red")
        bar = "█" * int(pct * 10) + "░" * (10 - int(pct * 10))
        seg.append(f"[{col}]ctx [{bar}] {pct:.0%}[/{col}]")

    if agent.session_stats:
        s = agent.session_stats
        parts: list[str] = []
        if s.agents_spawned:
            parts.append(f"{s.agents_spawned}↓")
        parts.append(f"{s.tool_calls}⚙")
        total_tok = s.input_tokens + s.output_tokens
        if total_tok:
            parts.append(f"{total_tok:,} tok")
        seg.append(f"[dim]{' · '.join(parts)}[/dim]")

    if agent.memory_store:
        n = len(agent.memory_store.all())
        if n:
            seg.append(f"[dim]{n}♦ mem[/dim]")

    seg.append(f"[dim]{_uptime_str()}[/dim]")
    return "  " + "  [dim]│[/dim]  ".join(seg)


async def _get_input(agent: Agent, prompt_str: str) -> str:
    """
    Read user input with both separator rules visible while typing.

    Prints the status info, top rule, prompt line, and bottom rule —
    then uses ANSI cursor-up to reposition the cursor on the prompt line
    so the bottom rule is already visible below before the user types.
    After Enter, advances past the bottom rule so subsequent output is clean.
    """
    width = console.width
    rule_plain = "─" * width

    # Status + top rule via Rich (handles markup / colour resets cleanly)
    console.print(_build_status_line(agent))
    console.rule(characters="─", style="dim")

    # Print the prompt text, a newline, then the bottom rule, then a newline.
    # This puts cursor at the start of a blank line below the bottom rule.
    sys.stdout.write(prompt_str + "\n")
    sys.stdout.write("\x1b[2m" + rule_plain + "\x1b[0m\n")

    # Move cursor back up 2 lines and right to end of prompt text,
    # so input() echoes on the prompt line with the bottom rule below.
    sys.stdout.write(f"\x1b[2A\x1b[{len(prompt_str)}C")
    sys.stdout.flush()

    loop = asyncio.get_event_loop()
    try:
        line = await loop.run_in_executor(None, lambda: input(""))
    except (EOFError, KeyboardInterrupt):
        raise

    # After Enter the cursor sits on the bottom rule line — skip past it.
    sys.stdout.write("\n")
    sys.stdout.flush()
    return line


# ---------------------------------------------------------------------------
# Streaming display
# ---------------------------------------------------------------------------

class _StreamRenderer:
    """
    Renders streaming text chunks to the terminal.
    Qwen3 wraps thinking in <think>...</think> — shown in dim italic.
    """
    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def chunk(self, text: str) -> None:
        self._buf += text
        while True:
            if not self._in_think:
                if "<think>" in self._buf:
                    before, _, rest = self._buf.partition("<think>")
                    if before:
                        console.print(before, end="", highlight=False)
                    console.print("[dim italic]", end="", highlight=False)
                    self._in_think = True
                    self._buf = rest
                else:
                    safe = self._buf if "<" not in self._buf else self._buf[:self._buf.rfind("<")]
                    if safe:
                        console.print(safe, end="", highlight=False)
                        self._buf = self._buf[len(safe):]
                    break
            else:
                if "</think>" in self._buf:
                    inside, _, rest = self._buf.partition("</think>")
                    if inside:
                        console.print(inside, end="", highlight=False)
                    console.print("[/dim italic]", end="", highlight=False)
                    self._in_think = False
                    self._buf = rest
                else:
                    safe = self._buf if "<" not in self._buf else self._buf[:self._buf.rfind("<")]
                    if safe:
                        console.print(safe, end="", highlight=False)
                        self._buf = self._buf[len(safe):]
                    break

    def flush(self) -> None:
        if self._buf:
            console.print(self._buf, end="", highlight=False)
            self._buf = ""
        if self._in_think:
            console.print("[/dim italic]", end="", highlight=False)
            self._in_think = False


# ---------------------------------------------------------------------------
# Thinking indicator — works for both streaming and non-streaming
# ---------------------------------------------------------------------------

async def _run_with_spinner(coro) -> str:
    """
    Non-streaming: live 'Thinking… Xs' counter while the model works,
    then a frozen 'Thought for Xs' header above the printed response.
    """
    start = time.monotonic()
    task = asyncio.ensure_future(coro)

    with Live(console=console, refresh_per_second=8, transient=True) as live:
        while not task.done():
            elapsed = time.monotonic() - start
            live.update(Text.assemble(
                ("  ◆ Thinking… ", "dim yellow"),
                (f"{elapsed:.0f}s", "bold dim yellow"),
            ))
            await asyncio.sleep(0.12)

    elapsed_thinking = time.monotonic() - start
    console.print(Text.assemble(
        ("  ◆ Thought for ", "dim yellow"),
        (f"{elapsed_thinking:.0f}s", "bold yellow"),
    ))
    console.print()
    return await task


async def _run_streaming(agent: Agent, user_input: str) -> str:
    """
    Streaming: show 'Thinking… Xs' until the first token arrives, then
    transition to live streaming output.

    Any chunks that arrive while the Live display is still active are buffered
    and flushed immediately after — nothing is lost.
    """
    start = time.monotonic()
    first_chunk_event = asyncio.Event()
    live_done = asyncio.Event()
    buffered: list[str] = []
    renderer = _StreamRenderer()

    def on_chunk(text: str) -> None:
        first_chunk_event.set()
        if not live_done.is_set():
            # Live display still active — buffer rather than printing now
            buffered.append(text)
        else:
            renderer.chunk(text)

    agent.on_stream_chunk = on_chunk
    task = asyncio.ensure_future(agent.run(user_input))

    # Live "Thinking…" counter until first token arrives (transient → erased on exit)
    with Live(console=console, refresh_per_second=8, transient=True) as live:
        while not first_chunk_event.is_set() and not task.done():
            elapsed = time.monotonic() - start
            live.update(Text.assemble(
                ("  ◆ Thinking… ", "dim yellow"),
                (f"{elapsed:.0f}s", "bold dim yellow"),
            ))
            await asyncio.sleep(0.12)

    # Freeze elapsed time, print a persistent "Thought for Xs" header.
    # This stays visible above the response while it streams.
    elapsed_thinking = time.monotonic() - start
    console.print(Text.assemble(
        ("  ◆ Thought for ", "dim yellow"),
        (f"{elapsed_thinking:.0f}s", "bold yellow"),
    ))
    console.print()

    # Unlock direct rendering and flush chunks buffered during the Live window
    live_done.set()
    for chunk in buffered:
        renderer.chunk(chunk)

    # Wait for completion; remaining tokens arrive via on_chunk → renderer.chunk
    answer = await task
    renderer.flush()
    console.print()
    return answer


# ---------------------------------------------------------------------------
# Tool + spawn callbacks
# ---------------------------------------------------------------------------

def on_tool(name: str, inputs: dict, result: str, depth: int = 0) -> None:
    indent = "  " * (depth + 1)

    if name == "__compact__":
        console.print(Text(f"{indent}◎  [context compacted] {result}", style="bold yellow"))
        return

    summary = repr(result.strip()[:72])
    arg_str = ", ".join(f"{k}={repr(str(v)[:36])}" for k, v in inputs.items())
    style = "dim cyan" if depth == 0 else "dim magenta"
    console.print(Text(f"{indent}⚙  {name}({arg_str}) → {summary}", style=style))


def on_spawn(event: SpawnEvent) -> None:
    prefix = _tree_prefix(event.depth)
    role_tag = f"[{event.role}] " if event.role else ""
    task_preview = event.task[:64] + ("…" if len(event.task) > 64 else "")

    if event.kind == "start":
        console.print(Text(f"{prefix}⬡ agent depth={event.depth} {role_tag}← {task_preview!r}",
                           style="bold yellow"))
    elif event.kind == "retry":
        console.print(Text(f"{prefix}↺ retrying depth={event.depth} {role_tag}",
                           style="bold orange3"))
    elif event.kind == "cached":
        console.print(Text(f"{prefix}⚡ cached depth={event.depth} {role_tag}← {task_preview!r}",
                           style="dim green"))
    else:
        calls = event.result.tool_calls_made if event.result else 0
        retried = " (retried)" if (event.result and event.result.retried) else ""
        console.print(Text(f"{prefix}✓ depth={event.depth} done ({calls} tool calls{retried})",
                           style="dim green"))


# ---------------------------------------------------------------------------
# Permissions  (6.3)
# ---------------------------------------------------------------------------

async def confirm_tool(name: str, inputs: dict) -> bool:
    preview = ", ".join(f"{k}={repr(str(v)[:60])}" for k, v in inputs.items())
    console.print(f"\n[yellow]⚠  Permission required:[/yellow] [bold]{name}[/bold]({preview})")
    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: input("  Allow? [y/N] ")
        )
        return answer.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ---------------------------------------------------------------------------
# Hooks  (6.4)
# ---------------------------------------------------------------------------

def run_hooks(hooks: list[dict], timing: str, tool_name: str) -> None:
    for hook in hooks:
        if hook.get("on") != timing:
            continue
        pattern = hook.get("tool", "*")
        if pattern != "*" and pattern != tool_name:
            continue
        cmd = hook.get("cmd", "")
        if cmd:
            try:
                subprocess.run(cmd, shell=True, check=False,
                               env={"FORGE_TOOL": tool_name})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Export  (6.6)
# ---------------------------------------------------------------------------

def _export_markdown(agent: Agent, path: str) -> None:
    lines = [f"# Forge Session Export\n"]
    for msg in agent.messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            lines.append(f"## {role.title()}\n\n{content}\n")
        elif isinstance(content, list):
            header_done = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text" and block["text"].strip():
                    if not header_done:
                        lines.append(f"## {role.title()}\n")
                        header_done = True
                    lines.append(f"{block['text']}\n")
                elif btype == "tool_use":
                    lines.append(f"\n**Tool:** `{block['name']}`\n```json\n{json.dumps(block['input'], indent=2)}\n```\n")
                elif btype == "tool_result":
                    val = block.get("content", "")
                    lines.append(f"\n**Result:**\n```\n{str(val)[:500]}\n```\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stats + distillation
# ---------------------------------------------------------------------------

def _print_stats(agent: Agent) -> None:
    if agent.session_stats:
        console.print(f"[dim]Session: {agent.session_stats.summary()}[/dim]")


async def _run_distill(agent: Agent) -> None:
    if not agent.memory_store:
        console.print("[dim]No memory store attached.[/dim]")
        return
    console.print("[dim]Distilling...[/dim]")
    written = await distill(agent.messages, agent.client, agent.memory_store)
    if written:
        console.print(f"[dim]Saved {len(written)} memor{'y' if len(written)==1 else 'ies'}: "
                      f"{', '.join(written)}[/dim]")
    else:
        console.print("[dim]Nothing new to remember.[/dim]")


def _print_footer(agent: Agent, elapsed: float) -> None:
    """Print the post-response summary line: elapsed · tokens · ctx bar."""
    elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m {elapsed % 60:.0f}s"
    parts: list[str] = [f"elapsed {elapsed_str}"]

    if agent.session_stats:
        s = agent.session_stats
        total_tok = s.input_tokens + s.output_tokens
        if total_tok:
            parts.append(f"{total_tok:,} tok")
        if s.cache_hits:
            parts.append(f"{s.cache_hits} cached")

    if agent.context_manager:
        cm = agent.context_manager
        pct = cm.usage_fraction
        col = "green" if pct < 0.70 else ("yellow" if pct < 0.85 else "red")
        filled = int(pct * 20)
        bar = "█" * filled + "░" * (20 - filled)
        parts.append(f"[{col}]ctx [{bar}] {cm.usage_summary}[/{col}]")

    console.print(Text.from_markup("  [dim]" + "  ·  ".join(parts) + "[/dim]"))


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def _handle_command(raw: str, agent: Agent) -> bool:
    parts = raw.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        console.print("\n[dim]Goodbye.[/dim]")
        return True
    if cmd == "/clear":
        agent.messages.clear()
        console.print("[dim]History cleared.[/dim]")
    elif cmd == "/history":
        console.print(f"[dim]{len(agent.messages)} messages in history.[/dim]")
    elif cmd == "/save":
        path = arg or "forge_checkpoint.json"
        checkpoint.save(agent.messages, path, agent.system)
        console.print(f"[dim]Checkpoint saved to {path}[/dim]")
    elif cmd == "/load":
        path = arg or "forge_checkpoint.json"
        try:
            msgs, _ = checkpoint.load(path)
            agent.messages.clear()
            agent.messages.extend(msgs)
            console.print(f"[dim]Loaded {len(msgs)} messages from {path}[/dim]")
        except FileNotFoundError:
            console.print(f"[red]File not found:[/red] {path}")
    elif cmd == "/memory":
        if not agent.memory_store:
            console.print("[dim]No memory store attached.[/dim]")
        else:
            entries = agent.memory_store.all()
            if not entries:
                console.print("[dim]Memory is empty.[/dim]")
            else:
                console.print(Rule("Memory", style="dim"))
                for e in entries:
                    tag_str = f" [dim][{', '.join(e.tags)}][/dim]" if e.tags else ""
                    console.print(f"  [bold cyan]{e.key}[/bold cyan]{tag_str}: {e.content}")
                console.print(Rule(style="dim"))
    elif cmd == "/forget":
        if not agent.memory_store:
            console.print("[dim]No memory store.[/dim]")
        elif not arg:
            console.print("[red]Usage:[/red] /forget <key>")
        elif agent.memory_store.delete(arg):
            console.print(f"[dim]Forgot: {arg}[/dim]")
        else:
            console.print(f"[red]No memory:[/red] {arg}")
    elif cmd == "/export":
        fmt = arg.strip().lower() or "md"
        if fmt in ("md", "markdown"):
            path = "forge_export.md"
            _export_markdown(agent, path)
            console.print(f"[dim]Exported to {path}[/dim]")
        elif fmt == "json":
            path = "forge_export.json"
            checkpoint.save(agent.messages, path, agent.system)
            console.print(f"[dim]Exported to {path}[/dim]")
        else:
            console.print(f"[red]Unknown format:[/red] {fmt}. Use 'md' or 'json'.")
    elif cmd == "/help":
        console.print(Rule("Commands", style="dim"))
        for c, desc in COMMANDS.items():
            console.print(f"  [bold cyan]{c:<24}[/bold cyan] {desc}")
        console.print(Rule(style="dim"))
    else:
        console.print(f"[red]Unknown command:[/red] {cmd}  (type /help)")
    return False


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

async def repl(agent: Agent, stream: bool = True, hooks: list[dict] | None = None) -> None:
    _print_banner(agent)
    _hooks = hooks or []

    if _hooks:
        orig_execute = agent.registry.execute

        async def hooked_execute(name: str, inputs: dict) -> str:
            run_hooks(_hooks, "pre", name)
            result = await orig_execute(name, inputs)
            run_hooks(_hooks, "post", name)
            return result

        agent.registry.execute = hooked_execute  # type: ignore

    while True:
        # ── boxed input: status · top rule · prompt · bottom rule ───────────
        try:
            user_input = await _get_input(agent, "forge> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            parts_split = user_input.strip().split(None, 1)
            first = parts_split[0].lower()
            if first == "/distill":
                await _run_distill(agent)
                continue
            if first == "/stats":
                _print_stats(agent)
                continue
            if _handle_command(user_input, agent):
                _print_stats(agent)
                if agent.memory_store and agent.messages:
                    console.print("[dim]Distilling session memories...[/dim]")
                    await _run_distill(agent)
                break
            continue

        start_time = time.monotonic()

        # ── run the agent ────────────────────────────────────────────────────
        if stream:
            try:
                await _run_streaming(agent, user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                agent.on_stream_chunk = None
                if agent.messages and agent.messages[-1]["role"] == "user":
                    agent.messages.pop()
                continue
            finally:
                agent.on_stream_chunk = None
        else:
            try:
                answer = await _run_with_spinner(agent.run(user_input))
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                if agent.messages and agent.messages[-1]["role"] == "user":
                    agent.messages.pop()
                continue
            console.print()
            console.print(Markdown(answer))
            console.print()

        # ── post-response footer ─────────────────────────────────────────────
        elapsed = time.monotonic() - start_time
        _print_footer(agent, elapsed)
        console.print()
