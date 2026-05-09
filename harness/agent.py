import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass

from harness.client import DEFAULT_MAX_TOKENS, BaseClient, TextChunkFn
from harness.context.manager import ContextManager
from harness.loop import run
from harness.memory.store import MemoryStore
from harness.roles import get_role
from harness.session import ResultCache, SessionStats
from harness.tools.registry import Tool, ToolRegistry

MAX_DEPTH = 4
MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Result & events
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    task: str
    answer: str
    tool_calls_made: int
    depth: int
    cached: bool = False
    retried: bool = False

    def to_str(self) -> str:
        return json.dumps({
            "answer": self.answer,
            "tool_calls_made": self.tool_calls_made,
            "depth": self.depth,
            "cached": self.cached,
        }, indent=2)


@dataclass
class SpawnEvent:
    kind: str          # "start" | "end" | "retry" | "cached"
    task: str
    depth: int
    role: str = ""
    result: AgentResult | None = None


ToolCallbackFn = Callable[[str, dict, str, int], None]
SpawnCallbackFn = Callable[[SpawnEvent], None]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    A self-contained agent: owns its client, registry, system prompt,
    and message history.

    Orchestrators and sub-agents are the same class. What differs:
      - depth          : 0 = orchestrator; increments with each spawn
      - registry       : sub-agents receive a scoped tool set
      - system         : sub-agents receive a focused system prompt
    """

    def __init__(
        self,
        client: BaseClient,
        registry: ToolRegistry,
        system: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_iterations: int = 20,
        depth: int = 0,
        on_tool_call: ToolCallbackFn | None = None,
        on_spawn: SpawnCallbackFn | None = None,
        context_manager: ContextManager | None = None,
        memory_store: MemoryStore | None = None,
        session_stats: SessionStats | None = None,
        result_cache: ResultCache | None = None,
        on_stream_chunk: TextChunkFn | None = None,
    ):
        self.client = client
        self.registry = registry
        self.system = system
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.depth = depth
        self.on_tool_call = on_tool_call
        self.on_spawn = on_spawn
        self.context_manager = context_manager
        self.memory_store = memory_store
        self.session_stats = session_stats
        self.result_cache = result_cache
        self.on_stream_chunk = on_stream_chunk
        self.messages: list[dict] = []
        self._tool_call_count = 0

    def _tracking_callback(self, name: str, inputs: dict, result: str) -> None:
        self._tool_call_count += 1
        if self.session_stats and name not in ("__compact__",):
            self.session_stats.record_tool()
        if self.on_tool_call:
            self.on_tool_call(name, inputs, result, self.depth)

    def _build_system(self) -> str:
        parts = [self.system]
        if self.memory_store and len(self.memory_store) > 0:
            parts.append(self.memory_store.format_for_prompt())
        return "\n\n".join(filter(None, parts))

    async def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        return await run(
            messages=self.messages,
            client=self.client,
            registry=self.registry,
            system=self._build_system(),
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
            on_tool_call=self._tracking_callback,
            context_manager=self.context_manager,
            session_stats=self.session_stats,
            on_stream_chunk=self.on_stream_chunk,
        )

    async def run_subtask(self, task: str) -> AgentResult:
        self._tool_call_count = 0
        answer = await self.run(task)
        return AgentResult(
            task=task,
            answer=answer,
            tool_calls_made=self._tool_call_count,
            depth=self.depth,
        )


# ---------------------------------------------------------------------------
# Spawn tool factory  (recursive, with cache + retry + verify)
# ---------------------------------------------------------------------------

def _scoped_registry(worker_registry: ToolRegistry, tool_names: list[str]) -> ToolRegistry:
    scoped = ToolRegistry()
    for name in tool_names:
        tool = worker_registry.get(name)
        if tool:
            scoped.add(tool)
    return scoped


def _build_child_registry(
    child: "Agent",
    worker_registry: ToolRegistry,
    role_tools: list[str] | None,
) -> ToolRegistry:
    base = (
        _scoped_registry(worker_registry, role_tools)
        if role_tools is not None
        else worker_registry
    )
    if child.depth >= MAX_DEPTH:
        return base
    registry = ToolRegistry()
    for tool in base._tools.values():
        registry.add(tool)
    registry.add(make_spawn_tool(child, worker_registry))
    return registry


def _looks_like_error(answer: str) -> bool:
    low = answer.strip().lower()
    return low.startswith("[error]") or low.startswith("error:")


def make_spawn_tool(parent: "Agent", worker_registry: ToolRegistry) -> Tool:

    async def spawn_agent(
        task: str,
        context: str = "",
        role: str = "",
        verify: bool = False,
    ) -> str:
        if parent.depth >= MAX_DEPTH:
            return f"[error] max agent depth ({MAX_DEPTH}) reached — handle this directly"

        # ── Cache hit ──────────────────────────────────────────────────────
        if parent.result_cache:
            cached = parent.result_cache.get(task, context)
            if cached:
                if parent.session_stats:
                    parent.session_stats.record_cache_hit()
                if parent.on_spawn:
                    parent.on_spawn(SpawnEvent(kind="cached", task=task, depth=parent.depth + 1, role=role))
                return cached

        # ── Build child ────────────────────────────────────────────────────
        role_obj = get_role(role) if role else None
        role_system = role_obj.system if role_obj else (
            "You are a focused sub-agent. Complete the assigned task using your tools. "
            "Return only the result — no preamble."
        )
        role_tools: list[str] | None = role_obj.tools if role_obj else None

        child_system = "\n\n".join(filter(None, [
            role_system,
            f"Context from orchestrator:\n{context}" if context else "",
        ]))

        child_ctx = ContextManager(
            model=parent.client.model,
            compact_at=0.80,
            budget_fraction=0.85,
        ) if parent.context_manager else None

        def _make_child(system_override: str | None = None) -> Agent:
            return Agent(
                client=parent.client,
                registry=worker_registry,
                system=system_override or child_system,
                max_tokens=parent.max_tokens,
                max_iterations=parent.max_iterations,
                depth=parent.depth + 1,
                on_tool_call=parent.on_tool_call,
                on_spawn=parent.on_spawn,
                context_manager=child_ctx,
                memory_store=parent.memory_store,
                session_stats=parent.session_stats,
                result_cache=parent.result_cache,
            )

        # ── First attempt ─────────────────────────────────────────────────
        if parent.on_spawn:
            parent.on_spawn(SpawnEvent(kind="start", task=task, depth=parent.depth + 1, role=role))
        if parent.session_stats:
            parent.session_stats.record_spawn()

        child = _make_child()
        child.registry = _build_child_registry(child, worker_registry, role_tools)
        result = await child.run_subtask(task)

        # ── Retry on error ────────────────────────────────────────────────
        if _looks_like_error(result.answer):
            if parent.on_spawn:
                parent.on_spawn(SpawnEvent(kind="retry", task=task, depth=parent.depth + 1, role=role))
            if parent.session_stats:
                parent.session_stats.record_retry()

            retry_context = (
                f"{context}\n\n"
                f"IMPORTANT: A previous attempt at this task failed with:\n{result.answer}\n"
                f"Try a different approach."
            ).strip()
            retry_system = "\n\n".join(filter(None, [
                role_system,
                f"Context from orchestrator:\n{retry_context}",
            ]))
            retry_child = _make_child(retry_system)
            retry_child.registry = _build_child_registry(retry_child, worker_registry, role_tools)
            result = await retry_child.run_subtask(task)
            result.retried = True

        # ── Optional verification ─────────────────────────────────────────
        if verify and not _looks_like_error(result.answer):
            verifier_system = (
                "You are a verification sub-agent. "
                "Check the provided result for correctness and completeness. "
                "If correct, reply: 'VERIFIED: <brief confirmation>'. "
                "If not, reply: 'FAILED: <what is wrong>'."
            )
            verifier_context = (
                f"Original task: {task}\n\n"
                f"Result to verify:\n{result.answer}"
            )
            ver_child = _make_child(
                "\n\n".join([verifier_system, f"Context:\n{verifier_context}"])
            )
            ver_child.registry = _build_child_registry(ver_child, worker_registry, ["bash", "read_file"])
            ver_result = await ver_child.run_subtask(f"Verify: {task}")
            result.answer += f"\n\n[Verification] {ver_result.answer}"

        # ── Finish ────────────────────────────────────────────────────────
        if parent.on_spawn:
            parent.on_spawn(SpawnEvent(kind="end", task=task, depth=parent.depth + 1, role=role, result=result))

        result_str = result.to_str()

        # Cache successful results
        if parent.result_cache and not _looks_like_error(result.answer):
            parent.result_cache.put(task, context, result_str)

        return result_str

    return Tool(
        name="spawn_agent",
        description=(
            "Spawn an isolated sub-agent with its own context window to complete a task.\n"
            "Use this to delegate focused units of work, run tasks in parallel, or keep "
            "orchestrator context clean.\n\n"
            "Available roles: planner, reader, executor, writer, synthesizer, verifier.\n"
            "Set verify=true to automatically run a verifier after the main agent.\n"
            "Results are cached within a session — identical task+context combos run once.\n"
            "Returns JSON: {answer, tool_calls_made, depth, cached}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task":    {"type": "string", "description": "Clear, self-contained description of what the sub-agent must do."},
                "context": {"type": "string", "description": "Background context: paths, prior results, constraints."},
                "role":    {"type": "string", "description": "Role preset: planner, reader, executor, writer, synthesizer, verifier."},
                "verify":  {"type": "boolean", "description": "If true, auto-run a verifier after the main agent completes."},
            },
            "required": ["task"],
        },
        executor=spawn_agent,
    )
