import json
from collections.abc import Callable
from dataclasses import dataclass

from harness.budget import HarnessAbort
from harness.client import DEFAULT_MAX_TOKENS, BaseClient, TextChunkFn
from harness.context.manager import ContextManager
from harness.loop import INCOMPLETE_PREFIX, run
from harness.memory.store import MemoryStore
from harness.roles import CACHEABLE_ROLES, ROLES, get_role
from harness.session import MUTATING_TOOLS, ResultCache, SessionStats
from harness.tools.registry import Tool, ToolRegistry

MAX_DEPTH = 4   # default; Agent.max_depth overrides (config: max_depth)


# ---------------------------------------------------------------------------
# Result & events
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    task: str
    answer: str
    tool_calls_made: int
    depth: int
    status: str = "ok"            # ok | error | incomplete | verify_failed
    cached: bool = False
    retried: bool = False
    verification: str = ""

    def to_str(self) -> str:
        data = {
            "status": self.status,
            "answer": self.answer,
            "tool_calls_made": self.tool_calls_made,
            "depth": self.depth,
        }
        if self.cached:
            data["cached"] = True
        if self.retried:
            data["retried"] = True
        if self.verification:
            data["verification"] = self.verification
        return json.dumps(data, indent=2)


@dataclass
class SpawnEvent:
    kind: str          # "start" | "end" | "retry" | "cached"
    task: str
    depth: int
    role: str = ""
    result: AgentResult | None = None


ToolCallbackFn = Callable[[str, dict, str, int], None]
SpawnCallbackFn = Callable[[SpawnEvent], None]


def classify(answer: str) -> str:
    low = answer.strip().lower()
    if low.startswith("[error]") or low.startswith("error:"):
        return "error"
    if low.startswith(INCOMPLETE_PREFIX):
        return "incomplete"
    return "ok"


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
        max_depth: int = MAX_DEPTH,
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
        self.max_depth = max_depth
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
        if name == "__compact__":
            if self.on_tool_call:
                self.on_tool_call(name, inputs, result, self.depth)
            return
        self._tool_call_count += 1
        if self.session_stats:
            self.session_stats.record_tool()
        if self.result_cache and name in MUTATING_TOOLS:
            self.result_cache.invalidate()
        if self.on_tool_call:
            self.on_tool_call(name, inputs, result, self.depth)

    def _build_system(self, task: str = "") -> str:
        parts = [self.system]
        if self.memory_store and len(self.memory_store) > 0:
            parts.append(self.memory_store.format_for_prompt(query=task))
        return "\n\n".join(filter(None, parts))

    async def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        return await run(
            messages=self.messages,
            client=self.client,
            registry=self.registry,
            system=self._build_system(task),
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
            status=classify(answer),
        )


# ---------------------------------------------------------------------------
# Spawning  (recursive, with cache + retry + verify)
# ---------------------------------------------------------------------------

DEFAULT_SUBAGENT_SYSTEM = (
    "You are a focused sub-agent. Complete the assigned task using your tools. "
    "Return only the result — no preamble."
)

VERIFIER_SYSTEM = (
    "You are a verification sub-agent. Check the provided result for correctness and "
    "completeness — run commands or read files to check rather than trusting the claim. "
    "Your reply MUST start with 'VERIFIED:' or 'FAILED:' followed by a brief reason."
)


def _child_registry(child: "Agent", worker_registry: ToolRegistry, tools: list[str] | None) -> ToolRegistry:
    # subset() carries permissions and hooks over — a fresh ToolRegistry would
    # silently drop require_confirm for every role-scoped sub-agent.
    registry = worker_registry.subset(tools)
    # Tool-less roles (planner, synthesizer) only think; giving them spawn_agent
    # would let them start doing the work they were asked to plan.
    if child.depth < child.max_depth and tools != []:
        registry.add(make_spawn_tool(child, worker_registry))
    return registry


async def spawn_subagent(
    parent: "Agent",
    worker_registry: ToolRegistry,
    task: str,
    context: str = "",
    role: str = "",
    verify: bool = False,
) -> AgentResult:
    """
    Run one sub-agent under `parent`. Used by the spawn_agent tool and by the
    plan executor. Raises HarnessAbort if the run budget is spent.
    """
    depth = parent.depth + 1
    if parent.depth >= parent.max_depth:
        return AgentResult(task, f"[error] max agent depth ({parent.max_depth}) reached — handle this directly",
                           0, depth, status="error")

    role_obj = get_role(role) if role else None
    if role and role_obj is None:
        return AgentResult(task, f"[error] unknown role {role!r}. Available: {', '.join(ROLES)}",
                           0, depth, status="error")

    cacheable = bool(parent.result_cache) and role_obj is not None \
        and role_obj.name in CACHEABLE_ROLES and not verify
    if cacheable:
        cached = parent.result_cache.get(task, context, role_obj.name)
        if cached:
            if parent.session_stats:
                parent.session_stats.record_cache_hit()
            if parent.on_spawn:
                parent.on_spawn(SpawnEvent(kind="cached", task=task, depth=depth, role=role))
            result = AgentResult(**{k: v for k, v in json.loads(cached).items()
                                    if k in AgentResult.__dataclass_fields__}, task=task)
            result.cached = True
            return result

    role_system = role_obj.system if role_obj else DEFAULT_SUBAGENT_SYSTEM
    role_tools = role_obj.tools if role_obj else None

    def make_child(system: str, tools: list[str] | None) -> Agent:
        child_ctx = None
        if parent.context_manager:
            pc = parent.context_manager
            child_ctx = ContextManager(
                context_window=pc.base_window,
                compact_at=pc.compact_at,
                budget_fraction=0.85,
                offload_dir=pc.offload_dir,
            )
        child = Agent(
            client=parent.client,
            registry=worker_registry,
            system=system,
            max_tokens=parent.max_tokens,
            max_iterations=parent.max_iterations,
            depth=depth,
            max_depth=parent.max_depth,
            on_tool_call=parent.on_tool_call,
            on_spawn=parent.on_spawn,
            context_manager=child_ctx,
            memory_store=parent.memory_store,
            session_stats=parent.session_stats,
            result_cache=parent.result_cache,
        )
        child.registry = _child_registry(child, worker_registry, tools)
        return child

    def with_context(extra: str) -> str:
        return "\n\n".join(filter(None, [role_system, f"Context from orchestrator:\n{extra}" if extra else ""]))

    if parent.session_stats:
        parent.session_stats.record_spawn()
        parent.session_stats.check_budget()
    if parent.on_spawn:
        parent.on_spawn(SpawnEvent(kind="start", task=task, depth=depth, role=role))

    result = await make_child(with_context(context), role_tools).run_subtask(task)

    # Retry once on a hard error, with the failure as context. "incomplete"
    # (ran out of iterations) is not retried: same budget, same outcome.
    if result.status == "error":
        if parent.on_spawn:
            parent.on_spawn(SpawnEvent(kind="retry", task=task, depth=depth, role=role))
        if parent.session_stats:
            parent.session_stats.record_retry()
            parent.session_stats.check_budget()
        retry_ctx = (
            f"{context}\n\nIMPORTANT: A previous attempt at this task failed with:\n"
            f"{result.answer}\nTry a different approach."
        ).strip()
        result = await make_child(with_context(retry_ctx), role_tools).run_subtask(task)
        result.retried = True

    if verify and result.status == "ok":
        verifier = make_child(
            f"{VERIFIER_SYSTEM}\n\nContext:\nOriginal task: {task}\n\nResult to verify:\n{result.answer}",
            ["bash", "read_file"],
        )
        verdict = (await verifier.run_subtask(f"Verify: {task}")).answer.strip()
        result.verification = verdict
        if not verdict.upper().startswith("VERIFIED"):
            result.status = "verify_failed"

    if parent.on_spawn:
        parent.on_spawn(SpawnEvent(kind="end", task=task, depth=depth, role=role, result=result))

    if cacheable and result.status == "ok":
        parent.result_cache.put(task, context, result.to_str(), role_obj.name)

    return result


def make_spawn_tool(parent: "Agent", worker_registry: ToolRegistry) -> Tool:

    async def spawn_agent(task: str, context: str = "", role: str = "", verify: bool = False) -> str:
        result = await spawn_subagent(parent, worker_registry, task, context, role, verify)
        return result.to_str()

    return Tool(
        name="spawn_agent",
        description=(
            "Spawn an isolated sub-agent with its own context window to complete a task.\n"
            "Use this to delegate focused units of work, run tasks in parallel, or keep "
            "orchestrator context clean.\n\n"
            f"Available roles: {', '.join(ROLES)}.\n"
            "Set verify=true to have a verifier check the result.\n"
            "Returns JSON: {status, answer, tool_calls_made, depth}. "
            "status is ok, error, incomplete (ran out of turns — split the task) or verify_failed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task":    {"type": "string", "description": "Clear, self-contained description of what the sub-agent must do."},
                "context": {"type": "string", "description": "Background context: file paths, prior results, constraints."},
                "role":    {"type": "string", "description": f"Role preset: {', '.join(ROLES)}."},
                "verify":  {"type": "boolean", "description": "If true, auto-run a verifier after the main agent completes."},
            },
            "required": ["task"],
        },
        executor=spawn_agent,
    )


__all__ = [
    "Agent", "AgentResult", "SpawnEvent", "HarnessAbort",
    "make_spawn_tool", "spawn_subagent", "classify",
]
