"""
Builds the orchestrator and its tool registries from a ForgeConfig.

main.py and the `forge` CLI used to each carry their own copy of this wiring;
they now share this one.
"""

from dataclasses import dataclass
from pathlib import Path

from harness.agent import Agent, make_spawn_tool
from harness.budget import RunBudget
from harness.client import BaseClient, make_client
from harness.config import ForgeConfig
from harness.context.manager import ContextManager
from harness.memory import tools as memory_tools
from harness.memory.store import MemoryStore
from harness.planning import make_plan_tool
from harness.prompt import build_system_prompt
from harness.session import ResultCache, SessionStats
from harness.tools import bash, files
from harness.tools.registry import ToolRegistry
from harness.tools.workspace import Workspace

ORCHESTRATOR_GUIDANCE = "\n".join([
    "## Orchestration strategy",
    "You are the orchestrator. For any non-trivial multi-step task:",
    "1. Call plan_task first to produce a structured execution plan.",
    "2. Execute the plan using spawn_agent — spawn independent steps in parallel.",
    "   - Use the `role` parameter to give sub-agents focused capabilities.",
    "   - Set `verify=true` for critical steps that need confirmation.",
    "   - A result with status 'incomplete' means the task was too big: split it.",
    "3. Synthesize results and respond to the user.",
    "Reserve direct tool use (bash, files) for quick single-step actions only.",
    "Use memory_write to persist anything worth remembering across sessions.",
])


@dataclass
class Forge:
    cfg: ForgeConfig
    client: BaseClient
    orchestrator: Agent
    worker_registry: ToolRegistry
    workspace: Workspace
    escalation_client: BaseClient | None = None


def make_budget(cfg: ForgeConfig) -> RunBudget | None:
    budget = RunBudget(
        max_total_tokens=cfg.budget_max_tokens,
        max_spawns=cfg.budget_max_spawns,
        max_tool_calls=cfg.budget_max_tool_calls,
        max_wall_seconds=cfg.budget_max_minutes * 60,
    )
    limits = (budget.max_total_tokens, budget.max_spawns, budget.max_tool_calls, budget.max_wall_seconds)
    return budget if any(limits) else None


def build_forge(
    cfg: ForgeConfig,
    client: BaseClient | None = None,
    workspace: str | None = None,
    on_tool_call=None,
    on_spawn=None,
    confirm_fn=None,
    budget: RunBudget | None = None,
) -> Forge:
    ws = Workspace(workspace or cfg.workspace or None)
    client = client or make_client(backend=cfg.backend, model=cfg.model, base_url=cfg.base_url)
    store = MemoryStore(Path(cfg.memory_path).expanduser())
    stats = SessionStats(budget=budget if budget is not None else make_budget(cfg))
    cache = ResultCache()
    ctx = ContextManager(
        model=cfg.model,
        context_window=cfg.context_window,
        compact_at=cfg.compact_at,
        budget_fraction=cfg.budget_fraction,
        offload_dir=(ws.root / ".forge" / "offload") if ws.root else None,
    )

    # Workers get read-only memory: they read untrusted files and pages, and
    # injected text must not be able to write itself into future sessions.
    worker_registry = ToolRegistry()
    bash.register(worker_registry, ws)
    files.register(worker_registry, ws)
    memory_tools.register(worker_registry, store, read_only=True)

    orch_registry = ToolRegistry()
    bash.register(orch_registry, ws)
    files.register(orch_registry, ws)
    memory_tools.register(orch_registry, store)

    orchestrator = Agent(
        client=client,
        registry=orch_registry,
        system=build_system_prompt(extra=ORCHESTRATOR_GUIDANCE, cwd=str(ws.cwd)),
        max_tokens=cfg.max_tokens,
        max_iterations=cfg.max_iterations,
        max_depth=cfg.max_depth,
        on_tool_call=on_tool_call,
        on_spawn=on_spawn,
        context_manager=ctx,
        memory_store=store,
        session_stats=stats,
        result_cache=cache,
    )
    orch_registry.add(make_plan_tool(client, store, stats))
    orch_registry.add(make_spawn_tool(orchestrator, worker_registry))

    for reg in (orch_registry, worker_registry):
        if cfg.require_confirm and confirm_fn:
            reg.set_permissions(cfg.require_confirm, confirm_fn)
        if cfg.hooks:
            reg.set_hooks(cfg.hooks)

    escalation = None
    if cfg.escalation_backend and cfg.escalation_model:
        escalation = make_client(backend=cfg.escalation_backend, model=cfg.escalation_model)

    return Forge(cfg, client, orchestrator, worker_registry, ws, escalation)
