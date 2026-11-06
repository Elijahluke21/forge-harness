"""
plan_task tool — asks the model to decompose an objective into a structured
execution plan before any sub-agents are spawned.

Returns a JSON array of steps:
  [{"id": "1", "task": "...", "role": "executor", "depends_on": [], "hint": "..."}]

The plan is validated (unique ids, known roles, real dependencies, no cycles)
before it is returned; an invalid plan is sent back to the model once with the
error so it can correct it. The autopilot runs validated plans through
harness.executor instead of trusting the model to sequence them.
"""

from harness.client import BaseClient
from harness.executor import PlanError, Step, parse_plan, plan_to_json
from harness.memory.store import MemoryStore
from harness.roles import ROLES
from harness.session import SessionStats
from harness.text import extract_json
from harness.tools.registry import Tool

PLAN_SYSTEM = f"""
You are a task decomposition expert.
Given an objective, output ONLY a valid JSON array of steps — no preamble, no explanation.

Each step must have:
  "id"         : unique string ("1", "2", ...)
  "task"       : clear, self-contained description of the work, naming exact files where known
  "role"       : one of: {", ".join(ROLES)}
  "depends_on" : list of step ids this step waits for ([] if independent)
  "hint"       : one sentence of context to pass to the sub-agent

Rules:
- Keep steps small enough for one focused agent (one file or one concern each).
- Steps with no shared dependencies should have empty depends_on so they run in parallel.
- Never have two parallel steps edit the same file.
- Use 3–8 steps. Do not add steps for work that was not asked for.
Output JSON only. No markdown fences.
""".strip()

PLAN_MAX_TOKENS = 4_000   # reasoning models spend tokens thinking before the JSON


async def generate_plan(
    client: BaseClient,
    objective: str,
    context: str = "",
    session_stats: SessionStats | None = None,
) -> list[Step]:
    """Ask the model for a plan; one correction round if it is invalid. Raises PlanError."""
    prompt = f"Objective: {objective}"
    if context:
        prompt += f"\n\nContext:\n{context}"
    messages = [{"role": "user", "content": prompt}]

    last_error = ""
    for _ in range(2):
        if session_stats:
            session_stats.check_budget()
        resp = await client.send(messages=messages, system=PLAN_SYSTEM, max_tokens=PLAN_MAX_TOKENS)
        if session_stats:   # planning tokens count against the run budget too
            session_stats.record_tokens(resp.input_tokens, resp.output_tokens)
        try:
            return parse_plan(extract_json(resp.text))
        except (PlanError, ValueError) as exc:
            last_error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": resp.text or "(empty)"},
                {"role": "user", "content": f"That plan is invalid: {exc}. Output the corrected JSON array only."},
            ]
    raise PlanError(last_error)


def make_plan_tool(
    client: BaseClient,
    memory_store: MemoryStore | None = None,
    session_stats: SessionStats | None = None,
) -> Tool:

    async def plan_task(objective: str, store_as: str = "") -> str:
        try:
            steps = await generate_plan(client, objective, session_stats=session_stats)
        except PlanError as exc:
            return f"[error] plan_task: {exc}"

        if store_as and memory_store:
            memory_store.set(store_as, plan_to_json(steps), tags=["plan"])

        lines = [f"Plan for: {objective}", ""]
        for s in steps:
            deps = f"  (after {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"Step {s.id} [{s.role}]{deps}: {s.task}")
            if s.hint:
                lines.append(f"  → hint: {s.hint}")
        lines += ["", "Full JSON:", plan_to_json(steps)]
        return "\n".join(lines)

    return Tool(
        name="plan_task",
        description=(
            "Decompose a complex objective into a structured, ordered execution plan. "
            "Call this BEFORE spawning sub-agents for any multi-step task. "
            "Returns a human-readable plan + JSON you can use to sequence spawn_agent calls. "
            "Independent steps can be spawned in parallel (same response). "
            "Call again mid-execution to revise the plan based on new information."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "objective": {"type": "string", "description": "The high-level goal to decompose into steps."},
                "store_as":  {"type": "string", "description": "Optional memory key to persist the plan for later reference."},
            },
            "required": ["objective"],
        },
        executor=plan_task,
    )
