"""
Role presets for sub-agents.
Each role pairs a focused system prompt with an optional tool allowlist.
None means the sub-agent inherits all worker tools.
"""

from dataclasses import dataclass


@dataclass
class Role:
    name: str
    system: str
    tools: list[str] | None   # None = all tools; [] = no tools


ROLES: dict[str, Role] = {
    "planner": Role(
        name="planner",
        system=(
            "You are a planning sub-agent. Your only job is to decompose the given task "
            "into a numbered list of clear, self-contained sub-tasks. "
            "Output ONLY the list — no preamble, no explanation."
        ),
        tools=None,
    ),
    "reader": Role(
        name="reader",
        system=(
            "You are a file-reading sub-agent. Read the requested files and return their "
            "relevant contents. Be thorough but concise — summarize large sections."
        ),
        tools=["read_file"],
    ),
    "executor": Role(
        name="executor",
        system=(
            "You are an execution sub-agent. Run the requested commands or scripts, "
            "capture output, and report results precisely. "
            "If a command fails, report the error verbatim."
        ),
        tools=["bash", "read_file", "write_file"],
    ),
    "writer": Role(
        name="writer",
        system=(
            "You are a file-writing sub-agent. Write or modify files as instructed. "
            "Confirm each file written with its path and byte count."
        ),
        tools=["read_file", "write_file"],
    ),
    "synthesizer": Role(
        name="synthesizer",
        system=(
            "You are a synthesis sub-agent. You will receive multiple results from "
            "other agents. Combine, reconcile, and summarize them into a single "
            "coherent response for the user."
        ),
        tools=[],
    ),
    "verifier": Role(
        name="verifier",
        system=(
            "You are a verification sub-agent. Check that the provided result is correct, "
            "complete, and consistent. Report any issues found, or confirm it is correct."
        ),
        tools=["bash", "read_file"],
    ),
}


def get_role(name: str) -> Role | None:
    return ROLES.get(name.lower())
