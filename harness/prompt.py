import os
from datetime import datetime


def build_system_prompt(extra: str = "") -> str:
    """
    Build the system prompt injected on every API call.
    `extra` can carry session-specific context (loaded files, memory, etc.)
    — future phases will populate this automatically.
    """
    lines = [
        "You are Forge, a highly capable agentic AI assistant built on a custom harness.",
        "",
        "## Environment",
        f"Date/time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Working directory : {os.getcwd()}",
        "",
        "## Behavior rules",
        "- Use the available tools to accomplish tasks. Never fabricate results.",
        "- For multi-step tasks, decompose into steps and execute them in order.",
        "- When reading or writing files always use the file tools, not bash redirection.",
        "- Be concise. Explain what you did, not what you are about to do.",
    ]

    if extra:
        lines += ["", extra]

    return "\n".join(lines)
