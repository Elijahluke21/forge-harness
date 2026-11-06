import os
from datetime import datetime


def build_system_prompt(extra: str = "", cwd: str | None = None) -> str:
    """
    Build the system prompt injected on every API call.
    `extra` carries role-specific guidance; `cwd` is the workspace the tools
    operate in (defaults to the process working directory).
    """
    lines = [
        "You are Forge, a highly capable agentic AI assistant built on a custom harness.",
        "",
        "## Environment",
        f"Date : {datetime.now().strftime('%Y-%m-%d')}",
        f"Working directory : {cwd or os.getcwd()}",
        "",
        "## Behavior rules",
        "- Use the available tools to accomplish tasks. Never fabricate results.",
        "- For multi-step tasks, decompose into steps and execute them in order.",
        "- When reading or writing files always use the file tools, not bash redirection.",
        "- To change part of an existing file use edit_file rather than rewriting it.",
        "- Be concise. Explain what you did, not what you are about to do.",
    ]

    if extra:
        lines += ["", extra]

    return "\n".join(lines)
