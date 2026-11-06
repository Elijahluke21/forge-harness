"""File/bash tools, workspace confinement, memory, compaction."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from conftest import FakeClient, text
from harness.context.manager import ContextManager
from harness.memory import tools as memory_tools
from harness.memory.store import MemoryStore
from harness.tools import bash, files
from harness.tools.registry import ToolRegistry
from harness.tools.workspace import Workspace


@pytest.fixture
def ws_reg(tmp_path):
    ws = Workspace(tmp_path)
    reg = ToolRegistry()
    files.register(reg, ws)
    bash.register(reg, ws)
    return tmp_path, reg


# ── files ─────────────────────────────────────────────────────────────────

async def test_file_tools_confined_to_workspace(ws_reg, tmp_path):
    root, reg = ws_reg
    assert (await reg.execute("write_file", {"path": "a/b.txt", "content": "hi"})).startswith("wrote")
    assert (root / "a" / "b.txt").read_text() == "hi"
    for bad in ("../escape.txt", "/etc/passwd", "a/../../x"):
        assert "outside the workspace" in await reg.execute("read_file", {"path": bad})
        assert "outside the workspace" in await reg.execute("write_file", {"path": bad, "content": "x"})


async def test_symlink_escape_blocked(ws_reg, tmp_path_factory):
    root, reg = ws_reg
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret").write_text("s")
    try:
        os.symlink(outside, root / "link")
    except OSError:
        pytest.skip("symlinks need admin or Developer Mode on Windows")
    assert "outside the workspace" in await reg.execute("read_file", {"path": "link/secret"})


async def test_edit_file(ws_reg):
    root, reg = ws_reg
    (root / "f.py").write_text("x = 1\ny = 1\n")
    assert "appears 2 times" in await reg.execute("edit_file", {"path": "f.py", "old_text": "= 1", "new_text": "= 2"})
    assert "not found" in await reg.execute("edit_file", {"path": "f.py", "old_text": "z", "new_text": "q"})
    await reg.execute("edit_file", {"path": "f.py", "old_text": "y = 1", "new_text": "y = 9"})
    assert (root / "f.py").read_text() == "x = 1\ny = 9\n"
    await reg.execute("edit_file", {"path": "f.py", "old_text": "= ", "new_text": "== ", "replace_all": True})
    assert (root / "f.py").read_text() == "x == 1\ny == 9\n"


async def test_read_file_truncation_says_where_to_continue(ws_reg):
    root, reg = ws_reg
    (root / "big.txt").write_text("".join(f"line {i} " + "x" * 80 + "\n" for i in range(500)))
    out = await reg.execute("read_file", {"path": "big.txt"})
    assert "use offset=" in out


# ── bash ──────────────────────────────────────────────────────────────────

async def test_bash_runs_in_workspace(ws_reg):
    root, reg = ws_reg
    # Git Bash prints /c/Users/... (or /tmp/...); `pwd -W` gives the native C:/... form.
    pwd = "pwd -W" if sys.platform == "win32" else "pwd"
    out = (await reg.execute("bash", {"command": pwd})).strip()
    assert Path(out).resolve() == root.resolve()


async def test_bash_timeout_kills_whole_process_group(ws_reg):
    root, reg = ws_reg
    marker = root / "child_alive"
    # The background child would create the marker after 3s if it survived.
    cmd = f"(sleep 3; touch {marker.as_posix()}) & sleep 30"
    start = time.monotonic()
    out = await reg.execute("bash", {"command": cmd, "timeout": 1})
    assert "timed out" in out and time.monotonic() - start < 10
    await asyncio.sleep(4)
    assert not marker.exists()


async def test_bash_long_output_keeps_head_and_tail(ws_reg):
    _, reg = ws_reg
    out = await reg.execute("bash", {"command": "echo START; seq 1 5000; echo END_MARKER"})
    assert "START" in out and "END_MARKER" in out and "truncated" in out


# ── memory ────────────────────────────────────────────────────────────────

def test_memory_prompt_is_relevant_and_bounded(memory_path):
    store = MemoryStore(memory_path)
    for i in range(40):
        store.set(f"fact_{i}", f"unrelated detail number {i} " + "y" * 100)
    store.set("swift_style", "SwiftUI views use MVVM", tags=["ios"])
    store.set("user_name", "Eli", tags=["pinned"])
    block = store.format_for_prompt(query="build the SwiftUI settings screen", max_chars=600)
    assert "Eli" in block and "MVVM" in block
    assert len(block) < 900
    assert "memories total" in block


def test_memory_corrupt_file_is_backed_up(memory_path):
    memory_path.write_text("{not json")
    store = MemoryStore(memory_path)
    assert len(store) == 0
    assert memory_path.with_suffix(".corrupt.json").exists()


def test_worker_memory_is_read_only(memory_path):
    reg = ToolRegistry()
    memory_tools.register(reg, MemoryStore(memory_path), read_only=True)
    assert "memory_write" not in reg.names() and "memory_delete" not in reg.names()
    assert "memory_search" in reg.names()


# ── compaction ────────────────────────────────────────────────────────────

def _history(rounds: int) -> list[dict]:
    msgs = [{"role": "user", "content": "the task"}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"step {i}"},
            {"type": "tool_use", "id": f"u{i}", "name": "bash", "input": {"command": "ls"}},
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"u{i}", "content": f"out {i}"},
        ]})
    return msgs


def _assert_valid(msgs: list[dict]) -> None:
    assert msgs[0]["role"] == "user"
    for a, b in zip(msgs, msgs[1:]):
        assert a["role"] != b["role"], "roles must alternate"
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and isinstance(m["content"], list):
            ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
            if ids:
                nxt = msgs[i + 1]["content"]
                assert ids == {b["tool_use_id"] for b in nxt if b.get("type") == "tool_result"}
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    prev = msgs[i - 1]
                    assert any(x.get("id") == b["tool_use_id"] for x in prev["content"])


@pytest.mark.parametrize("rounds", [5, 6, 9, 12])
async def test_compaction_keeps_tool_calls_paired(rounds):
    msgs = _history(rounds)
    cm = ContextManager(context_window=100)
    out = await cm._compact(msgs, FakeClient(lambda *a: text("<think>x</think>summary of work")))
    assert "compacted" in out
    _assert_valid(msgs)
    assert "the task" in msgs[0]["content"] and "summary of work" in msgs[0]["content"]
    assert "<think>" not in msgs[0]["content"]


async def test_offload_lands_in_given_dir(tmp_path):
    cm = ContextManager(offload_dir=tmp_path / "off", offload_chars=10)
    out = cm.offload_if_large("x" * 100)
    assert str(tmp_path / "off") in out
