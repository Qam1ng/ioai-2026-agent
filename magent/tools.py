"""SDK custom tools for the multiagent branch.

Reuses the battle-tested kaggle/memory/skill tool functions from
agent/tools/registry.py, exposed to the Claude Agent SDK as an in-process MCP
server. File/bash tools come from the SDK itself (Read/Write/Bash/Glob/Grep).
"""
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import tool, create_sdk_mcp_server

from agent.tools import registry as R

_CTX: R.Ctx | None = None  # set once by main


def init_ctx(workspace: Path, slug: str, budget, trace) -> None:
    global _CTX
    _CTX = R.Ctx(workspace=workspace, slug=slug, budget=budget, trace=trace)


def _run(fn, args) -> dict:
    try:
        out = fn(args, _CTX)
        return {"content": [{"type": "text", "text": str(out)}]}
    except Exception as e:  # noqa: BLE001
        return {"content": [{"type": "text",
                             "text": f"[tool error] {type(e).__name__}: {e}"}],
                "is_error": True}


@tool("kaggle_download", "Download+unzip this competition's data into input/ "
      "(skips if already present).", {})
async def kaggle_download(args):
    return _run(R.kaggle_download, args)


@tool("kaggle_push_kernel", "Push a kernel dir (containing kernel-metadata.json)"
      " to Kaggle and start a cloud run. Code competitions score notebooks, "
      "not CSVs. Note the version number in the output.",
      {"kernel_dir": str})
async def kaggle_push_kernel(args):
    return _run(R.kaggle_push_kernel, args)


@tool("kaggle_kernel_status", "Status of a pushed kernel (default: last pushed).",
      {"kernel_ref": str})
async def kaggle_kernel_status(args):
    return _run(R.kaggle_kernel_status, args)


@tool("kaggle_kernel_log", "Fetch a finished kernel's run log.",
      {"kernel_ref": str})
async def kaggle_kernel_log(args):
    return _run(R.kaggle_kernel_log, args)


@tool("kaggle_submit", "Submit the finished kernel's output to the competition."
      " Budget-gated by the harness.",
      {"kernel_ref": str, "kernel_version": int, "message": str,
       "file_name": str})
async def kaggle_submit(args):
    return _run(R.kaggle_submit, args)


@tool("kaggle_submissions", "List this competition's submissions and scores.", {})
async def kaggle_submissions(args):
    return _run(R.kaggle_submissions, args)


@tool("memory_recall", "Recall distilled lessons from past runs (optional "
      "query filter). Use early.", {"query": str})
async def memory_recall(args):
    return _run(R.memory_recall, args)


@tool("memory_write", "Save a durable, task-solving lesson for future runs. "
      "No competition logistics.", {"name": str, "content": str})
async def memory_write(args):
    return _run(R.memory_write, args)


@tool("skill_list", "List available skill playbooks.", {})
async def skill_list(args):
    return _run(R.skill_list, args)


@tool("skill_load", "Load a skill playbook by name.", {"name": str})
async def skill_load(args):
    return _run(R.skill_load, args)


SERVER = create_sdk_mcp_server(
    name="ioai",
    version="1.0.0",
    tools=[kaggle_download, kaggle_push_kernel, kaggle_kernel_status,
           kaggle_kernel_log, kaggle_submit, kaggle_submissions,
           memory_recall, memory_write, skill_list, skill_load],
)

# fully-qualified tool names as the SDK exposes them
TOOL_NAMES = [f"mcp__ioai__{n}" for n in
              ["kaggle_download", "kaggle_push_kernel", "kaggle_kernel_status",
               "kaggle_kernel_log", "kaggle_submit", "kaggle_submissions",
               "memory_recall", "memory_write", "skill_list", "skill_load"]]
