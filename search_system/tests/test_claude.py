from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ioai_agent_system.claude import ClaudeRunner, SecretBroker


MOCK_CLAUDE = r"""#!/usr/bin/env python3
import json
import pathlib
import re
import shutil
import sys

prompt = sys.stdin.read()
cwd = pathlib.Path.cwd()
model = "claude-opus-5"
print(json.dumps({
    "type": "system", "subtype": "init", "model": model,
    "session_id": "mock-session", "note": "sk-ant-dummysecret0123456789"
}), flush=True)

if "Required deliverables" in prompt:
    match = re.search(r"Read-only competition data:\s*(.+)", prompt)
    data = pathlib.Path(match.group(1).strip())
    candidate = cwd / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data / "sample_submission.csv", candidate / "submission.csv")
    (candidate / "metrics.json").write_text(json.dumps({
        "official_oof_score": 0.5, "runtime_s": 0.01
    }))
    (candidate / "solution.py").write_text("# mock\n")
    (candidate / "hypothesis_card.json").write_text("{}\n")
    (candidate / "README.md").write_text("mock\n")

print(json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "result": "OK", "total_cost_usd": 0.001, "session_id": "mock-session"
}), flush=True)
"""


class ClaudeRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_broker_helper_and_redacted_trace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            mock = root / "mock_claude.py"
            mock.write_text(textwrap.dedent(MOCK_CLAUDE), encoding="utf-8")
            mock.chmod(0o755)
            socket_path = root / "secret.sock"
            package_root = Path(__file__).resolve().parents[1]
            self.assertTrue(
                (package_root / "ioai_agent_system" / "secret_helper.py").is_file()
            )
            runner = ClaudeRunner(
                claude_bin=mock,
                python_bin=Path(sys.executable),
                package_root=package_root,
                model="claude-opus-5",
                effort="high",
                secret_socket=socket_path,
            )
            async with SecretBroker("sk-ant-unit-test-secret-0123456789", socket_path):
                result = await runner.preflight(root / "work", root / "traces")
            self.assertEqual(result.resolved_model, "claude-opus-5")
            trace = result.trace_path.read_text(encoding="utf-8")
            self.assertNotIn("dummysecret", trace)
            self.assertIn("<redacted>", trace)
            self.assertNotIn("ANTHROPIC_API_KEY", runner.environment())
            isolated = runner.environment(root / "isolated")
            self.assertEqual(isolated["HOME"], str(root / "isolated" / ".agent_home"))
            self.assertNotEqual(isolated["HOME"], os.environ.get("HOME"))
            self.assertNotIn("KAGGLE_CONFIG_DIR", isolated)
            self.assertIn("--model", runner.argv())
            persistent = runner.argv(persist_session=True)
            self.assertNotIn("--no-session-persistence", persistent)
            resumed = runner.argv(resume_session_id="mock-session")
            self.assertIn("--resume", resumed)
            self.assertIn("mock-session", resumed)


if __name__ == "__main__":
    unittest.main()
