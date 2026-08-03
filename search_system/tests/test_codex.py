from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ioai_agent_system.codex import CodexRunner


class CodexRunnerTests(unittest.TestCase):
    def test_native_search_resume_and_sanitized_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            runner = CodexRunner(
                codex_bin=Path("/opt/homebrew/bin/codex"),
                model="gpt-test",
                effort="high",
            )
            argv = runner.argv(
                workdir=root,
                last_message_path=root / "last.md",
                resume_session_id="12345678-1234-1234-1234-123456789abc",
            )
            self.assertIn("--search", argv)
            self.assertIn("resume", argv)
            self.assertIn("workspace-write", argv)
            ephemeral = runner.argv(
                workdir=root,
                last_message_path=root / "ephemeral.md",
            )
            self.assertIn("--ephemeral", ephemeral)
            persistent = runner.argv(
                workdir=root,
                last_message_path=root / "persistent.md",
                persist_session=True,
            )
            self.assertNotIn("--ephemeral", persistent)
            with mock.patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "secret",
                    "KAGGLE_API_TOKEN": "secret",
                    "HOME": "/tmp/test-home",
                },
                clear=False,
            ):
                env = runner.environment()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("KAGGLE_API_TOKEN", env)
            self.assertEqual(env["HOME"], "/tmp/test-home")


if __name__ == "__main__":
    unittest.main()
