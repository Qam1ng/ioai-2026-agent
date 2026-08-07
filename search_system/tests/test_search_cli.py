from __future__ import annotations

import contextlib
import io
import unittest

from ioai_agent_system.search_cli import build_parser


class SearchCliTests(unittest.TestCase):
    def test_duration_and_competition_mode_are_explicit(self) -> None:
        base = ["--assets-dir", "/tmp/assets", "--output-root", "/tmp/runs"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(base)
        args = build_parser().parse_args(
            [
                *base,
                "--duration-seconds",
                "1500",
                "--competition-mode",
                "formal",
                "--research-backends",
                "claude,codex,claude,codex",
            ]
        )
        self.assertEqual(args.duration_seconds, 1500)
        self.assertEqual(args.research_backends[1], "codex")


if __name__ == "__main__":
    unittest.main()
