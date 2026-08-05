"""A deliberately small fallback system for IOAI 2026.

Three tasks run in parallel in one process. Each task gets two solver agents --
Codex on gpt-5.6-sol and Claude Code on claude-fable-5 -- that share a single
workspace, plus a select manager that decides which of their candidates is worth
a submission. Every model call goes through OpenRouter.

It exists so that a bug in the main system on competition day costs a rank, not
the whole day.
"""

__all__ = ["config", "board", "runners", "submit", "part", "run"]
