from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from ioai_agent_system.models import AgentResult
from ioai_agent_system.search_assets import AssetPolicyError
from ioai_agent_system.search_orchestrator import SearchOrchestrator, SearchRunConfig


class FakeSearchRunner:
    def __init__(self) -> None:
        self.research_starts: list[float] = []
        self.resume_ids: list[str | None] = []

    @staticmethod
    def _path(prompt: str, label: str) -> Path:
        match = re.search(re.escape(label) + r"\s*\n\s*(.+)", prompt)
        if match is None:
            raise AssertionError(f"missing path label: {label}")
        return Path(match.group(1).strip())

    async def run(
        self,
        *,
        agent_id: str,
        role: str,
        prompt: str,
        workdir: Path,
        trace_dir: Path,
        timeout_s: float,
        persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        del timeout_s, persist_session
        started = time.monotonic()
        workdir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace = trace_dir / f"{agent_id}.jsonl"
        trace.write_text(
            json.dumps({"type": "mock", "agent_id": agent_id}) + "\n",
            encoding="utf-8",
        )
        session_id = resume_session_id or "11111111-1111-1111-1111-111111111111"

        if role == "task_analyst_stage_a":
            stage_a = self._path(prompt, "你的唯一可写目录：")
            stage_a.mkdir(parents=True, exist_ok=True)
            (stage_a / "ASSET_MAP.md").write_text("# assets\n", encoding="utf-8")
            (stage_a / "TASK_ANALYSIS_V1.md").write_text(
                "# analysis\n[LOCAL_FACT] mock\n", encoding="utf-8"
            )
            parts = [
                "research_plan:",
                "  enabled_research_ids: [R1, R2, R3, R4]",
                "  reason: test parallelism",
                "",
            ]
            for research_id in ("R1", "R2", "R3", "R4"):
                parts.extend(
                    [
                        f"research_id: {research_id}",
                        "time_budget_minutes: 1",
                        f"primary_scope: scope-{research_id}",
                        "",
                    ]
                )
            (stage_a / "RESEARCH_CHARTERS.md").write_text(
                "\n".join(parts), encoding="utf-8"
            )
        elif role == "research_agent":
            self.research_starts.append(time.monotonic())
            await asyncio.sleep(0.08)
            output = self._path(prompt, "必须写入的唯一最终报告：")
            output.parent.mkdir(parents=True, exist_ok=True)
            research_id = agent_id.removeprefix("research_")
            output.write_text(
                f"# {research_id} Research Report\n\n## 0. Top-5 可行动方向\n"
                f"{research_id}-F001 mock finding\n",
                encoding="utf-8",
            )
        elif role == "task_analyst_stage_b":
            self.resume_ids.append(resume_session_id)
            bundle = self._path(prompt, "最终 Search Bundle 目录：")
            (bundle / "SEARCH_OUTPUT.md").write_text(
                "# Search Output\n\n完整性状态：通过\n", encoding="utf-8"
            )
            (bundle / "ASSET_MAP.md").write_text("# assets final\n", encoding="utf-8")
            (bundle / "TASK_ANALYSIS.md").write_text(
                "# task final\n[LOCAL_FACT] mock\n", encoding="utf-8"
            )
            (bundle / "RESEARCH_SYNTHESIS.md").write_text(
                "# synthesis\n", encoding="utf-8"
            )
        return AgentResult(
            agent_id=agent_id,
            role=role,
            workdir=workdir,
            trace_path=trace,
            exit_code=0,
            timed_out=False,
            duration_s=time.monotonic() - started,
            session_id=session_id,
            result_text="OK",
        )

    async def terminate_all(self, *, grace_s: float = 8.0) -> None:
        del grace_s


class HangingSearchRunner:
    def __init__(self) -> None:
        self.terminated = False

    async def run(self, **_: object) -> AgentResult:
        await asyncio.sleep(60)
        raise AssertionError("global deadline did not cancel the runner")

    async def terminate_all(self, *, grace_s: float = 8.0) -> None:
        del grace_s
        self.terminated = True


class SearchOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def test_output_root_cannot_be_nested_in_assets(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            assets = root / "assets"
            assets.mkdir()
            config = SearchRunConfig(
                assets_dir=assets,
                output_root=assets / "runs",
                prompt_spec=Path(__file__).resolve().parents[1]
                / "SEARCH_PROMPTS_ZH.md",
                duration_s=10,
                competition_mode="formal",
                analyst_backend="claude",
                research_backends=("claude", "claude", "claude", "claude"),
            )
            with self.assertRaises(AssetPolicyError):
                SearchOrchestrator(config, runners={"claude": FakeSearchRunner()})

    async def test_end_to_end_preserves_assets_resumes_and_runs_research_in_parallel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            assets = root / "assets"
            assets.mkdir()
            original = b"raw\x00asset\xff"
            (assets / "data.bin").write_bytes(original)
            output = root / "runs"
            output.mkdir()
            prompt_spec = Path(__file__).resolve().parents[1] / "SEARCH_PROMPTS_ZH.md"
            runner = FakeSearchRunner()
            config = SearchRunConfig(
                assets_dir=assets,
                output_root=output,
                prompt_spec=prompt_spec,
                duration_s=8,
                competition_mode="formal",
                analyst_backend="claude",
                research_backends=("claude", "claude", "claude", "claude"),
            )
            session = await SearchOrchestrator(
                config, runners={"claude": runner}
            ).execute()
            status = json.loads((session / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete", status)
            bundle = Path(status["bundle_dir"])
            self.assertEqual((bundle / "ORIGINAL_ASSETS" / "data.bin").read_bytes(), original)
            self.assertEqual(len(list((bundle / "research_raw").glob("*.md"))), 4)
            self.assertEqual(
                runner.resume_ids,
                ["11111111-1111-1111-1111-111111111111"],
            )
            self.assertLess(max(runner.research_starts) - min(runner.research_starts), 0.05)
            self.assertEqual(len(list((session / "trajectories").glob("*.jsonl"))), 6)
            previous = "0" * 64
            for raw_line in (session / "controller_trajectory.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                event = json.loads(raw_line)
                self.assertEqual(event["previous_hash"], previous)
                event_hash = event.pop("event_hash")
                canonical = json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual(event_hash, hashlib.sha256(canonical).hexdigest())
                previous = event_hash

    async def test_global_deadline_stops_a_hanging_search_run(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            assets = root / "assets"
            assets.mkdir()
            (assets / "task.txt").write_text("task", encoding="utf-8")
            output = root / "runs"
            output.mkdir()
            runner = HangingSearchRunner()
            config = SearchRunConfig(
                assets_dir=assets,
                output_root=output,
                prompt_spec=Path(__file__).resolve().parents[1]
                / "SEARCH_PROMPTS_ZH.md",
                duration_s=1,
                competition_mode="practice",
                analyst_backend="claude",
                research_backends=("claude", "claude", "claude", "claude"),
            )
            started = time.monotonic()
            session = await SearchOrchestrator(
                config, runners={"claude": runner}
            ).execute()
            elapsed = time.monotonic() - started
            status = json.loads((session / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "timeout")
            self.assertTrue(runner.terminated)
            self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
