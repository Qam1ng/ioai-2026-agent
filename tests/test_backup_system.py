"""Offline tests for the backup system: no API key, no Kaggle, no model calls."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from datetime import UTC, datetime
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backup_system.board import Board  # noqa: E402
from backup_system.config import (  # noqa: E402
    DEFAULT_ANTHROPIC_BASE,
    DEFAULT_OPENAI_BASE,
    BackupConfig,
)
from backup_system.runners import ClaudeRunner, CodexRunner  # noqa: E402
from backup_system.submit import (  # noqa: E402
    DryRunKaggle,
    find_submission_template,
    floor_kernel,
    validate_kernel,
    validate_submission,
)

CONFIG = ROOT / "configs" / "backup_system.toml"


def _template(directory: Path, rows: int = 20) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "submission.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target"])
        for index in range(rows):
            writer.writerow([f"r{index}", 0])
    return path


def _candidate(root: Path, name: str, *, author: str = "codex",
               kernel: str = "print('x')\nopen('submission.csv','w')\n",
               extra: dict | None = None) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "kernel.py").write_text(kernel)
    (directory / "meta.json").write_text(json.dumps({
        "author": author, "idea": f"idea for {name}", "accelerator": "cpu",
        **(extra or {}),
    }))
    (directory / "READY").touch()
    return directory


# --------------------------------------------------------------------- config
def test_base_urls_do_not_double_the_version_segment() -> None:
    """Claude's SDK appends /v1/messages; Codex appends /responses."""
    assert not DEFAULT_ANTHROPIC_BASE.rstrip("/").endswith("/v1")
    assert f"{DEFAULT_ANTHROPIC_BASE}/v1/messages".endswith("/anthropic/v1/messages")
    assert DEFAULT_OPENAI_BASE.endswith("/openai/v1")
    assert f"{DEFAULT_OPENAI_BASE}/responses".endswith("/openai/v1/responses")


def test_shipped_config_wires_each_client_to_its_own_endpoint_shape() -> None:
    config = BackupConfig.load(CONFIG, repo_root=ROOT)
    assert config.codex.model == "gpt-5.6-sol"
    assert config.claude.model == "claude-fable-5"
    assert config.manager.model == "claude-fable-5"
    assert config.codex.base_url.endswith("/openai/v1")
    assert config.claude.base_url.endswith("/anthropic")
    assert config.manager.base_url.endswith("/anthropic")
    assert 1 <= config.run.kaggle_concurrency <= 2
    assert config.run.max_submissions <= 50


def test_config_rejects_a_concurrency_kaggle_will_not_honour(tmp_path: Path) -> None:
    text = CONFIG.read_text().replace("kaggle_concurrency = 2",
                                      "kaggle_concurrency = 6")
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="kaggle_concurrency"):
        BackupConfig.load(bad, repo_root=ROOT)


# -------------------------------------------------------------------- runners
def test_neither_agent_process_can_see_kaggle_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IOAI_LLM_API_KEY", "azure-test-key")
    monkeypatch.setenv("KAGGLE_USERNAME", "operator")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/Users/operator/.kaggle")
    claude = ClaudeRunner(binary=Path("claude"), model="m", effort="high",
                          api_key_env="IOAI_LLM_API_KEY")
    codex = CodexRunner(binary=Path("codex"), model="m", effort="high",
                        api_key_env="IOAI_LLM_API_KEY")
    for env in (claude.environment(tmp_path / "a", require_key=True),
                codex.environment(tmp_path / "b", require_key=True)):
        assert not [key for key in env if key.startswith("KAGGLE")]
        # HOME is redirected, so ~/.kaggle/kaggle.json is not discoverable either.
        assert env["HOME"].startswith(str(tmp_path))


def test_claude_runner_targets_openrouter_and_carries_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IOAI_LLM_API_KEY", "azure-test-key")
    runner = ClaudeRunner(binary=Path("claude"), model="claude-fable-5",
                          effort="high", api_key_env="IOAI_LLM_API_KEY")
    env = runner.environment(tmp_path, require_key=True)
    assert env["ANTHROPIC_BASE_URL"] == DEFAULT_ANTHROPIC_BASE
    assert env["ANTHROPIC_AUTH_TOKEN"] == "azure-test-key"
    assert env["ANTHROPIC_API_KEY"] == "azure-test-key"
    # Load-bearing: without it Claude Code sends an anthropic-beta header the
    # Azure endpoint rejects with 400.
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "claude-fable-5" in runner.argv(resume=None)


def test_solvers_are_not_bare_because_bare_has_no_write_tool() -> None:
    """`--bare` grants only Bash/Edit/Read, and no flag adds Write back."""
    runner = ClaudeRunner(binary=Path("claude"), model="m", effort="high",
                          api_key_env="IOAI_LLM_API_KEY")
    argv = runner.argv(resume=None)
    assert "--bare" not in argv
    assert "Write" in argv[argv.index("--tools") + 1]


def test_manager_is_given_no_tools_at_all() -> None:
    runner = ClaudeRunner(binary=Path("claude"), model="m", effort="high",
                          api_key_env="IOAI_LLM_API_KEY", tools="")
    argv = runner.argv(resume=None)
    assert argv[argv.index("--tools") + 1] == ""
    # Without tools there is nothing to permit, so no bypass either.
    assert "bypassPermissions" not in argv


def test_a_lane_can_be_switched_between_clients_without_changing_its_model(
    tmp_path: Path,
) -> None:
    """The escape hatch for Codex's OpenRouter auth failure."""
    text = CONFIG.read_text().replace('runner = "codex"', 'runner = "claude_code"')
    swapped = tmp_path / "swapped.toml"
    swapped.write_text(text)
    config = BackupConfig.load(swapped, repo_root=ROOT)
    assert config.codex.runner == "claude_code"
    assert config.codex.model == "gpt-5.6-sol"     # model diversity preserved
    assert config.codex.binary.name == "claude"


def test_unknown_runner_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text().replace('runner = "codex"', 'runner = "aider"')
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="runner"):
        BackupConfig.load(bad, repo_root=ROOT)


def test_codex_argv_points_at_the_configured_endpoint_and_hides_the_key() -> None:
    runner = CodexRunner(binary=Path("codex"), model="gpt-5.6-sol",
                         effort="high", api_key_env="IOAI_LLM_API_KEY")
    argv = runner.argv(workdir=Path("/tmp/w"), last_message=Path("/tmp/m"),
                       resume=None)
    joined = " ".join(argv)
    assert DEFAULT_OPENAI_BASE in joined
    assert 'wire_api="responses"' in joined
    assert 'shell_environment_policy.filters.IOAI_LLM_API_KEY="exclude"' in joined


def test_runner_requires_the_key_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IOAI_LLM_API_KEY", raising=False)
    runner = ClaudeRunner(binary=Path("claude"), model="m", effort="high",
                          api_key_env="IOAI_LLM_API_KEY")
    runner.environment(tmp_path, require_key=False)      # doctor path: fine
    with pytest.raises(RuntimeError, match="IOAI_LLM_API_KEY"):
        runner.environment(tmp_path, require_key=True)


# ------------------------------------------------------------------ submission
def test_template_is_found_under_the_ioai_name_not_only_sample_submission(
    tmp_path: Path,
) -> None:
    _template(tmp_path / "assets")
    found = find_submission_template(tmp_path / "assets")
    assert found is not None and found.name == "submission.csv"


def test_missing_template_fails_closed_instead_of_skipping_every_check(
    tmp_path: Path,
) -> None:
    (tmp_path / "assets").mkdir()
    bad = tmp_path / "cand.csv"
    bad.write_text("id,target\nr0,0\n")
    valid, errors = validate_submission(tmp_path / "assets", bad)
    assert not valid
    assert any("template" in error for error in errors)


def test_wrong_row_count_and_wrong_ids_are_both_caught(tmp_path: Path) -> None:
    _template(tmp_path / "assets", rows=20)
    short = tmp_path / "short.csv"
    short.write_text("id,target\n" + "".join(f"r{i},0\n" for i in range(5)))
    valid, errors = validate_submission(tmp_path / "assets", short)
    assert not valid and any("row count" in error for error in errors)

    scrambled = tmp_path / "scrambled.csv"
    scrambled.write_text("id,target\n" + "".join(f"x{i},0\n" for i in range(20)))
    valid, errors = validate_submission(tmp_path / "assets", scrambled)
    assert not valid and any("first-column" in error for error in errors)


def test_a_correct_submission_passes(tmp_path: Path) -> None:
    template = _template(tmp_path / "assets", rows=20)
    good = tmp_path / "good.csv"
    good.write_text(template.read_text())
    valid, errors = validate_submission(tmp_path / "assets", good)
    assert valid, errors


def test_kernel_must_be_a_single_offline_self_contained_script(tmp_path: Path) -> None:
    directory = _candidate(tmp_path / "c", "multi")
    (directory / "helper.py").write_text("X = 1\n")
    valid, errors = validate_kernel(directory)
    assert not valid and any("only kernel.py" in error for error in errors)

    installer = _candidate(tmp_path / "d", "pip",
                           kernel="!pip install torch\nopen('submission.csv','w')\n")
    valid, errors = validate_kernel(installer)
    assert not valid and any("offline" in error for error in errors)


def test_floor_kernel_copies_the_template_and_compiles() -> None:
    compile(floor_kernel(), "floor.py", "exec")
    assert "submission.csv" in floor_kernel()


# ---------------------------------------------------------------------- board
def test_board_never_exceeds_its_budget_under_concurrent_reservation(
    tmp_path: Path,
) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=3)
    for index in range(8):
        _candidate(board.candidates, f"cand{index}")
    taken = [board.reserve(c, "milestone")
             for c in board.ready_candidates()]
    assert sum(1 for t in taken if t is not None) == 3
    assert board.remaining() == 0


def test_one_candidate_cannot_hold_two_slots_at_once(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=10)
    _candidate(board.candidates, "only")
    candidate = board.ready_candidates()[0]
    assert board.reserve(candidate, "floor") is not None
    assert board.reserve(candidate, "milestone") is None


def test_a_transient_failure_is_retried_and_a_permanent_one_is_not(
    tmp_path: Path,
) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=10)
    _candidate(board.candidates, "flaky")
    _candidate(board.candidates, "broken")

    flaky = next(c for c in board.ready_candidates() if c["candidate_id"] == "flaky")
    attempt = board.reserve(flaky, "milestone")
    board.update(attempt.submission_id, status="deferred", retry_after=0.0)
    assert "flaky" in {c["candidate_id"]
                       for c in board.submittable(retry_attempts=3)}

    broken = next(c for c in board.ready_candidates() if c["candidate_id"] == "broken")
    attempt = board.reserve(broken, "milestone")
    board.update(attempt.submission_id, status="rejected", detail="two .py files")
    assert "broken" not in {c["candidate_id"]
                            for c in board.submittable(retry_attempts=3)}


def test_backoff_hides_a_candidate_until_retry_after_passes(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=10)
    _candidate(board.candidates, "waiting")
    attempt = board.reserve(board.ready_candidates()[0], "milestone")
    board.update(attempt.submission_id, status="deferred",
                 retry_after=time.time() + 300)
    assert board.submittable(retry_attempts=3) == []
    board.update(attempt.submission_id, retry_after=time.time() - 1)
    assert len(board.submittable(retry_attempts=3)) == 1


def test_retry_budget_is_finite(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=10)
    _candidate(board.candidates, "hopeless")
    for _ in range(3):
        candidate = board.submittable(retry_attempts=3)[0]
        attempt = board.reserve(candidate, "milestone")
        board.update(attempt.submission_id, status="retryable", retry_after=0.0)
    assert board.submittable(retry_attempts=3) == []


def test_half_written_candidates_are_invisible_until_ready(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=5)
    staging = board.candidates / ".staging-x"
    staging.mkdir(parents=True)
    (staging / "kernel.py").write_text("print(1)\n")
    (staging / "meta.json").write_text("{}")
    assert board.ready_candidates() == []
    (staging / "READY").touch()
    assert len(board.ready_candidates()) == 1


def test_digest_reports_scores_and_never_invents_one(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=5)
    assert "nothing has scored yet" in board.digest()
    _candidate(board.candidates, "cand0", author="claude")
    attempt = board.reserve(board.ready_candidates()[0], "floor")
    board.update(attempt.submission_id, status="scored", lb_score=0.8745)
    digest = board.digest()
    assert "0.8745" in digest and "claude" in digest
    assert board.best().lb_score == 0.8745


def test_feedback_is_appended_where_both_agents_read_it(tmp_path: Path) -> None:
    board = Board(tmp_path / "b", slug="t", max_submissions=5)
    board.feedback({"event": "leaderboard_score", "leaderboard_score": 0.5})
    board.feedback({"event": "leaderboard_score", "leaderboard_score": 0.6})
    lines = board.feedback_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["leaderboard_score"] == 0.6


# ------------------------------------------------------------------- dry run
def test_dry_run_adapter_still_enforces_the_kernel_contract(tmp_path: Path) -> None:
    adapter = DryRunKaggle("slug", tmp_path / "k", tmp_path / "assets")
    good = _candidate(tmp_path / "c", "ok")
    outcome = adapter.submit({"path": str(good), "candidate_id": "ok"}, "s1", "m")
    assert outcome.status == "submitted"

    bad = _candidate(tmp_path / "d", "bad")
    (bad / "helper.py").write_text("X=1\n")
    outcome = adapter.submit({"path": str(bad), "candidate_id": "bad"}, "s2", "m")
    assert outcome.status == "rejected"


def test_kaggle_slots_are_shared_across_the_three_parts() -> None:
    """Two concurrent kernels account-wide, whichever tasks want them."""

    async def scenario() -> int:
        slots = asyncio.Semaphore(2)
        peak = 0
        live = 0
        lock = asyncio.Lock()

        async def one() -> None:
            nonlocal peak, live
            async with slots:
                async with lock:
                    live += 1
                    peak = max(peak, live)
                await asyncio.sleep(0.02)
                async with lock:
                    live -= 1

        await asyncio.gather(*(one() for _ in range(9)))
        return peak

    assert asyncio.run(scenario()) == 2


def test_flipping_only_the_runner_moves_the_lane_to_the_matching_endpoint(
    tmp_path: Path,
) -> None:
    """The escape hatch must be genuinely one line: endpoints derive from it."""
    base = BackupConfig.load(CONFIG, repo_root=ROOT)
    assert base.codex.base_url.endswith("/openai/v1")

    flipped = tmp_path / "flipped.toml"
    flipped.write_text(
        CONFIG.read_text().replace('runner = "codex"', 'runner = "claude_code"')
    )
    config = BackupConfig.load(flipped, repo_root=ROOT)
    assert config.codex.runner == "claude_code"
    assert config.codex.model == "gpt-5.6-sol"        # model diversity kept
    assert config.codex.base_url.endswith("/anthropic")   # endpoint followed


def test_a_base_url_that_mismatches_its_client_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(CONFIG.read_text().replace(
        'runner = "claude_code"\nmodel = "claude-fable-5"',
        'runner = "claude_code"\nbase_url = "https://h/anthropic/v1"\n'
        'model = "claude-fable-5"',
    ))
    with pytest.raises(ValueError, match="must NOT end in /v1"):
        BackupConfig.load(bad, repo_root=ROOT)


def test_changing_the_host_moves_both_endpoints(tmp_path: Path) -> None:
    moved = tmp_path / "moved.toml"
    moved.write_text(CONFIG.read_text().replace(
        'llm_host = "https://prosgrow-aoai-prod.services.ai.azure.com"',
        'llm_host = "https://other.example.com"',
    ))
    config = BackupConfig.load(moved, repo_root=ROOT)
    assert config.codex.base_url == "https://other.example.com/openai/v1"
    assert config.claude.base_url == "https://other.example.com/anthropic"
    assert config.manager.base_url == "https://other.example.com/anthropic"


def test_the_api_key_never_appears_in_any_tracked_file() -> None:
    """The key lives in .env only. This test fails loudly if it leaks."""
    env = ROOT / ".env"
    if not env.is_file():
        pytest.skip(".env not present in this checkout")
    secret = ""
    for line in env.read_text().splitlines():
        if line.startswith("IOAI_LLM_API_KEY="):
            secret = line.partition("=")[2].strip()
    assert secret, "IOAI_LLM_API_KEY missing from .env"
    for path in (CONFIG, ROOT / ".env.example",
                 ROOT / "backup_system" / "README.md"):
        assert secret not in path.read_text(), f"key leaked into {path}"


# ------------------------------------------------------- IOAI rule compliance
from backup_system.intake import (  # noqa: E402
    IntakeError,
    build_task,
    coerce_timeout,
    extract_json,
    parse_deadline,
)
from backup_system.submit import Kaggle  # noqa: E402

_LIMITS = {"num_allowed_now": 50, "num_today": 0}


def test_push_passes_the_kernel_timeout_flag_ioai_requires() -> None:
    """A solution pushed without --timeout is invalid however well it scores."""
    import inspect
    from swarm.submit import kernel as k
    source = inspect.getsource(k.push_kernel)
    assert "--timeout" in source or '"-t"' in source
    assert "kernel_timeout_s" in inspect.signature(k.push_kernel).parameters


def test_backup_adapter_actually_forwards_the_task_time_limit() -> None:
    import inspect
    source = inspect.getsource(Kaggle.submit)
    assert "kernel_timeout_s=self.kernel_run_timeout_s" in source


def test_intake_refuses_to_start_without_a_kernel_time_limit() -> None:
    with pytest.raises(IntakeError, match="kernel time limit"):
        build_task({"slug": "some-comp", "deadline_iso": "2099-01-01T00:00:00+00:00"},
                   raw_prompt="x", limits=_LIMITS, fallback_minutes=60)


def test_intake_refuses_an_implausible_slug() -> None:
    with pytest.raises(IntakeError, match="slug"):
        build_task({"slug": "https://kaggle.com/c/foo", "kernel_timeout_seconds": 60},
                   raw_prompt="x", limits=_LIMITS, fallback_minutes=60)


def test_intake_refuses_a_timeout_beyond_kaggles_own_ceiling() -> None:
    assert coerce_timeout(13 * 3600) is None
    assert coerce_timeout(0) is None
    assert coerce_timeout("3600") == 3600


def test_a_deadline_without_a_timezone_is_refused_not_guessed() -> None:
    assert parse_deadline("2026-08-04T14:00:00") is None
    assert parse_deadline("2026-08-04T14:00:00Z") is not None
    assert parse_deadline("tomorrow afternoon") is None


def test_deadline_falls_back_to_the_operator_window_only_when_named() -> None:
    task = build_task(
        {"slug": "comp-x", "kernel_timeout_seconds": 1800},
        raw_prompt="x", limits=_LIMITS, fallback_minutes=90,
    )
    assert task.deadline_source == "operator_minutes"
    assert 89 <= task.minutes_left() <= 90
    with pytest.raises(IntakeError, match="deadline"):
        build_task({"slug": "comp-x", "kernel_timeout_seconds": 1800},
                   raw_prompt="x", limits=_LIMITS, fallback_minutes=None)


def test_an_absolute_deadline_in_the_prompt_wins_over_the_operator() -> None:
    import time as _t
    future = datetime.fromtimestamp(_t.time() + 7200, UTC).isoformat()
    task = build_task(
        {"slug": "comp-x", "kernel_timeout_seconds": 1800, "deadline_iso": future},
        raw_prompt="x", limits=_LIMITS, fallback_minutes=10,
    )
    assert task.deadline_source == "starter_prompt"
    assert 119 <= task.minutes_left() <= 120


def test_the_budget_comes_from_kaggle_not_from_the_task_text() -> None:
    task = build_task(
        {"slug": "comp-x", "kernel_timeout_seconds": 900,
         "max_submissions": 999},              # the text does not get a vote
        raw_prompt="x", limits={"num_allowed_now": 7, "num_today": 3},
        fallback_minutes=60,
    )
    assert task.max_submissions == 7
    assert task.submissions_used_today == 3
    with pytest.raises(IntakeError, match="budget"):
        build_task({"slug": "comp-x", "kernel_timeout_seconds": 900},
                   raw_prompt="x", limits={"num_allowed_now": None},
                   fallback_minutes=60)


def test_the_starter_prompt_reaches_the_solver_verbatim() -> None:
    """Report instructions we never anticipated must survive intake untouched."""
    starter = ("SOLVE THIS TASK.\n"
               "Your submitted code MUST begin with a short Report describing it.\n"
               "Kernel time limit: 9 hours.")
    from backup_system.prompts import solver_prompt as sp
    text = sp(agent="codex", peer="claude", slug="c", workdir=Path("/w"),
              candidates_dir=Path("/w/c"), assets_dir=Path("/a"),
              feedback_path=Path("/w/f.jsonl"), minutes_left=60, board="-",
              starter_prompt=starter, kernel_timeout_seconds=32400)
    for line in starter.splitlines():
        assert line in text, f"starter prompt line was lost: {line!r}"
    assert "32400" in text


def test_the_system_downloads_the_data_itself() -> None:
    """The rules make this the agent's step one, not the operator's."""
    import inspect
    assert "competitions" in inspect.getsource(Kaggle.download_data)
    assert "download" in inspect.getsource(Kaggle.download_data)
    from backup_system.run import BackupRun as _BR
    assert "download_data" in inspect.getsource(_BR.fetch_data)


def test_live_runs_refuse_the_operator_prepared_data_path() -> None:
    """--task hands over a directory a human built; a scored run may not."""
    import subprocess
    done = subprocess.run(
        [sys.executable, "-m", "backup_system", "run",
         "--task", f"slug={ROOT}", "--kernel-timeout-seconds", "60", "--live",
         "--kaggle-user", "someone"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert done.returncode != 0
    assert "rehearsal path" in done.stderr


def test_rehearsal_still_demands_a_kernel_timeout() -> None:
    import subprocess
    done = subprocess.run(
        [sys.executable, "-m", "backup_system", "run", "--task", f"slug={ROOT}"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert done.returncode != 0
    assert "kernel-timeout-seconds" in done.stderr


def test_every_agent_prompt_states_the_scoring_goal_and_its_two_hard_limits() -> None:
    """Being told to be resourceful is useless if it gets the entry disqualified."""
    from backup_system.prompts import INTAKE_PROMPT, manager_prompt
    from backup_system.prompts import solver_prompt as sp
    solver = sp(agent="codex", peer="claude", slug="c", workdir=Path("/w"),
                candidates_dir=Path("/w/c"), assets_dir=Path("/a"),
                feedback_path=Path("/w/f"), minutes_left=60, board="-")
    manager = manager_prompt(slug="c", board="-", candidates="-",
                             minutes_left=60, remaining=10)
    for name, text in (("solver", solver), ("manager", manager)):
        assert "排行榜分数做到最高" in text, f"{name} prompt lost the goal"
        assert "绕过 Kaggle 的提交系统" in text, f"{name} prompt lost the rule boundary"
        assert "隐藏测试集的标签" in text, f"{name} prompt lost the label boundary"
    assert "取得尽可能高的排行榜分数" in INTAKE_PROMPT


def test_a_zero_timeout_is_refused_before_it_reaches_kaggle() -> None:
    """Measured: Kaggle happily accepts `--timeout 0`, which is the invalid state."""
    from swarm.submit.kernel import push_kernel
    import tempfile, json as _json
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "kernel-metadata.json").write_text(_json.dumps({"id": "u/k"}))
        (d / "solution.py").write_text("print(1)\n")
        for bad in (0, -30):
            ok, _v, raw = push_kernel(d, kernel_timeout_s=bad)
            assert not ok and "positive" in raw
