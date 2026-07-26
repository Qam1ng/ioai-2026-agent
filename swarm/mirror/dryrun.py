"""Local rehearsal of a Kaggle kernel, before a real submission is spent.

The failure this module prevents is the expensive one.  A generated kernel
looks fine, gets pushed, sits in the Kaggle queue for twenty minutes, and then
dies on a typo, a missing package, a wrong ``/kaggle/input`` path, or a
submission file with the wrong number of rows.  One of the 50 submissions is
gone and — worse on a 6-hour clock — so is the wall time.

So we rehearse.  ``dry_run`` stages a fake ``/kaggle`` tree from local data,
rewrites the two Kaggle path prefixes into it, runs the kernel in a real
subprocess under a hard wall-clock timeout, and reports what came out.
``validate_submission`` then compares the produced file against the sample
submission the way the Kaggle grader will.

Design decisions worth stating:

* **Rewrite, don't mock.**  We cannot create ``/kaggle`` on a laptop without
  root, and monkeypatching ``open``/``os.path`` inside the child is fragile
  and lies about what the kernel does.  Rewriting the two literal prefixes is
  boring, deterministic, and inspectable — the rewritten file is left on disk.
* **Report, never refuse.**  Static violations from ``packages.py`` are
  recorded on the result and the kernel runs anyway.  A false alarm must not
  be able to block a rehearsal; a real one shows up in ``violations`` next to
  the traceback that confirms it.
* **Subprocess, not exec.**  A kernel that segfaults, leaks memory, or calls
  ``sys.exit`` must not take the swarm down with it.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .packages import check_kernel

#: What the kernel believes its filesystem looks like.
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

#: How much of each stream we keep.  Enough for a traceback plus context,
#: small enough to paste into a blackboard event.
TAIL_CHARS = 6000

_MAX_FIELD = 10_000_000  # radar submissions are ~9k columns wide


def _widen_csv_limits() -> None:
    """Let ``csv`` handle the very wide submissions some tasks produce."""
    try:
        if csv.field_size_limit() < _MAX_FIELD:
            csv.field_size_limit(_MAX_FIELD)
    except OverflowError:  # pragma: no cover - platform dependent
        pass


def _tail(text: str, n: int = TAIL_CHARS) -> str:
    if len(text) <= n:
        return text
    return "...[truncated]...\n" + text[-n:]


# ------------------------------------------------------------------- staging


def kaggle_root_of(competition_dir: Path) -> Path:
    """Find the fake ``/kaggle`` root that owns ``competition_dir``.

    ``competition_dir`` is expected to be ``<root>/kaggle/input/competitions/<slug>``.
    We walk up looking for a directory literally named ``kaggle`` so callers can
    nest the sandbox wherever they like; if there is none we fall back to three
    levels up, which is the same place for the canonical layout.
    """
    for parent in competition_dir.parents:
        if parent.name == "kaggle":
            return parent
    parents = list(competition_dir.parents)
    return parents[2] if len(parents) >= 3 else competition_dir.parent


def stage_input(dest: Path, sources: dict[str, Path]) -> Path:
    """Build a fake competition-data directory and a sibling ``working`` dir.

    ``dest`` is the competition directory itself, i.e. the local stand-in for
    ``/kaggle/input/competitions/<slug>/`` — one level deeper than people
    expect, which is exactly the trap recorded in
    ``memory/lessons/kaggle-kernel-input-paths.md``.  Staging at the real depth
    means a kernel that hardcodes the shallow path fails *here* instead of on
    the leaderboard.

    ``sources`` maps the name the kernel will see (``"train.csv"``,
    ``"test_scenarios.pkl"``, ``"training_set"``) to a local file or directory.
    Entries are symlinked when possible — the radar mirror alone is 760 MB and
    copying it per rehearsal would dominate the run.  If symlinking is not
    available (some sandboxes, Windows) we fall back to copying.

    Returns ``dest``.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (kaggle_root_of(dest) / "working").mkdir(parents=True, exist_ok=True)

    for name, src in sources.items():
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(f"stage_input: source for {name!r} does not exist: {src}")
        target = dest / name
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            _mirror_tree(src, target)
        else:
            _link_or_copy(src, target)
    return dest


def _link_or_copy(src: Path, target: Path) -> None:
    try:
        target.symlink_to(src.resolve())
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        shutil.copy2(src, target)


def _mirror_tree(src: Path, target: Path) -> None:
    """Recreate ``src``'s directory structure with real dirs and symlinked files.

    A single symlink to the whole directory would be cheaper still, but
    ``os.walk`` does not descend into symlinked directories by default — and
    walking ``/kaggle/input`` is the discovery pattern every kernel is required
    to use, because the competition mount point is not knowable in advance. A
    directory symlink therefore makes the staged tree look *empty* to exactly
    the code we are trying to rehearse, turning a healthy kernel into a false
    failure. Mirroring directories and linking only leaves keeps the walk
    faithful while still avoiding a copy of the data.
    """
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        dst = target / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                _link_or_copy(path, dst)


def rewrite_kaggle_paths(code: str, input_root: Path, working_root: Path) -> str:
    """Point a Kaggle-path kernel at the staged tree.

    Only the two absolute prefixes are touched, longest first so that
    ``/kaggle/working`` is not half-rewritten by the ``/kaggle`` in
    ``/kaggle/input``.  Everything else — including the ``os.walk`` discovery
    logic every good kernel should have — is left exactly as submitted.
    """
    return code.replace(KAGGLE_WORKING, str(working_root)).replace(
        KAGGLE_INPUT, str(input_root)
    )


# -------------------------------------------------------------------- result


@dataclass
class DryRunResult:
    """Everything the swarm needs to decide whether this kernel is submittable."""

    ok: bool = False
    exit_code: int = -1
    wall_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    violations: list[str] = field(default_factory=list)
    submission_path: str = ""
    submission_rows: int = 0
    timed_out: bool = False
    kernel_path: str = ""
    working_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "wall_s": round(self.wall_s, 1),
            "violations": self.violations,
            "submission_path": self.submission_path,
            "submission_rows": self.submission_rows,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail[-2000:],
            "stderr_tail": self.stderr_tail[-2000:],
        }

    def summary(self) -> str:
        state = "OK" if self.ok else ("TIMEOUT" if self.timed_out else "FAIL")
        return (
            f"[dry-run {state}] exit={self.exit_code} wall={self.wall_s:.1f}s "
            f"rows={self.submission_rows} violations={len(self.violations)}"
        )


# ------------------------------------------------------------------- running


def _find_submission(working: Path) -> tuple[str, int]:
    """Locate the produced submission and count its data rows.

    Prefers ``submission.csv`` (the Kaggle default), then any other top-level
    CSV, then the zip artifacts some IOAI tasks ask for (``predictions.zip``).
    Row count is data rows, header excluded; for a zip we report 0 rows because
    row semantics are task-specific.
    """
    _widen_csv_limits()
    candidates = [working / "submission.csv"]
    candidates += sorted(p for p in working.glob("*.csv") if p.name != "submission.csv")
    candidates += sorted(working.glob("*.zip"))
    for cand in candidates:
        if not cand.exists() or not cand.is_file():
            continue
        if cand.suffix == ".zip":
            return str(cand), 0
        rows = 0
        with cand.open(newline="") as fh:
            reader = csv.reader(fh)
            for i, _row in enumerate(reader):
                if i == 0:
                    continue
                rows += 1
        return str(cand), rows
    return "", 0


def dry_run(
    code: str,
    slug: str,
    input_sources: dict[str, Path],
    workdir: Path,
    timeout_s: float = 1800,
    python_exe: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> DryRunResult:
    """Run a kernel locally exactly as if it were the Kaggle scoring kernel.

    ``timeout_s`` defaults to 30 minutes because that is roughly the training
    budget an IOAI kernel is expected to fit into; a rehearsal that overruns it
    is itself the finding.  The child is started in its own process group and
    the whole group is killed on timeout, so a kernel that spawns workers
    (``DataLoader(num_workers=…)`` is the usual culprit) cannot leave orphans
    behind.

    Static checks run first and land in ``result.violations``.  They never
    prevent execution: a false alarm must not be able to block a rehearsal, and
    a real one is easier to trust when the traceback agrees with it.
    """
    t0 = time.time()
    workdir = Path(workdir).resolve()
    kaggle = workdir / "kaggle"
    comp_dir = kaggle / "input" / "competitions" / slug
    stage_input(comp_dir, input_sources)
    input_root = kaggle / "input"
    working = kaggle / "working"
    working.mkdir(parents=True, exist_ok=True)

    result = DryRunResult(working_dir=str(working))
    result.violations = check_kernel(code)

    rewritten = rewrite_kaggle_paths(code, input_root, working)
    kernel_path = workdir / "kernel_dryrun.py"
    kernel_path.write_text(rewritten)
    result.kernel_path = str(kernel_path)

    env = dict(os.environ)
    # A kernel is free to read these instead of hardcoding paths; the staged
    # tree and the env agree, so either style works under rehearsal.
    env.update(
        {
            "KAGGLE_INPUT": str(input_root),
            "KAGGLE_WORKING": str(working),
            "KAGGLE_ROOT": str(kaggle),
            "KAGGLE_COMPETITION_DIR": str(comp_dir),
            "KAGGLE_KERNEL_RUN_TYPE": "Interactive",
            "IOAI_DRY_RUN": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if env_overrides:
        env.update(env_overrides)

    exe = python_exe or sys.executable
    popen_kwargs: dict = {}
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [exe, str(kernel_path)],
        cwd=str(working),  # Kaggle kernels run with cwd == /kaggle/working
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **popen_kwargs,
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result.timed_out = True
        _kill_group(proc)
        out, err = proc.communicate()
        err = (err or "") + f"\n[dry-run] killed after {timeout_s:.0f}s wall-clock timeout"

    result.exit_code = proc.returncode if proc.returncode is not None else -1
    result.stdout_tail = _tail(out or "")
    result.stderr_tail = _tail(err or "")
    result.wall_s = time.time() - t0

    sub_path, rows = _find_submission(working)
    result.submission_path = sub_path
    result.submission_rows = rows
    result.ok = (
        not result.timed_out and result.exit_code == 0 and bool(sub_path) and rows >= 0
    )
    return result


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child and anything it spawned."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - platform dependent
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
        try:
            proc.kill()
        except Exception:
            pass


# ----------------------------------------------------------------- validation


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    _widen_csv_limits()
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return [h.strip() for h in header], [row for row in reader if row]


def _as_float(cell: str) -> float | None:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def validate_submission(
    path: Path,
    sample_path: Path,
    id_col: str | None = None,
    value_range: tuple[float, float] | None = None,
    max_report: int = 5,
) -> list[str]:
    """Compare a produced submission against the sample the way the grader does.

    Returns a list of problems; an empty list means the file is structurally
    valid.  This is the check the existing system lacks — every one of these
    failures scores as an error on Kaggle and costs a submission, while being
    completely detectable offline in milliseconds.

    Checked:

    * file exists and is non-empty;
    * identical column names **in identical order** (Kaggle matches by name,
      but an order mismatch almost always means the kernel built the frame
      wrong, so it is reported);
    * identical row count;
    * identical id-column values in identical order — the single most common
      real failure, because a shuffled ``DataLoader`` or a ``glob`` without
      ``sorted()`` silently reorders predictions;
    * no NaN / inf / empty cells;
    * values inside the plausible label range.

    The range check is honest about what it can know.  Sample submissions are
    usually a constant placeholder, so a range derived from them would be
    degenerate; when that happens we skip the range test and instead require
    that a sample of integers stays integers (which catches the real bug of
    writing class probabilities where class ids belong).  Pass ``value_range``
    explicitly when the task card states the true label range.
    """
    problems: list[str] = []
    path, sample_path = Path(path), Path(sample_path)

    if not path.exists():
        return [f"submission not found: {path}"]
    if path.stat().st_size == 0:
        return [f"submission is empty: {path}"]
    if not sample_path.exists():
        return [f"sample submission not found: {sample_path}"]

    sub_header, sub_rows = _read_csv(path)
    smp_header, smp_rows = _read_csv(sample_path)

    if not sub_header:
        return [f"submission has no header row: {path}"]
    if not smp_header:
        return [f"sample submission has no header row: {sample_path}"]

    # --- columns -------------------------------------------------------
    if sub_header != smp_header:
        missing = [c for c in smp_header if c not in sub_header]
        extra = [c for c in sub_header if c not in smp_header]
        if missing or extra:
            problems.append(
                f"column set differs: missing={missing[:max_report]} "
                f"extra={extra[:max_report]} "
                f"(sample has {len(smp_header)} columns, submission has {len(sub_header)})"
            )
        else:
            problems.append(
                f"column order differs: sample starts {smp_header[:max_report]}, "
                f"submission starts {sub_header[:max_report]}"
            )

    # --- row count -----------------------------------------------------
    if len(sub_rows) != len(smp_rows):
        problems.append(
            f"row count differs: sample has {len(smp_rows)}, submission has {len(sub_rows)}"
        )

    # --- id column -----------------------------------------------------
    key = id_col or smp_header[0]
    if key not in sub_header:
        problems.append(f"id column {key!r} missing from submission")
    else:
        si, mi = sub_header.index(key), smp_header.index(key)
        sub_ids = [r[si] if si < len(r) else "" for r in sub_rows]
        smp_ids = [r[mi] if mi < len(r) else "" for r in smp_rows]
        if sub_ids != smp_ids:
            if sorted(sub_ids) == sorted(smp_ids):
                first = next(
                    (i for i, (a, b) in enumerate(zip(sub_ids, smp_ids)) if a != b), 0
                )
                problems.append(
                    f"id column {key!r} has the right values in the WRONG ORDER "
                    f"(first difference at row {first}: submission {sub_ids[first]!r} "
                    f"vs sample {smp_ids[first]!r}) — row order is part of the format"
                )
            else:
                miss = [i for i in smp_ids if i not in set(sub_ids)][:max_report]
                extra = [i for i in sub_ids if i not in set(smp_ids)][:max_report]
                problems.append(
                    f"id column {key!r} values differ: missing={miss} extra={extra}"
                )

    # --- cell-level checks ---------------------------------------------
    value_cols = [c for c in sub_header if c != key]
    if value_cols and sub_rows:
        col_idx = [sub_header.index(c) for c in value_cols]
        bad_cells: list[str] = []
        nonnumeric = 0
        lo_seen, hi_seen = math.inf, -math.inf
        all_int = True
        for r_i, row in enumerate(sub_rows):
            if len(row) != len(sub_header):
                bad_cells.append(f"row {r_i} has {len(row)} fields, expected {len(sub_header)}")
                continue
            for c_i in col_idx:
                cell = row[c_i].strip()
                if cell == "" or cell.lower() in ("nan", "none", "null", "na"):
                    bad_cells.append(f"row {r_i} col {sub_header[c_i]!r} is empty/NaN")
                    continue
                v = _as_float(cell)
                if v is None:
                    nonnumeric += 1
                    continue
                if math.isnan(v) or math.isinf(v):
                    bad_cells.append(f"row {r_i} col {sub_header[c_i]!r} is {cell!r}")
                    continue
                lo_seen, hi_seen = min(lo_seen, v), max(hi_seen, v)
                if v != int(v):
                    all_int = False
            if len(bad_cells) > max_report:
                break
        if bad_cells:
            problems.append(
                f"{len(bad_cells)}+ invalid cells; first: " + "; ".join(bad_cells[:max_report])
            )

        # Range: explicit > derived-from-sample > skipped (with an int guard).
        rng = value_range
        smp_all_int = True
        if rng is None and smp_rows:
            smp_idx = [smp_header.index(c) for c in value_cols if c in smp_header]
            lo, hi = math.inf, -math.inf
            for row in smp_rows:
                for c_i in smp_idx:
                    if c_i >= len(row):
                        continue
                    v = _as_float(row[c_i])
                    if v is None or math.isnan(v) or math.isinf(v):
                        continue
                    lo, hi = min(lo, v), max(hi, v)
                    if v != int(v):
                        smp_all_int = False
            if lo < hi:  # non-degenerate sample -> a usable range
                rng = (lo, hi)
        if rng is not None and nonnumeric == 0 and lo_seen <= hi_seen:
            lo, hi = rng
            if lo_seen < lo or hi_seen > hi:
                problems.append(
                    f"values outside the expected label range [{lo:g}, {hi:g}]: "
                    f"submission spans [{lo_seen:g}, {hi_seen:g}]"
                )
        elif rng is None and smp_all_int and not all_int and nonnumeric == 0:
            problems.append(
                "sample submission holds integer labels but the submission holds "
                "non-integer values — writing probabilities/logits where class ids "
                "are expected scores as garbage, not as an error"
            )
    return problems
