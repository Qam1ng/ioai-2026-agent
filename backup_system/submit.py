"""The Kaggle path. No language model anywhere in this file.

Pushing a kernel, waiting for it, reading the output back and calling submit
involves no judgement, and every failure the earlier system had on this path was
plumbing. The kernel push/poll primitives come from `swarm.submit.kernel`, which
is the most heavily tested code in this repository -- rewriting it here would
mean re-earning bugs that are already fixed (the `KernelWorkerStatus.COMPLETE`
status parse, above all).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from agent.tools.registry import _kaggle_bin
from swarm.submit.kernel import (
    make_metadata,
    poll_kernel,
    push_kernel,
    unique_kernel_id,
    write_kernel,
)

#: Our own marker in the submission description; the only way to map a Kaggle
#: row back to a local attempt.
TOKEN = "bk"
_TOKEN_RE = re.compile(rf"\[{TOKEN}:([^\]]+)\]")


@dataclass
class SubmitOutcome:
    status: str          # submitted | deferred | retryable | rejected | execution_error
    detail: str = ""
    kernel_ref: str = ""
    kernel_version: int | None = None


def find_submission_template(assets: Path) -> Path | None:
    """Locate the official submission template.

    IOAI ships it as plain `submission.csv`, NOT `sample_submission.csv`. A
    matcher that only accepts the latter finds nothing, silently skips every
    structural check, and lets a malformed file consume a submission.
    """
    ranked: list[tuple[int, Path]] = []
    for path in Path(assets).rglob("*.csv"):
        name = path.name.lower().replace("-", "_")
        if "sample" in name and "submission" in name:
            ranked.append((0, path))
        elif name == "submission.csv":
            ranked.append((1, path))
    if not ranked:
        return None
    return min(ranked, key=lambda item: (item[0], len(item[1].parts), str(item[1])))[1]


def validate_submission(assets: Path, candidate_csv: Path) -> tuple[bool, list[str]]:
    """Check what a deterministic script can actually establish about a CSV."""
    candidate_csv = Path(candidate_csv)
    if not candidate_csv.is_file() or candidate_csv.stat().st_size == 0:
        return False, ["submission.csv is missing or empty"]
    try:
        with candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        return False, [f"submission.csv unreadable: {exc}"]
    if not rows or not rows[0]:
        return False, ["submission.csv has no header"]
    errors: list[str] = []
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        errors.append("rows have inconsistent column counts")
    template = find_submission_template(assets)
    if template is None:
        errors.append("no official submission template found in the assets")
        return not errors, errors
    with template.open(newline="", encoding="utf-8-sig") as handle:
        expected = list(csv.reader(handle))
    if not expected or not expected[0]:
        return False, ["official submission template has no header"]
    if rows[0] != expected[0]:
        errors.append(f"header mismatch: expected {expected[0]}, got {rows[0]}")
    if len(rows) != len(expected):
        errors.append(
            f"row count mismatch: expected {len(expected) - 1}, got {len(rows) - 1}"
        )
    if not errors and width > 1:
        if [r[0] for r in rows[1:]] != [r[0] for r in expected[1:]]:
            errors.append("first-column IDs or their order differ from the template")
    return not errors, errors


def validate_kernel(candidate_dir: Path) -> tuple[bool, list[str]]:
    """Kaggle uploads exactly one script; anything else is a silent crash."""
    kernel = Path(candidate_dir) / "kernel.py"
    if not kernel.is_file():
        return False, ["kernel.py is missing"]
    text = kernel.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    if not text.strip():
        errors.append("kernel.py is empty")
    if "submission.csv" not in text:
        errors.append("kernel.py never mentions submission.csv")
    for pattern, message in (
        (r"^\s*(?:!|%)\s*pip\s+install", "kernel installs packages; scoring kernels are offline"),
        (r"subprocess\.[a-z_]+\(\s*\[?['\"](?:pip|uv)\b", "kernel shells out to pip; scoring kernels are offline"),
    ):
        if re.search(pattern, text, re.MULTILINE):
            errors.append(message)
    helpers = [
        path.name for path in Path(candidate_dir).glob("*.py")
        if path.name != "kernel.py"
    ]
    if helpers:
        errors.append(
            "Kaggle uploads only kernel.py; inline these helpers instead: "
            + ", ".join(sorted(helpers))
        )
    return not errors, errors


class DryRunKaggle:
    """Everything except the two calls that cost something."""

    def __init__(self, slug: str, root: Path, assets: Path):
        self.slug = slug
        self.root = Path(root)
        self.assets = Path(assets)

    def remaining_today(self) -> int | None:
        return None

    def submit(self, candidate: dict, submission_id: str, message: str) -> SubmitOutcome:
        ok, errors = validate_kernel(Path(candidate["path"]))
        if not ok:
            return SubmitOutcome("rejected", "; ".join(errors))
        return SubmitOutcome("submitted", f"dry-run: {candidate['candidate_id']}")

    def scores(self) -> dict[str, float]:
        return {}


class Kaggle:
    """The only object in this system allowed to authenticate to Kaggle."""

    def __init__(self, *, slug: str, root: Path, user: str, assets: Path,
                 kernel_timeout_s: float, kernel_run_timeout_s: int | None = None):
        self.slug = slug
        self.root = Path(root)
        self.user = user
        self.assets = Path(assets)
        #: How long we wait locally for a kernel to finish.
        self.kernel_timeout_s = kernel_timeout_s
        #: The task's own stated limit, passed to Kaggle as `push --timeout`.
        #: IOAI invalidates any solution pushed without it.
        self.kernel_run_timeout_s = kernel_run_timeout_s

    def submission_limits(self) -> dict:
        """What Kaggle says about our budget. Also our proof the slug is real.

        For these private competitions this is the ONE informative endpoint:
        `competitions_list` returns nothing and `competition_get_settings` is
        403, but submission-limits answers. A wrong slug fails here, which is
        exactly the check we want before anything else runs.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            value = api.competition_get_submission_limits(self.slug)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        allowed = getattr(value, "num_allowed_now", None)
        today = getattr(value, "num_today", None)
        return {
            "ok": isinstance(allowed, int),
            "num_allowed_now": allowed if isinstance(allowed, int) else None,
            "num_today": today if isinstance(today, int) else None,
            "num_total": getattr(value, "num_total", None),
            "limited_by_total": bool(getattr(value, "limited_by_total", False)),
        }

    def download_data(self, destination: Path) -> tuple[bool, str]:
        """Fetch the competition data ourselves.

        The rules list downloading the data as the agent's job, not the
        operator's, so the system asks Kaggle for it rather than being handed a
        directory that someone prepared by hand.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            done = subprocess.run(
                [_kaggle_bin(), "competitions", "download", "-c", self.slug,
                 "-p", str(destination), "-o", "-q"],
                capture_output=True, text=True, timeout=3600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        output = (done.stdout or "") + (done.stderr or "")
        archives = sorted(destination.glob("*.zip"))
        for archive in archives:
            try:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(destination)
                archive.unlink()
            except (zipfile.BadZipFile, OSError) as exc:
                return False, f"could not unpack {archive.name}: {exc}"
        files = [p for p in destination.rglob("*") if p.is_file()]
        if not files:
            return False, f"download produced no files: {output[-400:]}"
        return True, f"{len(files)} files under {destination}"

    def remaining_today(self) -> int | None:
        limits = self.submission_limits()
        if limits.get("ok"):
            return max(0, int(limits["num_allowed_now"]))
        try:
            done = subprocess.run(
                [_kaggle_bin(), "competitions", "submission-limits",
                 "--json", self.slug],
                capture_output=True, text=True, timeout=90,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        try:
            value = json.loads(done.stdout or "{}")
            if "numAllowedNow" in value:
                return max(0, int(value["numAllowedNow"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        match = re.search(r"Remaining today:\s*(\d+)", done.stdout or "")
        return int(match.group(1)) if match else None

    def submit(self, candidate: dict, submission_id: str, message: str) -> SubmitOutcome:
        directory = Path(candidate["path"])
        ok, errors = validate_kernel(directory)
        if not ok:
            return SubmitOutcome("rejected", "; ".join(errors))

        code = (directory / "kernel.py").read_text(encoding="utf-8")
        ref = unique_kernel_id(self.user, self.slug, candidate["candidate_id"],
                               submission_id)
        package = self.root / "kernels" / submission_id
        if package.exists():
            shutil.rmtree(package)
        accelerator = candidate.get("accelerator", "cpu")
        write_kernel(
            package, code,
            make_metadata(ref, f"solution_{submission_id.replace('-', '_')}.py",
                          self.slug, accelerator),
        )
        # --timeout is not optional: IOAI invalidates a solution pushed without
        # it that then runs past the task's stated limit.
        pushed, version, raw = push_kernel(
            package, kernel_timeout_s=self.kernel_run_timeout_s
        )
        if not pushed:
            return SubmitOutcome("retryable", f"kernel push failed: {raw[-600:]}")

        status, log = poll_kernel(ref, timeout_s=self.kernel_timeout_s)
        if status != "complete":
            # A missing /kaggle/input path is usually a mount hiccup, not a bug
            # in the kernel; everything else is the kernel's own fault.
            transient = "filenotfounderror" in log.lower() and "/kaggle/input" in log.lower()
            return SubmitOutcome(
                "retryable" if transient else "execution_error",
                f"kernel {status}: {log[-600:]}", kernel_ref=ref,
                kernel_version=version,
            )

        output = self.root / "outputs" / submission_id
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.kernels_output(ref, str(output), force=True, quiet=True)
        except Exception as exc:  # noqa: BLE001
            return SubmitOutcome(
                "retryable", f"output download failed: {type(exc).__name__}: {exc}",
                kernel_ref=ref, kernel_version=version,
            )

        valid, errors = validate_submission(self.assets, output / "submission.csv")
        if not valid:
            return SubmitOutcome(
                "execution_error", "kernel output invalid: " + "; ".join(errors),
                kernel_ref=ref, kernel_version=version,
            )
        try:
            api.competition_submit_code(
                "submission.csv", message, self.slug,
                kernel=ref, kernel_version=version,
            )
        except Exception as exc:  # noqa: BLE001
            # The call may well have landed. Reconciliation by token decides.
            return SubmitOutcome(
                "ambiguous", f"submit call failed: {type(exc).__name__}: {exc}",
                kernel_ref=ref, kernel_version=version,
            )
        return SubmitOutcome("submitted", "submitted", kernel_ref=ref,
                             kernel_version=version)

    def scores(self) -> dict[str, float]:
        """Map our own token back to a public score; unknown rows are ignored."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            rows = api.competition_submissions(self.slug, page_size=100) or []
        except Exception:  # noqa: BLE001
            return {}
        found: dict[str, float] = {}
        for row in rows:
            match = _TOKEN_RE.search(str(getattr(row, "description", "") or ""))
            raw = getattr(row, "public_score", None)
            if not match or raw in (None, ""):
                continue
            try:
                found[match.group(1)] = float(raw)
            except (TypeError, ValueError):
                continue
        return found

    def seen_tokens(self) -> set[str]:
        """Tokens Kaggle has actually recorded, scored or not."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            rows = api.competition_submissions(self.slug, page_size=100) or []
        except Exception:  # noqa: BLE001
            return set()
        out = set()
        for row in rows:
            match = _TOKEN_RE.search(str(getattr(row, "description", "") or ""))
            if match:
                out.add(match.group(1))
        return out


def floor_kernel(template_name: str = "submission.csv") -> str:
    """Insurance: copy the official template through unchanged.

    It scores whatever a constant prediction scores, which is usually near
    nothing -- the point is to prove the whole chain works while there is still
    time to fix it if it does not.
    """
    return f'''import os, shutil

TARGET = "{template_name}"
src = None
for root, _dirs, files in os.walk("/kaggle/input"):
    for name in files:
        if name.lower() in ("sample_submission.csv", TARGET):
            src = os.path.join(root, name)
            break
    if src:
        break
if not src:
    raise SystemExit("no submission template found under /kaggle/input")
os.makedirs("/kaggle/working", exist_ok=True)
shutil.copyfile(src, "/kaggle/working/submission.csv")
print("floor: copied", src, flush=True)
'''
