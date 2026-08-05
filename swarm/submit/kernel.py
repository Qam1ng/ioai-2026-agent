"""Kaggle kernel packaging, pushing and polling.

Everything Kaggle-specific and mechanical lives here so the broker above can
stay a pure policy layer. Three facts drive the shape of this module:

* IOAI submissions are **code competition kernels**: a ``.py`` that trains on
  the Kaggle machine. No checkpoint upload, so the accelerator choice in the
  metadata *is* the compute budget decision (``docs/QA-NOTES.md``).
* ``enable_gpu`` must be the **string** ``"true"``/``"false"`` and the
  accelerator is selected by ``machine_shape``. Kaggle's own client accepts a
  JSON bool too, but the organizers documented the string form and a wrong
  ``machine_shape`` silently downgrades the machine — so the string form is
  what we emit.
* Parallel candidates must not share a kernel ref. Pushing two candidates to
  one ``id`` makes the second overwrite the first's version, and the
  submission then scores the wrong code.

The existing ``agent/tools/registry.py::kaggle_push_kernel`` returns the raw
CLI text and drops the version number; ``run.py`` recovers it by scanning for
"any digit token", which happily picks a digit out of the progress URL. Here
the version is parsed off the documented message and, failing that, queried.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# The Kaggle CLI is not on PATH for a detached launcher: ten of fifteen broker
# submissions in the first live run died with "kaggle CLI not found on PATH"
# while the binary sat in the venv beside sys.executable.
from agent.tools.registry import _kaggle_bin

#: accelerator label -> machine_shape string Kaggle expects
ACCEL_MACHINE_SHAPE: dict[str, str] = {
    "cpu": "",
    "p100": "NvidiaTeslaP100",
    "t4": "NvidiaTeslaT4",
}

#: Kaggle rejects kernel slugs beyond ~50 chars; stay comfortably inside.
MAX_KERNEL_SLUG = 44
#: Kaggle rejects kernel titles beyond ~50 characters with an opaque 400 on
#: SaveKernel that names no field. Every push in the third live run failed this
#: way; the account, the disk, competition_sources, the GPU and the wheel
#: dataset were each cleared individually before the title was tried.
MAX_KERNEL_TITLE = 44

#: ``kernels push`` prints: "Kernel version 7 successfully pushed.  Please
#: check progress at <url>" (kaggle 2.2.4, ``kernels_push_cli``).
_PUSH_VERSION_RE = re.compile(r"kernel version\s+(\d+)\s+successfully pushed", re.I)
_LOOSE_VERSION_RE = re.compile(r"\bversion\s+(\d+)\b", re.I)
#: ``kernels status`` prints: '<ref> has status "COMPLETE"'
#: Real CLI output is `<ref> has status "KernelWorkerStatus.COMPLETE"` — the
#: enum name, not the bare word. The first live run polled a kernel that had
#: genuinely finished until the poll timed out, because this pattern only
#: matched the bare form, and the submission was never made.
_STATUS_RE = re.compile(
    r'has\s+status\s+"(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', re.IGNORECASE
)

#: Kaggle's KernelWorkerStatus enum -> our four terminal/latent buckets.
_TERMINAL = {
    "COMPLETE": "complete",
    "ERROR": "error",
    "CANCEL_REQUESTED": "cancelled",
    "CANCEL_ACKNOWLEDGED": "cancelled",
}
_RUNNING = {"QUEUED", "RUNNING", "NEW_SCRIPT"}

_DURATION_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*seconds?", re.I),
    re.compile(r"runtime[^0-9]{0,12}(\d+(?:\.\d+)?)", re.I),
    re.compile(r"duration[^0-9]{0,12}(\d+(?:\.\d+)?)", re.I),
)
_HMS_RE = re.compile(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(\d+(?:\.\d+)?)\s*s\b", re.I)


@dataclass
class PushResult:
    """Structured form of :func:`push_kernel`'s tuple, for callers that want names."""

    ok: bool
    version: int | None
    raw: str


def _run(cmd: list[str], timeout: float = 900.0) -> tuple[int, str]:
    """Run a kaggle CLI command; never raise, the text is the diagnosis."""
    if cmd and cmd[0] == "kaggle":
        beside = Path(sys.executable).parent / "kaggle"
        cmd = [str(beside) if beside.is_file() else (shutil.which("kaggle") or "kaggle"), *cmd[1:]]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "[kaggle CLI not found on PATH]"
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s: {' '.join(cmd)}]"
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out.strip()


def slugify(text: str) -> str:
    """Kaggle slug rules: lowercase, alphanumerics and single dashes."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower())
    return re.sub(r"-{2,}", "-", s).strip("-")


# --------------------------------------------------------------------- ids
def unique_kernel_id(
    user: str, slug: str, candidate_id: str, discriminator: str = ""
) -> str:
    """``<user>/<kernel-slug>`` unique per candidate AND per submission.

    Parallel candidates that share a kernel ref clobber each other's versions,
    and the submission that follows then scores whichever code landed last.
    The competition slug is truncated (it can be 40+ chars) but a short digest
    of the *full* slug is kept so two competitions sharing a prefix cannot
    collide.

    ``discriminator`` exists because a candidate is pushed more than once: the
    floor lane and the milestone lane both submit the same candidate, and in
    the first live run both threads pushed and polled the *same* Kaggle kernel
    concurrently, so neither submission completed. Pass the submission id (or
    the lane) to keep concurrent pushes on separate kernels.
    """
    comp = slugify(slug)
    digest = hashlib.sha1(slug.encode()).hexdigest()[:4]
    cand = slugify(candidate_id) or "c0"
    disc = slugify(discriminator)
    tail = f"{cand}-{disc}" if disc else cand
    head_budget = MAX_KERNEL_SLUG - len(digest) - len(tail) - 2
    head = comp[: max(4, head_budget)].strip("-")
    kernel_slug = f"{head}-{digest}-{tail}"
    return f"{slugify(user)}/{kernel_slug}"


# ---------------------------------------------------------------- metadata
def make_metadata(
    kernel_id: str,
    code_file: str,
    slug: str,
    accelerator: str,
    dataset_sources: list[str] | None = None,
    kernel_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
) -> dict:
    """kernel-metadata.json for one submission kernel.

    ``dataset_sources`` exists for the documented task-2 workaround: when
    ``competition_sources`` mounting was misconfigured the data had to be
    re-uploaded as a private dataset and mounted that way
    (``docs/QA-NOTES.md``). ``competition_sources`` is kept in that case too,
    so the kernel still counts as a competition kernel.
    """
    accel = (accelerator or "cpu").lower()
    if accel not in ACCEL_MACHINE_SHAPE:
        raise ValueError(
            f"unknown accelerator {accelerator!r}; use one of {sorted(ACCEL_MACHINE_SHAPE)}"
        )
    title = kernel_id.split("/", 1)[-1]
    if len(title) < 5:  # Kaggle rejects titles under five characters
        title = f"{title}-kernel"
    if len(title) > MAX_KERNEL_TITLE:
        # Truncate the id with it, or Kaggle warns that the title does not
        # resolve to the id.
        title = title[:MAX_KERNEL_TITLE]
        kernel_id = f"{kernel_id.split('/', 1)[0]}/{title}"
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # Strings, not booleans: this is the form the organizers documented.
        "enable_gpu": "true" if accel != "cpu" else "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": ACCEL_MACHINE_SHAPE[accel],
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": [slug],
        "kernel_sources": list(kernel_sources or []),
        "model_sources": list(model_sources or []),
    }


def write_kernel(dir: Path, code: str, metadata: dict) -> Path:
    """Materialise a pushable kernel directory; returns the directory."""
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    code_file = metadata.get("code_file") or "kernel.py"
    (d / code_file).write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return d


# -------------------------------------------------------------------- push
def parse_push_version(raw: str) -> int | None:
    """Pull the version number out of ``kernels push`` output.

    The loose fallback is anchored on the word "version" so it cannot pick a
    digit out of the progress URL the way ``run.py``'s "last digit token" scan
    does.
    """
    m = _PUSH_VERSION_RE.search(raw or "")
    if m:
        return int(m.group(1))
    m = _LOOSE_VERSION_RE.search(raw or "")
    return int(m.group(1)) if m else None


def lookup_current_version(kernel_ref: str) -> int | None:
    """Ask Kaggle for the kernel's current version number.

    ``kaggle kernels list`` only prints ref/title/author/lastRunTime/totalVotes,
    so the CLI cannot answer this; the Python client's ``kernels_list`` returns
    ``ApiKernelMetadata.current_version_number``, which can. Falls back to a
    CLI existence check so the caller at least learns whether the push landed.
    """
    slug = kernel_ref.split("/", 1)[-1]
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        for k in api.kernels_list(mine=True, search=slug, page_size=50) or []:
            if k is not None and getattr(k, "ref", "") == kernel_ref:
                v = getattr(k, "current_version_number", 0) or 0
                return int(v) or None
    except Exception:  # noqa: BLE001 - a lookup failure must not fail the push
        pass
    _run([_kaggle_bin(), "kernels", "list", "-m", "-s", slug], timeout=120)
    return None


def push_kernel(dir: Path, timeout_s: float = 900.0) -> tuple[bool, int | None, str]:
    """Push a kernel directory. Returns ``(ok, version, raw_output)``.

    ``version`` is what ``competition_submit_code`` needs; submitting without
    it silently scores whatever version Kaggle considers current, which under
    parallel candidates is a coin flip.
    """
    d = Path(dir)
    meta_path = d / "kernel-metadata.json"
    if not meta_path.exists():
        return False, None, f"[error] kernel-metadata.json missing in {d}"

    rc, raw = _run([_kaggle_bin(), "kernels", "push", "-p", str(d)], timeout=timeout_s)
    # The CLI exits 0 even when it prints "Kernel push error: ...", so the exit
    # code alone is not a success signal.
    ok = rc == 0 and "push error" not in raw.lower()

    version = parse_push_version(raw)
    if version is None and ok:
        try:
            kernel_ref = json.loads(meta_path.read_text()).get("id", "")
        except (json.JSONDecodeError, OSError):
            kernel_ref = ""
        if kernel_ref:
            version = lookup_current_version(kernel_ref)
    return ok, version, raw


# -------------------------------------------------------------------- poll
def parse_status(raw: str) -> str | None:
    """Kaggle's raw worker status from ``kernels status`` output, if present."""
    m = _STATUS_RE.search(raw or "")
    return m.group(1).upper() if m else None


def fetch_log(kernel_ref: str, tail_chars: int = 8000) -> str:
    """Best-effort kernel log. ``kernels logs`` first, ``kernels output`` second."""
    rc, raw = _run([_kaggle_bin(), "kernels", "logs", kernel_ref], timeout=300)
    if rc == 0 and raw and "usage:" not in raw[:200].lower():
        return raw[-tail_chars:]
    with tempfile.TemporaryDirectory() as tmp:
        _run([_kaggle_bin(), "kernels", "output", kernel_ref, "-p", tmp], timeout=600)
        logs = sorted(Path(tmp).glob("*.log"))
        if not logs:
            return raw[-tail_chars:] if raw else "[no log available]"
        text = logs[-1].read_text(errors="replace")
        try:  # Kaggle writes the log as a JSON array of {stream, data} records
            entries = json.loads(text)
            text = "\n".join(
                e.get("data", "") for e in entries if str(e.get("data", "")).strip()
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return text[-tail_chars:]


def kernel_status_once(ref: str) -> tuple[str, str]:
    """Read one remote status without turning a still-running job into an error."""
    _rc, raw = _run([_kaggle_bin(), "kernels", "status", ref], timeout=180)
    status = parse_status(raw)
    if status in _TERMINAL:
        return _TERMINAL[status], raw
    if status in _RUNNING:
        return "running", raw
    if status is None and "not found" in raw.lower():
        return "error", raw
    return "unknown", raw


def poll_kernel(
    ref: str, timeout_s: float = 2700.0, interval_s: float = 30.0
) -> tuple[str, str]:
    """Block until the kernel run reaches a terminal state.

    Returns ``(final_status, log_tail)`` where ``final_status`` is one of
    ``complete | error | cancelled | timeout``. The log is only fetched when
    something went wrong — on the happy path it is a wasted round trip and the
    kernel's stdout is not information we act on.
    """
    deadline = time.time() + timeout_s
    last_raw = ""
    while time.time() < deadline:
        current, raw = kernel_status_once(ref)
        last_raw = raw
        if current in {"complete", "cancelled"}:
            log = fetch_log(ref) if current != "complete" else raw
            return current, log
        if current == "error":
            log = fetch_log(ref)
            if log and "no log available" not in log:
                raw = log
            return "error", raw
        status = parse_status(raw)
        if status is not None and status not in _RUNNING:
            # A status Kaggle added since this was written: treat as non-terminal
            # but keep the text, so a stuck poll is diagnosable from the log tail.
            last_raw = f"{raw}\n[unrecognised kernel status {status!r}]"
        time.sleep(interval_s)
    return "timeout", fetch_log(ref) or last_raw


# ------------------------------------------------------------ gpu accounting
def estimate_gpu_seconds(status_output: str, fallback: float = 1800.0) -> float:
    """Best-effort run duration in seconds from Kaggle status/log text.

    Kaggle's ``kernels status`` does not currently report a duration, so this
    is opportunistic: it recognises the shapes we have seen in kernel logs
    ("... 1234 seconds", "runtime: 812", "1h 05m 30s"). When nothing parses,
    the caller's reservation estimate is returned — over-counting GPU quota is
    safe, under-counting is what gets three problems locked out on day two.
    """
    text = status_output or ""
    m = _HMS_RE.search(text)
    if m and any(m.groups()):
        h = float(m.group(1) or 0)
        mi = float(m.group(2) or 0)
        s = float(m.group(3) or 0)
        total = h * 3600 + mi * 60 + s
        if total > 0:
            return total
    for pat in _DURATION_PATTERNS:
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            if val > 0:
                return val
    return float(fallback)
