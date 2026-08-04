from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.tools.registry import _kaggle_bin
from swarm.submit.kernel import (
    make_metadata,
    poll_kernel,
    push_kernel,
    unique_kernel_id,
    write_kernel,
)


@dataclass
class SubmitResult:
    consumed: bool
    status: str
    detail: str = ""
    external_ref: str = ""
    kernel_ref: str = ""
    kernel_version: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DryRunAdapter:
    def __init__(self, slug: str, root: Path):
        self.slug = slug
        self.root = Path(root)

    def remaining_today(self) -> int | None:
        return None

    def detect_mode(self) -> str:
        return "kernel"

    def submit(self, record: dict, submission_id: str, submission_class: str) -> SubmitResult:
        return SubmitResult(
            consumed=True,
            status="dry_run",
            detail=f"dry-run {submission_class}: {record['candidate_id']}",
            external_ref=f"dry:{submission_id}",
        )

    def scores(self) -> dict[str, float]:
        return {}


class KaggleAdapter:
    """The only class in the final system allowed to authenticate to Kaggle."""

    def __init__(
        self,
        *,
        slug: str,
        root: Path,
        kaggle_user: str,
        submission_mode: str,
        kernel_timeout_s: float = 2700,
    ):
        self.slug = slug
        self.root = Path(root)
        self.kaggle_user = kaggle_user
        self.submission_mode = submission_mode
        self.kernel_timeout_s = kernel_timeout_s

    def remaining_today(self) -> int | None:
        try:
            result = subprocess.run(
                [
                    _kaggle_bin(), "competitions", "submission-limits",
                    "--json", self.slug,
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
        except Exception:  # noqa: BLE001
            return None
        try:
            value = json.loads(result.stdout or "{}")
            if "numAllowedNow" in value:
                return max(0, int(value["numAllowedNow"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # 兼容没有 --json 的旧 Kaggle CLI 输出。
        match = re.search(r"Remaining today:\s*(\d+)", result.stdout or "")
        return int(match.group(1)) if match else None

    def detect_mode(self) -> str:
        if self.submission_mode in {"csv", "kernel"}:
            return self.submission_mode
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi(); api.authenticate()
            result = api.competitions_list(search=self.slug)
            for competition in getattr(result, "competitions", None) or []:
                ref = str(getattr(competition, "ref", ""))
                if ref.endswith(self.slug) or self.slug in ref:
                    return (
                        "kernel" if bool(
                            getattr(competition, "is_kernels_submissions_only", True)
                        ) else "csv"
                    )
        except Exception:  # noqa: BLE001
            pass
        # 正式 IOAI 是 code competition；private competition 无法被 list API
        # 看见时，严格路径是 kernel，而不是错误地尝试 CSV。
        return "kernel"

    def submit(self, record: dict, submission_id: str, submission_class: str) -> SubmitResult:
        mode = record.get("submission_mode")
        if mode == "unknown":
            mode = self.detect_mode()
        configured = self.detect_mode()
        if mode != configured:
            return SubmitResult(
                consumed=False,
                status="rejected",
                detail=f"candidate mode {mode} differs from competition mode {configured}",
            )
        message = (
            f"[fas:{submission_id}] {submission_class} "
            f"{record['source_lane']} {record['candidate_id']}"
        )[:140]
        if mode == "csv":
            return self._submit_csv(record, message)
        return self._submit_kernel(record, submission_id, message)

    def _submit_csv(self, record: dict, message: str) -> SubmitResult:
        csv_path = Path(record["snapshot_path"]) / "out" / "submission.csv"
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi(); api.authenticate()
            result = api.competition_submit(str(csv_path), message, self.slug)
            return SubmitResult(
                consumed=True,
                status="submitted",
                detail=str(result)[:1000],
                external_ref=str(getattr(result, "ref", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(
                consumed=False,
                status="ambiguous",
                detail=f"{type(exc).__name__}: {str(exc)[:1000]}",
            )

    def _submit_kernel(self, record: dict, submission_id: str, message: str) -> SubmitResult:
        snapshot = Path(record["snapshot_path"])
        source = snapshot / "out" / "kernel"
        if not source.is_dir():
            source = snapshot / "kernel"
        metadata_path = source / "kernel-metadata.json"
        if not metadata_path.is_file():
            return SubmitResult(False, "rejected", "kernel-metadata.json is missing")
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            code_file = source / source_metadata["code_file"]
            code = code_file.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(False, "rejected", f"kernel package unreadable: {exc}")

        kernel_ref = unique_kernel_id(
            self.kaggle_user, self.slug, record["candidate_id"], submission_id
        )
        package = self.root / "kernels" / submission_id
        if package.exists():
            shutil.rmtree(package)
        broker_code_file = f"solution_{submission_id.replace('-', '_')}.py"
        metadata = make_metadata(
            kernel_ref,
            broker_code_file,
            self.slug,
            record.get("accelerator", "cpu"),
            dataset_sources=source_metadata.get("dataset_sources") or [],
        )
        write_kernel(package, code, metadata)
        ok, version, raw = push_kernel(package)
        if not ok:
            return SubmitResult(False, "error", f"kernel push failed: {raw[-1000:]}")
        status, log = poll_kernel(kernel_ref, timeout_s=self.kernel_timeout_s)
        if status != "complete":
            return SubmitResult(
                False, "error", f"kernel {status}: {log[-1000:]}",
                kernel_ref=kernel_ref, kernel_version=version,
            )
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi(); api.authenticate()
            result = api.competition_submit_code(
                "submission.csv",
                message,
                self.slug,
                kernel=kernel_ref,
                kernel_version=version,
            )
            return SubmitResult(
                consumed=True,
                status="submitted",
                detail=str(result)[:1000],
                external_ref=str(getattr(result, "ref", "") or ""),
                kernel_ref=kernel_ref,
                kernel_version=version,
            )
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(
                False,
                "ambiguous",
                f"competition_submit_code failed: {type(exc).__name__}: {exc}",
                kernel_ref=kernel_ref,
                kernel_version=version,
            )

    def scores(self) -> dict[str, float]:
        """Map our stable fas token to a public score; unmatched rows are ignored."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi(); api.authenticate()
            rows = api.competition_submissions(self.slug) or []
        except Exception:  # noqa: BLE001
            return {}
        output: dict[str, float] = {}
        for row in rows:
            description = str(getattr(row, "description", "") or "")
            match = re.search(r"\[fas:([^\]]+)\]", description)
            raw = getattr(row, "public_score", None)
            if match and raw not in (None, ""):
                try:
                    output[match.group(1)] = float(raw)
                except (TypeError, ValueError):
                    continue
        return output
