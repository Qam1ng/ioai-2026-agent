from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.tools.registry import _kaggle_bin
from swarm.submit.kernel import (
    estimate_gpu_seconds,
    kernel_status_once,
    lookup_current_version,
    make_metadata,
    poll_kernel,
    push_kernel,
    unique_kernel_id,
    write_kernel,
)

from .day_gate import DayResourceGate
from .evaluation import validate_kernel_package, validate_submission_csv


@dataclass
class SubmitResult:
    consumed: bool
    status: str
    detail: str = ""
    external_ref: str = ""
    kernel_ref: str = ""
    kernel_version: int | None = None
    resource_lease_id: str = ""

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
        assets: Path,
        resource_gate: DayResourceGate,
        competition_deadline_epoch: float,
        default_gpu_kernel_minutes: float = 45,
        default_cpu_kernel_minutes: float = 10,
        kernel_start_margin_minutes: float = 5,
        kernel_timeout_s: float | None = None,
    ):
        self.slug = slug
        self.root = Path(root)
        self.kaggle_user = kaggle_user
        self.submission_mode = submission_mode
        self.assets = Path(assets)
        self.resource_gate = resource_gate
        self.competition_deadline_epoch = float(competition_deadline_epoch)
        self.default_gpu_kernel_minutes = float(default_gpu_kernel_minutes)
        self.default_cpu_kernel_minutes = float(default_cpu_kernel_minutes)
        self.kernel_start_margin_minutes = float(kernel_start_margin_minutes)
        self.kernel_timeout_s = (
            float(kernel_timeout_s) if kernel_timeout_s is not None else None
        )

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

    def _runtime_minutes(self, record: dict) -> float:
        accelerator = str(record.get("accelerator", "cpu")).lower()
        default_minutes = (
            self.default_cpu_kernel_minutes
            if accelerator == "cpu" else self.default_gpu_kernel_minutes
        )
        # Agent 声明只能把预算调大，不能用过小估计绕过账号配额和截止门。
        return max(
            default_minutes,
            float(record.get("estimated_kernel_minutes") or default_minutes),
        )

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
            result = self._submit_csv(record, message)
            if result.consumed and submission_class == "floor":
                self.resource_gate.mark_floor(self.slug)
            return result
        return self._submit_kernel(
            record, submission_id, submission_class, message
        )

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

    def _submit_kernel(
        self,
        record: dict,
        submission_id: str,
        submission_class: str,
        message: str,
    ) -> SubmitResult:
        snapshot = Path(record["snapshot_path"])
        package_check = validate_kernel_package(snapshot)
        if not package_check.get("valid"):
            return SubmitResult(
                False, "rejected", "; ".join(package_check.get("errors", []))
            )
        source = Path(package_check["source"])
        metadata_path = source / "kernel-metadata.json"
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            code_file = source / source_metadata["code_file"]
            code = code_file.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(False, "rejected", f"kernel package unreadable: {exc}")

        accelerator = str(record.get("accelerator", "cpu")).lower()
        runtime_minutes = self._runtime_minutes(record)
        latest_start = self.competition_deadline_epoch - (
            runtime_minutes + self.kernel_start_margin_minutes
        ) * 60
        if time.time() >= latest_start:
            return SubmitResult(
                False,
                "deadline_blocked",
                f"not enough time for estimated {runtime_minutes:.1f} minute kernel",
            )
        lease, gate_status, reason = self.resource_gate.acquire(
            submission_id=submission_id,
            slug=self.slug,
            submission_class=submission_class,
            accelerator=accelerator,
            estimated_seconds=runtime_minutes * 60,
            deadline_epoch=latest_start,
        )
        if lease is None:
            return SubmitResult(False, gate_status, reason)
        lease_id = str(lease["lease_id"])

        try:
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
                accelerator,
                dataset_sources=source_metadata.get("dataset_sources") or [],
                kernel_sources=source_metadata.get("kernel_sources") or [],
                model_sources=source_metadata.get("model_sources") or [],
            )
            write_kernel(package, code, metadata)
        except Exception as exc:  # noqa: BLE001
            self.resource_gate.release(lease_id)
            return SubmitResult(False, "rejected", f"kernel packaging failed: {exc}")
        ok, version, raw = push_kernel(package)
        if not ok:
            self.resource_gate.release(lease_id)
            return SubmitResult(
                False, "retryable", f"kernel push failed: {raw[-1000:]}"
            )
        self.resource_gate.mark_running(lease_id, kernel_ref)
        poll_timeout = runtime_minutes * 60
        if self.kernel_timeout_s is not None:
            poll_timeout = min(poll_timeout, self.kernel_timeout_s)
        status, log = poll_kernel(kernel_ref, timeout_s=poll_timeout)
        if status == "timeout":
            self.resource_gate.mark_external_unknown(lease_id)
            return SubmitResult(
                False, "external_running", f"kernel still running: {log[-1000:]}",
                kernel_ref=kernel_ref, kernel_version=version,
                resource_lease_id=lease_id,
            )
        # 优先采用 Kaggle 返回的运行时；无法解析时使用声明预算，而不是把
        # 本地打包、push 和远端排队时间都记成 GPU 执行时间。
        gpu_seconds = estimate_gpu_seconds(
            log, fallback=runtime_minutes * 60
        )
        self.resource_gate.settle(lease_id, gpu_seconds)
        if status != "complete":
            retryable_mount = (
                "filenotfounderror" in log.lower()
                and "/kaggle/input" in log.lower()
            )
            return SubmitResult(
                False, "retryable" if retryable_mount else "execution_error",
                f"kernel {status}: {log[-1000:]}",
                kernel_ref=kernel_ref, kernel_version=version,
            )
        return self._finish_completed_kernel(
            submission_id=submission_id,
            submission_class=submission_class,
            message=message,
            kernel_ref=kernel_ref,
            version=version,
        )

    def _finish_completed_kernel(
        self,
        *,
        submission_id: str,
        submission_class: str,
        message: str,
        kernel_ref: str,
        version: int | None,
    ) -> SubmitResult:
        exact_version = version or lookup_current_version(kernel_ref)
        if exact_version is None:
            return SubmitResult(
                False,
                "kernel_complete_pending_output",
                "completed kernel version could not be resolved",
                kernel_ref=kernel_ref,
            )
        output_dir = self.root / "kernel_outputs" / submission_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            api.kernels_output(kernel_ref, str(output_dir), force=True, quiet=True)
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(
                False, "kernel_complete_pending_output",
                f"kernel output download failed: {type(exc).__name__}: {exc}",
                kernel_ref=kernel_ref, kernel_version=exact_version,
            )
        output_csv = output_dir / "submission.csv"
        output_check = validate_submission_csv(self.assets, output_csv)
        if not output_check.get("valid"):
            return SubmitResult(
                False,
                "execution_error",
                "runtime submission.csv invalid: "
                + "; ".join(output_check.get("errors", [])),
                kernel_ref=kernel_ref,
                kernel_version=exact_version,
            )
        try:
            result = api.competition_submit_code(
                "submission.csv",
                message,
                self.slug,
                kernel=kernel_ref,
                kernel_version=exact_version,
            )
            submitted = SubmitResult(
                consumed=True,
                status="submitted",
                detail=str(result)[:1000],
                external_ref=str(getattr(result, "ref", "") or ""),
                kernel_ref=kernel_ref,
                kernel_version=exact_version,
            )
            if submission_class == "floor":
                self.resource_gate.mark_floor(self.slug)
            return submitted
        except Exception as exc:  # noqa: BLE001
            return SubmitResult(
                False,
                "ambiguous",
                f"competition_submit_code failed: {type(exc).__name__}: {exc}",
                kernel_ref=kernel_ref,
                kernel_version=exact_version,
            )

    def reconcile_external(self, record: dict, row: dict) -> SubmitResult | None:
        """Resume a running kernel or the download/submit tail of a completed one."""
        result = row.get("result", {})
        result_status = result.get("status")
        if result_status == "kernel_complete_pending_output":
            kernel_ref = str(result.get("kernel_ref", ""))
            if not kernel_ref:
                return SubmitResult(
                    False, "execution_error", "completed kernel has no remote ref"
                )
            message = (
                f"[fas:{row['submission_id']}] {row['submission_class']} "
                f"{record['source_lane']} {record['candidate_id']}"
            )[:140]
            return self._finish_completed_kernel(
                submission_id=row["submission_id"],
                submission_class=row["submission_class"],
                message=message,
                kernel_ref=kernel_ref,
                version=result.get("kernel_version"),
            )
        if result_status != "external_running":
            return None
        kernel_ref = str(result.get("kernel_ref", ""))
        if not kernel_ref:
            return SubmitResult(False, "execution_error", "missing remote kernel ref")
        status, raw = kernel_status_once(kernel_ref)
        if status in {"running", "unknown"}:
            return None
        lease_id = str(result.get("resource_lease_id", ""))
        if lease_id:
            self.resource_gate.settle(
                lease_id,
                estimate_gpu_seconds(
                    raw, fallback=self._runtime_minutes(record) * 60
                ),
            )
        if status != "complete":
            retryable_mount = (
                "filenotfounderror" in raw.lower()
                and "/kaggle/input" in raw.lower()
            )
            return SubmitResult(
                False,
                "retryable" if retryable_mount else "execution_error",
                f"reconciled kernel {status}: {raw[-1000:]}",
                kernel_ref=kernel_ref,
                kernel_version=result.get("kernel_version"),
            )
        message = (
            f"[fas:{row['submission_id']}] {row['submission_class']} "
            f"{record['source_lane']} {record['candidate_id']}"
        )[:140]
        return self._finish_completed_kernel(
            submission_id=row["submission_id"],
            submission_class=row["submission_class"],
            message=message,
            kernel_ref=kernel_ref,
            version=result.get("kernel_version"),
        )

    def _reconcile_orphaned_resource_leases(self) -> None:
        """Let any live controller release terminal jobs left by a crashed peer."""
        for lease in self.resource_gate.external_unknown_leases():
            kernel_ref = str(lease.get("external_ref", ""))
            if not kernel_ref:
                continue
            status, raw = kernel_status_once(kernel_ref)
            if status not in {"complete", "error", "cancelled"}:
                continue
            fallback = max(0.0, float(lease.get("estimated_seconds", 0.0)))
            self.resource_gate.settle(
                str(lease["lease_id"]),
                estimate_gpu_seconds(raw, fallback=fallback),
            )

    def on_reconciled_submission(self, row: dict) -> None:
        if row.get("submission_class") == "floor":
            self.resource_gate.mark_floor(self.slug)

    def submission_states(self) -> dict[str, dict]:
        """Map stable fas tokens to submission existence, status and public score."""
        self._reconcile_orphaned_resource_leases()
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi(); api.authenticate()
            rows = api.competition_submissions(self.slug, page_size=100) or []
        except Exception:  # noqa: BLE001
            return {}
        output: dict[str, dict] = {}
        for row in rows:
            description = str(getattr(row, "description", "") or "")
            match = re.search(r"\[fas:([^\]]+)\]", description)
            if not match:
                continue
            raw = getattr(row, "public_score", None)
            score = None
            if raw not in (None, ""):
                try:
                    score = float(raw)
                except (TypeError, ValueError):
                    score = None
            output[match.group(1)] = {
                "exists": True,
                "status": str(getattr(row, "status", "") or ""),
                "public_score": score,
            }
        return output

    def scores(self) -> dict[str, float]:
        return {
            key: float(value["public_score"])
            for key, value in self.submission_states().items()
            if value.get("public_score") is not None
        }
