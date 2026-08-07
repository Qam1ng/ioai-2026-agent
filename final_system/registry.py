from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .evaluation import (
    evaluate_candidate,
    validate_kernel_package,
    validate_submission_csv,
)
from .io import atomic_json, locked, read_json, tree_hash

SOURCE_LANES = ("hearsay", "codex", "claude")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_FORBIDDEN_NAMES = {".env", "kaggle.json", "credentials.json", "auth.json"}


def _candidate_files(source: Path) -> list[Path]:
    """Copy runnable artifacts, never agent homes, caches, data, or checkpoints."""
    source = Path(source)
    files: list[Path] = []
    roots = [name for name in ("out", "kernel", "code", "evidence") if (source / name).exists()]
    for name in roots:
        for path in sorted((source / name).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"candidate contains symlink: {path.relative_to(source)}")
            if path.is_file():
                # HearSay evaluator 会异步重写此派生文件；公共评估器从不信任它。
                # 排除它可以避免同一预测仅因 score.json 晚到而生成重复快照。
                if path.relative_to(source).as_posix() == "out/score.json":
                    continue
                files.append(path)
    for path in sorted(source.iterdir() if source.exists() else []):
        if path.is_symlink():
            raise ValueError(f"candidate contains symlink: {path.name}")
        if path.is_file() and (
            path.name == "candidate.json"
            or path.suffix.lower() in {".py", ".sh", ".toml", ".yaml", ".yml", ".md"}
        ):
            files.append(path)
    unique = sorted(set(files))
    for path in unique:
        relative = path.relative_to(source)
        if relative.name.lower() in _FORBIDDEN_NAMES:
            raise ValueError(f"credential material is not a candidate artifact: {relative}")
    return unique


def candidate_fingerprint(source: Path) -> str:
    source = Path(source).resolve()
    files = _candidate_files(source)
    return tree_hash(source, include={path.relative_to(source).as_posix() for path in files})


def _copy_files(source: Path, destination: Path, files: list[Path]) -> None:
    destination.mkdir(parents=True)
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


class CandidateRegistry:
    """Content-addressed, immutable candidate snapshots shared by all three lanes."""

    def __init__(self, root: Path, *, assets: Path, evaluation: Path):
        self.root = Path(root)
        self.assets = Path(assets)
        self.evaluation = Path(evaluation)
        self.index_path = self.root / "index.json"
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            atomic_json(self.index_path, {"schema_version": 1, "candidates": {}})

    def _load(self) -> dict:
        return read_json(self.index_path, {"schema_version": 1, "candidates": {}})

    def records(self) -> list[dict]:
        return list(self._load().get("candidates", {}).values())

    def get(self, candidate_id: str) -> dict | None:
        return self._load().get("candidates", {}).get(candidate_id)

    def register(
        self,
        source: Path,
        *,
        source_lane: str,
        manifest: dict[str, Any] | None = None,
    ) -> dict:
        if source_lane not in SOURCE_LANES:
            raise ValueError(f"source_lane must be one of {SOURCE_LANES}")
        source = Path(source).resolve()
        if manifest is None:
            manifest_path = source / "candidate.json"
            if not manifest_path.is_file():
                raise ValueError("candidate.json is missing")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_id = str(manifest.get("candidate_id", ""))
        if not _SAFE_ID.fullmatch(raw_id):
            raise ValueError("candidate_id must be 1-80 safe filename characters")
        declared_lane = manifest.get("source_lane")
        if declared_lane and declared_lane != source_lane:
            raise ValueError("candidate source_lane differs from its actual outbox")
        mode = str(manifest.get("submission_mode", "unknown"))
        if mode not in {"csv", "kernel", "unknown"}:
            raise ValueError("submission_mode must be csv, kernel, or unknown")
        accelerator = str(manifest.get("accelerator", "cpu")).lower()
        if accelerator not in {"cpu", "p100", "t4"}:
            raise ValueError("accelerator must be cpu, p100, or t4")
        estimated_kernel_minutes = manifest.get("estimated_kernel_minutes")
        if estimated_kernel_minutes is not None:
            try:
                estimated_kernel_minutes = float(estimated_kernel_minutes)
            except (TypeError, ValueError) as exc:
                raise ValueError("estimated_kernel_minutes must be numeric") from exc
            if not 0 < estimated_kernel_minutes <= 360:
                raise ValueError("estimated_kernel_minutes must be in (0, 360]")

        before = candidate_fingerprint(source)
        canonical_id = f"{source_lane}-{raw_id}-{before[:10]}"
        with locked(self.index_path):
            index = self._load()
            existing = index["candidates"].get(canonical_id)
            if existing:
                return existing

            files = _candidate_files(source)
            temporary = self.snapshots / f".{canonical_id}.{os.getpid()}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            _copy_files(source, temporary, files)
            after = candidate_fingerprint(source)
            copied = candidate_fingerprint(temporary)
            if before != after or before != copied:
                shutil.rmtree(temporary, ignore_errors=True)
                raise RuntimeError("candidate changed while its snapshot was created")
            snapshot = self.snapshots / canonical_id
            os.replace(temporary, snapshot)

            format_result = validate_submission_csv(
                self.assets, snapshot / "out" / "submission.csv"
            )
            kernel_result = (
                validate_kernel_package(snapshot)
                if mode in {"kernel", "unknown"}
                else {"valid": True, "errors": [], "source": "", "metadata": {}}
            )
            evaluation_result: dict[str, Any] = {"status": "pending_contract"}
            if (self.evaluation / "contract.json").is_file():
                evaluation_result = evaluate_candidate(self.evaluation, snapshot)
            duplicate_of = None
            submission_hash = format_result.get("submission_sha256")
            kernel_source = Path(str(kernel_result.get("source", "")))
            execution_hash = (
                tree_hash(kernel_source) if mode in {"kernel", "unknown"}
                and kernel_source.is_dir() else submission_hash
            )
            for candidate in index["candidates"].values():
                if execution_hash and candidate.get("execution_sha256") == execution_hash:
                    duplicate_of = candidate["candidate_id"]
                    break
            valid_artifact = bool(
                format_result.get("valid") and kernel_result.get("valid")
            )
            status = (
                "invalid_format" if not valid_artifact
                else "duplicate" if duplicate_of
                else "eligible"
            )
            record = {
                "schema_version": 1,
                "candidate_id": canonical_id,
                "raw_candidate_id": raw_id,
                "source_lane": source_lane,
                "created_at": time.time(),
                "source_fingerprint": before,
                "snapshot_path": str(snapshot),
                "submission_mode": mode,
                "accelerator": accelerator,
                "estimated_kernel_minutes": estimated_kernel_minutes,
                "purpose": str(manifest.get("purpose", ""))[:500],
                "parent_id": str(manifest.get("parent_id", ""))[:100],
                "claimed_local_score": manifest.get("claimed_local_score"),
                "submission_sha256": submission_hash,
                "execution_sha256": execution_hash,
                "format": format_result,
                "kernel_format": {
                    key: value for key, value in kernel_result.items()
                    if key != "metadata"
                },
                "evaluation": evaluation_result,
                "status": status,
                "duplicate_of": duplicate_of,
                "submitted": False,
            }
            atomic_json(snapshot / "sealed_candidate.json", record)
            index["candidates"][canonical_id] = record
            atomic_json(self.index_path, index)
            return record

    def refresh_evaluations(self) -> int:
        if not (self.evaluation / "contract.json").is_file():
            return 0
        changed = 0
        with locked(self.index_path):
            index = self._load()
            for record in index["candidates"].values():
                if record.get("evaluation", {}).get("status") == "ok":
                    continue
                snapshot = Path(record["snapshot_path"])
                result = evaluate_candidate(self.evaluation, snapshot)
                if result != record.get("evaluation"):
                    record["evaluation"] = result
                    changed += 1
            if changed:
                atomic_json(self.index_path, index)
        return changed

    def mark_submitted(self, candidate_id: str) -> None:
        with locked(self.index_path):
            index = self._load()
            if candidate_id in index["candidates"]:
                index["candidates"][candidate_id]["submitted"] = True
                atomic_json(self.index_path, index)
