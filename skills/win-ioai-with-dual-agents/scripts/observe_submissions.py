#!/usr/bin/env python3
"""只读发现新的 Kaggle competition submissions，并可归档唯一 slug 的 latest source。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


KERNEL_URL_RE = re.compile(r"(?:https?://www\.kaggle\.com)?/code/([^/?#]+)/([^/?#]+)")
SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CANDIDATE_RE = re.compile(r"^((?:CX|CL)\d{2,4})\|", re.IGNORECASE)


def run_cli(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Kaggle CLI 只读调用超过 120 秒") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Kaggle CLI error"
        raise RuntimeError(f"Kaggle CLI 失败（退出码 {result.returncode}）：{detail}")
    return result.stdout


def load_json_output(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = min((i for i in (stripped.find("["), stripped.find("{")) if i >= 0), default=-1)
        if start < 0:
            raise RuntimeError("Kaggle CLI 没有返回 JSON") from None
        payload = json.loads(stripped[start:])
    if isinstance(payload, dict):
        for key in ("submissions", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle CLI 返回的 submissions JSON 结构未知")
    return [item for item in payload if isinstance(item, dict)]


def first(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def submission_key(item: dict[str, Any]) -> str:
    stable = first(item, "ref", "id", "submissionId", "submission_id")
    if stable is not None:
        return str(stable)
    fallback = {
        "file": first(item, "fileName", "file_name"),
        "date": first(item, "date", "submittedDate", "submitted_at"),
        "description": first(item, "description", "message"),
    }
    return json.dumps(fallback, sort_keys=True, ensure_ascii=False)


def normalize(item: dict[str, Any]) -> dict[str, Any]:
    url = first(item, "url", "notebookUrl", "notebook_url", "kernelUrl", "kernel_url")
    kernel_ref = None
    if isinstance(url, str):
        match = KERNEL_URL_RE.search(url)
        if match:
            kernel_ref = f"{match.group(1)}/{match.group(2)}"
    return {
        "key": submission_key(item),
        "ref": first(item, "ref", "id", "submissionId", "submission_id"),
        "date": first(item, "date", "submittedDate", "submitted_at"),
        "description": first(item, "description", "message"),
        "status": first(item, "status"),
        "public_score": first(item, "publicScore", "public_score"),
        "private_score": first(item, "privateScore", "private_score"),
        "url": url,
        "kernel_ref": kernel_ref,
    }


def candidate_id(record: dict[str, Any]) -> str | None:
    description = record.get("description")
    if not isinstance(description, str):
        return None
    match = CANDIDATE_RE.match(description.strip())
    return match.group(1).lower() if match else None


def fetch_candidate_kernel_refs(kaggle_bin: str, competition: str) -> dict[str, list[str]]:
    raw = run_cli([
        kaggle_bin,
        "kernels",
        "list",
        "--mine",
        "--competition",
        competition,
        "--sort-by",
        "dateRun",
        "--page-size",
        "200",
        "--format",
        "json",
    ])
    result: dict[str, list[str]] = {}
    for item in load_json_output(raw):
        ref = first(item, "ref")
        title = first(item, "title")
        haystack = f"{ref or ''} {title or ''}".lower()
        if not isinstance(ref, str):
            continue
        for match in re.finditer(r"(?:^|[^a-z0-9])((?:cx|cl)\d{2,4})(?:[^a-z0-9]|$)", haystack):
            bucket = result.setdefault(match.group(1), [])
            if ref not in bucket:
                bucket.append(ref)
    return result


def record_fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "description": record.get("description"),
            "status": record.get("status"),
            "public_score": record.get("public_score"),
            "private_score": record.get("private_score"),
            "kernel_ref": record.get("kernel_ref"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_state(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], bool]:
    if not path.exists():
        return {}, {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 state 文件 {path}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("fingerprints"), dict):
        fingerprints = {str(key): str(value) for key, value in payload["fingerprints"].items()}
        archives = payload.get("archives", {})
        if not isinstance(archives, dict):
            raise RuntimeError(f"state 文件 {path} 的 archives 不是对象")
        initialized = bool(payload.get("initialized", True))
        return fingerprints, {str(key): value for key, value in archives.items() if isinstance(value, dict)}, initialized
    values = payload.get("seen", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise RuntimeError(f"旧 state 文件 {path} 的 seen 不是列表")
    return {str(value): "legacy" for value in values}, {}, True


def save_state(path: Path, fingerprints: dict[str, str], archives: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {"initialized": True, "fingerprints": fingerprints, "archives": archives}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def archive_kernel(kaggle_bin: str, kernel_ref: str, root: Path, key: str) -> dict[str, Any]:
    safe_key = SAFE_RE.sub("_", key)[:80] or "submission"
    safe_ref = SAFE_RE.sub("_", kernel_ref)
    target = root / f"{safe_key}__{safe_ref}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        run_cli([kaggle_bin, "kernels", "pull", kernel_ref, "-p", str(target), "-m"])
    except RuntimeError as exc:
        return {"ok": False, "path": str(target), "error": str(exc)}
    return {"ok": True, "path": str(target)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("competition", help="Kaggle competition slug")
    parser.add_argument("--state", type=Path, required=True, help="已见 submission key 的 JSON 文件")
    parser.add_argument("--archive-dir", type=Path, help="可选：归档可解析出的 peer Kernel latest source")
    parser.add_argument("--baseline-existing", action="store_true", help="首次运行时静默把已有 submissions 设为基线")
    parser.add_argument("--archive-retries", type=int, default=3, help="源码归档瞬时失败的有限重试次数")
    parser.add_argument("--kaggle-bin", default=os.environ.get("KAGGLE_BIN", "kaggle"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = run_cli([
        args.kaggle_bin,
        "competitions",
        "submissions",
        args.competition,
        "--format",
        "json",
        "--page-size",
        "200",
    ])
    records = [normalize(item) for item in load_json_output(raw)]
    fingerprints, archives, initialized = load_state(args.state)
    changed_records = [r for r in records if fingerprints.get(str(r["key"])) != record_fingerprint(r)]
    if not initialized and args.baseline_existing:
        changed_records = []

    record_by_key = {str(record["key"]): record for record in records}
    pending_records = [
        record_by_key[key]
        for key, state in archives.items()
        if key in record_by_key
        and state.get("status") == "retryable"
        and int(state.get("attempts", 0)) < args.archive_retries
        and key not in {str(record["key"]) for record in changed_records}
    ]

    kernel_refs: dict[str, list[str]] = {}
    kernel_lookup_error: str | None = None
    if args.archive_dir and (changed_records or pending_records):
        try:
            kernel_refs = fetch_candidate_kernel_refs(args.kaggle_bin, args.competition)
        except RuntimeError as exc:
            kernel_lookup_error = str(exc)

    def try_archive(record: dict[str, Any]) -> dict[str, Any]:
        key = str(record["key"])
        previous = archives.get(key, {})
        attempts = int(previous.get("attempts", 0)) + 1
        event_ref = record.get("kernel_ref")
        cid = candidate_id(record)
        matches = kernel_refs.get(cid or "", [])
        if not event_ref and cid and len(matches) == 1:
            event_ref = matches[0]
        if event_ref:
            result = archive_kernel(args.kaggle_bin, str(event_ref), args.archive_dir, key)
            return {**result, "status": "success" if result["ok"] else "retryable", "attempts": attempts, "kernel_ref": event_ref}
        if kernel_lookup_error or (cid and len(matches) == 0):
            return {
                "ok": False,
                "status": "retryable",
                "attempts": attempts,
                "error": kernel_lookup_error or f"候选 {cid} 尚未匹配到 Kernel",
            }
        reason = (
            f"候选 {cid} 匹配到 {len(matches)} 个 Kernel；description 和唯一 slug 必须使用相同 CX/CL ID"
            if cid
            else "description 未以 CXnn| 或 CLnn| 开头，无法解析唯一 Kernel"
        )
        return {"ok": False, "status": "permanent", "attempts": attempts, "error": reason}

    events: list[dict[str, Any]] = []
    for record in sorted(changed_records, key=lambda value: str(value.get("date") or "")):
        event = dict(record)
        key = str(record["key"])
        event["event_type"] = "new" if key not in fingerprints else "updated"
        if args.archive_dir:
            existing_archive = archives.get(key)
            if existing_archive and existing_archive.get("status") == "success":
                event["archive"] = existing_archive
            else:
                archives[key] = try_archive(record)
                event["archive"] = archives[key]
        events.append(event)
        fingerprints[key] = record_fingerprint(record)

    archive_retries: list[dict[str, Any]] = []
    for record in pending_records:
        key = str(record["key"])
        archives[key] = try_archive(record)
        archive_retries.append({"key": key, "archive": archives[key]})

    if not initialized and args.baseline_existing:
        fingerprints.update({str(record["key"]): record_fingerprint(record) for record in records})
    save_state(args.state, fingerprints, archives)
    print(json.dumps({
        "competition": args.competition,
        "new_count": sum(event["event_type"] == "new" for event in events),
        "updated_count": sum(event["event_type"] == "updated" for event in events),
        "events": events,
        "archive_retries": archive_retries,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
