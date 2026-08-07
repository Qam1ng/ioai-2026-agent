#!/usr/bin/env python3
"""以原子账本、candidate SHA 和 deadline 门保护唯一 Kaggle kernels push。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from preflight_kernel import load_contract
from resource_slots import (
    active_slot_refs,
    fetch_kernel_statuses,
    reconcile_slots,
    transition_slot,
    try_acquire_slot,
)
from submit_with_diagnostics import redact as safe_redact


CANDIDATE_RE = re.compile(r"^(CX|CL)\d{2,4}$", re.IGNORECASE)
AMBIGUOUS_PUSH_STATUSES = {"push_returned_nonzero", "push_cli_timeout"}
RELEASED_PUSH_STATUSES = {"aborted_before_push", "safe_not_created"}
REMOTE_ABSENCE_WAIT_SECONDS = 60
ORPHAN_RESERVATION_SECONDS = 300


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def subprocess_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"submission_deadline_utc 不是 ISO-8601：{value}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("submission_deadline_utc 必须带时区")
    return parsed.astimezone(timezone.utc)


def candidate_hashes(root: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    solution = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        if path.name == "kernel-metadata.json":
            try:
                normalized = json.loads(data)
                normalized.pop("id", None)
                normalized.pop("title", None)
                data = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
            except (json.JSONDecodeError, AttributeError):
                pass
        solution.update(len(relative).to_bytes(4, "big"))
        solution.update(relative)
        solution.update(len(data).to_bytes(8, "big"))
        solution.update(data)
    return digest.hexdigest(), solution.hexdigest()


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"账本 {path} 第 {line_number} 行损坏：{exc}") from exc
        if not isinstance(event, dict) or not event.get("reservation_id"):
            raise RuntimeError(f"账本 {path} 第 {line_number} 行缺少 reservation_id")
        events.append(event)
    return events


def latest_reservations(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[str(event["reservation_id"])] = event
    return latest


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def candidate_matches_ref(candidate: str, kernel_ref: str) -> bool:
    """只在 slug 的独立边界匹配 candidate，避免 CX01 误匹配 CX010。"""
    slug = kernel_ref.rsplit("/", 1)[-1].lower()
    return re.search(rf"(?:^|[-_]){re.escape(candidate.lower())}(?:[-_]|$)", slug) is not None


def competition_slug_tag(competition: str) -> str:
    return hashlib.sha256(competition.encode()).hexdigest()[:8]


def slug_has_token(kernel_ref: str, token: str) -> bool:
    slug = kernel_ref.rsplit("/", 1)[-1].lower()
    return re.search(rf"(?:^|[-_]){re.escape(token.lower())}(?:[-_]|$)", slug) is not None


def event_age_seconds(event: dict[str, Any], now: datetime) -> float:
    try:
        return max(0.0, (now - parse_utc(str(event["time_utc"]))).total_seconds())
    except (KeyError, RuntimeError):
        return 0.0


def append_transition(path: Path, event: dict[str, Any], now: datetime, status: str) -> None:
    updated = dict(event)
    updated.update({"time_utc": iso(now), "status": status})
    append_event(path, updated)


def reconcile_ambiguous_pushes(
    ledger: Path,
    latest: dict[str, dict[str, Any]],
    remote_refs: set[str],
    now: datetime,
) -> None:
    """两次间隔确认远端不存在后才释放；远端出现则恢复为成功。"""
    for event in list(latest.values()):
        status = event.get("status")
        kernel_ref = str(event.get("kernel_ref") or "")
        if status in AMBIGUOUS_PUSH_STATUSES:
            if kernel_ref in remote_refs:
                append_transition(ledger, event, now, "remote_kernel_confirmed")
            elif event_age_seconds(event, now) >= REMOTE_ABSENCE_WAIT_SECONDS:
                append_transition(ledger, event, now, "remote_absent_once")
        elif status == "remote_absent_once":
            if kernel_ref in remote_refs:
                append_transition(ledger, event, now, "remote_kernel_confirmed")
            elif event_age_seconds(event, now) >= REMOTE_ABSENCE_WAIT_SECONDS:
                append_transition(ledger, event, now, "safe_not_created")
        elif status == "reserved":
            if kernel_ref in remote_refs:
                append_transition(ledger, event, now, "remote_kernel_confirmed")
            elif event_age_seconds(event, now) >= ORPHAN_RESERVATION_SECONDS:
                append_transition(ledger, event, now, "remote_absent_once")


def remote_kernel_refs(
    kaggle_bin: str,
    competition: str | None = None,
    search: str | None = None,
    page_size: int = 200,
    sort_by: str | None = None,
    all_pages: bool = False,
    max_pages: int = 20,
) -> list[str]:
    if max_pages <= 0:
        raise RuntimeError("远端 Kernel 对账 max_pages 必须为正")
    refs: set[str] = set()
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(max_pages):
        command = [
            kaggle_bin,
            "kernels",
            "list",
            "--mine",
            "--page-size",
            str(page_size),
            "--format",
            "json",
        ]
        if competition:
            command += ["--competition", competition]
        if search:
            command += ["--search", search]
        if sort_by:
            command += ["--sort-by", sort_by]
        if page_token:
            command += ["--page-token", page_token]
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("远端 Kernel 对账超时") from exc
        if result.returncode != 0:
            raise RuntimeError(f"远端 Kernel 对账失败（exit={result.returncode}，原始响应未记录）")
        output = result.stdout.strip()
        next_token: str | None = None
        lines = output.splitlines()
        if lines and lines[0].startswith("Next Page Token = "):
            next_token = lines[0].split("=", 1)[1].strip()
            if not next_token:
                raise RuntimeError("远端 Kernel 分页 token 为空")
            output = "\n".join(lines[1:]).strip()
        if output.lower() == "not found":
            payload: Any = []
        else:
            try:
                payload = json.loads(output)
            except json.JSONDecodeError as exc:
                raise RuntimeError("远端 Kernel 列表不是 JSON") from exc
        if not isinstance(payload, list):
            raise RuntimeError("远端 Kernel 列表结构未知")
        refs.update(
            str(item["ref"])
            for item in payload
            if isinstance(item, dict) and item.get("ref")
        )
        if not all_pages or next_token is None:
            return sorted(refs)
        if next_token in seen_tokens:
            raise RuntimeError("远端 Kernel 分页 token 循环")
        seen_tokens.add(next_token)
        page_token = next_token
    raise RuntimeError(f"远端 Kernel 对账超过 {max_pages} 页，保守拒绝 push")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_dir", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", required=True, help="CX07 或 CL08 形式，且必须出现在唯一 Kernel slug 中")
    parser.add_argument("--actor", choices=("CX", "CL"), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--priority", choices=("normal", "final"), default="normal")
    parser.add_argument("--late-report-only", action="store_true")
    parser.add_argument("--kaggle-bin", default=os.environ.get("KAGGLE_BIN", "kaggle"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.ledger = args.ledger.resolve()
    root = args.kernel_dir.resolve()
    contract_path = args.contract.resolve()
    candidate = args.candidate.upper()
    if not CANDIDATE_RE.fullmatch(candidate) or not candidate.startswith(args.actor):
        print("PUSH REFUSED: candidate 必须形如 CX07/CL08 且与 actor 一致", file=sys.stderr)
        return 2
    try:
        contract = load_contract(contract_path)
        deadline = parse_utc(str(contract["submission_deadline_utc"]))
    except RuntimeError as exc:
        print(f"PUSH REFUSED: {exc}", file=sys.stderr)
        return 2
    if contract["mode"] == "official" and contract_path.stat().st_mode & 0o222:
        print("PUSH REFUSED: official TASK_CONTRACT.json 必须先 chmod 444 冻结", file=sys.stderr)
        return 2
    if contract["mode"] == "official":
        task_root_raw = os.environ.get("IOAI_TASK_ROOT")
        account_root_raw = os.environ.get("IOAI_ACCOUNT_ROOT")
        if not task_root_raw or not account_root_raw:
            print("PUSH REFUSED: official launcher 必须固定 IOAI_TASK_ROOT 与 IOAI_ACCOUNT_ROOT", file=sys.stderr)
            return 2
        task_root = Path(task_root_raw).resolve()
        account_root = Path(account_root_raw).resolve()
        if account_root == Path("/"):
            print("PUSH REFUSED: IOAI_ACCOUNT_ROOT 不得是文件系统根目录", file=sys.stderr)
            return 2
        if contract_path != task_root / "TASK_CONTRACT.json" or args.ledger.resolve() != task_root / "push-ledger.jsonl":
            print("PUSH REFUSED: official contract/ledger 必须使用 IOAI_TASK_ROOT canonical 路径", file=sys.stderr)
            return 2
        slot_ledger = account_root / "kernel-slot-ledger.jsonl"
    else:
        slot_ledger = contract_path.parent / ".ioai-account" / "kernel-slot-ledger.jsonl"
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    preflight = Path(__file__).with_name("preflight_kernel.py")
    checked = subprocess.run([sys.executable, str(preflight), str(root), "--contract", str(contract_path)], text=True)
    if checked.returncode != 0:
        print("PUSH REFUSED: preflight 未通过", file=sys.stderr)
        return checked.returncode

    try:
        metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"PUSH REFUSED: metadata 读取失败：{exc}", file=sys.stderr)
        return 2
    kernel_ref = str(metadata.get("id") or "")
    if not candidate_matches_ref(candidate, kernel_ref):
        print(f"PUSH REFUSED: 唯一 Kernel slug {kernel_ref!r} 必须以边界形式包含候选 ID {candidate}", file=sys.stderr)
        return 2
    competition_tag = competition_slug_tag(contract["competition_slug"])
    if not slug_has_token(kernel_ref, competition_tag):
        print(
            f"PUSH REFUSED: Kernel slug 必须以独立边界包含本题稳定标签 {competition_tag}，避免跨题 version 冲突",
            file=sys.stderr,
        )
        return 2

    now = utc_now()
    if args.late_report_only:
        if contract["late_report_window_seconds"] <= 0:
            print("PUSH REFUSED: 合同未授权赛后 Report 窗口", file=sys.stderr)
            return 2
        latest_start = deadline + timedelta(
            seconds=(
                contract["late_report_window_seconds"]
                - contract["kernel_timeout_seconds"]
                - contract["latest_start_safety_seconds"]
            )
        )
    else:
        latest_start = deadline - timedelta(
            seconds=contract["kernel_timeout_seconds"] + contract["latest_start_safety_seconds"]
        )

    try:
        remote_refs = remote_kernel_refs(args.kaggle_bin, contract["competition_slug"])
        global_matching_refs = remote_kernel_refs(
            args.kaggle_bin,
            search=kernel_ref.rsplit("/", 1)[-1],
        )
        accounted_slot_refs = active_slot_refs(slot_ledger)
        account_refs = set(
            remote_kernel_refs(
                args.kaggle_bin,
                page_size=100,
                sort_by="dateRun",
                all_pages=True,
                max_pages=20,
            )
        )
        slot_statuses = fetch_kernel_statuses(
            args.kaggle_bin,
            accounted_slot_refs | account_refs,
        )
        list_status_conflicts = {
            ref: slot_statuses.get(ref, "unknown")
            for ref in account_refs
            if slot_statuses.get(ref) == "absent"
        }
        if list_status_conflicts:
            raise RuntimeError(
                "账号 Kernel list 与 status 返回冲突；单次 404 不足以证明远端不存在"
            )
        untracked = {
            ref: slot_statuses.get(ref, "unknown")
            for ref in account_refs - accounted_slot_refs
            if slot_statuses.get(ref) != "terminal"
        }
        if untracked:
            raise RuntimeError(
                "账号最近 Kernel 中存在未纳入共享槽位账本的 active/unknown 运行；"
                "先等待终态或把所有写入者切到 guarded wrapper"
            )
        reconcile_slots(
            slot_ledger,
            limits=contract["kernel_slots"],
            statuses=slot_statuses,
        )
    except RuntimeError as exc:
        print(f"PUSH REFUSED: {exc}", file=sys.stderr)
        return 3
    now = utc_now()

    digest, solution_digest = candidate_hashes(root)
    reservation_id = str(uuid.uuid4())
    lock_path = args.ledger.with_suffix(args.ledger.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            events = read_events(args.ledger)
            latest = latest_reservations(events)
            exact_remote_refs = set(remote_refs)
            if kernel_ref in set(global_matching_refs):
                exact_remote_refs.add(kernel_ref)
            reconcile_ambiguous_pushes(args.ledger, latest, exact_remote_refs, now)
            events = read_events(args.ledger)
            latest = latest_reservations(events)
            used = [event for event in latest.values() if event.get("status") not in RELEASED_PUSH_STATUSES]
            contract_hashes = {event.get("contract_sha256") for event in used if event.get("contract_sha256")}
            if contract_hashes and contract_hashes != {contract_sha}:
                print("PUSH REFUSED: 当前合同 SHA 与本题账本不一致", file=sys.stderr)
                return 4
            known_refs = {str(event.get("kernel_ref")) for event in used if event.get("kernel_ref")}
            for remote_ref in remote_refs:
                if remote_ref in known_refs:
                    continue
                external_id = "external:" + hashlib.sha256(remote_ref.encode()).hexdigest()[:24]
                append_event(args.ledger, {
                    "reservation_id": external_id,
                    "time_utc": iso(now),
                    "status": "external_remote_kernel",
                    "actor": "EXTERNAL",
                    "candidate": None,
                    "kernel_ref": remote_ref,
                    "competition": contract["competition_slug"],
                    "contract_sha256": contract_sha,
                })
            if remote_refs:
                events = read_events(args.ledger)
                latest = latest_reservations(events)
                used = [event for event in latest.values() if event.get("status") not in RELEASED_PUSH_STATUSES]
            exact_existing = [
                event
                for event in used
                if event.get("candidate") == candidate
                and event.get("kernel_ref") == kernel_ref
                and event.get("candidate_sha256") == digest
                and event.get("solution_sha256") == solution_digest
                and event.get("contract_sha256") == contract_sha
            ]
            if len(exact_existing) == 1 and exact_existing[0].get("status") in {
                "push_returned_0",
                "remote_kernel_confirmed",
            }:
                print(f"PUSH CONFIRMED: 远端/账本已有同一候选 {candidate}，不重复创建 version")
                return 0
            if len(exact_existing) == 1 and exact_existing[0].get("status") == "remote_absent_once":
                print("PUSH DEFERRED: 第一次确认远端不存在；60 秒后重跑同一命令", file=sys.stderr)
                return 4
            if len(exact_existing) == 1 and exact_existing[0].get("status") in {
                "reserved",
                "push_returned_nonzero",
                "push_cli_timeout",
            }:
                print("PUSH DEFERRED: 既有 push 仍在进行或结果歧义；稍后重跑同一命令", file=sys.stderr)
                return 4
            if kernel_ref in set(global_matching_refs):
                print(
                    f"PUSH REFUSED: Kaggle 账号全局已存在 Kernel ref {kernel_ref}；"
                    "必须换全新 slug，不能创建 version 2+",
                    file=sys.stderr,
                )
                return 4
            gate_now = utc_now()
            if args.late_report_only:
                if gate_now < deadline or gate_now >= latest_start:
                    print(
                        f"PUSH REFUSED: 不在可完成赛后 Report Kernel 的窗口 "
                        f"[{iso(deadline)}, {iso(latest_start)})",
                        file=sys.stderr,
                    )
                    return 3
            elif gate_now >= latest_start:
                print(f"PUSH REFUSED: 已晚于 latest_start={iso(latest_start)}；只做远端对账", file=sys.stderr)
                return 3
            if any(event.get("candidate") == candidate for event in used):
                print(f"PUSH REFUSED: candidate ID {candidate} 已使用", file=sys.stderr)
                return 4
            if any(event.get("kernel_ref") == kernel_ref for event in used):
                print(f"PUSH REFUSED: Kernel ref {kernel_ref} 已使用，必须一候选一 slug", file=sys.stderr)
                return 4
            if any(
                event.get("candidate_sha256") == digest or event.get("solution_sha256") == solution_digest
                for event in used
            ):
                print(f"PUSH REFUSED: package/solution SHA {digest[:12]}/{solution_digest[:12]} 已在账本中", file=sys.stderr)
                return 4
            if len(used) >= contract["notebook_version_limit"]:
                print(f"PUSH REFUSED: version 账本已用 {len(used)}/{contract['notebook_version_limit']}", file=sys.stderr)
                return 5
            resource_class = "gpu" if metadata.get("enable_gpu") is True else "cpu"
            decision = try_acquire_slot(
                path=slot_ledger,
                lease_id=reservation_id,
                actor=args.actor,
                candidate=candidate,
                competition=contract["competition_slug"],
                kernel_ref=kernel_ref,
                resource_class=resource_class,
                priority=args.priority,
                contract_sha256=contract_sha,
                limits=contract["kernel_slots"],
                deadline=deadline,
                now=gate_now,
                statuses=slot_statuses,
            )
            if not decision.acquired:
                print(
                    f"RESOURCE DEFERRED: {decision.reason}; retry_after={decision.retry_after_seconds}s; "
                    "重跑同一 guarded wrapper，不要直接调用 kaggle kernels push",
                    file=sys.stderr,
                )
                return 75
            try:
                append_event(args.ledger, {
                    "reservation_id": reservation_id,
                    "time_utc": iso(now),
                    "status": "reserved",
                    "actor": args.actor,
                    "candidate": candidate,
                    "candidate_sha256": digest,
                    "solution_sha256": solution_digest,
                    "kernel_ref": kernel_ref,
                    "competition": contract["competition_slug"],
                    "contract_sha256": contract_sha,
                    "late_report_only": args.late_report_only,
                    "resource_class": resource_class,
                    "priority": args.priority,
                    "slot_ledger": str(slot_ledger),
                })
            except Exception:
                transition_slot(slot_ledger, reservation_id, "released_push_reservation_failed")
                raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    if candidate_hashes(root) != (digest, solution_digest):
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            append_event(args.ledger, {
                "reservation_id": reservation_id,
                "time_utc": iso(utc_now()),
                "status": "aborted_before_push",
                "actor": args.actor,
                "candidate": candidate,
                "candidate_sha256": digest,
                "solution_sha256": solution_digest,
                "kernel_ref": kernel_ref,
                "reason": "candidate_changed_after_reservation",
                "contract_sha256": contract_sha,
                "late_report_only": args.late_report_only,
            })
        transition_slot(slot_ledger, reservation_id, "released_before_push")
        print("PUSH REFUSED: 预留后候选发生变化，未调用 Kaggle", file=sys.stderr)
        return 6

    transition_slot(slot_ledger, reservation_id, "push_started")
    command = [
        args.kaggle_bin,
        "kernels",
        "push",
        "-p",
        str(root),
        "--timeout",
        str(contract["kernel_timeout_seconds"]),
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=180)
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        status = "push_returned_0" if returncode == 0 else "push_returned_nonzero"
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = subprocess_output(exc.stdout)
        stderr = subprocess_output(exc.stderr) + "\nKaggle CLI call timed out after 180 seconds"
        status = "push_cli_timeout"
    transition_slot(
        slot_ledger,
        reservation_id,
        "remote_active" if status == "push_returned_0" else "push_ambiguous",
    )
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        append_event(args.ledger, {
            "reservation_id": reservation_id,
            "time_utc": iso(utc_now()),
            "status": status,
            "actor": args.actor,
            "candidate": candidate,
            "candidate_sha256": digest,
            "solution_sha256": solution_digest,
            "kernel_ref": kernel_ref,
            "returncode": returncode,
            "contract_sha256": contract_sha,
            "late_report_only": args.late_report_only,
        })
    if returncode == 0:
        if stdout:
            print(safe_redact(str(stdout).rstrip()))
        if stderr:
            print(safe_redact(str(stderr).rstrip()), file=sys.stderr)
    elif stdout or stderr:
        print("Kaggle push 非零响应正文已抑制；重跑同一 guarded 命令做远端对账", file=sys.stderr)
    print(
        f"ledger: {args.ledger} | CLI 歧义时重新运行同一命令进行远端对账；"
        "只有两次间隔确认不存在才会释放"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
