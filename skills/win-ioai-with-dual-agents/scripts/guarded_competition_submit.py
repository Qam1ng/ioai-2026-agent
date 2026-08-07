#!/usr/bin/env python3
"""下载并校验真实 Kernel 输出，再用原子账本保护 IOAI competition submit。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from guarded_kernel_push import (
    append_event,
    candidate_matches_ref,
    iso,
    latest_reservations,
    read_events,
    subprocess_output,
)
from preflight_kernel import load_contract
from resource_slots import transition_slot
from submit_with_diagnostics import redact as safe_redact


CANDIDATE_RE = re.compile(r"^(CX|CL)\d{2,4}$", re.IGNORECASE)
AMBIGUOUS_SUBMIT_STATUSES = {"submit_returned_nonzero", "submit_cli_timeout"}
RELEASED_SUBMIT_STATUSES = {"aborted_before_submit", "safe_not_created"}
SUCCESS_SUBMIT_STATUSES = {"submit_returned_0", "remote_submit_confirmed"}
REMOTE_ABSENCE_WAIT_SECONDS = 60
# reservation 后最坏路径约 540 秒（status 60 + output 180 + 两个 validator 60+60 + submit 180）。
# TTL 必须严格高于该上界，避免把仍存活的提交误判为孤儿。
ORPHAN_RESERVATION_SECONDS = 720
DIAGNOSTIC_PREFIX = "KAGGLE_SUBMIT_DIAGNOSTIC="
HELPER_UNAVAILABLE_EXIT = 86
UNSAFE_DESCRIPTION_RE = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\b(?:file|data):|KGAT_[A-Za-z0-9_-]{8,}|"
    r"sk-(?:ant-api|or-v1)-[A-Za-z0-9_-]{8,}|"
    r"(?:authorization|password|sessionid|cookie|token|api[_-]?key)\s*[:=]|[\r\n])",
    re.IGNORECASE,
)
CONFIRMED_REMOTE_STATUSES = {"pending", "complete", "completed"}


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("submission_deadline_utc 不是 ISO-8601") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("submission_deadline_utc 必须带时区")
    return parsed.astimezone(timezone.utc)


def event_age_seconds(event: dict[str, Any], now: datetime) -> float:
    try:
        return max(0.0, (now - parse_utc(str(event["time_utc"]))).total_seconds())
    except (KeyError, RuntimeError):
        return 0.0


def validation_command(contract: dict[str, Any], actual: Path) -> list[str]:
    schema = contract["submission_schema"]
    validator = Path(__file__).with_name("validate_submission.py")
    command = [
        sys.executable,
        str(validator),
        str(actual),
        "--sample",
        schema["sample_submission"],
    ]
    for column in schema["id_columns"]:
        command += ["--id-column", column]
    if schema["allow_row_reorder"]:
        command.append("--allow-row-reorder")
    if schema["allow_column_reorder"]:
        command.append("--allow-column-reorder")
    for column in schema["numeric_columns"]:
        command += ["--numeric-column", column]
    for column in schema["integer_columns"]:
        command += ["--integer-column", column]
    for column, (low, high) in schema["ranges"].items():
        command += ["--range", f"{column}:{low}:{high}"]
    return command


def task_validation_command(contract: dict[str, Any], actual: Path) -> list[str]:
    return [
        sys.executable,
        contract["task_validator"]["path"],
        str(actual),
        "--sample",
        contract["submission_schema"]["sample_submission"],
    ]


def run_read_cli(command: list[str], timeout: int, label: str) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label}超时") from exc
    if result.returncode != 0:
        raise RuntimeError(f"{label}失败（exit={result.returncode}，原始响应未记录）")
    return result.stdout


def extract_diagnostic(stderr: str) -> tuple[dict[str, Any] | None, str]:
    diagnostic: dict[str, Any] | None = None
    kept: list[str] = []
    for line in stderr.splitlines():
        if line.startswith(DIAGNOSTIC_PREFIX):
            try:
                payload = json.loads(line[len(DIAGNOSTIC_PREFIX):])
            except json.JSONDecodeError:
                kept.append("Kaggle submit helper 返回了无法解析的脱敏诊断")
            else:
                if isinstance(payload, dict):
                    diagnostic = payload
            continue
        kept.append(line)
    return diagnostic, "\n".join(kept)


def json_list(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    # Kaggle CLI 2.2.4 在该比赛还没有任何提交时不会输出 JSON，而是输出这句固定文案。
    if not stripped or stripped == "No submissions found":
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = min((index for index in (stripped.find("["), stripped.find("{")) if index >= 0), default=-1)
        if start < 0:
            raise RuntimeError("Kaggle CLI 没有返回 JSON") from None
        payload = json.loads(stripped[start:])
    if isinstance(payload, dict):
        for key in ("submissions", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle submissions JSON 结构未知")
    return [item for item in payload if isinstance(item, dict)]


def remote_submissions(kaggle_bin: str, competition: str) -> list[dict[str, Any]]:
    output = run_read_cli(
        [
            kaggle_bin,
            "competitions",
            "submissions",
            competition,
            "--format",
            "json",
            "--page-size",
            "200",
        ],
        60,
        "远端 submission 对账",
    )
    return json_list(output)


def submission_binding(kernel_ref: str, version: int) -> str:
    return hashlib.sha256(f"{kernel_ref}@{version}".encode()).hexdigest()[:10]


def valid_description(description: str, kernel_ref: str, version: int) -> bool:
    if (
        len(description) > 500
        or UNSAFE_DESCRIPTION_RE.search(description)
        or safe_redact(description) != description
    ):
        return False
    fields = {
        key: value
        for part in description.split("|")[1:]
        if "=" in part
        for key, value in [part.split("=", 1)]
    }
    return fields.get("r") == submission_binding(kernel_ref, version)


def remote_submission_matches(
    item: dict[str, Any],
    *,
    description: str,
    kernel_ref: str,
    version: int,
) -> bool:
    value = item.get("description", item.get("message"))
    status = str(item.get("status") or "").strip().lower()
    if value != description or status not in CONFIRMED_REMOTE_STATUSES:
        return False
    remote_kernel = next(
        (item[key] for key in ("kernelRef", "kernel_ref", "kernel") if item.get(key)),
        None,
    )
    if remote_kernel is not None and str(remote_kernel) != kernel_ref:
        return False
    remote_version = next(
        (item[key] for key in ("kernelVersion", "kernel_version", "version") if item.get(key) is not None),
        None,
    )
    if remote_version is not None:
        try:
            if int(remote_version) != version:
                return False
        except (TypeError, ValueError):
            return False
    return True


def require_complete_kernel(kaggle_bin: str, kernel_ref: str) -> None:
    output = run_read_cli(
        [kaggle_bin, "kernels", "status", kernel_ref],
        60,
        "Kernel 状态查询",
    )
    lowered = output.lower()
    if re.search(r"\b(?:error|failed|failure|cancelled|canceled)\b", lowered):
        raise RuntimeError("Kernel 不是成功终态（远端状态为失败或取消）")
    if not re.search(r"\bcomplete(?:d)?\b", lowered):
        raise RuntimeError("Kernel 尚未完成或状态无法唯一确认")


def download_remote_output(
    kaggle_bin: str,
    kernel_ref: str,
    output_filename: str,
    target: Path,
) -> Path:
    run_read_cli(
        [kaggle_bin, "kernels", "output", kernel_ref, "-p", str(target)],
        180,
        "Kernel 输出下载",
    )
    matches = [path for path in target.rglob(output_filename) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"远端输出中应恰有一个 {output_filename}，实际 {len(matches)} 个")
    return matches[0]


def append_transition(
    path: Path,
    event: dict[str, Any],
    now: datetime,
    status: str,
    description: str,
) -> None:
    updated = dict(event)
    updated.update({"time_utc": iso(now), "status": status, "description": description})
    append_event(path, updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--actor", choices=("CX", "CL"), required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--push-ledger", type=Path, required=True)
    parser.add_argument("--submit-ledger", type=Path, required=True)
    parser.add_argument("--late-report-only", action="store_true")
    parser.add_argument("--kaggle-bin", default=os.environ.get("KAGGLE_BIN", "kaggle"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.push_ledger = args.push_ledger.resolve()
    args.submit_ledger = args.submit_ledger.resolve()
    candidate = args.candidate.upper()
    contract_path = args.contract.resolve()
    if not CANDIDATE_RE.fullmatch(candidate) or not candidate.startswith(args.actor):
        print("SUBMIT REFUSED: candidate 必须形如 CX07/CL08 且与 actor 一致", file=sys.stderr)
        return 2
    if (
        args.version != 1
        or not args.description.startswith(candidate + "|")
        or not valid_description(args.description, args.kernel_ref, args.version)
    ):
        print(
            "SUBMIT REFUSED: version 必须为 1；description 必须以 candidate| 开头、含正确 r= 绑定且无凭证/URL",
            file=sys.stderr,
        )
        return 2
    if not candidate_matches_ref(candidate, args.kernel_ref):
        print("SUBMIT REFUSED: kernel-ref slug 未以边界形式包含 candidate", file=sys.stderr)
        return 2
    try:
        contract = load_contract(contract_path)
        deadline = parse_utc(str(contract["submission_deadline_utc"]))
    except RuntimeError as exc:
        print(f"SUBMIT REFUSED: {exc}", file=sys.stderr)
        return 2
    if contract["mode"] == "official" and contract_path.stat().st_mode & 0o222:
        print("SUBMIT REFUSED: official TASK_CONTRACT.json 必须为只读", file=sys.stderr)
        return 2
    if contract["mode"] == "official":
        task_root_raw = os.environ.get("IOAI_TASK_ROOT")
        account_root_raw = os.environ.get("IOAI_ACCOUNT_ROOT")
        if not task_root_raw or not account_root_raw:
            print("SUBMIT REFUSED: official launcher 必须固定 IOAI_TASK_ROOT 与 IOAI_ACCOUNT_ROOT", file=sys.stderr)
            return 2
        task_root = Path(task_root_raw).resolve()
        account_root = Path(account_root_raw).resolve()
        if (
            contract_path != task_root / "TASK_CONTRACT.json"
            or args.push_ledger.resolve() != task_root / "push-ledger.jsonl"
            or args.submit_ledger.resolve() != task_root / "submit-ledger.jsonl"
        ):
            print("SUBMIT REFUSED: official contract/ledgers 必须使用 IOAI_TASK_ROOT canonical 路径", file=sys.stderr)
            return 2
        expected_slot_ledger = account_root / "kernel-slot-ledger.jsonl"
    else:
        expected_slot_ledger = contract_path.parent / ".ioai-account" / "kernel-slot-ledger.jsonl"
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    now = datetime.now(timezone.utc)
    if args.late_report_only:
        if contract["late_report_window_seconds"] <= 0 or "late-report" not in args.description.lower():
            print("SUBMIT REFUSED: late-report 模式未获合同授权，或 description 未含 late-report", file=sys.stderr)
            return 2
        cutoff = deadline + timedelta(
            seconds=contract["late_report_window_seconds"] - contract["submit_command_safety_seconds"]
        )
    else:
        cutoff = deadline - timedelta(seconds=contract["submit_command_safety_seconds"])

    push_latest = latest_reservations(read_events(args.push_ledger))
    successful_push = [
        event
        for event in push_latest.values()
        if event.get("status") in {"push_returned_0", "remote_kernel_confirmed"}
        and event.get("candidate") == candidate
        and event.get("kernel_ref") == args.kernel_ref
        and event.get("contract_sha256") == contract_sha
    ]
    if len(successful_push) != 1:
        print("SUBMIT REFUSED: 找不到唯一、合同一致且成功/远端确认的 push 记录", file=sys.stderr)
        return 4
    push_event = successful_push[0]
    if Path(str(push_event.get("slot_ledger") or "")).resolve() != expected_slot_ledger.resolve():
        print("SUBMIT REFUSED: push 记录未绑定 canonical 账号级槽位账本", file=sys.stderr)
        return 4

    try:
        submissions = remote_submissions(args.kaggle_bin, contract["competition_slug"])
    except RuntimeError as exc:
        print(f"SUBMIT REFUSED: {exc}", file=sys.stderr)
        return 3
    exact_remote_count = sum(
        remote_submission_matches(
            item,
            description=args.description,
            kernel_ref=args.kernel_ref,
            version=args.version,
        )
        for item in submissions
    )
    same_description_count = sum(
        item.get("description", item.get("message")) == args.description
        for item in submissions
    )
    if exact_remote_count > 1:
        print("SUBMIT REFUSED: 远端存在多个同 description 提交，无法唯一对账", file=sys.stderr)
        return 5
    if same_description_count and exact_remote_count == 0:
        print(
            "SUBMIT REFUSED: 远端同 description 记录的状态或 kernel/version 绑定不匹配，不能假确认为成功",
            file=sys.stderr,
        )
        return 5

    submit_key = f"{contract['competition_slug']}|{args.kernel_ref}|{args.version}"
    reservation_id = str(uuid.uuid4())
    lock_path = args.submit_ledger.with_suffix(args.submit_ledger.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing = latest_reservations(read_events(args.submit_ledger))
            related = [event for event in existing.values() if event.get("submit_key") == submit_key]

            if exact_remote_count == 1:
                if related:
                    target = max(related, key=lambda event: str(event.get("time_utc") or ""))
                    append_transition(
                        args.submit_ledger,
                        target,
                        now,
                        "remote_submit_confirmed",
                        args.description,
                    )
                else:
                    append_event(args.submit_ledger, {
                        "reservation_id": "external:" + hashlib.sha256(args.description.encode()).hexdigest()[:24],
                        "time_utc": iso(now),
                        "status": "remote_submit_confirmed",
                        "actor": "EXTERNAL",
                        "candidate": candidate,
                        "submit_key": submit_key,
                        "kernel_ref": args.kernel_ref,
                        "version": args.version,
                        "description": args.description,
                        "contract_sha256": contract_sha,
                    })
                print("SUBMIT CONFIRMED: 远端已有完全相同 description，不重复调用 submit")
                return 0

            active = [event for event in related if event.get("status") not in RELEASED_SUBMIT_STATUSES]
            if any(event.get("status") in SUCCESS_SUBMIT_STATUSES for event in active):
                print(f"SUBMIT CONFIRMED: 本地账本已记录成功：{submit_key}")
                return 0
            if len(active) > 1:
                print("SUBMIT REFUSED: 同一 submit_key 有多个活跃 reservation", file=sys.stderr)
                return 5
            if active:
                event = active[0]
                status = event.get("status")
                age = event_age_seconds(event, now)
                if status in AMBIGUOUS_SUBMIT_STATUSES and age >= REMOTE_ABSENCE_WAIT_SECONDS:
                    append_transition(args.submit_ledger, event, now, "remote_absent_once", args.description)
                    print("SUBMIT DEFERRED: 第一次确认远端不存在；60 秒后重跑同一命令", file=sys.stderr)
                    return 5
                if status == "remote_absent_once" and age >= REMOTE_ABSENCE_WAIT_SECONDS:
                    append_transition(args.submit_ledger, event, now, "safe_not_created", args.description)
                elif status == "submit_reserved" and age >= ORPHAN_RESERVATION_SECONDS:
                    append_transition(args.submit_ledger, event, now, "remote_absent_once", args.description)
                    print("SUBMIT DEFERRED: 恢复孤儿 reservation，60 秒后重跑同一命令", file=sys.stderr)
                    return 5
                else:
                    print("SUBMIT REFUSED: 既有 submit 仍在进行或结果歧义；稍后重跑同一命令对账", file=sys.stderr)
                    return 5

            gate_now = datetime.now(timezone.utc)
            if args.late_report_only:
                if gate_now < deadline or gate_now >= cutoff:
                    print(f"SUBMIT REFUSED: 不在赛后 Report 窗口 [{iso(deadline)}, {iso(cutoff)})", file=sys.stderr)
                    return 3
            elif gate_now >= cutoff:
                print(f"SUBMIT REFUSED: 已晚于 submit safety cutoff={iso(cutoff)}", file=sys.stderr)
                return 3

            append_event(args.submit_ledger, {
                "reservation_id": reservation_id,
                "time_utc": iso(datetime.now(timezone.utc)),
                "status": "submit_reserved",
                "actor": args.actor,
                "candidate": candidate,
                "submit_key": submit_key,
                "kernel_ref": args.kernel_ref,
                "version": args.version,
                "description": args.description,
                "late_report_only": args.late_report_only,
                "contract_sha256": contract_sha,
            })
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    output_sha256: str | None = None
    try:
        require_complete_kernel(args.kaggle_bin, args.kernel_ref)
        transition_slot(expected_slot_ledger, str(push_event["reservation_id"]), "released_terminal")
        with tempfile.TemporaryDirectory(prefix="ioai-kernel-output-") as temp_dir:
            output = download_remote_output(
                args.kaggle_bin,
                args.kernel_ref,
                contract["submission_schema"]["output_filename"],
                Path(temp_dir),
            )
            try:
                validation = subprocess.run(validation_command(contract, output), text=True, timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("通用输出校验超过 60 秒") from exc
            if validation.returncode != 0:
                raise RuntimeError("远端真实输出校验失败")
            try:
                task_validation = subprocess.run(task_validation_command(contract, output), text=True, timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("题目专属输出校验超过 60 秒") from exc
            if task_validation.returncode != 0:
                raise RuntimeError("远端真实输出未通过题目专属校验")
            output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except RuntimeError as exc:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            append_event(args.submit_ledger, {
                "reservation_id": reservation_id,
                "time_utc": iso(datetime.now(timezone.utc)),
                "status": "aborted_before_submit",
                "actor": args.actor,
                "candidate": candidate,
                "submit_key": submit_key,
                "kernel_ref": args.kernel_ref,
                "version": args.version,
                "description": args.description,
                "contract_sha256": contract_sha,
                "reason": str(exc),
            })
        print(f"SUBMIT REFUSED: {exc}", file=sys.stderr)
        return 6

    gate_now = datetime.now(timezone.utc)
    phase_closed = (
        gate_now >= cutoff
        or (args.late_report_only and gate_now < deadline)
    )
    if phase_closed:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            append_event(args.submit_ledger, {
                "reservation_id": reservation_id,
                "time_utc": iso(datetime.now(timezone.utc)),
                "status": "aborted_before_submit",
                "actor": args.actor,
                "candidate": candidate,
                "submit_key": submit_key,
                "kernel_ref": args.kernel_ref,
                "version": args.version,
                "description": args.description,
                "contract_sha256": contract_sha,
                "reason": "submission_phase_closed_after_output_validation",
            })
        print(f"SUBMIT REFUSED: 输出校验后已离开允许提交窗口，cutoff={iso(cutoff)}", file=sys.stderr)
        return 3

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        current = latest_reservations(read_events(args.submit_ledger)).get(reservation_id)
        if not current or current.get("status") != "submit_reserved":
            print("SUBMIT REFUSED: reservation fencing 已失效，旧进程不得继续 submit", file=sys.stderr)
            return 7
        fenced_now = datetime.now(timezone.utc)
        if fenced_now >= cutoff or (args.late_report_only and fenced_now < deadline):
            append_transition(
                args.submit_ledger,
                current,
                fenced_now,
                "aborted_before_submit",
                args.description,
            )
            print("SUBMIT REFUSED: 获得最终 fencing 后已离开提交窗口", file=sys.stderr)
            return 3

        helper = Path(__file__).with_name("submit_with_diagnostics.py")
        command = [
            sys.executable,
            str(helper),
            "--competition",
            contract["competition_slug"],
            "--kernel",
            args.kernel_ref,
            "--version",
            str(args.version),
            "--file",
            contract["submission_schema"]["output_filename"],
            "--message",
            args.description,
        ]
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=180)
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
            if returncode == HELPER_UNAVAILABLE_EXIT:
                command = [
                    args.kaggle_bin,
                    "competitions",
                    "submit",
                    contract["competition_slug"],
                    "-k",
                    args.kernel_ref,
                    "-v",
                    str(args.version),
                    "-f",
                    contract["submission_schema"]["output_filename"],
                    "-m",
                    args.description,
                ]
                result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=180)
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            status = "submit_returned_0" if returncode == 0 else "submit_returned_nonzero"
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = subprocess_output(exc.stdout)
            stderr = subprocess_output(exc.stderr) + "\nKaggle submit call timed out after 180 seconds"
            status = "submit_cli_timeout"
        diagnostic, stderr = extract_diagnostic(str(stderr))
        append_event(args.submit_ledger, {
            "reservation_id": reservation_id,
            "time_utc": iso(datetime.now(timezone.utc)),
            "status": status,
            "actor": args.actor,
            "candidate": candidate,
            "submit_key": submit_key,
            "kernel_ref": args.kernel_ref,
            "version": args.version,
            "description": args.description,
            "late_report_only": args.late_report_only,
            "contract_sha256": contract_sha,
            "output_sha256": output_sha256,
            "returncode": returncode,
            "diagnostic": diagnostic,
        })
    if stdout and returncode == 0:
        print(safe_redact(str(stdout).rstrip()))
    elif stdout:
        print("Kaggle submit stdout 已抑制；使用 submit ledger 中的脱敏 diagnostic", file=sys.stderr)
    if stderr and returncode == 0:
        print(safe_redact(str(stderr).rstrip()), file=sys.stderr)
    elif stderr:
        print("Kaggle submit stderr 已抑制；使用 submit ledger 中的脱敏 diagnostic", file=sys.stderr)
    print(
        f"submit ledger: {args.submit_ledger} | CLI 歧义时重跑同一命令对账；"
        "完全相同 description 已存在时不会重复提交"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
