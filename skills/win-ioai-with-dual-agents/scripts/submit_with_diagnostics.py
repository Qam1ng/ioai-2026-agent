#!/usr/bin/env python3
"""提交一次 code competition，并安全记录同一 HTTP 失败的脱敏诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any


HELPER_UNAVAILABLE_EXIT = 86
MAX_BODY_BYTES = 4096
MAX_VALUE_CHARS = 512
SAFE_FIELDS = {"code", "status", "message", "detail", "reason", "errors"}
SECRET_PATTERNS = (
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"\b(?:file|data):[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"KGAT_[A-Za-z0-9_-]{8,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(
        r"(?i)(authorization|x-api-key|api[_-]?key|(?:access[_-]?)?token|password|signature|"
        r"session[_-]?id|cookie|xsrf|csrf)(\s*[:=]\s*)([^\s,;\]}]+)"
    ),
)


def redact(value: str) -> str:
    cleaned = "".join(character if character in "\n\t" or ord(character) >= 32 else " " for character in value)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            cleaned = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned[:MAX_VALUE_CHARS]


def collect_safe_fields(value: Any, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            safe_key = str(key).lower()
            if safe_key in SAFE_FIELDS:
                child_path = f"{path}.{safe_key}"
                if isinstance(child, (str, int, float, bool)) or child is None:
                    safe_value = redact(child) if isinstance(child, str) else child
                    found.append({"path": child_path, "value": safe_value})
                elif isinstance(child, (dict, list)):
                    found.extend(collect_safe_fields(child, child_path))
            else:
                # 父级 JSON key 不在白名单中，不能让 URL/header/token 借 path 泄漏。
                found.extend(collect_safe_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value[:20]):
            found.extend(collect_safe_fields(child, f"{path}[{index}]"))
    return found[:40]


def safe_error_summary(response: Any, error_class: str) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    raw_body = bytes(getattr(response, "content", b"") or b"")
    body = raw_body[:MAX_BODY_BYTES]
    total_length = len(raw_body)
    summary: dict[str, Any] = {
        "error_class": error_class,
        "status_code": status_code if isinstance(status_code, int) else None,
        "body_bytes": total_length,
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
        "body_truncated": total_length > len(body),
    }
    content_type = str(getattr(response, "headers", {}).get("Content-Type", "")).lower()
    stripped = body.lstrip()
    if "json" in content_type or stripped.startswith((b"{", b"[")):
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            summary["body_kind"] = "invalid_json"
        else:
            summary["body_kind"] = "json"
            summary["safe_fields"] = collect_safe_fields(parsed)
    else:
        # 无结构正文无法可靠区分错误信息与凭证/私有 URL，只保留长度和哈希。
        summary["body_kind"] = "opaque"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--message", required=True)
    return parser.parse_args()


def emit_diagnostic(response: Any, error_class: str) -> None:
    payload = safe_error_summary(response, error_class)
    print("KAGGLE_SUBMIT_DIAGNOSTIC=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("KAGGLE_SUBMIT_HELPER_UNAVAILABLE", file=sys.stderr)
        return HELPER_UNAVAILABLE_EXIT

    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_submit_code(
            file_name=args.file,
            message=args.message,
            competition=args.competition,
            kernel=args.kernel,
            kernel_version=args.version,
            quiet=True,
        )
    except Exception as exc:  # Kaggle SDK 可抛 requests.HTTPError 或 SDK 包装异常。
        response = getattr(exc, "response", None)
        if response is not None:
            emit_diagnostic(response, type(exc).__name__)
        else:
            print(
                "KAGGLE_SUBMIT_DIAGNOSTIC="
                + json.dumps({"error_class": type(exc).__name__, "body_kind": "unavailable"}, sort_keys=True),
                file=sys.stderr,
            )
        return 1

    print("KAGGLE_SUBMIT_API_RETURNED_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
