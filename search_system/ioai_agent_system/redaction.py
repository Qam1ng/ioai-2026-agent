"""Small, dependency-free secret redaction helpers.

The control plane persists prompts, events, and subprocess diagnostics.  These
helpers are deliberately applied at the persistence boundary so an accidental
credential in an agent response cannot enter SQLite or the audit JSONL.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"

# Anthropic keys currently start with ``sk-ant-``.  The lower bound avoids
# changing harmless prose that merely mentions the prefix.
_SK_ANT = re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{16,}")
_KAGGLE_TOKEN = re.compile(r"(?<![A-Za-z0-9])KGAT_[A-Za-z0-9_-]{16,}")
_OPENAI_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:(?:RSA|OPENSSH|EC) )?PRIVATE KEY-----.*?"
    r"-----END (?:(?:RSA|OPENSSH|EC) )?PRIVATE KEY-----",
    re.S,
)

# Handle JSON/YAML-like quoted fields before the more general header patterns;
# this preserves the surrounding document syntax.
_QUOTED_SENSITIVE_FIELD = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?
        (?:authorization|x-api-key|anthropic_api_key)
        ["']?
        \s*[:=]\s*
    )
    (?P<quote>["'])
    (?P<value>(?:\\.|(?! (?P=quote) ).)*)
    (?P=quote)
    """
)

_AUTHORIZATION = re.compile(
    r"""(?ix)
    (?P<prefix>\bauthorization\b\s*[:=]\s*)
    (?P<quote>["']?)
    (?:(?:bearer|basic)\s+)?
    (?P<secret>[A-Za-z0-9._~+/=-]{8,})
    (?P=quote)
    """
)

_API_KEY_FIELD = re.compile(
    r"""(?ix)
    (?P<prefix>\b(?:x-api-key|anthropic_api_key)\b\s*[:=]\s*)
    (?P<quote>["']?)
    (?P<secret>[A-Za-z0-9._~+/=-]{8,})
    (?P=quote)
    """
)

_SENSITIVE_KEYS = {
    "authorization",
    "x-api-key",
    "x_api_key",
    "anthropic_api_key",
}


def _quoted_field_replacement(match: re.Match[str]) -> str:
    return (
        f"{match.group('prefix')}{match.group('quote')}"
        f"{REDACTED}{match.group('quote')}"
    )


def _header_replacement(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{match.group('prefix')}{quote}{REDACTED}{quote}"


def redact_text(text: str) -> str:
    """Return *text* with supported credential forms removed.

    Redaction is idempotent and intentionally retains header names so traces
    remain useful while their values do not.
    """

    redacted = _SK_ANT.sub(REDACTED, text)
    redacted = _KAGGLE_TOKEN.sub(REDACTED, redacted)
    redacted = _OPENAI_TOKEN.sub(REDACTED, redacted)
    redacted = _PRIVATE_KEY_BLOCK.sub(REDACTED, redacted)
    redacted = _QUOTED_SENSITIVE_FIELD.sub(
        _quoted_field_replacement, redacted
    )
    redacted = _AUTHORIZATION.sub(_header_replacement, redacted)
    redacted = _API_KEY_FIELD.sub(_header_replacement, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact strings and values under credential-like keys.

    The result contains only ordinary Python containers.  Tuples are retained
    as tuples; other non-string sequences are returned as lists.
    """

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _SENSITIVE_KEYS:
                result[key] = REDACTED
            else:
                result[key] = redact_value(item)
        return result
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        return [redact_value(item) for item in value]
    return value


def contains_secret(value: Any) -> bool:
    """Return whether *value* contains a supported, unredacted secret form."""

    if isinstance(value, str):
        return redact_text(value) != value
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in _SENSITIVE_KEYS:
                if item not in (None, "", REDACTED):
                    return True
            if contains_secret(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return any(contains_secret(item) for item in value)
    return False


__all__ = ["REDACTED", "contains_secret", "redact_text", "redact_value"]
