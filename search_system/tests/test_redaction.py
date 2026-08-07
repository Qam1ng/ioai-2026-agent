from __future__ import annotations

import unittest

from ioai_agent_system.redaction import (
    REDACTED,
    contains_secret,
    redact_text,
    redact_value,
)


class RedactionTests(unittest.TestCase):
    def test_redacts_anthropic_key_without_retaining_fragments(self) -> None:
        secret = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"
        result = redact_text(f"before {secret} after")
        self.assertEqual(result, f"before {REDACTED} after")
        self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUVWXYZ", result)

    def test_redacts_kaggle_openai_and_private_key_material(self) -> None:
        value = (
            "KGAT_abcdefghijklmnopqrstuvwxyz012345 "
            "sk-proj-abcdefghijklmnopqrstuvwxyz012345 "
            "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----"
        )
        result = redact_text(value)
        self.assertNotIn("KGAT_", result)
        self.assertNotIn("sk-proj-", result)
        self.assertNotIn("BEGIN PRIVATE KEY", result)

    def test_redacts_authorization_header_and_json_field(self) -> None:
        header = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
        self.assertEqual(
            redact_text(header), f"Authorization: {REDACTED}"
        )
        jsonish = (
            '{"Authorization":"Bearer abcdefghijklmnop",'
            '"safe":"retained"}'
        )
        self.assertEqual(
            redact_text(jsonish),
            f'{{"Authorization":"{REDACTED}","safe":"retained"}}',
        )

    def test_redacts_x_api_key_in_multiple_forms(self) -> None:
        self.assertEqual(
            redact_text("x-api-key=abcdefghijklmnop"),
            f"x-api-key={REDACTED}",
        )
        self.assertEqual(
            redact_text("'x-api-key': 'abcdefghijklmnop'"),
            f"'x-api-key': '{REDACTED}'",
        )

    def test_recursive_redaction_preserves_nonsecret_structure(self) -> None:
        value = {
            "Authorization": "Bearer abcdefghijklmnop",
            "nested": [
                {"x-api-key": "abcdefghijklmnop"},
                "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345",
                7,
            ],
            "safe": "hello",
        }
        result = redact_value(value)
        self.assertEqual(result["Authorization"], REDACTED)
        self.assertEqual(result["nested"][0]["x-api-key"], REDACTED)
        self.assertEqual(result["nested"][1], REDACTED)
        self.assertEqual(result["nested"][2], 7)
        self.assertEqual(result["safe"], "hello")
        self.assertFalse(contains_secret(result))

    def test_contains_secret_detects_sensitive_mapping_value(self) -> None:
        self.assertTrue(
            contains_secret({"x-api-key": "not-even-a-token"})
        )
        self.assertTrue(
            contains_secret(
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
            )
        )
        self.assertFalse(contains_secret({"x-api-key": REDACTED}))

    def test_redaction_is_idempotent(self) -> None:
        original = (
            "Authorization: Bearer abcdefghijklmnop "
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"
        )
        once = redact_text(original)
        self.assertEqual(redact_text(once), once)


if __name__ == "__main__":
    unittest.main()
