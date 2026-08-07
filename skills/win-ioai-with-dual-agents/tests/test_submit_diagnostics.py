from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from guarded_competition_submit import (  # noqa: E402
    extract_diagnostic,
    json_list,
    remote_submission_matches,
    submission_binding,
    valid_description,
)
import guarded_competition_submit as guarded_submit  # noqa: E402
from guarded_kernel_push import subprocess_output  # noqa: E402
import submit_with_diagnostics  # noqa: E402
from submit_with_diagnostics import safe_error_summary  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, content: bytes, content_type: str) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type, "Authorization": "Bearer never-log-this"}


class SubmitDiagnosticTests(unittest.TestCase):
    def test_empty_remote_submission_message_is_normalized(self) -> None:
        self.assertEqual(json_list("No submissions found\n"), [])

    def test_submission_list_accepts_cli_next_page_prefix(self) -> None:
        output = 'Next Page Token = opaque-token\n[{"status":"complete"}]\n'
        self.assertEqual(json_list(output), [{"status": "complete"}])

    def test_nested_json_is_allowlisted_and_redacted(self) -> None:
        body = json.dumps({
            "message": "hardware P100 denied KGAT_secretvalue123",
            "debug": {
                "authorization": "Bearer secretsecret",
                "detail": "GPU not allowed; see https://internal.test/private?session_id=LEAKME",
            },
            "errors": [{"reason": "machine_shape"}],
            "https://private.example/raw?token=LEAKME": {"message": "safe message"},
            "Authorization: Bearer HEADERSECRET123": {"detail": "safe detail"},
        }).encode()
        summary = safe_error_summary(FakeResponse(400, body, "application/json"), "HTTPError")
        rendered = json.dumps(summary)
        self.assertEqual(summary["status_code"], 400)
        self.assertIn("GPU not allowed", rendered)
        self.assertNotIn("KGAT_", rendered)
        self.assertNotIn("secretsecret", rendered)
        self.assertNotIn("never-log-this", rendered)
        self.assertNotIn("internal.test", rendered)
        self.assertNotIn("LEAKME", rendered)
        self.assertNotIn("private.example", rendered)
        self.assertNotIn("HEADERSECRET123", rendered)

    def test_html_is_opaque_but_hashed(self) -> None:
        body = b"<html><body>private diagnostics</body></html>"
        summary = safe_error_summary(FakeResponse(500, body, "text/html"), "HTTPError")
        self.assertEqual(summary["body_kind"], "opaque")
        self.assertEqual(summary["body_sha256"], hashlib.sha256(body).hexdigest())
        self.assertNotIn("text", summary)
        self.assertNotIn("private diagnostics", json.dumps(summary))

    def test_plain_text_is_opaque(self) -> None:
        body = (b"Bearer verysecretcredential token=supersecretvalue " + b"x" * 1000)
        summary = safe_error_summary(FakeResponse(403, body, "text/plain"), "HTTPError")
        self.assertEqual(summary["body_kind"], "opaque")
        self.assertNotIn("text", summary)
        self.assertNotIn("verysecretcredential", json.dumps(summary))

    def test_wrapper_extracts_only_sentinel_json(self) -> None:
        diagnostic = {"status_code": 400, "body_kind": "json"}
        parsed, rest = extract_diagnostic(
            "before\nKAGGLE_SUBMIT_DIAGNOSTIC=" + json.dumps(diagnostic) + "\nafter"
        )
        self.assertEqual(parsed, diagnostic)
        self.assertEqual(rest, "before\nafter")

    def test_description_is_bound_and_rejects_secrets_or_urls(self) -> None:
        binding = submission_binding("owner/kernel", 1)
        valid = f"CX01|p=-|f=model|cv=.5|d=x|h=abc|r={binding}"
        self.assertTrue(valid_description(valid, "owner/kernel", 1))
        self.assertFalse(valid_description(valid.replace(binding, "bad"), "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|url=https://private.test", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|token=secret", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=Bearer verysecretcredential", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=Basic verysecretcredential", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=sk-proj-verysecretcredential", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=ftp://user:secret@host", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=s3://private-bucket/key", "owner/kernel", 1))
        self.assertFalse(valid_description(valid + "|d=file:///private/path", "owner/kernel", 1))

    def test_timeout_partial_output_bytes_are_normalized(self) -> None:
        self.assertEqual(subprocess_output(b"partial\xff"), "partial�")

    def test_kernel_status_failure_does_not_echo_raw_output(self) -> None:
        raw = "KernelWorkerStatus.ERROR Bearer SUPERSECRET123 https://private.example/path"
        with patch.object(guarded_submit, "run_read_cli", return_value=raw):
            with self.assertRaises(RuntimeError) as caught:
                guarded_submit.require_complete_kernel("kaggle", "owner/kernel")
        rendered = str(caught.exception)
        self.assertNotIn("SUPERSECRET123", rendered)
        self.assertNotIn("private.example", rendered)

    def test_remote_confirmation_requires_status_and_available_binding_fields(self) -> None:
        binding = submission_binding("owner/kernel", 1)
        description = f"CX01|p=-|f=model|cv=.5|d=x|h=abc|r={binding}"
        self.assertTrue(remote_submission_matches(
            {"description": description, "status": "pending", "kernelRef": "owner/kernel", "kernelVersion": 1},
            description=description,
            kernel_ref="owner/kernel",
            version=1,
        ))
        self.assertFalse(remote_submission_matches(
            {"description": description, "status": "error", "kernelRef": "owner/kernel", "kernelVersion": 1},
            description=description,
            kernel_ref="owner/kernel",
            version=1,
        ))
        self.assertFalse(remote_submission_matches(
            {"description": description, "status": "complete", "kernelRef": "owner/other", "kernelVersion": 1},
            description=description,
            kernel_ref="owner/kernel",
            version=1,
        ))

    def test_helper_issues_exactly_one_submit_call(self) -> None:
        calls: list[dict] = []

        class FakeApi:
            def authenticate(self) -> None:
                pass

            def competition_submit_code(self, **kwargs):
                calls.append(kwargs)

        module = types.ModuleType("kaggle.api.kaggle_api_extended")
        module.KaggleApi = FakeApi
        argv = [
            "submit_with_diagnostics.py",
            "--competition", "task",
            "--kernel", "owner/kernel",
            "--version", "1",
            "--file", "submission.csv",
            "--message", "CX01|test",
        ]
        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": module}), patch.object(sys, "argv", argv):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                returncode = submit_with_diagnostics.main()
        self.assertEqual(returncode, 0)
        self.assertEqual(len(calls), 1)

    def test_http_failure_is_diagnosed_without_retry(self) -> None:
        calls = 0

        class SubmitError(Exception):
            def __init__(self) -> None:
                self.response = FakeResponse(400, b'{"message":"P100 is not allowed"}', "application/json")

        class FakeApi:
            def authenticate(self) -> None:
                pass

            def competition_submit_code(self, **kwargs):
                nonlocal calls
                calls += 1
                raise SubmitError()

        module = types.ModuleType("kaggle.api.kaggle_api_extended")
        module.KaggleApi = FakeApi
        argv = [
            "submit_with_diagnostics.py",
            "--competition", "task",
            "--kernel", "owner/kernel",
            "--version", "1",
            "--file", "submission.csv",
            "--message", "CX01|test",
        ]
        stderr = io.StringIO()
        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": module}), patch.object(sys, "argv", argv):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                returncode = submit_with_diagnostics.main()
        self.assertEqual(returncode, 1)
        self.assertEqual(calls, 1)
        self.assertIn("P100 is not allowed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
