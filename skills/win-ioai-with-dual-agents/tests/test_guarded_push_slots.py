from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))

from guarded_kernel_push import remote_kernel_refs  # noqa: E402
from resource_slots import try_acquire_slot  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GuardedPushSlotIntegrationTests(unittest.TestCase):
    def test_kernel_list_accepts_cli_next_page_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake_kaggle = Path(temp) / "fake-kaggle"
            fake_kaggle.write_text(
                "#!/usr/bin/env python3\n"
                "print('Next Page Token = opaque-token')\n"
                "print('[{\"ref\":\"owner/a\"},{\"ref\":\"owner/b\"}]')\n",
                encoding="utf-8",
            )
            fake_kaggle.chmod(0o755)
            self.assertEqual(
                remote_kernel_refs(str(fake_kaggle), page_size=20),
                ["owner/a", "owner/b"],
            )

    def test_kernel_list_follows_all_page_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake_kaggle = Path(temp) / "fake-kaggle"
            fake_kaggle.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if '--page-token' in sys.argv:\n"
                "    print('[{\"ref\":\"owner/page2\"}]')\n"
                "else:\n"
                "    print('Next Page Token = opaque-token')\n"
                "    print('[{\"ref\":\"owner/page1\"}]')\n",
                encoding="utf-8",
            )
            fake_kaggle.chmod(0o755)
            self.assertEqual(
                remote_kernel_refs(str(fake_kaggle), all_pages=True),
                ["owner/page1", "owner/page2"],
            )

    def test_resource_deferred_does_not_reserve_version_or_call_push(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kernel = root / "kernel"
            kernel.mkdir()
            calls = root / "calls.jsonl"
            fake_kaggle = root / "fake-kaggle"
            fake_kaggle.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                f"p=pathlib.Path({str(calls)!r})\n"
                "p.open('a').write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1:3] == ['kernels','list']:\n"
                "    if os.environ.get('FAKE_EXTERNAL') == '1' and '--competition' not in sys.argv and '--search' not in sys.argv:\n"
                "        print('[{\"ref\":\"owner/external\"}]')\n"
                "    else:\n"
                "        print('[]')\n"
                "elif sys.argv[1:3] == ['kernels','status']:\n"
                "    if os.environ.get('FAKE_STATUS_404') == '1':\n"
                "        print('404 Not Found', file=sys.stderr); raise SystemExit(1)\n"
                "    print('KernelWorkerStatus.RUNNING')\n"
                "elif sys.argv[1:3] == ['kernels','push']:\n"
                "    print('UNEXPECTED PUSH', file=sys.stderr); raise SystemExit(99)\n"
                "else:\n"
                "    raise SystemExit(98)\n",
                encoding="utf-8",
            )
            fake_kaggle.chmod(0o755)

            sample = root / "sample.csv"
            validator = root / "validator.py"
            rules = root / "rules.md"
            sample.write_text("id,prediction\n1,0\n", encoding="utf-8")
            validator.write_text("raise SystemExit(0)\n", encoding="utf-8")
            rules.write_text("rehearsal rules\n", encoding="utf-8")
            asset_manifest = root / "asset-manifest.json"
            asset_manifest.write_text(json.dumps({
                "discovery_complete": True,
                "files": [
                    {"path": "train.csv", "role": "train", "split": "train", "sha256": "1" * 64},
                    {"path": "test.csv", "role": "features", "split": "submission_test", "sha256": "2" * 64},
                ],
                "base_training_split_names": ["train"],
                "labeled_split_names": [],
                "overlap_checks": [],
            }), encoding="utf-8")
            competition = "rehearsal-task"
            tag = hashlib.sha256(competition.encode()).hexdigest()[:8]
            (kernel / "script.py").write_text("print('ok')\n", encoding="utf-8")
            (kernel / "kernel-metadata.json").write_text(json.dumps({
                "id": f"owner/{tag}-cx01-model",
                "title": "candidate",
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "competition_sources": [competition],
                "dataset_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }), encoding="utf-8")
            contract = root / "TASK_CONTRACT.json"
            contract.write_text(json.dumps({
                "mode": "rehearsal",
                "competition_slug": competition,
                "official_sources": [{
                    "url": "https://example.test/rules",
                    "retrieved_at": "2026-08-07T00:00:00Z",
                    "local_path": "rules.md",
                    "sha256": sha(rules),
                }],
                "official_labeled_splits": [],
                "asset_audit": {
                    "manifest_path": "asset-manifest.json",
                    "manifest_sha256": sha(asset_manifest),
                },
                "final_training_splits": ["train"],
                "start_time_utc": "2026-08-07T00:00:00Z",
                "submission_deadline_utc": "2099-08-07T02:00:00Z",
                "metric": {"name": "accuracy", "direction": "maximize"},
                "submission_schema": {
                    "sample_submission": "sample.csv",
                    "sample_submission_sha256": sha(sample),
                    "output_filename": "submission.csv",
                    "id_columns": ["id"],
                    "allow_row_reorder": False,
                    "allow_column_reorder": False,
                    "numeric_columns": [],
                    "integer_columns": ["prediction"],
                    "ranges": {"prediction": [0, 5]},
                },
                "task_validator": {"path": "validator.py", "sha256": sha(validator)},
                "notebook_version_limit": 20,
                "kernel_timeout_seconds": 300,
                "latest_start_safety_seconds": 0,
                "submit_command_safety_seconds": 0,
                "late_report_window_seconds": 0,
                "allowed_hardware": {
                    "allow_cpu": True,
                    "gpu_shapes": ["NvidiaTeslaT4"],
                    "only_cuda0": True,
                },
                "kernel_slots": {
                    "max_concurrent_cpu": 5,
                    "max_concurrent_gpu": 2,
                    "per_actor_inflight": 1,
                    "final_priority_window_seconds": 1500,
                    "final_waiter_ttl_seconds": 45,
                    "orphan_reservation_seconds": 300,
                },
                "allowed_sources": {
                    "competition_sources": [competition],
                    "dataset_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
                "allowed_source_provenance": [],
                "agent_research_network_policy": "allowed_methods_only",
                "kernel_internet_enabled": False,
                "required_report": False,
                "strict_kernel_files": True,
                "final_selection_policy": "kaggle_auto",
            }), encoding="utf-8")

            slot_ledger = root / ".ioai-account" / "kernel-slot-ledger.jsonl"
            now = datetime.now(timezone.utc)
            slot_limits = json.loads(contract.read_text())["kernel_slots"]
            for index in range(2):
                decision = try_acquire_slot(
                    path=slot_ledger,
                    lease_id=f"occupied{index}",
                    actor=f"other{index}",
                    candidate=f"other{index}",
                    competition="other-task",
                    kernel_ref=f"owner/occupied{index}",
                    resource_class="gpu",
                    priority="normal",
                    contract_sha256="b" * 64,
                    limits=slot_limits,
                    deadline=now + timedelta(hours=2),
                    now=now,
                    statuses={},
                )
                self.assertTrue(decision.acquired)

            push_ledger = root / "push-ledger.jsonl"
            ledger_alias = root / "alias-ledger.jsonl"
            ledger_alias.symlink_to(push_ledger)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "guarded_kernel_push.py"),
                    str(kernel),
                    "--contract",
                    str(contract),
                    "--candidate",
                    "CX01",
                    "--actor",
                    "CX",
                    "--ledger",
                    str(ledger_alias),
                    "--kaggle-bin",
                    str(fake_kaggle),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
            self.assertIn("RESOURCE DEFERRED", result.stderr)
            self.assertFalse(push_ledger.exists())
            self.assertTrue(push_ledger.with_suffix(".jsonl.lock").exists())
            self.assertFalse(ledger_alias.with_suffix(".jsonl.lock").exists())
            recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertFalse(any(call[:2] == ["kernels", "push"] for call in recorded))

            external = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "guarded_kernel_push.py"),
                    str(kernel),
                    "--contract",
                    str(contract),
                    "--candidate",
                    "CX01",
                    "--actor",
                    "CX",
                    "--ledger",
                    str(ledger_alias),
                    "--kaggle-bin",
                    str(fake_kaggle),
                ],
                env={**os.environ, "FAKE_EXTERNAL": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(external.returncode, 3)
            self.assertIn("未纳入共享槽位账本", external.stderr)

            conflict = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "guarded_kernel_push.py"),
                    str(kernel),
                    "--contract",
                    str(contract),
                    "--candidate",
                    "CX01",
                    "--actor",
                    "CX",
                    "--ledger",
                    str(ledger_alias),
                    "--kaggle-bin",
                    str(fake_kaggle),
                ],
                env={**os.environ, "FAKE_EXTERNAL": "1", "FAKE_STATUS_404": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(conflict.returncode, 3)
            self.assertIn("list 与 status 返回冲突", conflict.stderr)


if __name__ == "__main__":
    unittest.main()
