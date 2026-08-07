from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from preflight_kernel import (  # noqa: E402
    embedded_artifact_findings,
    load_contract,
    training_split_declaration,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "sample.csv").write_text("id,prediction\n1,0\n", encoding="utf-8")
        (self.root / "validator.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        (self.root / "rules.md").write_text("official validation may be used for training\n", encoding="utf-8")
        (self.root / "validation.csv").write_text("id,label\n2,1\n", encoding="utf-8")
        (self.root / "asset-manifest.json").write_text(json.dumps({
            "discovery_complete": True,
            "files": [
                {"path": "train.csv", "role": "train", "split": "train", "sha256": "1" * 64},
                {
                    "path": "validation.csv",
                    "role": "labels",
                    "split": "validation",
                    "sha256": sha(self.root / "validation.csv"),
                },
                {"path": "test.csv", "role": "features", "split": "submission_test", "sha256": "3" * 64},
            ],
            "base_training_split_names": ["train"],
            "labeled_split_names": ["validation"],
            "overlap_checks": [
                {
                    "left": "validation",
                    "right": "train",
                    "id_overlap_count": 0,
                    "content_overlap_count": 0,
                    "label_transfer_authorized": False,
                },
                {
                    "left": "validation",
                    "right": "submission_test",
                    "id_overlap_count": 0,
                    "content_overlap_count": 0,
                    "label_transfer_authorized": False,
                },
            ],
        }), encoding="utf-8")
        self.contract = {
            "mode": "rehearsal",
            "competition_slug": "rehearsal-task",
            "official_sources": [{
                "url": "https://example.test/rules",
                "retrieved_at": "2026-08-07T00:00:00Z",
                "local_path": "rules.md",
                "sha256": sha(self.root / "rules.md"),
            }],
            "official_labeled_splits": [{
                "name": "validation",
                "labels_path": "validation.csv",
                "labels_sha256": sha(self.root / "validation.csv"),
                "sample_count": 1,
                "group_count": 1,
                "label_columns": ["label"],
                "official_purpose": "development validation and final training",
                "training_use": "allowed",
                "source_sha256": sha(self.root / "rules.md"),
            }],
            "asset_audit": {
                "manifest_path": "asset-manifest.json",
                "manifest_sha256": sha(self.root / "asset-manifest.json"),
            },
            "final_training_splits": ["train", "validation"],
            "start_time_utc": "2026-08-07T00:00:00Z",
            "submission_deadline_utc": "2026-08-07T02:00:00Z",
            "metric": {"name": "accuracy", "direction": "maximize"},
            "submission_schema": {
                "sample_submission": "sample.csv",
                "sample_submission_sha256": sha(self.root / "sample.csv"),
                "output_filename": "submission.csv",
                "id_columns": ["id"],
                "allow_row_reorder": False,
                "allow_column_reorder": False,
                "numeric_columns": [],
                "integer_columns": ["prediction"],
                "ranges": {"prediction": [0, 5]},
            },
            "task_validator": {"path": "validator.py", "sha256": sha(self.root / "validator.py")},
            "notebook_version_limit": 20,
            "kernel_timeout_seconds": 300,
            "latest_start_safety_seconds": 0,
            "submit_command_safety_seconds": 0,
            "late_report_window_seconds": 0,
            "allowed_hardware": {"allow_cpu": True, "gpu_shapes": [], "only_cuda0": True},
            "kernel_slots": {
                "max_concurrent_cpu": 5,
                "max_concurrent_gpu": 0,
                "per_actor_inflight": 1,
                "final_priority_window_seconds": 1500,
                "final_waiter_ttl_seconds": 45,
                "orphan_reservation_seconds": 300,
            },
            "allowed_sources": {
                "competition_sources": ["rehearsal-task"],
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
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_contract(self) -> Path:
        path = self.root / "TASK_CONTRACT.json"
        path.write_text(json.dumps(self.contract), encoding="utf-8")
        return path

    def test_labeled_split_and_slot_limits_are_normalized(self) -> None:
        loaded = load_contract(self.write_contract())
        self.assertEqual(loaded["official_labeled_splits"][0]["training_use"], "allowed")
        self.assertTrue(Path(loaded["official_labeled_splits"][0]["labels_path"]).is_absolute())
        self.assertEqual(loaded["kernel_slots"]["max_concurrent_cpu"], 5)

    def test_unknown_training_use_is_explicitly_allowed_but_other_values_fail(self) -> None:
        self.contract["official_labeled_splits"][0]["training_use"] = "UNKNOWN"
        self.contract["final_training_splits"] = ["train"]
        self.assertEqual(load_contract(self.write_contract())["official_labeled_splits"][0]["training_use"], "UNKNOWN")
        self.contract["official_labeled_splits"][0]["training_use"] = "assumed_allowed"
        with self.assertRaisesRegex(RuntimeError, "training_use"):
            load_contract(self.write_contract())

    def test_validation_only_cannot_enter_final_training(self) -> None:
        self.contract["official_labeled_splits"][0]["training_use"] = "validation_only"
        with self.assertRaisesRegex(RuntimeError, "final_training_splits"):
            load_contract(self.write_contract())

    def test_asset_manifest_must_cover_overlap_checks(self) -> None:
        manifest_path = self.root / "asset-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["overlap_checks"] = manifest["overlap_checks"][:1]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.contract["asset_audit"]["manifest_sha256"] = sha(manifest_path)
        with self.assertRaisesRegex(RuntimeError, "重叠检查"):
            load_contract(self.write_contract())

    def test_extra_source_requires_official_provenance(self) -> None:
        self.contract["allowed_sources"]["dataset_sources"] = ["owner/private-cache"]
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            load_contract(self.write_contract())

    def test_training_marker_and_embedded_artifact_gate(self) -> None:
        self.assertEqual(
            training_split_declaration("# IOAI_FINAL_TRAINING_SPLITS: train,validation\nprint('ok')\n"),
            ["train", "validation"],
        )
        findings = embedded_artifact_findings("predictions = [0] * 31 + [1]\n")
        self.assertTrue(findings)
        findings = embedded_artifact_findings("predictions = [0," + "1," * 31 + "2]\n")
        self.assertTrue(findings)
        findings = embedded_artifact_findings("answers = np.array([" + "1," * 32 + "2])\n")
        self.assertTrue(findings)

    def test_official_mode_requires_strict_two_file_package(self) -> None:
        self.contract["mode"] = "official"
        self.contract["latest_start_safety_seconds"] = 180
        self.contract["submit_command_safety_seconds"] = 30
        self.contract["strict_kernel_files"] = False
        with self.assertRaisesRegex(RuntimeError, "strict_kernel_files"):
            load_contract(self.write_contract())

    def test_official_mode_requires_orphan_ttl_longer_than_push_call(self) -> None:
        self.contract["mode"] = "official"
        self.contract["latest_start_safety_seconds"] = 180
        self.contract["submit_command_safety_seconds"] = 30
        self.contract["kernel_slots"]["orphan_reservation_seconds"] = 60
        with self.assertRaisesRegex(RuntimeError, "orphan TTL"):
            load_contract(self.write_contract())


if __name__ == "__main__":
    unittest.main()
