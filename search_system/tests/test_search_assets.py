from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ioai_agent_system.search_assets import (
    AssetIntegrityError,
    AssetPolicyError,
    snapshot_assets,
    verify_snapshot,
)


class SearchAssetTests(unittest.TestCase):
    def test_snapshot_preserves_bytes_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "nested").mkdir()
            payload = b"\x00\x01exact bytes\xff"
            (source / "nested" / "data.bin").write_bytes(payload)
            bundle = root / "bundle"
            bundle.mkdir()
            snapshot = snapshot_assets(
                source,
                bundle / "ORIGINAL_ASSETS",
                sha256sums_path=bundle / "SHA256SUMS",
                manifest_path=bundle / "ASSET_MANIFEST.json",
            )
            copied = snapshot.snapshot_root / "nested" / "data.bin"
            self.assertEqual(copied.read_bytes(), payload)
            self.assertNotIn(
                str(source), snapshot.manifest_path.read_text(encoding="utf-8")
            )
            verify_snapshot(snapshot)
            copied.chmod(0o644)
            copied.write_bytes(b"tampered")
            with self.assertRaises(AssetIntegrityError):
                verify_snapshot(snapshot)

    def test_credentials_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "kaggle.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(AssetPolicyError):
                snapshot_assets(
                    source,
                    root / "copy1",
                    sha256sums_path=root / "sum1",
                    manifest_path=root / "manifest1",
                )
            (source / "kaggle.json").unlink()
            (source / "outside").symlink_to(root)
            with self.assertRaises(AssetPolicyError):
                snapshot_assets(
                    source,
                    root / "copy2",
                    sha256sums_path=root / "sum2",
                    manifest_path=root / "manifest2",
                )


if __name__ == "__main__":
    unittest.main()
