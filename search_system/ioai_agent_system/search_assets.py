from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class AssetPolicyError(RuntimeError):
    """Raised when the approved asset directory contains unsafe material."""


class AssetIntegrityError(RuntimeError):
    """Raised when a source or snapshot changes after the initial manifest."""


@dataclass(frozen=True, slots=True)
class AssetEntry:
    path: str
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    source_root: Path
    snapshot_root: Path
    entries: tuple[AssetEntry, ...]
    directories: tuple[str, ...]
    sha256sums_path: Path
    manifest_path: Path


_BUFFER_SIZE = 1024 * 1024
_FORBIDDEN_NAMES = {
    ".env",
    "access_token",
    "auth.json",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "kaggle.json",
}
_FORBIDDEN_DIRS = {".aws", ".kaggle", ".ssh"}
_SECRET_PATTERNS = (
    re.compile(rb"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"KGAT_[A-Za-z0-9_-]{16,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_disjoint_paths(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if _is_within(destination, source) or _is_within(source, destination):
        raise AssetPolicyError(
            f"asset source and output must be disjoint: {source} vs {destination}"
        )


def _check_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("asset snapshot exceeded the Search module deadline")


def _hash_file(
    path: Path, *, deadline_monotonic: float | None = None
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            _check_deadline(deadline_monotonic)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_relative_path(relative: Path) -> None:
    if "\n" in relative.as_posix() or "\r" in relative.as_posix():
        raise AssetPolicyError(f"newline in asset path: {relative!s}")
    if any(part in _FORBIDDEN_DIRS for part in relative.parts):
        raise AssetPolicyError(f"credential directory is not an asset: {relative!s}")
    if relative.name.lower() in _FORBIDDEN_NAMES:
        raise AssetPolicyError(f"credential file is not an asset: {relative!s}")


def _scan_obvious_secret(path: Path, relative: Path) -> None:
    if path.stat().st_size > 2 * 1024 * 1024:
        return
    data = path.read_bytes()
    if any(pattern.search(data) for pattern in _SECRET_PATTERNS):
        raise AssetPolicyError(f"obvious credential material in asset: {relative!s}")


def _enumerate_tree(
    root: Path,
    *,
    scan_secrets: bool,
    deadline_monotonic: float | None = None,
) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = [Path(".")]
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        _check_deadline(deadline_monotonic)
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        for name in sorted(dir_names):
            _check_deadline(deadline_monotonic)
            candidate = current_path / name
            relative = candidate.relative_to(root)
            _validate_relative_path(relative)
            if candidate.is_symlink():
                raise AssetPolicyError(f"symlink assets are not accepted: {relative!s}")
            mode = candidate.stat().st_mode
            if not stat.S_ISDIR(mode):
                raise AssetPolicyError(f"special directory entry: {relative!s}")
            directories.append(relative)
        for name in sorted(file_names):
            _check_deadline(deadline_monotonic)
            candidate = current_path / name
            relative = candidate.relative_to(root)
            _validate_relative_path(relative)
            if candidate.is_symlink():
                raise AssetPolicyError(f"symlink assets are not accepted: {relative!s}")
            mode = candidate.stat().st_mode
            if not stat.S_ISREG(mode):
                raise AssetPolicyError(f"special file is not an asset: {relative!s}")
            if scan_secrets:
                _scan_obvious_secret(candidate, relative)
            files.append(relative)
    return sorted(files), sorted(set(directories))


def _manifest(
    root: Path,
    files: list[Path],
    *,
    deadline_monotonic: float | None = None,
) -> tuple[AssetEntry, ...]:
    entries: list[AssetEntry] = []
    for relative in files:
        path = root / relative
        _check_deadline(deadline_monotonic)
        digest, size = _hash_file(path, deadline_monotonic=deadline_monotonic)
        entries.append(
            AssetEntry(
                path=relative.as_posix(),
                sha256=digest,
                size=size,
                mode=stat.S_IMODE(path.stat().st_mode),
            )
        )
    return tuple(entries)


def _write_manifests(snapshot: AssetSnapshot) -> None:
    snapshot.sha256sums_path.write_text(
        "".join(f"{entry.sha256}  {entry.path}\n" for entry in snapshot.entries),
        encoding="utf-8",
    )
    payload = {
        "algorithm": "sha256",
        "source_root": "approved_input",
        "snapshot_root": str(snapshot.snapshot_root),
        "file_count": len(snapshot.entries),
        "directories": list(snapshot.directories),
        "files": [asdict(entry) for entry in snapshot.entries],
    }
    snapshot.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def snapshot_assets(
    source_root: Path,
    snapshot_root: Path,
    *,
    sha256sums_path: Path,
    manifest_path: Path,
    scan_secrets: bool = True,
    deadline_monotonic: float | None = None,
) -> AssetSnapshot:
    source_root = source_root.expanduser().resolve()
    snapshot_root = snapshot_root.expanduser().absolute()
    if not source_root.is_dir():
        raise AssetPolicyError(f"asset directory does not exist: {source_root}")
    require_disjoint_paths(source_root, snapshot_root)
    if snapshot_root.exists():
        raise FileExistsError(snapshot_root)

    files, directories = _enumerate_tree(
        source_root,
        scan_secrets=scan_secrets,
        deadline_monotonic=deadline_monotonic,
    )
    before = _manifest(
        source_root, files, deadline_monotonic=deadline_monotonic
    )
    snapshot_root.mkdir(parents=True)
    for relative in directories:
        if relative != Path("."):
            (snapshot_root / relative).mkdir(parents=True, exist_ok=True)
    for entry in before:
        _check_deadline(deadline_monotonic)
        source = source_root / entry.path
        destination = snapshot_root / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_handle, destination.open("wb") as output_handle:
            while chunk := input_handle.read(_BUFFER_SIZE):
                _check_deadline(deadline_monotonic)
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        shutil.copystat(source, destination, follow_symlinks=False)
        # 保留原执行位，同时移除全部写位；Agent 只能读取或执行快照。
        destination.chmod(0o555 if entry.mode & 0o111 else 0o444)

    after_source = _manifest(
        source_root, files, deadline_monotonic=deadline_monotonic
    )
    if after_source != before:
        raise AssetIntegrityError("asset source changed while the snapshot was built")
    snapshot_files, snapshot_dirs = _enumerate_tree(
        snapshot_root,
        scan_secrets=False,
        deadline_monotonic=deadline_monotonic,
    )
    after_snapshot = _manifest(
        snapshot_root,
        snapshot_files,
        deadline_monotonic=deadline_monotonic,
    )
    if [(item.path, item.sha256, item.size) for item in after_snapshot] != [
        (item.path, item.sha256, item.size) for item in before
    ]:
        raise AssetIntegrityError("snapshot bytes differ from source bytes")

    for relative in reversed(snapshot_dirs):
        directory = snapshot_root if relative == Path(".") else snapshot_root / relative
        directory.chmod(0o555)

    snapshot = AssetSnapshot(
        source_root=source_root,
        snapshot_root=snapshot_root,
        entries=before,
        directories=tuple(item.as_posix() for item in directories),
        sha256sums_path=sha256sums_path,
        manifest_path=manifest_path,
    )
    sha256sums_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifests(snapshot)
    return snapshot


def verify_snapshot(
    snapshot: AssetSnapshot,
    *,
    verify_source: bool = True,
    deadline_monotonic: float | None = None,
) -> None:
    files, directories = _enumerate_tree(
        snapshot.snapshot_root,
        scan_secrets=False,
        deadline_monotonic=deadline_monotonic,
    )
    actual = _manifest(
        snapshot.snapshot_root,
        files,
        deadline_monotonic=deadline_monotonic,
    )
    expected_core = [(item.path, item.sha256, item.size) for item in snapshot.entries]
    actual_core = [(item.path, item.sha256, item.size) for item in actual]
    if actual_core != expected_core:
        raise AssetIntegrityError("ORIGINAL_ASSETS no longer matches SHA256SUMS")
    if tuple(item.as_posix() for item in directories) != snapshot.directories:
        raise AssetIntegrityError("ORIGINAL_ASSETS directory set changed")
    if verify_source:
        source_files, source_dirs = _enumerate_tree(
            snapshot.source_root,
            scan_secrets=False,
            deadline_monotonic=deadline_monotonic,
        )
        source = _manifest(
            snapshot.source_root,
            source_files,
            deadline_monotonic=deadline_monotonic,
        )
        if [(item.path, item.sha256, item.size) for item in source] != expected_core:
            raise AssetIntegrityError("original asset source changed during the run")
        if tuple(item.as_posix() for item in source_dirs) != snapshot.directories:
            raise AssetIntegrityError("original asset directory set changed")


def make_tree_read_only(root: Path) -> None:
    files, directories = _enumerate_tree(root, scan_secrets=False)
    for relative in files:
        path = root / relative
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(0o555 if mode & 0o111 else 0o444)
    for relative in reversed(directories):
        directory = root if relative == Path(".") else root / relative
        directory.chmod(0o555)


__all__ = [
    "AssetEntry",
    "AssetIntegrityError",
    "AssetPolicyError",
    "AssetSnapshot",
    "make_tree_read_only",
    "require_disjoint_paths",
    "snapshot_assets",
    "verify_snapshot",
]
