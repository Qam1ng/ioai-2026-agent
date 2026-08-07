"""Loaders for the radar-ioai-2025 heat-map segmentation mirror.

Each sample is a single ``torch.save``-d tensor in a ``*.mat.pt`` file:

* training file: ``(7, 50, 181) float64`` — 6 input channels + 1 label channel
* test file:     ``(6, 50, 181) float64`` — inputs only

Two things this module does that a naive loader would not:

**torch is imported inside the functions.** The swarm's control box has no torch
(only the Kaggle kernel does), and a module that cannot be imported cannot be
used to check anything.  Importing at module scope would make the task card,
the metric, and every test unavailable off-GPU.  The error you get when you
actually load without torch is explicit about the fix.

**There is a torch-free reader.**  ``backend="raw"`` parses the ``.pt``
container directly — it is a zip holding a pickle header plus one flat storage
blob — and returns the same float64 array.  This is not a party trick: it is how
the shapes above were *confirmed* rather than assumed, and it lets the swarm
inspect, validate and metric-check radar data on a machine with no torch.  It
only handles the exact single-tensor layout these files use and raises
otherwise, rather than pretending to be a general unpickler.
"""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

#: Confirmed by parsing the pickle headers of the shipped files.
FULL_SHAPE = (7, 50, 181)
N_INPUT_CHANNELS = 6
LABEL_CHANNEL = 6
HEIGHT, WIDTH = 50, 181
N_PIXELS = HEIGHT * WIDTH  # 9050
N_CLASSES = 5

#: Raw labels on disk are -1..3; the notebook adds 1 to get 0..4 for
#: CrossEntropyLoss and subtracts 1 again before writing the submission.
LABEL_SHIFT = 1
RAW_LABEL_RANGE = (-1, 3)
TRAIN_LABEL_RANGE = (0, N_CLASSES - 1)

SUBMISSION_ID_COL = "filename"
FILE_SUFFIX = ".mat.pt"

_TORCH_HINT = (
    "torch is required to load %s. Install it (the Kaggle kernel already has "
    "torch==2.7.1) or call load_tensor(..., backend='raw'), which parses the "
    "same file with numpy only."
)


# ------------------------------------------------------------------ discovery


def list_tensor_files(directory: str | Path) -> list[Path]:
    """Every ``*.mat.pt`` under ``directory``, in **numeric** filename order.

    Sorted numerically (``2.mat.pt`` before ``10.mat.pt``), not lexically.  Row
    order is part of the submission format, and the difference between these two
    orderings is a submission that scores as garbage while looking fine.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"radar data directory not found: {directory}")
    files = [p for p in directory.rglob("*" + FILE_SUFFIX) if p.is_file()]

    def key(p: Path) -> tuple[int, str]:
        m = re.match(r"^(\d+)", p.name)
        return (int(m.group(1)) if m else 1 << 60, p.name)

    return sorted(files, key=key)


def find_data_dir(
    names: Sequence[str] = ("training_set", "test_set"),
    roots: Iterable[str | Path] = ("/kaggle/input", "data", "."),
) -> dict[str, Path]:
    """Locate the ``training_set`` / ``test_set`` leaf directories.

    The zip nests them twice (``training_set/training_set/*.mat.pt``) and on
    Kaggle the whole thing sits under ``/kaggle/input/competitions/<slug>/`` —
    one level deeper than people write by hand
    (``memory/lessons/kaggle-kernel-input-paths.md``).  So we search for the
    directory that actually *contains* the tensors rather than trusting any
    prefix.
    """
    candidates: list[Path] = []
    for var in ("KAGGLE_COMPETITION_DIR", "KAGGLE_INPUT"):
        val = os.environ.get(var)
        if val:
            candidates.append(Path(val))
    candidates += [Path(r) for r in roots]

    found: dict[str, Path] = {}
    for root in candidates:
        if not root.exists():
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            if not any(f.endswith(FILE_SUFFIX) for f in filenames):
                continue
            p = Path(dirpath)
            for name in names:
                if name in p.parts and name not in found:
                    found[name] = p
        if len(found) == len(names):
            break
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(
            f"could not find {missing} containing *{FILE_SUFFIX} under "
            f"{[str(c) for c in candidates]}"
        )
    return found


# -------------------------------------------------------------------- loading


def _load_raw(path: Path) -> np.ndarray:
    """Read a single-tensor ``.pt`` without torch.

    ``torch.save`` writes a zip: ``*/data.pkl`` (a pickle whose only real
    content is a ``_rebuild_tensor_v2`` call carrying dtype, shape and stride)
    and ``*/data/0`` (the raw storage bytes).  We read the shape and dtype out
    of the pickle opcodes and reinterpret the blob.

    Deliberately narrow: contiguous, offset-0, single-storage tensors only —
    which is what every file in this dataset is.  Anything else raises instead
    of silently returning a wrong view.
    """
    import pickletools

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        blob_name = next((n for n in names if n.endswith("data/0")), None)
        if pkl_name is None or blob_name is None:
            raise ValueError(f"{path} is not a single-tensor torch archive")

        ops = list(pickletools.genops(zf.read(pkl_name)))
        dtype: np.dtype | None = None
        ints: list[int] = []
        for op, arg, _pos in ops:
            if op.name == "GLOBAL" and isinstance(arg, str):
                if arg == "torch DoubleStorage":
                    dtype = np.dtype("<f8")
                elif arg == "torch FloatStorage":
                    dtype = np.dtype("<f4")
                elif arg == "torch LongStorage":
                    dtype = np.dtype("<i8")
                elif arg == "torch IntStorage":
                    dtype = np.dtype("<i4")
                elif arg.startswith("torch ") and arg.endswith("Storage"):
                    raise ValueError(f"unsupported storage type in {path}: {arg}")
            elif op.name.startswith("BININT") or op.name == "LONG1":
                ints.append(int(arg))
        if dtype is None:
            raise ValueError(f"could not determine dtype of {path}")

        # Opcode layout: [storage numel] [offset] [shape...] [stride...]
        # The shape is the run of ints after the storage size and the offset
        # whose product divides the storage; take the 3 following the offset.
        if len(ints) < 5:
            raise ValueError(f"unexpected tensor header in {path}")
        numel, _offset, *rest = ints
        shape = tuple(rest[:3])
        if any(s <= 0 for s in shape) or int(np.prod(shape)) > numel:
            raise ValueError(f"unexpected tensor shape {shape} for storage of {numel} in {path}")

        buf = zf.read(blob_name)
        flat = np.frombuffer(buf, dtype=dtype, count=numel)
        # These files store all 7 channels even when the tensor view exposes 6;
        # reshape from the *storage*, then trim to the declared view so this
        # backend returns exactly what torch.load would (see CARD.md's note on
        # the label channel that survives in the test files).
        plane = shape[-2] * shape[-1]
        if numel % plane:
            raise ValueError(
                f"{path}: storage of {numel} is not a whole number of "
                f"{shape[-2]}x{shape[-1]} planes"
            )
        full = flat.reshape(numel // plane, shape[-2], shape[-1])
        return np.array(full[: shape[0]], dtype=np.float64)


def load_tensor(path: str | Path, backend: str = "torch") -> np.ndarray:
    """Load one ``*.mat.pt`` into a float64 array.

    ``backend``:

    * ``"torch"`` (default) — ``torch.load(weights_only=True)``, exactly what the
      baseline notebook and the Kaggle kernel do.  Raises a clear ``RuntimeError``
      naming the fix if torch is absent.
    * ``"raw"`` — the torch-free zip/pickle reader above.
    * ``"auto"`` — torch when importable, otherwise raw.

    The default is ``"torch"`` on purpose: the submitted kernel must use torch,
    so the path we exercise by default is the path that ships.
    """
    path = Path(path)
    if backend not in ("torch", "raw", "auto"):
        raise ValueError(f"unknown backend: {backend!r}")
    if backend == "raw":
        return _load_raw(path)

    try:
        import torch
    except ImportError as exc:
        if backend == "auto":
            return _load_raw(path)
        raise RuntimeError(_TORCH_HINT % path) from exc
    return torch.load(path, weights_only=True).numpy().astype(np.float64)


def load_sample(
    path: str | Path, with_labels: bool = True, backend: str = "torch"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(inputs (6,50,181) float32, labels (50,181) int64 | None)``.

    Labels come back **already shifted** into ``0..4`` (raw ``-1..3`` plus
    ``LABEL_SHIFT``), which is what a ``CrossEntropyLoss`` wants and what
    ``metric.py`` expects.  The −1 shift back happens once, in
    ``write_submission``.
    """
    arr = load_tensor(path, backend=backend)
    inputs = arr[:N_INPUT_CHANNELS].astype(np.float32)
    labels: np.ndarray | None = None
    if with_labels:
        if arr.shape[0] <= LABEL_CHANNEL:
            raise ValueError(
                f"{path} has {arr.shape[0]} channels — no label channel "
                f"(index {LABEL_CHANNEL}); pass with_labels=False for test files"
            )
        labels = (arr[LABEL_CHANNEL] + LABEL_SHIFT).astype(np.int64)
    return inputs, labels


def load_split(
    directory: str | Path,
    with_labels: bool = True,
    limit: int | None = None,
    backend: str = "torch",
) -> dict[str, Any]:
    """Load a whole split into memory.

    Returns ``{"X": (N,6,50,181) float32, "y": (N,50,181) int64 | None,
    "filenames": [str], "paths": [Path]}``.

    Sizing warning before you call this on everything: the 1,000 training files
    are ~1.1 GB as float32 inputs alone (``1000 * 6 * 9050 * 4``), plus 72 MB of
    int64 labels.  Use ``limit`` for exploration and stream in the kernel.
    """
    paths = list_tensor_files(directory)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no *{FILE_SUFFIX} files under {directory}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for p in paths:
        x, y = load_sample(p, with_labels=with_labels, backend=backend)
        xs.append(x)
        if y is not None:
            ys.append(y)
    return {
        "X": np.stack(xs),
        "y": np.stack(ys) if ys else None,
        "filenames": [p.name for p in paths],
        "paths": paths,
    }


# ---------------------------------------------------------------- submission


def submission_columns(n_pixels: int = N_PIXELS) -> list[str]:
    """``['filename', 'pixel_0', ..., 'pixel_9049']`` — the exact header."""
    return [SUBMISSION_ID_COL] + [f"pixel_{i}" for i in range(n_pixels)]


def write_submission(
    path: str | Path,
    filenames: Sequence[str],
    predictions: np.ndarray,
    shift_back: bool = True,
) -> Path:
    """Write ``filename,pixel_0..pixel_N`` with labels shifted back to raw.

    ``predictions`` is ``(N, 50, 181)`` or ``(N, 9050)`` holding **training-space**
    class ids ``0..4``; ``shift_back`` subtracts ``LABEL_SHIFT`` so the file
    carries the raw ``-1..3`` the grader compares against — the exact step the
    baseline does with ``preds = preds - 1``.  Forgetting it produces a
    perfectly-formed submission that is wrong on every pixel, which is why it is
    the default and not the caller's job.

    Rows are written in the order given; that must be the numeric filename order
    from ``list_tensor_files``.
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3:
        preds = preds.reshape(preds.shape[0], -1)
    if preds.ndim != 2:
        raise ValueError(f"predictions must be (N,H,W) or (N,P); got {preds.shape}")
    if len(filenames) != preds.shape[0]:
        raise ValueError(
            f"{len(filenames)} filenames but {preds.shape[0]} prediction rows"
        )
    if shift_back:
        preds = preds - LABEL_SHIFT

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = submission_columns(preds.shape[1])
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for name, row in zip(filenames, preds.astype(np.int64)):
            writer.writerow([name, *row.tolist()])
    return path


def read_submission(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a submission back as ``(filenames, (N,P) int64)`` in raw label space."""
    path = Path(path)
    try:
        if csv.field_size_limit() < 10_000_000:
            csv.field_size_limit(10_000_000)
    except OverflowError:  # pragma: no cover - platform dependent
        pass
    names: list[str] = []
    rows: list[list[int]] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            names.append(row[0])
            rows.append([int(float(v)) for v in row[1:]])
    return names, np.array(rows, dtype=np.int64)


# ------------------------------------------------------------------ describe


def describe(
    train_dir: str | Path | None = None,
    limit: int = 25,
    backend: str = "auto",
    stream: Any = None,
) -> dict[str, Any]:
    """Print (and return) split sizes and the class balance.

    Defaults to ``backend='auto'`` because this is an inspection helper and
    should work on a box with no torch.
    """
    import sys
    from collections import Counter

    out = stream or sys.stdout
    if train_dir is None:
        train_dir = find_data_dir(("training_set",))["training_set"]
    paths = list_tensor_files(train_dir)

    counts: Counter[int] = Counter()
    for p in paths[:limit]:
        _x, y = load_sample(p, with_labels=True, backend=backend)
        for v, c in zip(*np.unique(y, return_counts=True)):
            counts[int(v)] += int(c)
    total = sum(counts.values()) or 1

    stats = {
        "train_dir": str(train_dir),
        "n_files": len(paths),
        "sampled": min(limit, len(paths)),
        "full_shape": FULL_SHAPE,
        "n_pixels": N_PIXELS,
        "class_fractions": {k: counts[k] / total for k in sorted(counts)},
    }
    print(f"dir              : {stats['train_dir']}", file=out)
    print(f"files            : {stats['n_files']}  (sampled {stats['sampled']})", file=out)
    print(f"tensor           : {FULL_SHAPE} float64 — 6 inputs + label channel {LABEL_CHANNEL}", file=out)
    print(f"pixels per sample: {N_PIXELS} (50 x 181)", file=out)
    print("class balance (shifted 0..4):", file=out)
    for k in sorted(counts):
        print(
            f"    class {k} (raw {k - LABEL_SHIFT:+d}) : {counts[k]:9d}  "
            f"({100.0 * counts[k] / total:6.3f}%)",
            file=out,
        )
    print(
        "NOTE  class 0 is ~97.7% of pixels — pixel accuracy is meaningless here; "
        "score with metric.radar_metric().",
        file=out,
    )
    return stats


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    describe(sys.argv[1] if len(sys.argv) > 1 else None)
