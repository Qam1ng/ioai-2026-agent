"""Loaders for the robot-delivery pickles, returning plain numpy.

Three files, all pickle, all confirmed by loading them (see ``CARD.md`` for the
exact structures):

* ``train_demos.pkl``     — dict with 400 expert trajectories (5,327 decisions)
* ``valid_scenarios.pkl`` — list of 200 scenarios, no labels
* ``test_scenarios.pkl``  — list of 1,600 scenarios, no labels

Two deliberate choices:

* **numpy out, not torch.**  Nothing here needs a GPU, and the swarm has to be
  able to inspect and validate this data on a box with no torch installed.  The
  model code converts; the data layer does not.
* **Paths are discovered, never hardcoded.**  On Kaggle the data mounts under
  ``/kaggle/input/competitions/<slug>/`` — one level deeper than people write
  by hand (``memory/lessons/kaggle-kernel-input-paths.md``).  ``find_data_dir``
  walks for the filenames instead of trusting a prefix, so the same kernel runs
  under the dry-run sandbox and on Kaggle unchanged.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

TRAIN_DEMOS = "train_demos.pkl"
VALID_SCENARIOS = "valid_scenarios.pkl"
TEST_SCENARIOS = "test_scenarios.pkl"

#: Confirmed by loading the shipped files, not by reading the notebook.
GRID_SHAPE = (6, 8, 8)
VECTOR_DIM = 13
FLAT_DIM = 6 * 8 * 8 + VECTOR_DIM  # 397
N_ACTIONS = 6

#: Every field a scenario dict carries.
SCENARIO_KEYS = (
    "layout_id",
    "layout_seed",
    "episode_seed",
    "walls",
    "depots",
    "agent_pos",
    "package_location",
    "destination",
)

#: Where competition data can appear.  Checked in order.
SEARCH_ROOTS = ("/kaggle/input", "data", ".")


def find_data_dir(roots: Iterable[str | Path] = SEARCH_ROOTS) -> Path:
    """Locate the directory holding the three pickles.

    Walks each root looking for a directory that contains ``train_demos.pkl``;
    the ``KAGGLE_INPUT`` / ``KAGGLE_COMPETITION_DIR`` env vars (set by our
    dry-run sandbox) are searched first when present.  Raises with the roots it
    tried, because a silent wrong path is the failure that costs a submission.
    """
    candidates: list[Path] = []
    for var in ("KAGGLE_COMPETITION_DIR", "KAGGLE_INPUT"):
        val = os.environ.get(var)
        if val:
            candidates.append(Path(val))
    candidates += [Path(r) for r in roots]

    for root in candidates:
        if not root.exists():
            continue
        if (root / TRAIN_DEMOS).exists():
            return root
        for dirpath, _dirnames, filenames in os.walk(root):
            if TRAIN_DEMOS in filenames:
                return Path(dirpath)
    raise FileNotFoundError(
        f"{TRAIN_DEMOS} not found under any of: {[str(c) for c in candidates]}"
    )


def _load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_train_demos(path: str | Path) -> dict[str, Any]:
    """Load ``train_demos.pkl``.

    Returns the dict as shipped: ``trajectories`` (list of 400),
    ``total_trajectories`` (400) and ``environment_config`` (grid_size 8,
    n_depots 6, n_walls 8, max_steps 120, action_names).  ``environment_config``
    is the authoritative source for ``max_steps`` — read it rather than assuming
    120, since a rules change would land there first.
    """
    data = _load_pickle(path)
    if not isinstance(data, dict) or "trajectories" not in data:
        raise ValueError(f"{path} is not a train_demos payload (keys: {list(data)[:8]})")
    return data


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load ``valid_scenarios.pkl`` / ``test_scenarios.pkl`` (a plain list)."""
    data = _load_pickle(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a scenario list (got {type(data).__name__})")
    return data


def load_all(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load all three files at once.  ``data_dir=None`` triggers discovery."""
    d = Path(data_dir) if data_dir is not None else find_data_dir()
    return {
        "data_dir": d,
        "train": load_train_demos(d / TRAIN_DEMOS),
        "valid": load_scenarios(d / VALID_SCENARIOS),
        "test": load_scenarios(d / TEST_SCENARIOS),
    }


# ---------------------------------------------------------------- conversion


def to_arrays(trajectories: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Flatten trajectories into aligned per-decision arrays.

    Returns, with ``N = 5327`` for the shipped demonstrations:

    * ``grids``   ``(N, 6, 8, 8) float32`` — spatial planes, kept in 2-D so a
      conv model does not have to un-flatten what the baseline threw away;
    * ``vectors`` ``(N, 13) float32``
    * ``masks``   ``(N, 6) bool`` — the legal-action mask at each state;
    * ``states``  ``(N, 4) int64`` — ``(row, col, package_field, destination)``;
    * ``actions`` ``(N,) int64`` — the expert label;
    * ``traj_id`` ``(N,) int64`` — index into ``trajectories``;
    * ``layout``  ``(N,) <U16`` — the layout id.

    ``traj_id`` and ``layout`` exist so validation can be split **by layout**.
    The 400 demonstrations cover only 100 distinct layouts (4 episodes each), so
    a random per-decision split leaks the same map into train and validation and
    reports an optimistic number — the exact trap recorded in
    ``memory/lessons/local-cv-optimism.md``.
    """
    grids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    states: list[tuple] = []
    actions: list[int] = []
    traj_id: list[int] = []
    layouts: list[str] = []

    for t_i, traj in enumerate(trajectories):
        obs_list = traj["observations"]
        act_list = traj["actions"]
        if len(obs_list) != len(act_list):
            raise ValueError(
                f"trajectory {t_i} ({traj.get('layout_id')}) has {len(obs_list)} "
                f"observations but {len(act_list)} actions"
            )
        for obs, act in zip(obs_list, act_list):
            grids.append(np.asarray(obs["grid"], dtype=np.float32))
            vectors.append(np.asarray(obs["vector"], dtype=np.float32))
            masks.append(np.asarray(obs["action_mask"], dtype=bool))
            states.append(tuple(obs["state"]))
            actions.append(int(act))
            traj_id.append(t_i)
            layouts.append(str(traj["layout_id"]))

    if not actions:
        return {
            "grids": np.zeros((0, *GRID_SHAPE), dtype=np.float32),
            "vectors": np.zeros((0, VECTOR_DIM), dtype=np.float32),
            "masks": np.zeros((0, N_ACTIONS), dtype=bool),
            "states": np.zeros((0, 4), dtype=np.int64),
            "actions": np.zeros((0,), dtype=np.int64),
            "traj_id": np.zeros((0,), dtype=np.int64),
            "layout": np.array([], dtype="<U16"),
        }
    return {
        "grids": np.stack(grids),
        "vectors": np.stack(vectors),
        "masks": np.stack(masks),
        "states": np.array(states, dtype=np.int64),
        "actions": np.array(actions, dtype=np.int64),
        "traj_id": np.array(traj_id, dtype=np.int64),
        "layout": np.array(layouts),
    }


def to_supervised(trajectories: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """The baseline's view: ``(X (N, 397) float32, y (N,) int64)``.

    ``X`` is ``concat(grid.reshape(-1), vector)``, matching the notebook's
    ``flatten_observation``.  Provided for parity with the published baseline
    and for quick sklearn probes — but note that flattening is precisely the
    limitation the organizers call out (§8, §9): the 8x8 geometry is destroyed,
    and a model that keeps it should beat this representation.
    """
    arr = to_arrays(trajectories)
    n = arr["actions"].shape[0]
    X = np.concatenate([arr["grids"].reshape(n, -1), arr["vectors"]], axis=1).astype(np.float32)
    return X, arr["actions"]


def layout_groups(trajectories: Sequence[dict[str, Any]]) -> np.ndarray:
    """Per-decision layout ids, for ``GroupKFold``-style grouped validation."""
    return to_arrays(trajectories)["layout"]


# ------------------------------------------------------------------ describe


def describe(data_dir: str | Path | None = None, stream: Any = None) -> dict[str, Any]:
    """Print (and return) dataset statistics.

    Prints the numbers that actually change a modelling decision: how few
    decisions there are, how few distinct layouts back them, and how rare
    ``pickup``/``dropoff`` are.
    """
    import sys
    from collections import Counter

    out = stream or sys.stdout
    bundle = load_all(data_dir)
    train = bundle["train"]
    trajs = train["trajectories"]
    arr = to_arrays(trajs)

    steps = np.array([t["num_steps"] for t in trajs])
    action_counts = Counter(int(a) for a in arr["actions"])
    n = int(arr["actions"].shape[0])
    stats: dict[str, Any] = {
        "data_dir": str(bundle["data_dir"]),
        "environment_config": train.get("environment_config", {}),
        "n_trajectories": len(trajs),
        "n_decisions": n,
        "n_train_layouts": len(set(t["layout_id"] for t in trajs)),
        "episodes_per_layout": len(trajs) / max(1, len(set(t["layout_id"] for t in trajs))),
        "expert_success_rate": float(np.mean([bool(t["success"]) for t in trajs])),
        "steps_mean": float(steps.mean()),
        "steps_min": int(steps.min()),
        "steps_max": int(steps.max()),
        "action_counts": {k: action_counts.get(k, 0) for k in range(N_ACTIONS)},
        "feature_dim": FLAT_DIM,
        "n_valid": len(bundle["valid"]),
        "n_test": len(bundle["test"]),
        "n_valid_layouts": len({s["layout_id"] for s in bundle["valid"]}),
        "n_test_layouts": len({s["layout_id"] for s in bundle["test"]}),
    }

    names = train.get("environment_config", {}).get("action_names", {})
    print(f"data dir                : {stats['data_dir']}", file=out)
    print(f"environment_config      : {stats['environment_config']}", file=out)
    print(
        f"train demos             : {stats['n_trajectories']} trajectories over "
        f"{stats['n_train_layouts']} layouts "
        f"({stats['episodes_per_layout']:.1f} episodes/layout)",
        file=out,
    )
    print(
        f"train decisions         : {stats['n_decisions']} "
        f"(mean {stats['steps_mean']:.2f}, min {stats['steps_min']}, max {stats['steps_max']} per episode)",
        file=out,
    )
    print(f"expert success rate     : {stats['expert_success_rate']:.3f}", file=out)
    print("action balance          :", file=out)
    for a in range(N_ACTIONS):
        c = stats["action_counts"][a]
        label = names.get(str(a), names.get(a, str(a)))
        print(f"    {a} {label:<8} {c:6d}  ({100.0 * c / max(1, n):5.2f}%)", file=out)
    print(
        f"valid / test            : {stats['n_valid']} scenarios "
        f"({stats['n_valid_layouts']} layouts) / {stats['n_test']} scenarios "
        f"({stats['n_test_layouts']} layouts)",
        file=out,
    )
    print(
        f"flat feature dim        : {stats['feature_dim']} "
        f"(= 6*8*8 grid + {VECTOR_DIM} vector)",
        file=out,
    )
    print(
        "NOTE  layouts repeat 4x within each split — group validation by "
        "layout_id or local scores will be optimistic.",
        file=out,
    )
    return stats


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    describe(sys.argv[1] if len(sys.argv) > 1 else None)
