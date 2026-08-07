"""Tests for the Kaggle-mirror checks and the two new task adapters.

Constraints this file honours, because they are the point:

* **No network.**  Nothing here downloads anything; every fixture is built in
  a tmpdir or in memory.
* **No torch.**  The control box has none.  Anything that genuinely needs it
  goes through ``pytest.importorskip`` so it skips rather than fails.
* **No competition data on disk.**  Tests that would need the real pickles or
  the 760 MB radar archive construct equivalent fixtures instead, so the suite
  is green on a fresh checkout.
"""

from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.mirror import dryrun, packages  # noqa: E402
from tasks.task2_robot import data as robot_data  # noqa: E402
from tasks.task2_robot import metric as robot_metric  # noqa: E402
from tasks.task2_robot import simulator as robot_sim  # noqa: E402
from tasks.task3_radar import data as radar_data  # noqa: E402
from tasks.task3_radar import metric as radar_metric  # noqa: E402


# =============================================================== packages ===


def test_pinned_env_parses_and_contains_the_ml_stack():
    env = packages.parse_env()
    assert len(env) > 100, "pinned env looks truncated"
    for pkg in ("torch", "numpy", "pandas", "scikit-learn", "transformers"):
        assert pkg in env, f"{pkg} missing from the pinned env"
    assert env["torch"], "torch should carry a pinned version"


def test_parse_env_handles_comments_extras_markers_and_vcs(tmp_path):
    f = tmp_path / "env.txt"
    f.write_text(
        textwrap.dedent(
            """\
            # a comment
            numpy==2.2.6
            Some_Pkg[extra]==1.0 ; python_version < "3.13"

            git+https://github.com/openai/CLIP.git
            """
        )
    )
    env = packages.parse_env(f)
    assert env["numpy"] == "2.2.6"
    assert env["some-pkg"] == "1.0"  # canonicalised name, extras stripped
    assert "clip" in env


def test_scan_imports_takes_top_level_only_and_skips_relative():
    mods = packages.scan_imports(
        "import torch.nn.functional as F\n"
        "from sklearn.linear_model import Ridge\n"
        "from . import sibling\n"
        "import os, json\n"
    )
    assert mods == {"torch", "sklearn", "os", "json"}


def test_check_imports_passes_allowed_and_catches_disallowed():
    env = packages.parse_env()

    allowed = "import numpy as np\nimport torch\nfrom sklearn.svm import SVC\nimport os\n"
    assert packages.check_imports(allowed, env) == []

    bad = "import numpy as np\nimport definitely_not_a_kaggle_package\n"
    problems = packages.check_imports(bad, env)
    assert len(problems) == 1
    assert "definitely_not_a_kaggle_package" in problems[0]


def test_check_imports_resolves_import_name_aliases():
    """sklearn/cv2/PIL/yaml must not be reported just because they are spelled
    differently in the requirements file."""
    env = packages.parse_env()
    code = "import sklearn\nimport cv2\nimport PIL\nimport yaml\nimport skimage\n"
    assert packages.check_imports(code, env) == []


def test_check_imports_reports_a_syntax_error_instead_of_crashing():
    problems = packages.check_imports("def broken(:\n", {})
    assert problems and "does not parse" in problems[0]


def test_check_no_network_catches_requests_get():
    problems = packages.check_no_network(
        "import requests\nr = requests.get('http://example.com')\n"
    )
    joined = " ".join(problems)
    assert "requests" in joined
    assert any("import" in p for p in problems)
    assert any("downloads over the network" in p or "URL" in p for p in problems)


def test_check_no_network_catches_bare_from_pretrained():
    problems = packages.check_no_network(
        "from transformers import AutoModel\n"
        "m = AutoModel.from_pretrained('bert-base')\n"
    )
    assert any("bert-base" in p and "hub id" in p for p in problems)


def test_check_no_network_allows_from_pretrained_with_a_local_path():
    clean = packages.check_no_network(
        "from transformers import AutoModel\n"
        "m = AutoModel.from_pretrained('/kaggle/input/competitions/x/model')\n"
    )
    assert clean == []


def test_check_no_network_catches_pip_install():
    problems = packages.check_no_network("import os\nos.system('pip install timm')\n")
    assert any("pip install" in p for p in problems)


def test_check_no_network_catches_wget_and_kagglehub():
    assert any(
        "wget" in p
        for p in packages.check_no_network("import subprocess\nsubprocess.run('wget http://x/y')\n")
    )
    assert any(
        "kagglehub" in p
        for p in packages.check_no_network(
            "import kagglehub\np = kagglehub.model_download('org/m')\n"
        )
    )


def test_check_no_network_is_quiet_on_a_clean_offline_kernel():
    code = textwrap.dedent(
        """\
        import os
        import numpy as np

        def find_input():
            for root, _d, files in os.walk('/kaggle/input'):
                if 'train.csv' in files:
                    return root
            raise RuntimeError('not found')

        np.save('/kaggle/working/out.npy', np.zeros(3))
        """
    )
    assert packages.check_no_network(code) == []


# ================================================================= dryrun ===


KERNEL_OK = textwrap.dedent(
    """\
    import csv, os

    def find_input():
        for root, _d, files in os.walk('/kaggle/input'):
            if 'sample_submission.csv' in files:
                return root
        raise RuntimeError('input not found under /kaggle/input')

    INP = find_input()
    with open(os.path.join(INP, 'sample_submission.csv')) as fh:
        rows = list(csv.DictReader(fh))

    with open('/kaggle/working/submission.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'target'])
        for r in rows:
            w.writerow([r['id'], 1])
    print('wrote', len(rows), 'rows')
    """
)


@pytest.fixture()
def sample_submission(tmp_path) -> Path:
    p = tmp_path / "sample_submission.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "target"])
        for i in range(5):
            w.writerow([f"row_{i}", 0])
    return p


def test_stage_input_builds_the_nested_kaggle_tree(tmp_path, sample_submission):
    comp = tmp_path / "sandbox" / "kaggle" / "input" / "competitions" / "some-slug"
    out = dryrun.stage_input(comp, {"sample_submission.csv": sample_submission})
    assert out == comp.resolve()
    assert (comp / "sample_submission.csv").exists()
    # the working dir is created next to input/, not inside the competition dir
    assert (tmp_path / "sandbox" / "kaggle" / "working").is_dir()


def test_stage_input_rejects_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        dryrun.stage_input(tmp_path / "c", {"nope.csv": tmp_path / "nope.csv"})


def test_dry_run_executes_a_kaggle_path_kernel_and_finds_the_submission(
    tmp_path, sample_submission
):
    res = dryrun.dry_run(
        KERNEL_OK,
        slug="some-slug",
        input_sources={"sample_submission.csv": sample_submission},
        workdir=tmp_path / "run",
        timeout_s=120,
    )
    assert res.ok, f"{res.summary()}\n{res.stderr_tail}"
    assert res.exit_code == 0
    assert res.submission_rows == 5
    assert Path(res.submission_path).name == "submission.csv"
    assert res.violations == []
    assert "wrote 5 rows" in res.stdout_tail


def test_dry_run_reports_a_crash_without_raising(tmp_path, sample_submission):
    res = dryrun.dry_run(
        "raise SystemError('boom')\n",
        slug="s",
        input_sources={"sample_submission.csv": sample_submission},
        workdir=tmp_path / "run",
        timeout_s=120,
    )
    assert not res.ok
    assert res.exit_code != 0
    assert "boom" in res.stderr_tail


def test_dry_run_records_violations_but_still_runs(tmp_path, sample_submission):
    code = "import requests  # noqa\n" + KERNEL_OK
    res = dryrun.dry_run(
        code,
        slug="s",
        input_sources={"sample_submission.csv": sample_submission},
        workdir=tmp_path / "run",
        timeout_s=120,
    )
    assert res.violations, "the requests import must be reported"
    # It still executed: report, never refuse.
    assert res.submission_rows == 5


def test_dry_run_enforces_the_wall_clock_timeout(tmp_path, sample_submission):
    res = dryrun.dry_run(
        "import time\ntime.sleep(30)\n",
        slug="s",
        input_sources={"sample_submission.csv": sample_submission},
        workdir=tmp_path / "run",
        timeout_s=2,
    )
    assert res.timed_out
    assert not res.ok
    assert res.wall_s < 25


# ------------------------------------------------------- validate_submission


def _write_csv(path: Path, header, rows) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture()
def sample_pair(tmp_path):
    sample = _write_csv(
        tmp_path / "sample.csv",
        ["id", "target"],
        [[f"row_{i}", 0] for i in range(5)],
    )
    return tmp_path, sample


def test_validate_submission_accepts_a_correct_file(sample_pair):
    tmp_path, sample = sample_pair
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "target"], [[f"row_{i}", i % 3] for i in range(5)]
    )
    assert dryrun.validate_submission(sub, sample) == []


def test_validate_submission_catches_wrong_row_count(sample_pair):
    tmp_path, sample = sample_pair
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "target"], [[f"row_{i}", 1] for i in range(4)]
    )
    problems = dryrun.validate_submission(sub, sample)
    assert any("row count differs" in p for p in problems)


def test_validate_submission_catches_wrong_column_set(sample_pair):
    tmp_path, sample = sample_pair
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "prediction"], [[f"row_{i}", 1] for i in range(5)]
    )
    problems = dryrun.validate_submission(sub, sample)
    assert any("column set differs" in p for p in problems)


def test_validate_submission_catches_reordered_ids(sample_pair):
    tmp_path, sample = sample_pair
    order = [0, 2, 1, 3, 4]
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "target"], [[f"row_{i}", 1] for i in order]
    )
    problems = dryrun.validate_submission(sub, sample)
    assert any("WRONG ORDER" in p for p in problems)


def test_validate_submission_catches_nan_cells(sample_pair):
    tmp_path, sample = sample_pair
    rows = [[f"row_{i}", 1] for i in range(5)]
    rows[2][1] = "nan"
    sub = _write_csv(tmp_path / "sub.csv", ["id", "target"], rows)
    problems = dryrun.validate_submission(sub, sample)
    assert any("NaN" in p for p in problems)


def test_validate_submission_honours_an_explicit_value_range(sample_pair):
    tmp_path, sample = sample_pair
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "target"], [[f"row_{i}", 99] for i in range(5)]
    )
    problems = dryrun.validate_submission(sub, sample, value_range=(0, 4))
    assert any("outside the expected label range" in p for p in problems)


def test_validate_submission_flags_floats_where_the_sample_holds_ints(sample_pair):
    tmp_path, sample = sample_pair
    sub = _write_csv(
        tmp_path / "sub.csv", ["id", "target"], [[f"row_{i}", 0.73] for i in range(5)]
    )
    problems = dryrun.validate_submission(sub, sample)
    assert any("non-integer" in p for p in problems)


def test_validate_submission_reports_a_missing_file(tmp_path, sample_pair):
    _, sample = sample_pair
    problems = dryrun.validate_submission(tmp_path / "nope.csv", sample)
    assert problems and "not found" in problems[0]


# ============================================================ task2 robot ===


def _scenario(
    layout_id="unit_0000",
    agent_pos=(0, 0),
    package_location=0,
    destination=1,
    walls=(),
    depots=((2, 2), (5, 5), (0, 7), (7, 0), (3, 6), (6, 3)),
    episode_seed=1,
):
    """A hand-built, wall-free scenario: trivially solvable, easy to reason about."""
    return {
        "layout_id": layout_id,
        "layout_seed": 0,
        "episode_seed": episode_seed,
        "walls": tuple(walls),
        "depots": tuple(depots),
        "agent_pos": tuple(agent_pos),
        "package_location": package_location,
        "destination": destination,
    }


TRIVIAL_SCENARIOS = [
    _scenario(layout_id="unit_0000", agent_pos=(0, 0), package_location=0, destination=1, episode_seed=1),
    _scenario(layout_id="unit_0001", agent_pos=(7, 7), package_location=1, destination=2, episode_seed=2),
    _scenario(layout_id="unit_0002", agent_pos=(4, 1), package_location=3, destination=4, episode_seed=3),
]


def always_west(_obs):
    """Deliberately broken: never picks up, so it can never deliver."""
    return 3


def test_simulator_observation_shapes_match_the_demonstrations():
    sim = robot_sim.DeliverySimulator8x8()
    sim.reset(TRIVIAL_SCENARIOS[0])
    obs = sim.observation()
    assert obs["grid"].shape == (6, 8, 8) and obs["grid"].dtype == np.float32
    assert obs["vector"].shape == (13,) and obs["vector"].dtype == np.float32
    assert obs["action_mask"].shape == (6,) and obs["action_mask"].dtype == bool
    assert len(obs["state"]) == 4
    assert robot_sim.FLAT_DIM == 397


def test_simulator_pickup_and_dropoff_semantics():
    sim = robot_sim.DeliverySimulator8x8()
    sc = _scenario(agent_pos=(2, 2), package_location=0, destination=1)  # start on depot 0
    sim.reset(sc)
    assert sim.valid_action_mask()[robot_sim.PICKUP]

    _s, done, _t, info = sim.step(robot_sim.DROPOFF)  # dropoff while empty -> invalid
    assert info["invalid_pickup_or_dropoff"] and not done

    _s, done, _t, info = sim.step(robot_sim.PICKUP)
    assert sim.carrying and not done and not info["invalid_pickup_or_dropoff"]

    state = sim.state()
    assert state[2] == robot_sim.N_DEPOTS, "package field must read 6 while carrying"


def test_simulator_move_into_a_wall_is_a_no_op_that_still_costs_a_step():
    sim = robot_sim.DeliverySimulator8x8()
    sim.reset(_scenario(agent_pos=(0, 0), walls=((0, 1),)))
    before = sim.agent_pos
    sim.step(2)  # east, into the wall
    assert sim.agent_pos == before
    assert sim.step_count == 1


def test_evaluate_policy_scores_one_on_a_solvable_scenario_with_an_optimal_policy():
    res = robot_metric.evaluate_policy(
        robot_sim.greedy_reference_policy, TRIVIAL_SCENARIOS, keep_episodes=True
    )
    assert res["success_rate"] == 1.0
    assert res["n"] == 3 and res["n_success"] == 3
    assert res["failure_reasons"] == {}
    assert 0 < res["mean_steps"] < robot_sim.MAX_STEPS
    assert all(ep["failure_reason"] == "success" for ep in res["episodes"])


def test_evaluate_policy_scores_zero_for_a_deliberately_broken_policy():
    res = robot_metric.evaluate_policy(always_west, TRIVIAL_SCENARIOS)
    assert res["success_rate"] == 0.0
    assert res["n_success"] == 0
    # It never picks up, which is exactly what the diagnostic should say.
    assert res["failure_reasons"] == {"never_picked_up": 3}
    assert res["mean_steps"] == robot_sim.MAX_STEPS


def test_evaluate_policy_on_no_scenarios_returns_zero_not_nan():
    res = robot_metric.evaluate_policy(always_west, [])
    assert res["success_rate"] == 0.0 and res["n"] == 0


def test_replay_actions_matches_the_policy_rollout():
    """The grader replays actions; our replay must agree with our rollout."""
    for sc in TRIVIAL_SCENARIOS:
        rolled = robot_sim.run_episode(sc, robot_sim.greedy_reference_policy)
        replayed = robot_sim.replay_actions(sc, rolled["actions"])
        assert replayed["success"] == rolled["success"] is True
        assert replayed["steps"] == rolled["steps"]


def test_replay_actions_rejects_an_out_of_range_action_without_raising():
    res = robot_sim.replay_actions(TRIVIAL_SCENARIOS[0], [0, 1, 99])
    assert res["failure_reason"] == "invalid_action"
    assert res["success"] is False


def test_evaluate_predictions_matches_on_layout_and_seed_and_flags_gaps():
    preds = []
    for sc in TRIVIAL_SCENARIOS:
        ep = robot_sim.run_episode(sc, robot_sim.greedy_reference_policy)
        preds.append(
            {
                "layout_id": sc["layout_id"],
                "episode_seed": sc["episode_seed"],
                "actions": ep["actions"],
            }
        )
    full = robot_metric.evaluate_predictions(preds, TRIVIAL_SCENARIOS)
    assert full["success_rate"] == 1.0 and full["format_ok"]

    partial = robot_metric.evaluate_predictions(preds[:2], TRIVIAL_SCENARIOS)
    assert partial["n_missing"] == 1
    assert not partial["format_ok"]
    # A missing prediction is a failed delivery, not an excluded row.
    assert partial["success_rate"] == pytest.approx(2 / 3)


def test_action_accuracy_reports_per_action_recall_from_arrays():
    y = np.array([0, 0, 1, 4, 5])
    p = np.array([0, 0, 1, 0, 0])  # never emits pickup/dropoff
    res = robot_metric.action_accuracy(p, expert_actions=y)
    assert res["accuracy"] == pytest.approx(0.6)
    assert res["per_action"]["pickup"]["recall"] == 0.0
    assert res["per_action"]["dropoff"]["recall"] == 0.0
    assert res["per_action"]["south"]["support"] == 2


def test_action_accuracy_is_a_poor_predictor_of_success():
    """The documented trap, made numeric: 95% per-action still loses half the
    episodes at the shipped mean length."""
    assert robot_metric.expected_episode_success(0.95) < 0.55
    assert robot_metric.expected_episode_success(1.0) == 1.0


def test_to_arrays_shapes_and_grouping(tmp_path):
    """Built from the simulator rather than the real pickle, so this runs on a
    fresh checkout with no competition data."""
    trajs = []
    for i, sc in enumerate(TRIVIAL_SCENARIOS):
        sim = robot_sim.DeliverySimulator8x8()
        sim.reset(sc)
        observations, actions = [], []
        for _ in range(30):
            obs = sim.observation()
            a = robot_sim.greedy_reference_policy(obs)
            observations.append(obs)
            actions.append(a)
            _s, done, timed_out, _i = sim.step(a)
            if done or timed_out:
                break
        trajs.append(
            {
                "layout_id": sc["layout_id"],
                "episode_seed": sc["episode_seed"],
                "scenario": sc,
                "observations": observations,
                "actions": actions,
                "success": True,
                "num_steps": len(actions),
            }
        )

    arr = robot_data.to_arrays(trajs)
    n = arr["actions"].shape[0]
    assert arr["grids"].shape == (n, 6, 8, 8)
    assert arr["vectors"].shape == (n, 13)
    assert arr["masks"].shape == (n, 6) and arr["masks"].dtype == bool
    assert arr["states"].shape == (n, 4)
    assert set(np.unique(arr["layout"])) == {s["layout_id"] for s in TRIVIAL_SCENARIOS}

    X, y = robot_data.to_supervised(trajs)
    assert X.shape == (n, robot_data.FLAT_DIM) == (n, 397)
    assert y.shape == (n,)


def test_to_arrays_rejects_a_desynchronised_trajectory():
    bad = [{"layout_id": "x", "observations": [], "actions": [1]}]
    with pytest.raises(ValueError, match="observations"):
        robot_data.to_arrays(bad)


def test_load_scenarios_rejects_the_wrong_payload_shape(tmp_path):
    import pickle

    p = tmp_path / "scen.pkl"
    p.write_bytes(pickle.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="scenario list"):
        robot_data.load_scenarios(p)


# ============================================================ task3 radar ===


def test_radar_data_imports_without_torch_installed():
    """The module must be usable for inspection on a box with no torch."""
    assert radar_data.FULL_SHAPE == (7, 50, 181)
    assert radar_data.N_PIXELS == 9050
    assert radar_data.LABEL_SHIFT == 1


def test_radar_load_tensor_raises_a_clear_error_when_torch_is_missing(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed; the torch-missing path cannot be exercised")
    with pytest.raises(RuntimeError, match="torch is required"):
        radar_data.load_tensor(tmp_path / "1.mat.pt", backend="torch")


def test_radar_load_tensor_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    arr = torch.rand(7, 50, 181, dtype=torch.float64)
    arr[6] = torch.randint(-1, 4, (50, 181)).double()
    path = tmp_path / "1.mat.pt"
    torch.save(arr, path)

    x, y = radar_data.load_sample(path, backend="torch")
    assert x.shape == (6, 50, 181)
    assert y.shape == (50, 181)
    assert y.min() >= 0 and y.max() <= 4  # shifted into training space


def test_radar_files_sort_numerically_not_lexically(tmp_path):
    for n in (1, 2, 10, 100):
        (tmp_path / f"{n}.mat.pt").write_bytes(b"")
    names = [p.name for p in radar_data.list_tensor_files(tmp_path)]
    assert names == ["1.mat.pt", "2.mat.pt", "10.mat.pt", "100.mat.pt"]


def test_radar_submission_columns_and_roundtrip(tmp_path):
    cols = radar_data.submission_columns()
    assert cols[0] == "filename" and len(cols) == 9051
    assert cols[1] == "pixel_0" and cols[-1] == "pixel_9049"

    rng = np.random.default_rng(0)
    preds = rng.integers(0, 5, size=(3, 50, 181))
    path = radar_data.write_submission(
        tmp_path / "submission.csv", ["1.mat.pt", "2.mat.pt", "3.mat.pt"], preds
    )
    names, raw = radar_data.read_submission(path)
    assert names == ["1.mat.pt", "2.mat.pt", "3.mat.pt"]
    assert raw.shape == (3, 9050)
    assert raw.min() >= -1 and raw.max() <= 3, "submission must be in raw label space"
    assert np.array_equal(raw.reshape(3, 50, 181) + 1, preds)


def test_radar_write_submission_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="filenames"):
        radar_data.write_submission(tmp_path / "s.csv", ["a"], np.zeros((2, 9050), dtype=int))


def test_radar_metric_perfect_and_all_background():
    rng = np.random.default_rng(1)
    y = np.zeros((50, 181), dtype=np.int64)
    y[rng.random(y.shape) < 0.02] = 4
    y[0, :5] = [1, 2, 3, 1, 2]

    perfect = radar_metric.radar_metric(y, y)
    assert perfect["primary"] == pytest.approx(1.0)
    assert perfect["pixel_accuracy"] == pytest.approx(1.0)

    bg = radar_metric.radar_metric(y, np.zeros_like(y))
    # This is the whole reason pixel accuracy is not the metric.
    assert bg["pixel_accuracy"] > 0.95
    assert bg["macro_f1_fg"] == 0.0
    assert bg["primary"] == 0.0


def test_radar_metric_confusion_matrix_and_dropped_pixels():
    y = np.array([[0, 1], [2, 3]])
    p = np.array([[0, 1], [2, 9]])  # 9 is out of range -> dropped, not a crash
    m = radar_metric.radar_metric(y, p)
    assert m["n_pixels"] == 3
    assert m["n_dropped"] == 1
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (5, 5) and cm.sum() == 3


def test_radar_metric_reports_absent_classes_as_none_not_zero():
    y = np.zeros((4, 4), dtype=np.int64)
    m = radar_metric.radar_metric(y, y)
    assert m["per_class_f1"][0] == pytest.approx(1.0)
    assert m["per_class_f1"][4] is None  # absent from truth and prediction
    assert m["support"][4] == 0


def test_radar_calibration_table_separates_the_candidate_aggregations():
    rng = np.random.default_rng(2)
    y = np.zeros((50, 181), dtype=np.int64)
    y[rng.random(y.shape) < 0.02] = 4
    y[0, :6] = [1, 1, 2, 2, 3, 3]  # all 5 classes present, as in the real data
    scores = radar_metric.constant_prediction_scores(y)
    bg = scores[0]
    # An all-background submission must score very differently under each
    # candidate — that separation is what makes one calibration submission
    # enough to identify the real metric.
    assert bg["pixel_accuracy"] > 0.9
    assert bg["macro_f1"] < 0.3
    assert bg["macro_f1_fg"] == 0.0
    table = radar_metric.calibration_table(y)
    assert "pixel_accuracy" in table and "all 0" in table


def test_radar_primary_is_a_known_aggregation():
    assert radar_metric.PRIMARY in radar_metric.AGGREGATIONS
