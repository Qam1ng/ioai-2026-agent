"""Scoring for the robot-delivery task — and a warning about the easy proxy.

**The metric is episode success rate.**  ``SR = successful deliveries / episodes``
(notebook §7.1).  Nothing else is scored.

**Action accuracy is a misleading proxy and must not be optimised alone.**
The task statement says it outright: *"A single wrong action can move the robot
into states that were rare in the demonstrations, so high action accuracy does
not always mean high episode success"* (notebook §1), and repeats it as a
failure-analysis prompt in §9.  The mechanism is compounding distribution
shift, the classic behaviour-cloning failure: the demonstrations only ever show
states on an expert's optimal path, so the moment the model steps off that
path it is predicting on states it never trained on, and the errors feed each
other.  With ~13 actions per episode, a uniform 95%-accurate policy finishes an
episode cleanly only ~51% of the time — and real errors are not uniform, they
cluster exactly where the model is already lost.

The asymmetry is worse than that.  Of the 5,327 training decisions only 400 are
``pickup`` and 400 are ``dropoff`` (7.5% each); a model can score 85% action
accuracy while never learning ``pickup`` at all — and then its success rate is
exactly zero.  So: report both, gate on ``success_rate``, and use
``action_accuracy`` only as a cheap training-loop signal that must never be the
selection criterion.

Everything here is pure numpy; no torch, no GPU.  ``evaluate_policy`` on all
200 validation scenarios is a couple of seconds, so there is no excuse for
selecting on the proxy.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # importable both as ``tasks.task2_robot.metric`` and as a loose module
    from .simulator import (
        ACTION_NAMES,
        MAX_STEPS,
        N_ACTIONS,
        Policy,
        replay_actions,
        run_episode,
    )
except ImportError:  # pragma: no cover - direct-path import fallback
    from simulator import (  # type: ignore[no-redef]
        ACTION_NAMES,
        MAX_STEPS,
        N_ACTIONS,
        Policy,
        replay_actions,
        run_episode,
    )

#: The only thing the leaderboard scores.
PRIMARY = "success_rate"


def evaluate_policy(
    policy: Policy,
    scenarios: Sequence[dict[str, Any]],
    max_steps: int = MAX_STEPS,
    limit: int | None = None,
    keep_episodes: bool = False,
) -> dict[str, Any]:
    """Roll ``policy`` out on every scenario and report the real metric.

    Returns:

    * ``success_rate`` — the score (0.0 on an empty scenario list, not NaN, so
      callers can compare without special-casing);
    * ``n`` — episodes actually run;
    * ``mean_steps`` / ``mean_steps_success`` — average episode length overall
      and over successes only.  The overall mean is dominated by timeouts
      (every failure contributes ``max_steps``), so the success-only figure is
      the one that tells you whether the policy is walking efficiently;
    * ``mean_invalid_pickup_or_dropoff`` — how often it grabs at thin air;
    * ``failure_reasons`` — counts of ``never_picked_up`` vs
      ``carried_but_never_delivered``.  This split is the single most useful
      diagnostic: a policy that never picks up has not learned the rare action,
      while one that picks up and then wanders has a navigation problem;
    * ``episodes`` — full per-episode dicts, only when ``keep_episodes``
      (1,600 test episodes with action lists is a lot to hold by accident).
    """
    subset = list(scenarios)[:limit] if limit is not None else list(scenarios)
    results = [run_episode(s, policy, max_steps=max_steps) for s in subset]
    return _summarise(results, keep_episodes)


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]],
    scenarios: Sequence[dict[str, Any]],
    max_steps: int = MAX_STEPS,
    keep_episodes: bool = False,
) -> dict[str, Any]:
    """Score a ``predictions.jsonl``-shaped payload the way the grader will.

    ``predictions`` are dicts with ``layout_id``, ``episode_seed`` and
    ``actions``; they are matched to scenarios on the ``(layout_id,
    episode_seed)`` pair, because ``layout_id`` alone is *not* unique — the
    shipped files hold 4 episodes per layout (200 valid scenarios over 50
    layouts, 1,600 test scenarios over 400).  Matching on ``layout_id`` alone
    silently scores the same episode four times.

    Missing or extra predictions are reported rather than ignored: a submission
    that covers 1,599 of 1,600 scenarios is a format failure, not a 0.06% loss.
    """
    by_key = {(s["layout_id"], int(s["episode_seed"])): s for s in scenarios}
    seen: set[tuple[str, int]] = set()
    results: list[dict[str, Any]] = []
    unknown: list[tuple[str, int]] = []
    duplicates: list[tuple[str, int]] = []

    for pred in predictions:
        key = (pred.get("layout_id"), int(pred.get("episode_seed", -1)))
        scenario = by_key.get(key)
        if scenario is None:
            unknown.append(key)
            continue
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        results.append(replay_actions(scenario, pred.get("actions") or [], max_steps=max_steps))

    missing = [k for k in by_key if k not in seen]
    out = _summarise(results, keep_episodes)
    # Denominator is the scenario count, not the prediction count: an episode
    # with no prediction is a failed delivery, exactly as the grader sees it.
    out["success_rate"] = (
        float(sum(r["success"] for r in results)) / len(by_key) if by_key else 0.0
    )
    out["n_scenarios"] = len(by_key)
    out["n_missing"] = len(missing)
    out["n_unknown"] = len(unknown)
    out["n_duplicate"] = len(duplicates)
    out["format_ok"] = not (missing or unknown or duplicates)
    return out


def _summarise(results: Sequence[dict[str, Any]], keep_episodes: bool) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "success_rate": 0.0,
            "n": 0,
            "n_success": 0,
            "mean_steps": 0.0,
            "mean_steps_success": 0.0,
            "mean_invalid_pickup_or_dropoff": 0.0,
            "failure_reasons": {},
            "episodes": [],
        }
    successes = [r for r in results if r["success"]]
    reasons = Counter(r.get("failure_reason", "unknown") for r in results if not r["success"])
    return {
        "success_rate": float(len(successes)) / n,
        "n": n,
        "n_success": len(successes),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_steps_success": (
            float(np.mean([r["steps"] for r in successes])) if successes else 0.0
        ),
        "mean_invalid_pickup_or_dropoff": float(
            np.mean([r["invalid_pickup_or_dropoff"] for r in results])
        ),
        "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "episodes": list(results) if keep_episodes else [],
    }


def action_accuracy(
    predict: Callable[[dict[str, Any]], int] | np.ndarray,
    trajectories: Sequence[dict[str, Any]] | None = None,
    expert_actions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Teacher-forced agreement with the expert — the MISLEADING proxy.

    See the module docstring.  This measures agreement on *the expert's own
    states*; the deployed policy visits its own states, and the gap between the
    two is where episodes are lost.  Use it to tell whether training is
    progressing at all; never to choose between candidates.

    Two calling conventions, because both are convenient at different points:

    * ``action_accuracy(predict_fn, trajectories)`` — replays the stored
      observations through a callable;
    * ``action_accuracy(pred_array, expert_actions=y)`` — compares two already
      computed label arrays (what a training loop has to hand).

    Reports overall accuracy plus **per-action recall**, which is the part that
    matters: ``pickup``/``dropoff`` are 7.5% of the data each, so overall
    accuracy hides a model that never emits them, and a model that never emits
    them scores exactly 0.0 success.
    """
    if expert_actions is not None:
        y_true = np.asarray(expert_actions).astype(int).ravel()
        y_pred = np.asarray(predict).astype(int).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: predictions {y_pred.shape} vs expert {y_true.shape}"
            )
    else:
        if trajectories is None:
            raise ValueError("pass either trajectories or expert_actions")
        if not callable(predict):
            raise ValueError("predict must be callable when replaying trajectories")
        pairs = [
            (obs, int(a))
            for traj in trajectories
            for obs, a in zip(traj["observations"], traj["actions"])
        ]
        y_true = np.array([a for _o, a in pairs], dtype=int)
        y_pred = np.array([int(predict(o)) for o, _a in pairs], dtype=int)

    if y_true.size == 0:
        return {"accuracy": 0.0, "n": 0, "per_action": {}}

    correct = y_true == y_pred
    per_action: dict[str, dict[str, float | int]] = {}
    for a in range(N_ACTIONS):
        sel = y_true == a
        support = int(sel.sum())
        per_action[ACTION_NAMES[a]] = {
            "support": support,
            "recall": float(correct[sel].mean()) if support else 0.0,
            "predicted": int((y_pred == a).sum()),
        }
    return {
        "accuracy": float(correct.mean()),
        "n": int(y_true.size),
        "per_action": per_action,
        "note": "proxy only — select on evaluate_policy()['success_rate']",
    }


def expected_episode_success(accuracy: float, mean_episode_len: float = 13.32) -> float:
    """Rough translation of action accuracy into episode success.

    ``accuracy ** mean_len``, under the (charitable) assumption that mistakes
    are independent and unrecoverable.  With the shipped demonstrations
    averaging 13.32 steps, 0.95 accuracy maps to ~0.51 success and 0.99 to
    ~0.87.  Real behaviour is worse, because errors are correlated: the states
    the model gets wrong are the ones its own mistakes led it into.  Use this
    only to argue *against* trusting a high accuracy number, never to predict a
    score.
    """
    acc = min(max(float(accuracy), 0.0), 1.0)
    return float(acc**mean_episode_len)


def format_report(metrics: dict[str, Any]) -> str:
    """One-screen human summary for the blackboard / technical report."""
    lines = [
        f"success_rate            : {metrics.get('success_rate', 0.0):.4f} "
        f"({metrics.get('n_success', 0)}/{metrics.get('n', 0)})",
        f"mean_steps (all)        : {metrics.get('mean_steps', 0.0):.2f}",
        f"mean_steps (success)    : {metrics.get('mean_steps_success', 0.0):.2f}",
        f"invalid pickup/dropoff  : {metrics.get('mean_invalid_pickup_or_dropoff', 0.0):.3f}/ep",
    ]
    reasons = metrics.get("failure_reasons") or {}
    if reasons:
        lines.append("failures                : " + ", ".join(f"{k}={v}" for k, v in reasons.items()))
    for key in ("n_missing", "n_unknown", "n_duplicate"):
        if metrics.get(key):
            lines.append(f"FORMAT PROBLEM {key}   : {metrics[key]}")
    return "\n".join(lines)
