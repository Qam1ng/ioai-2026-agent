"""The 8x8 delivery environment, lifted out of the organizers' notebook.

Source: ``IOAI-2026/Home Task/Home-Task-2.ipynb`` (cells 5-6, 22).  The class
below is behaviourally identical to ``DeliverySimulator8x8`` there — same state
tuple, same observation tensor, same action semantics, same step limit.  That
fidelity is the whole point: our local success rate is only worth anything if
it is computed by the same dynamics the graders replay our actions through.

What changed, and why:

* Notebook-only globals are now module constants (``GRID_SIZE``, ``MAX_STEPS``,
  ``ACTION_DELTAS``, …), so this file imports cleanly with no notebook around
  it.  Their values are unchanged.
* ``max_steps`` became an instance attribute defaulting to ``MAX_STEPS``.  The
  notebook's ``step()`` closed over the global; that made shortened debug
  episodes impossible without editing the class.  Default behaviour is identical.
* The PIL rendering / GIF cells are dropped — they need a display and pull in
  imports the Kaggle kernel does not need.  ASCII ``render()`` is kept because
  it is the cheapest way to read a failed episode.
* ``replay_actions`` is new.  The official evaluator does not run our policy;
  it **replays a recorded action list** from the scenario's initial state
  (notebook §6.3).  Scoring a submission therefore has to go through replay,
  not through the policy, or we would be measuring something the grader isn't.
  It also tolerates out-of-range actions instead of raising, because a
  submission file is untrusted input.

There is no torch here.  The environment is pure numpy, which means the whole
evaluation loop runs on a laptop while the model training does not have to.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

GRID_SIZE = 8
N_DEPOTS = 6
MAX_STEPS = 120
N_ACTIONS = 6

ACTION_NAMES: dict[int, str] = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}

#: Row/column deltas for the four movement actions.  Note ``south`` is +row:
#: row 0 is the top of the rendered grid.
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (1, 0),
    1: (-1, 0),
    2: (0, 1),
    3: (0, -1),
}

PICKUP = 4
DROPOFF = 5

DEPOT_NAMES = ["A", "B", "C", "D", "E", "F"]

#: Observation dimensions, confirmed against the shipped ``train_demos.pkl``.
GRID_SHAPE = (6, GRID_SIZE, GRID_SIZE)  # walls, depots, agent, package, dest, carrying
VECTOR_DIM = 13
FLAT_DIM = GRID_SHAPE[0] * GRID_SIZE * GRID_SIZE + VECTOR_DIM  # 384 + 13 = 397

#: A policy takes one observation dict and returns an action id 0..5.
Policy = Callable[[dict[str, Any]], int]


class DeliverySimulator8x8:
    """Run one 8x8 delivery episode.

    Episode: the robot walks to the depot holding the package, picks it up,
    walks to the destination depot, drops it off.  Success is a completed
    dropoff within ``max_steps``.
    """

    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.carrying = False
        self.walls: set[tuple[int, int]] = set()
        self.depots: list[tuple[int, int]] = []
        self.agent_pos: tuple[int, int] = (0, 0)
        self.package_location = 0
        self.destination = 0

    # ----------------------------------------------------------- lifecycle
    def reset(self, scenario: dict[str, Any]) -> tuple[int, int, int, int]:
        """Start a scenario and return the compact state."""
        self.step_count = 0
        self.carrying = False
        self.walls = {tuple(cell) for cell in scenario["walls"]}
        self.depots = [tuple(cell) for cell in scenario["depots"]]
        self.agent_pos = tuple(scenario["agent_pos"])
        self.package_location = int(scenario["package_location"])
        self.destination = int(scenario["destination"])
        return self.state()

    def state(self) -> tuple[int, int, int, int]:
        """Return row, column, package field, and destination.

        ``package field`` is the depot index holding the package, or
        ``N_DEPOTS`` (6) while the robot is carrying it — the same trick the
        classic Taxi environment uses to fold "in transit" into one integer.
        """
        package_field = N_DEPOTS if self.carrying else self.package_location
        return (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(package_field),
            int(self.destination),
        )

    # ------------------------------------------------------------- queries
    def can_enter(self, row: int, col: int) -> bool:
        """Check whether the robot can occupy a cell."""
        return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and (row, col) not in self.walls

    def valid_action_mask(self) -> np.ndarray:
        """Return the currently valid actions as a length-6 boolean array.

        Movement into a wall or off the grid is *masked out here* but is still
        legal to attempt: ``step`` treats it as a no-op that burns a step.  Only
        pickup/dropoff are penalised as "invalid" in the diagnostics.
        """
        row, col, _, destination = self.state()
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for action, (dr, dc) in ACTION_DELTAS.items():
            mask[action] = self.can_enter(row + dr, col + dc)
        mask[PICKUP] = (not self.carrying) and self.agent_pos == self.depots[self.package_location]
        mask[DROPOFF] = self.carrying and self.agent_pos == self.depots[destination]
        return mask

    def observation(self) -> dict[str, Any]:
        """Build the model observation for the current state.

        Keys and shapes match the demonstrations exactly:
        ``grid (6,8,8) float32``, ``vector (13,) float32``,
        ``action_mask (6,) bool``, ``state`` 4-tuple.
        """
        row, col, package_field, destination = self.state()
        carrying = package_field == N_DEPOTS
        dest_row, dest_col = self.depots[destination]
        target_row, target_col = (
            (dest_row, dest_col) if carrying else self.depots[package_field]
        )

        grid = np.zeros(GRID_SHAPE, dtype=np.float32)
        for wr, wc in self.walls:
            grid[0, wr, wc] = 1.0
        for dr, dc in self.depots:
            grid[1, dr, dc] = 1.0
        grid[2, row, col] = 1.0
        if not carrying:
            pr, pc = self.depots[package_field]
            grid[3, pr, pc] = 1.0
        grid[4, dest_row, dest_col] = 1.0
        grid[5, :, :] = float(carrying)

        blocked_moves = [
            float(not self.can_enter(row + dr, col + dc)) for dr, dc in ACTION_DELTAS.values()
        ]
        vector = np.array(
            [
                row / (GRID_SIZE - 1),
                col / (GRID_SIZE - 1),
                package_field / N_DEPOTS,
                destination / (N_DEPOTS - 1),
                float(carrying),
                target_row / (GRID_SIZE - 1),
                target_col / (GRID_SIZE - 1),
                (target_row - row) / (GRID_SIZE - 1),
                (target_col - col) / (GRID_SIZE - 1),
                *blocked_moves,
            ],
            dtype=np.float32,
        )
        return {
            "grid": grid,
            "vector": vector,
            "action_mask": self.valid_action_mask(),
            "state": self.state(),
        }

    # ---------------------------------------------------------------- step
    def step(self, action: int) -> tuple[tuple[int, int, int, int], bool, bool, dict[str, Any]]:
        """Apply one action and report ``(state, done, timed_out, info)``.

        ``done`` is a successful dropoff.  ``timed_out`` is the step budget
        being exhausted without one.  A move into a wall silently costs a step;
        a pickup/dropoff in the wrong place costs a step and sets
        ``info['invalid_pickup_or_dropoff']``.
        """
        action = int(action)
        done = False
        info = {"invalid_pickup_or_dropoff": False}

        if action in ACTION_DELTAS:
            dr, dc = ACTION_DELTAS[action]
            row, col = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if self.can_enter(row, col):
                self.agent_pos = (row, col)
        elif (
            action == PICKUP
            and (not self.carrying)
            and self.agent_pos == self.depots[self.package_location]
        ):
            self.carrying = True
        elif action == DROPOFF and self.carrying and self.agent_pos == self.depots[self.destination]:
            done = True
            self.carrying = False
            self.package_location = self.destination
        elif action in (PICKUP, DROPOFF):
            info["invalid_pickup_or_dropoff"] = True
        else:
            raise ValueError(f"unknown action: {action}")

        self.step_count += 1
        return self.state(), done, self.step_count >= self.max_steps and not done, info

    # -------------------------------------------------------------- render
    def render(self) -> str:
        """Return an ASCII rendering of the current grid.

        ``#`` wall, ``A``-``F`` depots, ``T`` robot, ``T*`` robot carrying.
        """
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for row, col in self.walls:
            grid[row][col] = "#"
        for i, (row, col) in enumerate(self.depots):
            grid[row][col] = DEPOT_NAMES[i]

        agent_row, agent_col = self.agent_pos
        grid[agent_row][agent_col] = "T*" if self.carrying else "T"
        rows = [" ".join(f"{cell:>2}" for cell in row) for row in grid]
        package_name = "in taxi" if self.carrying else DEPOT_NAMES[self.package_location]
        rows.append(f"package={package_name}, destination={DEPOT_NAMES[self.destination]}")
        return "\n".join(rows)


# ------------------------------------------------------------------ episodes


def run_episode(
    scenario: dict[str, Any],
    action_fn: Policy,
    max_steps: int = MAX_STEPS,
    render: bool = False,
) -> dict[str, Any]:
    """Roll one policy out on one scenario.

    Returns ``success``, ``steps``, ``invalid_pickup_or_dropoff``, ``actions``,
    ``frames`` (ASCII, only when ``render``), plus ``picked_up`` and
    ``failure_reason`` — the two fields that turn a bare 0/1 into a debuggable
    result.  Almost every early failure is "never reached the package"; almost
    every late one is "carried it and got lost", and they call for different
    fixes.
    """
    sim = DeliverySimulator8x8(max_steps=max_steps)
    sim.reset(scenario)
    frames: list[str] = []
    actions: list[int] = []
    invalid = 0
    done = False
    picked_up = False

    if render:
        frames.append(sim.render())

    for _ in range(max_steps):
        action = int(action_fn(sim.observation()))
        _, done, timed_out, info = sim.step(action)
        actions.append(action)
        invalid += int(info["invalid_pickup_or_dropoff"])
        picked_up = picked_up or sim.carrying
        if render:
            frames.append(sim.render())
        if done or timed_out:
            break

    return {
        "success": done,
        "steps": len(actions),
        "invalid_pickup_or_dropoff": invalid,
        "actions": actions,
        "frames": frames,
        "picked_up": picked_up,
        "failure_reason": _failure_reason(done, picked_up, invalid),
    }


def replay_actions(
    scenario: dict[str, Any],
    actions: Sequence[int],
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Replay a recorded action list — this is what the grader does.

    Notebook §6.3: the evaluator reads the predicted actions, resets to the
    scenario state, and steps through them.  Scoring a ``predictions.jsonl``
    locally must go through here rather than through the policy, otherwise we
    would be validating our rollout code instead of our submission file.

    Out-of-range actions are reported as ``failure_reason='invalid_action'``
    rather than raising, because a submission file is untrusted input; the
    grader's behaviour on a malformed action is unspecified, so producing one
    is a bug we want to see locally, not an exception in the middle of scoring
    1,600 episodes.
    """
    sim = DeliverySimulator8x8(max_steps=max_steps)
    sim.reset(scenario)
    invalid = 0
    done = False
    picked_up = False
    used = 0
    bad_action = False

    for raw in list(actions)[:max_steps]:
        try:
            action = int(raw)
        except (TypeError, ValueError):
            bad_action = True
            break
        if action not in ACTION_NAMES:
            bad_action = True
            break
        _, done, timed_out, info = sim.step(action)
        used += 1
        invalid += int(info["invalid_pickup_or_dropoff"])
        picked_up = picked_up or sim.carrying
        if done or timed_out:
            break

    reason = "invalid_action" if bad_action else _failure_reason(done, picked_up, invalid)
    if not done and not bad_action and len(actions) < max_steps and used == len(actions):
        # Ran out of submitted actions before the step limit: the policy stopped
        # early, which is a different bug from wandering until timeout.
        reason = "actions_exhausted" if reason != "success" else reason
    return {
        "success": done,
        "steps": used,
        "invalid_pickup_or_dropoff": invalid,
        "actions": list(actions)[:used],
        "picked_up": picked_up,
        "failure_reason": reason,
    }


def _failure_reason(done: bool, picked_up: bool, invalid: int) -> str:
    if done:
        return "success"
    if not picked_up:
        return "never_picked_up"
    return "carried_but_never_delivered"


def greedy_reference_policy(obs: dict[str, Any]) -> int:
    """A hand-written policy, provided ONLY as a test fixture and sanity floor.

    It is deliberately *not* a solution: notebook §6.4 forbids rule-based or
    hard-coded submissions and forbids generating extra expert trajectories
    with search or planning.  Its legitimate uses are (a) proving the simulator
    and the metric agree on an obviously-solvable episode, and (b) giving an
    upper reference on wall-free layouts.  It has no wall avoidance beyond the
    action mask, so it deadlocks on real layouts — which is exactly why the
    task needs a learned model.
    """
    mask = obs["action_mask"]
    if mask[PICKUP]:
        return PICKUP
    if mask[DROPOFF]:
        return DROPOFF

    grid = obs["grid"]
    carrying = bool(grid[5, 0, 0] > 0.5)
    target_plane = grid[4] if carrying else grid[3]
    tr, tc = np.unravel_index(int(np.argmax(target_plane)), target_plane.shape)
    row, col = obs["state"][0], obs["state"][1]

    for want, cond in (
        (0, tr > row),
        (1, tr < row),
        (2, tc > col),
        (3, tc < col),
    ):
        if cond and mask[want]:
            return want
    for a in (0, 1, 2, 3):  # blocked on every useful axis: take any legal move
        if mask[a]:
            return a
    return 0


def iter_scenarios(scenarios: Iterable[dict[str, Any]], limit: int | None = None):
    """Yield at most ``limit`` scenarios — keeps eval loops one-liners."""
    for i, s in enumerate(scenarios):
        if limit is not None and i >= limit:
            return
        yield s
