# Task 2 — Robot Delivery Academy (At Home Practice) · task card

Source: `IOAI-2026/Home Task/Home-Task-2.ipynb` (organizers' notebook, fetched 2026-07-26).
Type: **imitation learning / behaviour cloning**, discrete 6-action policy on an 8×8 grid.
Every shape and count below was confirmed by loading the shipped `.pkl` files, not by
reading the notebook prose.

## The problem
An 8×8 city map. Each episode the robot starts somewhere, walks to the depot holding
the package, **picks it up**, walks to the destination depot, and **drops it off**.
8 cells are walls; 6 cells are depots; every map differs.

The point is *not* to solve it — a 20-line BFS solves it perfectly. The point is
whether a model can **learn the behaviour from a deliberately small demonstration
budget** and still act correctly on maps it has never seen.

## Data — 3 pickle files (confirmed by loading)

### `train_demos.pkl` → `dict`
```python
{
  "trajectories": [ ... 400 dicts ... ],
  "total_trajectories": 400,
  "environment_config": {
      "grid_size": 8, "n_depots": 6, "n_walls": 8, "max_steps": 120,
      "action_names": {0:"south",1:"north",2:"east",3:"west",4:"pickup",5:"dropoff"},
  },
}
```
Each trajectory:
```python
{
  "layout_id":    str,     # e.g. "train_0000"  (100 distinct layouts, 4 episodes each)
  "episode_seed": int,     # 100000..100399
  "scenario":     dict,    # same 8 keys as a valid/test scenario (see below)
  "observations": [ dict, ... ],   # len == num_steps
  "actions":      [ int, ... ],    # len == num_steps, ids 0..5
  "success":      bool,    # True for all 400
  "num_steps":    int,     # 4..30, mean 13.32
}
```
Each observation:

| key | type | shape | dtype | contents |
|---|---|---|---|---|
| `grid` | ndarray | `(6, 8, 8)` | `float32` | ch0 walls · ch1 depots · ch2 robot · ch3 package depot (all-zero while carrying) · ch4 destination depot · ch5 carrying flag broadcast over the whole plane |
| `vector` | ndarray | `(13,)` | `float32` | `row/7, col/7, package_field/6, destination/5, carrying, target_row/7, target_col/7, (target_row-row)/7, (target_col-col)/7, blocked_south, blocked_north, blocked_east, blocked_west` |
| `action_mask` | ndarray | `(6,)` | `bool` | which actions are legal right now |
| `state` | tuple | `(4,)` | int | `(row, col, package_field, destination)`; `package_field == 6` means *carrying* |

Flattened baseline feature dim = `6*8*8 + 13` = **397**.

### `valid_scenarios.pkl` / `test_scenarios.pkl` → `list[dict]`
200 and **1,600** scenarios, no labels. Each:
```python
{ "layout_id": "valid_0000",  "layout_seed": 20000, "episode_seed": 200000,
  "walls":  ((1,1), ... ),   # exactly 8 (row, col) tuples
  "depots": ((6,1), ... ),   # exactly 6 (row, col) tuples, indexed 0..5
  "agent_pos": (5, 6),
  "package_location": 3,     # depot index
  "destination": 5 }         # depot index, never equal to package_location
```
Layouts repeat: valid = 50 layouts × 4 episodes, test = 400 layouts × 4 episodes.
`(layout_id, episode_seed)` is the unique episode key — **`layout_id` alone is not.**

### Dataset scale (the actual constraint)
- **5,327** state→action decisions total. That is the entire supervision budget.
- Action balance: south 1066 (20.0%), north 1148 (21.6%), east 1120 (21.0%),
  west 1193 (22.4%), **pickup 400 (7.5%), dropoff 400 (7.5%)**.
  Exactly one pickup and one dropoff per episode — the two actions that decide
  success are the two rarest in the data.
- Expert success rate 1.000; expert episodes average 13.32 steps.

## Objective & metric
Train a behaviour-cloning action model: observation → action id 0..5, run it step by
step through complete episodes.

**Score = episode success rate**, `SR = successful deliveries / total episodes`.
Success = dropoff completed at the destination depot within `max_steps = 120`.
Diagnostics reported alongside (not scored): `avg_steps`,
`avg_invalid_pickup_or_dropoff`.

> **Action accuracy is a trap, and the organizers say so.** §1: *"A single wrong
> action can move the robot into states that were rare in the demonstrations, so
> high action accuracy does not always mean high episode success."* At 13.32 steps
> per episode, even a uniformly 95%-accurate policy finishes cleanly only ~51% of
> the time, and real errors are correlated — they cluster in the off-distribution
> states the model's own earlier mistakes created. Worse, a model can hit ~85%
> action accuracy while never once emitting `pickup`, and score **exactly 0.0**.
> Gate on `metric.evaluate_policy(...)["success_rate"]`. Never on the proxy.

## Environment dynamics (from the simulator, replicated in `simulator.py`)
- Actions 0–3 move; moving into a wall or off-grid is a **silent no-op that still
  burns a step**.
- Action 4 `pickup` only works standing on `depots[package_location]` and not
  carrying; action 5 `dropoff` only works carrying and standing on
  `depots[destination]`. Otherwise the step counts and
  `info["invalid_pickup_or_dropoff"]` is set.
- Successful dropoff ends the episode immediately.
- `timed_out` when `step_count >= 120` without a dropoff.

## Submission
`predictions.zip` containing `predictions.jsonl`, one JSON object per line:
```json
{"layout_id": "test_0000", "episode_seed": 300000, "actions": [1, 1, 2, 4, 0, 5]}
```
All 1,600 test scenarios. The grader **replays the recorded action list** from the
scenario's initial state — it does not run our policy. Local scoring must therefore
go through `metric.evaluate_predictions()` / `simulator.replay_actions()`, not
through the policy rollout, or we validate the wrong thing.

## Constraints (notebook §6.4 — verbatim intent)
- ✅ Train on the provided demonstrations.
- ❌ Do **not** use expert action labels for validation or test scenarios.
- ❌ Do **not** generate additional expert trajectories with search, planning, or
  another expert model. *(This kills the obvious BFS-relabelling / DAgger route —
  it is the single most tempting and most disqualifying idea in this task.)*
- ⚙️ Final prediction process must be **deterministic**.
- ⚙️ The submitted notebook must build `predictions.zip` from scratch.
- ⚠️ Rule-based or hard-coded solutions may be reviewed by the Scientific Committee.

Plus the standing IOAI rules from `docs/QA-NOTES.md`: the submitted `.py` trains on
the Kaggle kernel itself, no checkpoints uploaded, no internet, fixed package list.

## Local reference numbers (measured here, not guessed)
- Replaying all 400 expert action sequences through `simulator.py` → success rate
  **1.000**. This is the fidelity check that our simulator matches the organizers'.
- `simulator.greedy_reference_policy` (mask-aware greedy, *not a submission* — it is
  rule-based and forbidden) scores **0.540** on the 200 validation scenarios:
  55 never picked up, 37 carried but never delivered. Walls deadlock naive greedy,
  which is precisely why the task is non-trivial for a small learned policy.

## Modelling notes worth carrying in
- **Group validation by `layout_id`.** 4 episodes share each layout; a random
  per-decision split leaks the same map into both sides and reports an optimistic
  number (`memory/lessons/local-cv-optimism.md`).
- The baseline flattens `(6,8,8)` into 384 numbers and loses the geometry; the
  organizers name this as its main limitation (§8, §9). A conv/spatial model over
  the grid planes plus the 13-vector is the obvious first improvement.
- `action_mask` is available at inference. Masking illegal actions at rollout time
  costs nothing and removes a whole class of wasted steps — the baseline only uses
  it during training.
- Rare-action handling (`pickup`/`dropoff` at 7.5% each) is the highest-leverage
  imbalance fix: class weights, or a separate "am I on the target depot" head.

## Files
- `data.py` — loaders → numpy; `describe()` prints the statistics above.
- `simulator.py` — `DeliverySimulator8x8`, `run_episode`, `replay_actions`.
- `metric.py` — `evaluate_policy` (the real metric), `evaluate_predictions`
  (grader-equivalent replay), `action_accuracy` (the proxy, clearly labelled).
