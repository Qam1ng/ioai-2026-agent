# Task 3 — Radar heat-map semantic segmentation (IOAI-2025 mirror) · task card

Kaggle slug: `radar-ioai-2025-task-1` (paths in the baseline notebook are
`/kaggle/input/radar-ioai-2025-task-1/{training_set,test_set}/…`).
Type: **dense per-pixel multi-class classification** (semantic segmentation) on
radar heat maps. Used here as a *mirror* — a real past IOAI task to rehearse the
whole pipeline on, not a live leaderboard.

Every shape, dtype and class fraction below was measured by parsing the shipped
`*.mat.pt` files, not inferred from the notebook.

## The problem
Each sample is one radar frame rendered as a `50 × 181` map (range × azimuth
bins) with **6 input channels**. Predict a class label for every one of the
`50 * 181 = 9,050` pixels.

## Data — `*.mat.pt`, one `torch.save`-d tensor per file

| split | files | tensor | dtype |
|---|---|---|---|
| `training_set/training_set/` | 1,000 | `(7, 50, 181)` | `float64` (`DoubleStorage`) |
| `test_set/test_set/` | 500 | `(6, 50, 181)` | `float64` (`DoubleStorage`) |

- Channels `0..5` are the inputs; every one is in `[0, 1]`.
- Channel `6` (training only) is the **label** channel.
- Filenames are `<n>.mat.pt`. Training ids are sparse in `1..1800`; test ids are
  `1..500`. The directories are nested twice (`training_set/training_set/…`) and on
  Kaggle sit under `/kaggle/input/competitions/<slug>/` — discover, never hardcode
  (`memory/lessons/kaggle-kernel-input-paths.md`).
- There is **no `sample_submission.csv`** in the archive; the header has to be
  constructed (see below).

### The ±1 label shift
Raw labels on disk are `{-1, 0, 1, 2, 3}` — 5 classes. The baseline:

```python
labels = labels + 1        # dataset  : -1..3  ->  0..4  for CrossEntropyLoss
...
preds = torch.argmax(outputs, dim=1)
preds = preds - 1          # inference:  0..4  -> -1..3  for the submission
```

The model's last conv has `out_channels=5`, confirming 5 classes. **Class 0
(raw −1) is background.** Dropping either half of the shift produces a
perfectly well-formed submission that is wrong on every pixel — `data.py`
therefore shifts `+1` on load and `−1` on write by default, so the shift is
never the caller's responsibility.

### Class balance (measured over 30 training files)

| shifted | raw | fraction |
|---:|---:|---:|
| 0 | −1 | **97.66 %** |
| 1 | 0 | 0.147 % |
| 2 | +1 | 0.170 % |
| 3 | +2 | 0.079 % |
| 4 | +3 | 1.939 % |

This dominates every modelling decision. Predicting background everywhere scores
≈0.977 pixel accuracy, so **pixel accuracy is not a usable signal**; the
baseline's unweighted `CrossEntropyLoss` over all pixels is 97.7% background
gradient and will happily converge to exactly that degenerate solution.

## Metric — partly known, partly assumed
The competition uses a custom `RadarMetric` whose source was not published with
the mirror.

**KNOWN**: per-pixel multi-class prediction over 5 classes, labels shifted ±1.

**ASSUMED**: the aggregation. `metric.py` implements and exposes *all* the
plausible readings — `pixel_accuracy`, `mean_iou`, `mean_iou_fg`, `macro_f1`,
`macro_f1_fg`, `weighted_f1` — plus a macro-over-files variant
(`per_sample_metric`). `metric.PRIMARY = "macro_f1_fg"` (macro-F1 over the four
foreground classes) is our working choice: it ignores the trivially-easy
background and weights rare classes equally.

> ⚠️ **`PRIMARY` must be verified with one calibration submission before it is
> trusted.** `metric.calibration_table(y_true)` prints the score of every
> constant-class prediction under every aggregation; those columns are far apart
> (e.g. all-background scores 0.987 pixel-accuracy, 0.199 macro-F1, 0.000
> macro-F1-fg). Submit one constant map, read which column the leaderboard
> matches, then set `PRIMARY` accordingly. One submission out of 50 to eliminate
> the risk of optimising the wrong objective for six hours. Log it as a
> `ProbeResult(probe_kind="calibration")`.

## Submission format
```
filename,pixel_0,pixel_1,...,pixel_9049
1.mat.pt,-1,-1,3,...
```
- 9,051 columns; one row per test file; values in **raw** label space (`−1..3`).
- Row order follows the order files were listed. `data.list_tensor_files` sorts
  **numerically**, not lexically (`2.mat.pt` before `10.mat.pt`) — a lexical sort
  produces a valid-looking file scored against the wrong rows.

## Data-integrity note (mirror only — do not exploit)
The test `*.mat.pt` files declare a `(6, 50, 181)` tensor view, but the
underlying storage blob is still `7 × 50 × 181` and **contains the label
channel**, with the same `{-1,0,1,2,3}` distribution as training. `torch.load`
never exposes it (the view is 6 channels), so the leak is invisible through the
normal API, and `data.load_tensor(backend="raw")` deliberately trims to the
declared view so both backends behave identically.

Using it to manufacture leaderboard predictions would be cheating and is out of
the question. Its one legitimate use is offline: it gives this *ended, mirrored*
competition a ground truth for the 500 test frames, which is exactly what is
needed to sanity-check a metric implementation without a live leaderboard. Treat
it as a rehearsal fixture, never as a source of submitted values. Assume the 2026
tasks will not have this bug.

## Constraints (standing IOAI rules, `docs/QA-NOTES.md`)
- The submitted artifact is a `.py` Kaggle kernel that **trains on the kernel
  itself**; no checkpoints may be uploaded.
- Kernel runs with **internet disabled** and a **fixed package list** (no
  `pip install`) — check with `swarm.mirror.packages`.
- Accelerator chosen in `kernel-metadata.json`; 30 GPU-hours/week shared across
  the day's three problems.

## Modelling notes
- The baseline is three `3×3` convs (`6→16→32→5`), no pooling, no normalisation,
  40 epochs of Adam @ 1e-3, plain `CrossEntropyLoss`. It is a floor, not a target.
- Highest-leverage fixes, in order: class weighting or focal/Dice loss to survive
  the 97.7% background; a U-Net-ish encoder/decoder for receptive field; and
  input normalisation (channels are already `[0,1]`, but their scales differ).
- `50 × 181` is small and non-square — pooling twice already costs a lot of
  azimuth resolution. Prefer dilation or a shallow U-Net over aggressive
  downsampling.
- 1,000 training frames × 9,050 pixels is plenty of *pixels* but few *scenes*;
  validate by held-out **file**, and expect local scores to be optimistic if
  frames come from contiguous sequences (`memory/lessons/local-cv-optimism.md`).

## Files
- `data.py` — file discovery, `.mat.pt` loading (torch, plus a torch-free `raw`
  backend), label shifting, submission read/write, `describe()`.
- `metric.py` — `radar_metric`, `per_sample_metric`, `confusion_matrix`,
  `per_class_iou`, `per_class_f1`, `constant_prediction_scores`,
  `calibration_table`, `PRIMARY`.
