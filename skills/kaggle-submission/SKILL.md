# Submit to an IOAI code competition (what the generic playbook leaves out)

Read `kaggle-cli-official` for the mechanics of the CLI. This covers the parts
specific to these competitions, which is where the failures actually are.

## The environment block is mandatory and must come first

The real tasks run offline against a pinned package set shipped as a Kaggle
dataset of wheels, not against Kaggle's default image. The task description
carries an official starter with a `setup_ioai_env()` block. Keep it at the very
top of your script, unchanged, and write everything else underneath.

```json
{
  "id": "<your-username>/<notebook-slug>",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": false,
  "dataset_sources": ["kamalkhan/ioai-2026-wheel-dataset"],
  "competition_sources": ["<competition-slug>"]
}
```

```python
# ── the starter's setup block, unchanged ──
setup_ioai_env()

# ── your solution below ──
import torch
```

`setup_ioai_env()` must run **before** you import anything it installs. pip
cannot replace a module Python has already loaded, so an import above that line
silently keeps the wrong version and you find out from a score, not an error.
The install costs 30-40 seconds per push.

Confirm the dataset slug against the task description rather than trusting the
one above — it is what the practice competition used, and it can change.

## Never hard-code an input path

Mount layout differs between competition data and datasets and has changed
before.

```python
def find_input(name, root="/kaggle/input"):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not under {root}")
    return hits[0]
```

Write to `/kaggle/working/submission.csv`, keeping the sample's exact header and
row order.

## Timing: the deadline applies to the submit, not the run

The organisers tested five scenarios. What counts is that
`kaggle competitions submit` is *issued* before the deadline; the metric may
finish scoring afterwards and the entry still lands on the public leaderboard. A
kernel that is still running at the deadline does not, however — it completes,
it scores, and it is invisible.

So the last moment worth starting a kernel is `deadline − kernel runtime −
a few minutes of Kaggle latency`. Use `--timeout` on the push to cap a run
rather than hoping. An earlier organiser message said both steps had to finish
before the deadline; the later five-scenario test supersedes it.

## Push once

The generic playbook says to push twice on first creation, working around a race
where the data mount is not ready and the run dies instantly with
`FileNotFoundError`. Only two GPU kernels may run at once, so a reflexive second
push spends half your concurrency on a run you expect to fail. Push once, and
push again only if the first actually failed that way.

## Concurrency and quota

Two GPU kernels and five CPU kernels at a time, across the whole account. GPU
quota is 30 hours a week shared by all three problems of a competition day, so a
kernel that runs long is spending the next problem's budget. There is no way to
cancel a running kernel — give the script its own wall-clock budget and have it
write the best submission it has when time runs out.

## In this system you do not submit

You have no submit tool and the CLI path is blocked. Produce `out/kernel/` and
`out/submission.csv`; the evaluator decides what goes and the harness sends it.
Test that your kernel *runs* if you need to, but expect a push to be counted
against the concurrency you share with everyone else.
