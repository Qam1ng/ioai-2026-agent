# Independent live compliance and output review

- Reviewer: `trace_3` / independent compliance-data-output reviewer
- Review time: 2026-08-03 22:05:48 CST (UTC+8)
- Live competition: `ioai-2026-ai-models-track-practice-task-1-timed-deps`
- Scope: only this fresh live run's official Starter, `task_config.json`, downloaded competition files, and the current live candidate. No prior Practice One artifacts, predictions, reports, or configs were inspected. No hidden test structure was inferred. The reviewer did not push or submit.
- Candidate reviewed: `kernel/script.py` SHA-256 `fc47618e54bbaca9ee364b20bab21c22e66c5361e81bcd5a38c32f951e9d4b39`; `kernel/kernel-metadata.json` SHA-256 `21ec4b6db3fc6720ec8ef03278e718eaac017060c2678fae188be9c7fd6f93be`.

## Decision

**PASS for a live Kaggle execution attempt; no obvious fatal API, output, or compliance blocker was found in the reviewed hashes.** Local Python compilation and metadata JSON parsing pass. Remote execution is still required evidence: the remaining material risk is T4 runtime/memory, especially the 1,214-token AST at inference batch size 12 and the offline package-install fallback.

The original official Starter body was not a valid submission because it wrote one `id,prediction` dummy row. The reviewed candidate has replaced that body and now implements the required live output contract.

## Official live data audit

- `train.csv`: 296 labeled rows, schema `path,split,target,category`, exactly old labels 0-15.
- `fine_tune.csv`: 624 labeled rows, the same schema, exactly new labels 16-28.
- `submission.csv`: 363 test rows, schema `path,target`.
- Audio: 1,283 unique `.wav` files, exactly matching the union of the three CSV path sets. No duplicate path across the three CSVs, no missing referenced file, no unreferenced audio file, and no duplicate test path were found.
- The 1,260 files readable by the standard-library WAV parser include mono and stereo audio at varied sample rates; durations were median 5.0 s, p95 24.345 s, maximum about 120.001 s. Another 23 are valid non-PCM/extended WAV variants not handled by the standard-library parser. The candidate appropriately uses `soundfile`, converts stereo to mono, and deterministically resamples to 16 kHz.

Class support:

| Labels | Per-class counts |
|---|---|
| old 0-15 | `12,12,12,22,35,12,12,3,12,22,3,62,12,35,23,7` |
| new 16-28 | `24,45,45,45,45,45,45,45,45,60,60,60,60` |

The old data are strongly imbalanced and labels 7 and 10 have only three examples each. The candidate's deterministic within-class holdout leaves at least one validation and two training samples for every class; the class-weighted head loss is therefore well-defined, though old-class validation estimates for those two labels are necessarily noisy.

## Supplied checkpoint and preprocessing

- Architecture: `ASTForAudioClassification`, float32, 12 transformer layers, hidden size 768, 12 attention heads, FFN size 3,072, 16x16 patches, time/frequency strides 10, 128 mel bins, max feature length 1,024, and zero configured dropout.
- Safetensors: 203 tensors, about 329 MiB. The supplied classifier has layer norm plus dense weights `[16,768]` and bias `[16]`, exactly covering old labels 0-15.
- Feature extractor: 16 kHz, normalized with supplied mean `-4.2677393` and std `4.5689974`, right padding, max length 1,024.
- Candidate use is structurally appropriate: load the provided checkpoint locally, freeze all supplied parameters, preserve the old 16-row classifier, and train only a fresh 13-row head over the same AST embedding.

## Official environment and Kaggle source contract

The official `ioai-starter.py` setup remains at the very top, before imports of Torch, Transformers, SciPy, SoundFile, or NumPy. It:

1. Searches `/kaggle/input` for `ioai_env-*.whl` without assuming mount nesting.
2. Symlinks CUDA wheel filenames into normalized names where Kaggle stripped `+cu...`.
3. Installs `uv` offline with pip.
4. Installs the `ioai-env` meta-package offline with `python -m uv pip install --system`, using `UV_SKIP_WHEEL_FILENAME_CHECK=1`.
5. Falls back to offline pip if `uv` fails.

The fast path is documented by the organizer as about 40 seconds; the fallback is documented as about six minutes.

The reviewed kernel metadata has the required bindings:

- `dataset_sources`: exactly `kamalkhan/ioai-2026-wheel-dataset`
- `competition_sources`: `ioai-2026-ai-models-track-practice-task-1-timed-deps`
- GPU enabled; internet disabled; private Python script kernel; T4 requested.

These bindings are essential. Omitting the wheel dataset causes the setup to fail before model code imports. The candidate searches live mounts rather than hard-coding a local or historical path.

## Output contract

Required output is `/kaggle/working/submission.csv` with:

- exact columns and order: `path,target`;
- exactly the 363 template paths in exact template order;
- integer targets in `[0,28]`;
- no missing/NaN target.

The candidate writes rows by zipping predictions with the original `submission.csv` row list and then reopens the file to verify row count, path order, and target range. Integer construction makes NaN impossible. A useful non-blocking strengthening would be an explicit header equality assertion and `combined.shape == (len(test_rows), 29)`.

## Single-forward and resource compliance

The reviewed candidate satisfies the live structural restrictions:

- one deterministic waveform view per sample;
- exactly one call to `base.audio_spectrogram_transformer(...)` per sample, batched;
- one shared AST embedding feeding immutable old logits and the trained new head;
- final concatenated tensor has 29 logits before argmax;
- no TTA, second encoder, external model, external data, test-global assignment, path/row override, or historical prediction use;
- calibration is one scalar offset selected only on the labeled within-class holdout;
- supplied checkpoint is used, final internet is disabled, and execution targets one Kaggle GPU.

Repeated new-head calls during head training act only on cached labeled embeddings and do not repeat AST encoder calls. That remains consistent with the one-AST-forward contract and the live baseline branch allowing a frozen old classifier plus fresh new rows.

## Runtime assessment

- Session planning p95: 420 seconds.
- Fast dependency setup: about 40 seconds; slow pip fallback: about six minutes.
- The candidate encodes 920 labeled plus 363 test clips once, in batches of 12, then trains only a small 768-to-13 linear head. Head optimization and calibration should be seconds; decoding/resampling plus AST encoding dominate.
- Batch 12 is plausible on a 16 GiB T4 under inference mode and fp16 autocast, but AST uses about 1,214 tokens per clip, so only an actual Kaggle run confirms peak memory and throughput. If OOM occurs, reduce `encode` batch size to 8 or 6; this preserves semantics at a modest runtime cost.
- No full-encoder training occurs, so the 600-second training budget is unlikely to be consumed by optimization. The setup fallback plus feature encoding can still push total wall time toward the upper end, so queue and retry reserve remain important.

## Fastest safe validation gates

Before treating a remote run as valid, check in this order:

1. Kaggle version reaches `COMPLETE`, not merely upload/queue/running.
2. Log contains `IOAI environment ready`, `encoding 920 labelled and 363 test clips`, validation/calibration metrics, and `wrote ... with 363 rows`.
3. No OOM, missing-wheel, import, audio-decoder, or checkpoint-load exception appears.
4. Download the exact kernel output and independently verify filename, header, 363 rows, exact path/order equality to the live template, integer/range/no-NaN, and a SHA-256 binding to that kernel version.
5. Only after the completed output passes those gates may the authorized submission controller submit it. Preserve one attempt for recovery.

