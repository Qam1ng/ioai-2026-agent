# Official Q&A intel (Discord, 2026-07-22/23) — binding facts for our design

Distilled from organizer (Abhishek Divekar) answers. These override guesses.

## Timeline
- **Registration closes 31 Jul 2026 23:59 AoE** (email adivekar@utexas.edu, lab name in subject).
- **Competition: 4 Aug (Day 1) & 6 Aug (Day 2)**, 6-hour window, 3 problems/day.
- Technical report per problem due **within 30 min** of each day's end.
- Practice comps end ~31 Jul. Symposium: remote, 1–2 months post-IOAI; medalists give orals.

## Launch protocol (matches our design!)
- Organizers provide a **simple task-agnostic start prompt** per problem, e.g.
  "Solve the Kaggle competition <slug>. Follow your system instructions…"
  and a similar **continuation prompt**. One prompt per task (not one prompt
  fanning out to all 3).
- **Our system prompt must be task-agnostic** — exactly our SYSTEM + BRIEF split.
  (Our `--brief` remains valid: all task info will come via Kaggle download.)

## Kaggle submission environment (hard constraints)
- All task info comes **via Kaggle only** (kaggle-cli downloadable). Agents get
  a `.py`-based flow, not the human starter notebook.
- Agent-side compute is UNRESTRICTED (own GPUs fine, any packages, internet).
  Only the generated solution `.py` must run on standardized Kaggle hardware.
- Kernel accelerator is agent's choice in kernel-metadata.json:
  - CPU: `"enable_gpu":"false"`, `"machine_shape":""`
  - GPU P100: `"enable_gpu":"true"`, `"machine_shape":"NvidiaTeslaP100"`
  - GPU T4:  `"enable_gpu":"true"`, `"machine_shape":"NvidiaTeslaT4"`
- **GPU quota: 30h/week per Kaggle account; all 3 problems of a day share ONE
  account.** Day 1 and Day 2 may use different accounts (fresh quota).
- **No trained checkpoints may be submitted** — only the `.py`; the model must
  train on the Kaggle kernel itself (fair 1-GPU comparison).
- Kaggle kernel: **internet disabled**, **fixed package list** (Technical
  Appendix p.13; env file to be published; no pip install inside submission).
- No enforced 10-min clock guard, but training-time discipline still expected;
  scoring: best submission on private testset A is re-scored on private
  testset B after close.
- Known issue: task2 practice comp `competition_sources` mounting was
  misconfigured (empty /kaggle/input) — organizers debugging. Workaround seen in
  community: re-upload data as a private dataset via dataset_sources.

## Implications for our agent (action items)
1. Kernel metadata: support machine_shape (P100/T4/CPU) and teach the agent to
   CHOOSE the accelerator per problem + budget GPU quota (30h shared across 3
   problems/day). Add quota awareness to the submit gate.
2. Never rely on uploading artifacts/checkpoints — solution .py must train
   from scratch on Kaggle within quota.
3. Match the official launch protocol: task-agnostic system prompt + per-task
   start/continue prompts; support a "continue" entry (our crash-resume state
   machine already fits).
4. Pin our kernel code to the official package list once published; add a local
   dry-run env mirroring it.
5. Watch for the fixed env file + updated SKILL.md + possible starter notebook
   from organizers.
