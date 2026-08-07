# Agent memory — boundary rules (READ BEFORE ADDING FILES)

`memory/lessons/*.md` is the **agent's** memory, surfaced during task-solving via
the `memory_recall` tool. It must contain **task-solving knowledge ONLY**:

✅ ALLOWED (helps solve the ML problem)
- ML techniques & patterns (e.g. frozen-encoder linear probe for class-incremental)
- validation methodology (k-fold, significance, local↔LB calibration)
- coding gotchas (library API changes, kernel path discovery)
- Kaggle *mechanics needed to submit/run* (code-comp notebook flow, kernel
  accelerator config, GPU-quota frugality, offline/no-checkpoint constraints)

🚫 FORBIDDEN (competition logistics — the agent must never see this)
- registration deadlines, dates, who to email, the symposium, awards ceremony,
  timelines, org contacts, community/Discord chatter

Those logistics live in **`docs/`** (for the human team) and in the developer's
own notes — deliberately OUTSIDE the agent's reach. The agent's tools are
workspace-confined: `read_file` sees only `workspace/`, `memory_recall` sees only
this dir, `skill_load` sees only `skills/`. Nothing here can read `docs/`.

Rule of thumb: if a fact does not change *how the agent writes/trains/submits the
solution*, it does not belong in `memory/lessons/`.

## Optional scope tag
Prefer starting a lesson with a scope hint so recall stays relevant across
different tasks (a robotics task shouldn't dwell on audio specifics):
`scope: general` · `scope: kaggle` · `scope: audio` · `scope: <task-family>`.
