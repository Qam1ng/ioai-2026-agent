# Runbook

## Run HearSay

    cd ~/IOAI2026-agent
    export CUDA_VISIBLE_DEVICES=4,5,6,7          # only cards that are ours
    setsid nohup ~/venvs/ioai312/bin/python -u -m native.main \
        --mode competition --slug <competition> --solvers 3 \
        --deadline-min 120 --max-cost-usd 100 --max-submissions 4 \
        --model claude-opus-5 --effort high \
        > run.log 2>&1 < /dev/null &

    setsid nohup ./killswitch.sh 7200 >/dev/null 2>&1 </dev/null &

Two things about that second line. `setsid` because plain `nohup` has died on
ssh disconnect here, and the killswitch because the harness's own deadline has
been walked past before — it stops the launcher, every agent session, and
anything the solvers backgrounded out of the workspace. It is the only thing
that has ever shut a run down cleanly on the first attempt.

`--max-submissions` is a ceiling; the real allowance is whatever the daily quota
has left, checked at boot. Set it well under the daily cap on tight
competitions — chicken allows five a day and a run that submits on every
improvement will spend them all.

For a test-of-record, use `--mode clean-benchmark` and a fresh workspace. That
mode withholds leaderboard and web feedback from candidate lineages; an
attempted audit read is recorded as permanent taint. Never reuse a competition
workspace as clean evidence.

Before accepting a run, inspect these harness-owned artifacts together:

    run_contract.json
    route_coordination.json
    evidence/firewall.json
    evidence/candidates/<candidate>.json
    <candidate>/out/sample_locality_receipt.json

`route_coordination.json` says whether routes actually adopted one another's
fact IDs or merely ran independently. A valid CSV without the matching
contract, OOF/fold hashes, locality receipt, and untainted lineage is not a
clean benchmark result.

## Watch it

    ./watch                    # most recent workspace, once
    ./watch <slug> -l          # follow, refreshing every 20s

The board is the top of that view because it is where everything ends up. If
`!! STUCK GATE` appears, read it first: a gate that has refused the same thing
three times is more likely wrong than the run is.

## Stop it

    pkill -f "killswitch.sh"; pkill -9 -f "native.main"; pkill -9 -f "_bundled/claude"
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
      case "$(readlink /proc/$p/cwd 2>/dev/null)" in
        *hearsay*) kill -9 $p;;
      esac
    done

The loop matters. Solvers background their training with `nohup`, and those
children outlive the session that started them — one run's watcher kept
submitting eleven minutes after the harness printed its summary.

Then check, and do not trust `pgrep` here: its own command line contains the
pattern and it matches itself.

    ps -eo cmd --no-headers | grep -c "[n]ative.main --slug"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l

## Environment (UCLA box)

    ~/venvs/ioai312/bin/python          torch+cu130, kaggle, claude-agent-sdk
    /data/qyan/ioai/<slug>/             competition data, symlinked as input/

The venv was created with `uv` and has no `pip`; install with
`VIRTUAL_ENV=~/venvs/ioai312 ~/.local/bin/uv pip install <pkg>`.

Cards 0-3 belong to other people. Set `CUDA_VISIBLE_DEVICES` to 4-7 and note
that a child process setting it again *replaces* the restriction rather than
indexing into it — a run reached GPU 0 that way. The harness now refuses those
commands, but the launcher's own variable is what defines "ours".

## Checks

    python -m native.selftest                          # 201, hand-computed
    python -m native.monitor --slug <slug> --json
    python -m native.scripts.checkfolds --workspace <ws>
    python -m native.scripts.integrity --candidate <s> --workspace <ws>

## Data synthesis (chicken)

    python -m native.scripts.chicken_data_synthesis prepare \
        --run-root <run> --source-root <forge-v4-run> --timestamps <timestamps.json>
    python -m native.scripts.chicken_data_synthesis synthesize --run-root <run> --source-root <forge-v4-run>
    python -m native.scripts.chicken_data_synthesis validate --run-root <run>
    python -m native.scripts.chicken_data_synthesis seal --run-root <run>

`synthesize` needs torch and Pillow and takes about five minutes for 3,200
rows; `validate` calibrates the ruler and then scores every recipe on it, which
is leave-one-out here and runs about nine minutes.

`seal` exits 2 with `no recipe cleared the gate; the incumbent stands` when
nothing beats the incumbent under the calibrated ruler. That is the expected
outcome, not a failure — read `outputs/synthetic_validation_receipt.json`
before overriding anything, in particular `protocol_mismatch_audit`, which says
how much of a candidate's claimed margin was two rulers rather than progress.
