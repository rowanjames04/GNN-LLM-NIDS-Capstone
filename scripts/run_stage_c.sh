#!/usr/bin/env bash
# Stage C: the batched training campaign, as one unattended launch (D24).
#
#   scripts/run_stage_c.sh                 # everything in the default plan
#   scripts/run_stage_c.sh --dry-run       # print the commands, run nothing
#   scripts/run_stage_c.sh --steps gnn zeroday
#   scripts/run_stage_c.sh --steps layers moreseeds     # the optional runs
#
# Why a script and not six commands typed by hand: the order matters (evidence
# packs are built from a checkpoint, reports from the packs), torch and xgboost
# must never share a process (C3), and a campaign someone has to babysit at
# 2 a.m. is one that gets a step skipped.
#
# Safe to re-run. train_gnn.py and train_zeroday.py both resume: a finished
# (seed) or (family, seed) with the same configuration is reused, not retrained.
# Every step's output goes to results/logs/<timestamp>/<step>.log as well as the
# terminal.
#
# Default plan, about 7.5 hours on the M4:
#   archive    keep the 2026-08-23 checkpoints and transfer files      seconds
#   gnn        Phase 4: three ablations x 3 seeds, fixed budget        ~75 min
#   transfer   Phase 9 re-run on the new checkpoints + control         minutes
#   evidence   200 evidence packs from the new full_seed0              minutes
#   replay     replay metrics at full scale                            minutes
#   stub       the LLM study's free control arm, both prompt versions  seconds
#   zeroday    Phase 5: 8 families x 3 seeds                           ~5 h
#   baselines  the six non-XGBoost Phase 3 baselines                   ~58 min
#   report     summaries, figures and the report-facts file            seconds
#
# The quick steps that depend only on `gnn` run before `zeroday`, so they exist
# even if the long run is interrupted.
#
# Optional, not in the default plan:
#   layers     U3: K = 1 and K = 3 for the full model                  ~1.3 h
#   moreseeds  U2: full and channel1_only extended to 10 seeds         ~2 h

set -uo pipefail
cd "$(dirname "$0")/.."

DEFAULT_STEPS=(archive gnn transfer evidence replay stub zeroday baselines report)
STEPS=()
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --steps)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do STEPS+=("$1"); shift; done ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=("${DEFAULT_STEPS[@]}")

# Unbuffered, and inherited by the per-seed child processes the training scripts
# spawn, so a log reads in the order things happened.
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at .venv -- see README" >&2; exit 1; }

DS="NF-ToN-IoT-v2"
TONIOT_PRE="configs/preprocess_toniot.yaml"
CKPT="results/checkpoints"
LOG_DIR="results/logs/$(date +%Y-%m-%d_%H%M%S)"
[[ $DRY -eq 1 ]] || mkdir -p "$LOG_DIR"

FAILED=()
GNN_OK=1          # dependents of `gnn` are skipped if it fails in this launch

# run <step> <command...>: echo, log, and record a failure without stopping the
# campaign -- one failed step must not cost the independent ones after it.
run() {
  local step="$1"; shift
  echo
  echo "[$(date +%H:%M:%S)] $step: $*"
  [[ $DRY -eq 1 ]] && return 0
  if ! "$@" 2>&1 | tee -a "$LOG_DIR/$step.log"; then
    echo "[$(date +%H:%M:%S)] $step FAILED -- see $LOG_DIR/$step.log"
    FAILED+=("$step")
    return 1
  fi
}

needs_gnn() {
  if [[ $GNN_OK -eq 0 ]]; then
    echo; echo "skipping $1: it depends on the gnn step, which failed"
    return 1
  fi
}

step_archive() {
  # The checkpoints the settled transfer results and the first 60 evidence
  # packs were produced from. Stage C overwrites the live ones.
  local dst="$CKPT/archive_2026-08-23_early_stopping"
  if [[ ! -d "$dst" ]]; then
    run archive mkdir -p "$dst"
    run archive bash -c "cp -pn $CKPT/${DS}_*.pt $dst/ 2>/dev/null || true"
  fi
  # summarise_transfer.py pools every transfer_*.json in its folder. Results
  # from the old checkpoints must leave that folder or they would be averaged
  # with the re-run. (Not named archive*/ -- that pattern is git-ignored.)
  local old="results/metrics/transfer/superseded_2026-08"
  shopt -s nullglob
  local stale=(results/metrics/transfer/transfer_${DS}_to_*_seed[0-9].json
               results/metrics/transfer/transfer_${DS}_to_${DS}.json)
  shopt -u nullglob
  if [[ ${#stale[@]} -gt 0 ]]; then
    run archive mkdir -p "$old"
    run archive mv "${stale[@]}" "$old/"
  fi
}

step_gnn() {
  run gnn "$PY" -u scripts/train_gnn.py --seeds 3 || GNN_OK=0
}

step_transfer() {
  needs_gnn transfer || return 0
  for seed in 0 1 2; do
    run transfer "$PY" -u scripts/transfer_eval.py \
      --checkpoint "$CKPT/${DS}_full_seed${seed}.pt"
  done
  # The in-dataset control, through the same code path.
  run transfer "$PY" -u scripts/transfer_eval.py \
    --checkpoint "$CKPT/${DS}_full_seed0.pt" --target-preprocess "$TONIOT_PRE"
}

step_evidence() { needs_gnn evidence || return 0; run evidence "$PY" -u scripts/make_evidence.py; }
step_replay()   { needs_gnn replay   || return 0; run replay   "$PY" -u scripts/replay.py; }

step_stub() {
  needs_gnn stub || return 0
  # Free and offline: the template control arm on the new packs.
  run stub "$PY" -u scripts/run_llm_study.py --only stub --prompt-versions v1 v2
}

step_zeroday() { run zeroday "$PY" -u scripts/train_zeroday.py --seeds 3; }

step_baselines() {
  # Two invocations: the MLP imports torch, and although XGBoost is not being
  # re-run here, keeping torch out of the scikit-learn process costs nothing.
  run baselines "$PY" -u scripts/train_baselines.py --models logreg random_forest
  run baselines "$PY" -u scripts/train_baselines.py --models mlp
}

step_report() {
  run report "$PY" scripts/summarise_transfer.py
  run report "$PY" scripts/summarise_llm_study.py
  for s in summarise_evidence make_figures export_report_facts; do
    [[ -f "scripts/$s.py" ]] && run report "$PY" "scripts/$s.py"
  done
}

step_layers() {
  for k in 1 3; do
    run layers "$PY" -u scripts/train_gnn.py --ablation full --n-gnn-layers "$k" --seeds 3
  done
}

step_moreseeds() {
  run moreseeds "$PY" -u scripts/train_gnn.py \
    --ablation full channel1_only --seeds 10 --extend
}

echo "Stage C: ${STEPS[*]}$([[ $DRY -eq 1 ]] && echo '   [dry run]')"
main() {
  for step in "${STEPS[@]}"; do
    if declare -F "step_$step" >/dev/null; then
      "step_$step"
    else
      echo "unknown step: $step" >&2; FAILED+=("$step")
    fi
  done
}

# caffeinate keeps the Mac awake for the whole campaign; a lid-closed sleep at
# hour three is the likeliest way to lose a night.
if [[ $DRY -eq 0 ]] && command -v caffeinate >/dev/null; then
  caffeinate -is -w $$ &
fi
main

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Stage C finished with failures: ${FAILED[*]}"
  echo "Re-run the same command: finished training runs are reused."
  exit 1
fi
echo "Stage C finished: ${STEPS[*]}"
[[ $DRY -eq 1 ]] || echo "logs: $LOG_DIR"
