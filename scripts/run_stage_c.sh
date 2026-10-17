#!/usr/bin/env bash
# The training campaign, as one unattended launch (D24; rebuilt for D47).
#
#   scripts/run_stage_c.sh                 # everything in the default plan
#   scripts/run_stage_c.sh --dry-run       # print the commands, run nothing
#   scripts/run_stage_c.sh --steps gnn zeroday
#
# Rebuilt 2026-10-17 for the data-quality step back (D47): the flow-duration
# columns were an exporter artefact, preprocessing changed, and EVERY result has
# to be regenerated from the new data in one launch. The pre-mortem that shaped
# this file found the ways the old runner would have mixed old and new results:
# a feature cache never invalidated (C33), run reuse blind to the data (C32),
# old XGBoost and transfer files left where the summaries pool them, and failed
# steps recorded as passed. Each step below states which of those it closes.
#
# Safe to re-run. Training reuses finished (seed) and (family, seed) runs only
# when their fingerprint, which now includes the data version, matches. The
# archive step runs once (it leaves a marker) so a re-run never sweeps new
# results away.
#
# Default plan, roughly 10-11 hours on the M4:
#   archive          move every result built on the old data aside, once     seconds
#   preprocess       both datasets; refuses if the rows changed              ~15 min
#   audit            data-quality audit of the new preprocessing             ~2 min
#   baselines        xgboost alone, then logreg+rf, then mlp; both datasets  ~1.5 h
#   gnn              NF-ToN-IoT-v2: 3 ablations x 3 seeds                    ~75 min
#   gnn_unsw         NF-UNSW-NB15-v2: 3 ablations x 3 seeds                  ~45 min
#   transfer         GNN both directions x 3 seeds + both in-dataset controls ~15 min
#   transfer_trees   XGBoost both directions                                 ~10 min
#   evidence         200 evidence packs from the new full_seed0              ~3 min
#   replay           replay metrics                                          seconds
#   stub             the LLM study's template control on the new packs       seconds
#   zeroday          leave-one-attack-out, 8 families x 3 seeds              ~4.5 h
#   report           summaries, figures, report_facts.md                     seconds
#   moreseeds        full and channel1_only to 10 seeds (U2)                 ~2 h
#   report           again, with the extra seeds
#
# The language-model study itself is NOT here: it calls a hosted service on a
# free allowance and is run separately (scripts/run_llm_study.py, D44-D46).

set -uo pipefail
cd "$(dirname "$0")/.."

DEFAULT_STEPS=(archive preprocess audit baselines gnn gnn_unsw transfer transfer_trees
               evidence replay stub zeroday report moreseeds report)
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

export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at .venv -- see README" >&2; exit 1; }

TON="NF-ToN-IoT-v2"; UNSW="NF-UNSW-NB15-v2"
TON_PRE="configs/preprocess_toniot.yaml"; UNSW_PRE="configs/preprocess.yaml"
CKPT="results/checkpoints"
LABEL="2026-10-10_duration_artefact"
ARCHIVE="results/superseded/$LABEL"
LOG_DIR="results/logs/$(date +%Y-%m-%d_%H%M%S)"
[[ $DRY -eq 1 ]] || mkdir -p "$LOG_DIR"

FAILED=()
DATA_OK=1   # nothing trains on data that failed to preprocess
TON_OK=1    # dependents of the NF-ToN-IoT-v2 GNN
UNSW_OK=1   # dependents of the NF-UNSW-NB15-v2 GNN

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

gate() {   # gate <flag-value> <step> <what it needs>
  if [[ $1 -eq 0 ]]; then
    echo; echo "skipping $2: it needs $3, which failed in this launch"
    FAILED+=("$2(skipped)")
    return 1
  fi
}

step_archive() {
  # Every result built on the pre-D47 data moves to one folder, once. Metrics
  # stay in git there (results/superseded/ is not ignored) as the "with the
  # artefact" condition: the feature list is the only difference, so the two
  # sets measure what the artefact was worth. Checkpoints are git-ignored and
  # move beside the live ones.
  if [[ -f "$ARCHIVE/.done" ]]; then
    echo; echo "archive: already done ($ARCHIVE/.done) -- skipped so new results stay put"
    return 0
  fi
  run archive "$PY" - "$ARCHIVE" "$LABEL" <<'PYEOF'
import shutil, sys
from pathlib import Path
archive, label = Path(sys.argv[1]), sys.argv[2]
R = Path("results")
moved = []
def move(p: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dest)); moved.append(str(p))
def copy(p: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
for sub in ("gnn", "zeroday", "transfer", "baselines", "evidence", "replay"):
    for p in sorted((R / "metrics" / sub).glob("*")):
        if p.is_file() and not p.name.startswith("smoke_") and p.name != ".gitkeep" \
                and p.name != "gnn_NF-ToN-IoT-v2_fullonly.json":
            move(p, archive / "metrics" / sub / p.name)
for p in sorted((R / "evidence").glob("evidence_*.json")):
    move(p, archive / "evidence" / p.name)
for pat in ("reports_*.json", "*.partial.jsonl", "excerpts_*.md", "study_manifest.json"):
    for p in sorted((R / "reports").glob(pat)):
        if p.is_file() and not p.name.startswith("smoke_"):
            move(p, archive / "reports" / p.name)
for p in sorted((R / "checkpoints").glob("*.pt")):
    move(p, R / "checkpoints" / f"superseded_{label}" / p.name)
for p in [R / "report_facts.md", *sorted((R / "figures").glob("results_*")),
          *sorted((R / "metrics").glob("preprocess*.json"))]:
    if p.exists():
        copy(p, archive / p.relative_to(R))
archive.mkdir(parents=True, exist_ok=True)
(archive / "README.md").write_text(
    "Results built on the preprocessing before D47 (2026-10-17), when the model\n"
    "saw FLOW_DURATION_MILLISECONDS, DURATION_IN/OUT and the four throughput\n"
    "columns. Those columns are an exporter artefact (A8). Kept as the 'with the\n"
    "artefact' condition. Do not quote them as current results.\n")
(archive / ".done").write_text("\n".join(moved) + "\n")
print(f"moved {len(moved)} files into {archive}")
PYEOF
}

step_preprocess() {
  # Both datasets. Each refuses to write if its rows differ from the previous
  # run, so the comparison with the archived results stays a feature-list
  # comparison and nothing else. Deletes the stale graph cache itself (C33).
  run preprocess "$PY" -u scripts/preprocess.py --config "$TON_PRE" --no-sweep || DATA_OK=0
  run preprocess "$PY" -u scripts/preprocess.py --config "$UNSW_PRE" --no-sweep || DATA_OK=0
}

step_audit() {
  gate $DATA_OK audit "preprocessing" || return 0
  run audit "$PY" -u scripts/audit_data_quality.py
}

step_baselines() {
  # Before the GNN, because train_gnn.py stores the current baselines as its
  # comparator. Three invocations per dataset: xgboost must never share a
  # process with torch (C3), and the MLP imports torch. Entries from other
  # data are dropped on merge (C32), so nothing old survives.
  gate $DATA_OK baselines "preprocessing" || return 0
  for pre in "$TON_PRE" "$UNSW_PRE"; do
    run baselines "$PY" -u scripts/train_baselines.py --preprocess-config "$pre" --models xgboost
    run baselines "$PY" -u scripts/train_baselines.py --preprocess-config "$pre" --models logreg random_forest
    run baselines "$PY" -u scripts/train_baselines.py --preprocess-config "$pre" --models mlp
  done
}

step_gnn() {
  gate $DATA_OK gnn "preprocessing" || { TON_OK=0; return 0; }
  run gnn "$PY" -u scripts/train_gnn.py --seeds 3 || TON_OK=0
}

step_gnn_unsw() {
  gate $DATA_OK gnn_unsw "preprocessing" || { UNSW_OK=0; return 0; }
  run gnn_unsw "$PY" -u scripts/train_gnn.py --config configs/gnn_unsw.yaml --seeds 3 || UNSW_OK=0
}

step_transfer() {
  # Both directions, three seeds each, and both in-dataset controls through the
  # same code path. Every checkpoint is checked against the data it reads.
  if gate $TON_OK "transfer ToN-IoT->UNSW" "the NF-ToN-IoT-v2 GNN"; then
    for seed in 0 1 2; do
      run transfer "$PY" -u scripts/transfer_eval.py \
        --checkpoint "$CKPT/${TON}_full_seed${seed}.pt" \
        --source-preprocess "$TON_PRE" --target-preprocess "$UNSW_PRE"
    done
    run transfer "$PY" -u scripts/transfer_eval.py \
      --checkpoint "$CKPT/${TON}_full_seed0.pt" \
      --source-preprocess "$TON_PRE" --target-preprocess "$TON_PRE"
  fi
  if gate $UNSW_OK "transfer UNSW->ToN-IoT" "the NF-UNSW-NB15-v2 GNN"; then
    for seed in 0 1 2; do
      run transfer "$PY" -u scripts/transfer_eval.py \
        --checkpoint "$CKPT/${UNSW}_full_seed${seed}.pt" \
        --source-preprocess "$UNSW_PRE" --target-preprocess "$TON_PRE"
    done
    run transfer "$PY" -u scripts/transfer_eval.py \
      --checkpoint "$CKPT/${UNSW}_full_seed0.pt" \
      --source-preprocess "$UNSW_PRE" --target-preprocess "$UNSW_PRE"
  fi
}

step_transfer_trees() {
  # The tree arm of the transfer verdict, both directions (D32). No torch here.
  gate $DATA_OK transfer_trees "preprocessing" || return 0
  run transfer_trees "$PY" -u scripts/transfer_baselines.py \
    --source-preprocess "$TON_PRE" --target-preprocess "$UNSW_PRE" --seeds 3
  run transfer_trees "$PY" -u scripts/transfer_baselines.py \
    --source-preprocess "$UNSW_PRE" --target-preprocess "$TON_PRE" --seeds 3
}

step_evidence() { gate $TON_OK evidence "the NF-ToN-IoT-v2 GNN" || return 0
                  run evidence "$PY" -u scripts/make_evidence.py; }
step_replay()   { gate $TON_OK replay "the NF-ToN-IoT-v2 GNN" || return 0
                  run replay "$PY" -u scripts/replay.py; }

step_stub() {
  gate $TON_OK stub "the NF-ToN-IoT-v2 GNN" || return 0
  run stub "$PY" -u scripts/run_llm_study.py --only stub --prompt-versions v1 v2
}

step_zeroday() {
  gate $DATA_OK zeroday "preprocessing" || return 0
  run zeroday "$PY" -u scripts/train_zeroday.py --seeds 3
}

step_report() {
  run report "$PY" scripts/summarise_transfer.py
  run report "$PY" scripts/summarise_llm_study.py
  for s in export_llm_excerpts summarise_evidence make_figures export_report_facts; do
    [[ -f "scripts/$s.py" ]] && run report "$PY" "scripts/$s.py"
  done
}

step_moreseeds() {
  # U2: n = 3 gives p = 0.05 at best. Last, so every headline exists first.
  gate $TON_OK moreseeds "the NF-ToN-IoT-v2 GNN" || return 0
  run moreseeds "$PY" -u scripts/train_gnn.py --ablation full channel1_only --seeds 10 --extend
}

step_layers() {
  gate $DATA_OK layers "preprocessing" || return 0
  for k in 1 3; do
    run layers "$PY" -u scripts/train_gnn.py --ablation full --n-gnn-layers "$k" --seeds 3
  done
}

echo "Campaign: ${STEPS[*]}$([[ $DRY -eq 1 ]] && echo '   [dry run]')"
main() {
  for step in "${STEPS[@]}"; do
    if declare -F "step_$step" >/dev/null; then
      "step_$step"
    else
      echo "unknown step: $step" >&2; FAILED+=("$step")
    fi
  done
}

if [[ $DRY -eq 0 ]] && command -v caffeinate >/dev/null; then
  caffeinate -is -w $$ &
fi
main

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Campaign finished with failures: ${FAILED[*]}"
  echo "Re-run the same command: finished training runs are reused."
  exit 1
fi
echo "Campaign finished: ${STEPS[*]}"
