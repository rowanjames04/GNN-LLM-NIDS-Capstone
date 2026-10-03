# GNN-LLM-NIDS-Capstone

Explainable network intrusion detection: a graph neural network flags network
flows, and a language model writes the incident report.

A dual-channel graph neural network classifies each flow from its own features
and from what its two hosts are doing. For each detection it assembles a
structured evidence pack, which a language model turns into a plain-language
incident report. The project measures three things: what message passing adds
over a topology-blind model, how much of that survives on attack families and
networks the model has never seen, and how faithfully different cloud and
self-hosted language models relay the evidence.

UTS Honours capstone (41030), 2026. Supervisor: Dr Tanzeela Altaf.

All results are offline batch evaluation. The demo replays a recorded capture;
nothing here is a real-time system.

## Layout

```
src/gnnids/
  data/        loading, schema, splits, feature transforms, host features
  graph/       flows -> graph snapshots, windowing, memory-mapped inputs
  models/      dual-channel GNN, E-GraphSAGE layer, MLP baseline
  training/    the training loop, and campaign bookkeeping (tags, resume)
  eval/        metrics, prevalence standardisation, hold-out plans, transfer
  explain/     integrated gradients, neighbour occlusion, evidence packs
  llm/         model adapters, prompts, groundedness and fidelity scoring,
               resumable report logging, run provenance
  ui/          replay engine, neighbourhood graph, Streamlit demo
configs/       every experiment is a YAML file, never an edited constant
scripts/       thin command-line entry points over src/gnnids
notebooks/     the Colab launcher for the self-hosted language models
tests/         unit and end-to-end tests
results/       metrics, evidence packs, reports and figures (committed);
               checkpoints and logs (ignored)
data/          datasets and preprocessed arrays (ignored; see below)
```

## Setup

Python 3.13.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests/ -q
```

`torch` and `xgboost` must not be imported into the same process on macOS: both
bundle an OpenMP runtime and the process segfaults. The scripts keep them apart.

## Reproducing the results

Every script has a `--smoke` mode that proves its code path in seconds on a few
windows. Smoke output is written under a `smoke_` name and is never a result.

### 1. Data

```bash
python scripts/download_data.py                                      # NF-UNSW-NB15-v2
python scripts/download_data.py --config configs/dataset_toniot.yaml # NF-ToN-IoT-v2
python scripts/verify_dataset.py                    # what is actually in the file
python scripts/preprocess.py                                             # NF-UNSW-NB15-v2
python scripts/preprocess.py --config configs/preprocess_toniot.yaml --max-rows 6000000
```

NF-ToN-IoT-v2 is the primary dataset and is capped at 5.91 million of its 16.9
million flows for memory; every attack family is represented.

Preprocessing writes `raw_values.npy` beside the model's inputs: every feature
as it was measured. Evidence packs are built from it, because the model's
float16 inputs cannot be inverted back to exact values. For a dataset
preprocessed before that file existed:

```bash
python scripts/preprocess.py --config configs/preprocess_toniot.yaml --max-rows 6000000 --raw-only
```

### 2. Training and evaluation

```bash
scripts/run_stage_c.sh --dry-run        # print the plan
scripts/run_stage_c.sh                  # about 7.5 hours on an M4, unattended
```

This runs, in dependency order: the three model variants on three seeds, the
cross-dataset transfer evaluation, evidence packs, the replay, the template
control arm of the language-model study, leave-one-attack-out over eight attack
families, the topology-blind baselines, and the summaries and figures. It is
safe to re-run: a finished training run with the same configuration is reused.

Individual steps:

```bash
python scripts/train_gnn.py --seeds 3                 # the topology ablation
python scripts/train_zeroday.py --seeds 3             # leave-one-attack-out
python scripts/train_baselines.py --models xgboost    # baselines (run torch models separately)
python scripts/transfer_eval.py                       # cross-dataset, no retraining
python scripts/make_evidence.py                       # evidence packs
python scripts/train_gnn.py --ablation full --n-gnn-layers 3     # a tagged variant
```

### 3. Language-model study

```bash
python scripts/run_llm_study.py --estimate                           # projected cost, calls nothing
python scripts/run_llm_study.py --only paid free_tier --confirm-spend
```

The cloud models read `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` from the
environment. Nothing loads a `.env` file automatically, so export them first:

```bash
set -a; source .env; set +a      # if the keys are kept in a git-ignored .env
```

No paid call is made without `--confirm-spend`. The self-hosted models run in
Google Colab from `notebooks/colab_self_hosted_arm.ipynb`. Report generation is
resumable, and `python scripts/rescore_reports.py` re-applies the scoring to
saved reports without calling any model.

### 4. Tables, figures and the demo

```bash
python scripts/summarise_transfer.py
python scripts/summarise_llm_study.py
python scripts/summarise_evidence.py
python scripts/make_figures.py            # results/figures/, with a caption manifest
python scripts/export_report_facts.py     # results/report_facts.md: every number, with its source
streamlit run src/gnnids/ui/app.py
```

## Notes

Design rationale, derivations, the decision register and the defect register
are kept in the project notes, outside this repository.
