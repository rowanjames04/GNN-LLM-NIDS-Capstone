# GNN-LLM-NIDS-Capstone

Explainable Network Intrusion Detection via a GNN-LLM hybrid pipeline.

A dual-channel Graph Neural Network classifies network flows (including attack
families it has never seen during training), assembles a structured evidence
pack for each detection, and a Large Language Model translates that evidence
into a plain-language incident report. The project additionally benchmarks a
range of cloud-hosted and self-hosted language models on the quality and
factual groundedness of those reports.

UTS Honours capstone (41030), 2026. Supervisor: Dr Tanzeela Altaf.

## Layout

```
src/gnnids/
  data/        dataset loading, splits, feature handling
  graph/       flow -> graph construction, windowing
  models/      dual-channel GNN, baselines
  causality/   post-detection causal / attribution layer
  llm/         evidence-pack serialisation, prompting, model adapters
  eval/        metrics, zero-day protocol, report scoring
  ui/          Streamlit demo
configs/       experiment configs (YAML)
scripts/       entry points (download, preprocess, train, evaluate)
notebooks/     figure generation only; pipeline logic lives in src/
tests/         unit tests
results/       metrics (committed), figures (committed), checkpoints (ignored)
data/          all contents git-ignored
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Notes

Design rationale, mathematical derivations, and the decision register are kept
in an Obsidian vault outside this repository, not in the code.
