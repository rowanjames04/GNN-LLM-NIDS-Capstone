# Report facts

*Exported 2026-10-03T05:34:08 UTC by `scripts/export_report_facts.py`. Every number below was read from a results file at export time. None was typed in.*

**For whoever drafts from this file (human or model):**

1. Use these numbers exactly as written. Do not round further, recompute, or combine them.
2. A section marked **NOT YET MEASURED** has no numbers. Write `[NUM?]`.
3. A section marked **SUPERSEDED** has numbers that will be replaced. Do not quote them in final text.
4. `0.9887 ± 0.0030 (n = 3)` means mean ± standard deviation over 3 seeds.
5. Every PR-AUC is at the stated prevalence. A random model scores the prevalence itself.

## 1. Datasets and preprocessing

### NF-ToN-IoT-v2 (primary)

*Source: `results/metrics/preprocess_NF-ToN-IoT-v2.json`, generated 2026-08-22T04:57:00 UTC.*

- Flows used: **5,910,000** of 16,940,496 in the source file (capped for memory; every attack family is represented)
- Attack families: backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss (9, plus Benign)
- Features: 31 continuous (18 log-transformed), 5 categorical, 4 presence indicators, 2 port buckets
- Dropped as shortcuts: MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN
- Dropped as uninformative: DNS_QUERY_ID, ICMP_IPV4_TYPE
- Split method: whole 10,000-flow windows assigned to splits at random (the capture is ordered by scenario, not by time); rows purged between splits: 0

| Split | Flows | Attack rate |
|---|---|---|
| train | 4,130,000 | 65.49% |
| val | 590,000 | 69.33% |
| test | 1,190,000 | 62.80% |

### NF-UNSW-NB15-v2 (secondary)

*Source: `results/metrics/preprocess.json`, generated 2026-08-22T04:18:43 UTC.*

- Flows used: **2,390,275** of 2,390,275 in the source file
- Attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms (9, plus Benign)
- Features: 31 continuous (22 log-transformed), 5 categorical, 4 presence indicators, 2 port buckets
- Dropped as shortcuts: MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN
- Dropped as uninformative: DNS_QUERY_ID, ICMP_IPV4_TYPE
- Split method: contiguous by position in the capture; rows purged between splits: 40,000

| Split | Flows | Attack rate |
|---|---|---|
| train | 1,663,192 | 2.62% |
| val | 219,027 | 6.71% |
| test | 468,056 | 7.33% |

## 2. Baselines (models that cannot see topology)

### NF-ToN-IoT-v2

*Source: `results/metrics/baselines/baselines_NF-ToN-IoT-v2.json`, generated 2026-08-25T06:01:10 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).

| Model :: features | PR-AUC | F1 | FPR at 95% recall |
|---|---|---|---|
| xgboost::flow | 0.9860 ± 0.0005 (n = 3) | 0.9416 ± 0.0004 (n = 3) | 0.00284 ± 0.00006 (n = 3) |
| xgboost::flow+host | 0.9809 ± 0.0011 (n = 3) | 0.9497 ± 0.0011 (n = 3) | 0.00237 ± 0.00012 (n = 3) |

**Do not quote:** logreg::flow, logreg::flow+host, mlp::flow, mlp::flow+host, random_forest::flow, random_forest::flow+host. These were run before the reporting path was fixed and are at native prevalence, not 4%. Re-run with `python scripts/train_baselines.py`.

### NF-UNSW-NB15-v2

*Source: `results/metrics/baselines/baselines_NF-UNSW-NB15-v2.json`, generated 2026-08-17T05:48:25 UTC.*

**Do not quote:** logreg::flow, logreg::flow+host, mlp::flow, mlp::flow+host, random_forest::flow, random_forest::flow+host, xgboost::flow, xgboost::flow+host. These were run before the reporting path was fixed and are at native prevalence, not 4%. Re-run with `python scripts/train_baselines.py`.

## 3. Topology ablation on NF-ToN-IoT-v2

*Source: `results/metrics/gnn/gnn_NF-ToN-IoT-v2.json`, generated 2026-08-23T04:20:51 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).

> **SUPERSEDED.** These runs used early stopping. The final results use a fixed training budget (decision D39). Re-run `python scripts/train_gnn.py --seeds 3` and re-export before quoting anything in this section.

| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | Best epoch per seed |
|---|---|---|---|---|---|
| flow features only (channel 1) | 0.9792 ± 0.0012 (n = 3) | 0.9239 ± 0.0096 (n = 3) | 0.00387 ± 0.00030 (n = 3) | 0.9809, 0.9782, 0.9786 | 38, 28, 26 |
| neighbourhood only (channel 2) | 0.9647 ± 0.0112 (n = 3) | 0.9112 ± 0.0128 (n = 3) | 0.00563 ± 0.00056 (n = 3) | 0.9488, 0.9733, 0.9720 | 13, 16, 37 |
| both channels (full model) | 0.9887 ± 0.0030 (n = 3) | 0.9410 ± 0.0045 (n = 3) | 0.00248 ± 0.00054 (n = 3) | 0.9866, 0.9865, 0.9929 | 16, 15, 20 |

- **Topology gain** (full minus flow-features-only, PR-AUC): **+0.0094**
- Seed ranges: full 0.9865–0.9929; flow-features-only 0.9782–0.9809. **They do not overlap.**
- False-positive rate at 95% recall: 0.00387 → 0.00248 (-36%)
- Seeds per variant: 3. With so few, say whether the ranges overlap; do not use the word "significant".
- Mean share of the decision assigned to the neighbourhood channel (full model, seed 0, whole test split): 0.5052

### Attack-family head (full model)

**NOT YET MEASURED.** The attack-family head was not scored in these runs. To produce it: `python scripts/train_gnn.py --seeds 3`.
Write `[NUM?]` wherever a number from this section is needed.

## 4. The graph model against XGBoost

> **SUPERSEDED.** The graph-model figures here come from the early-stopped runs (see Section 3). Re-export after the re-run.

Same dataset, same prevalence, same reporting path.

| Model | PR-AUC (ranking) | F1 (operating point) |
|---|---|---|
| graph model (full) | 0.9887 ± 0.0030 (n = 3) | 0.9410 ± 0.0045 (n = 3) |
| xgboost::flow | 0.9860 ± 0.0005 (n = 3) | 0.9416 ± 0.0004 (n = 3) |
| xgboost::flow+host | 0.9809 ± 0.0011 (n = 3) | 0.9497 ± 0.0011 (n = 3) |

- Against the best XGBoost by PR-AUC (xgboost::flow): PR-AUC **+0.0027**
- Against the best XGBoost by F1 (xgboost::flow+host): F1 **-0.0087**
- **Both halves must be stated together.** The graph model ranks better and operates worse.

## 5. Topology ablation on NF-UNSW-NB15-v2 (the saturated dataset)

*Source: `results/metrics/gnn/gnn_NF-UNSW-NB15-v2.json`, generated 2026-08-31T04:54:19 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).
Training: fixed budget of 40 epochs, best-validation checkpoint kept.

| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | Best epoch per seed |
|---|---|---|---|---|---|
| flow features only (channel 1) | 0.9903 ± 0.0006 (n = 3) | 0.9481 ± 0.0022 (n = 3) | 0.00225 ± 0.00011 (n = 3) | 0.9897, 0.9911, 0.9900 | 35, 37, 40 |
| both channels (full model) | 0.9914 ± 0.0011 (n = 3) | 0.9542 ± 0.0037 (n = 3) | 0.00203 ± 0.00035 (n = 3) | 0.9919, 0.9925, 0.9899 | 38, 35, 33 |

- **Topology gain** (full minus flow-features-only, PR-AUC): **+0.0011**
- Seed ranges: full 0.9899–0.9925; flow-features-only 0.9897–0.9911. **They overlap.**
- False-positive rate at 95% recall: 0.00225 → 0.00203 (-10%)
- Seeds per variant: 3. With so few, say whether the ranges overlap; do not use the word "significant".
- Mean share of the decision assigned to the neighbourhood channel (full model, seed 0, whole test split): 0.1270

### Attack-family head (full model)

**NOT YET MEASURED.** The attack-family head was not scored in these runs. To produce it: `python scripts/train_gnn.py --seeds 3`.
Write `[NUM?]` wherever a number from this section is needed.

## 6. Zero-day: leave-one-attack-out

**NOT YET MEASURED.** Recall on attack families held out of training. This is the claim in the project's title and has never been run. To produce it: `python scripts/train_zeroday.py --seeds 3`.
Write `[NUM?]` wherever a number from this section is needed.

## 7. Cross-dataset transfer (no retraining)

*Source: every `results/metrics/transfer/transfer_*.json`, pooled by `scripts/summarise_transfer.py`.*

PR-AUC of a random model at this prevalence: 0.04.

| Model | Trained on → tested on | Kind | Seeds | PR-AUC | F1 | Out-of-vocabulary |
|---|---|---|---|---|---|---|
| gnn::full | ToN-IoT → ToN-IoT | in-dataset control | 1 | 0.9543 (one seed) | 0.8992 | 2.1% |
| gnn::full | UNSW-NB15 → UNSW-NB15 | in-dataset control | 1 | 0.9954 (one seed) | 0.9706 | 0.6% |
| xgboost::flow | ToN-IoT → ToN-IoT | in-dataset control | 1 | 0.8828 (one seed) | 0.7015 | 2.1% |
| xgboost::flow | UNSW-NB15 → UNSW-NB15 | in-dataset control | 1 | 0.9848 (one seed) | 0.9524 | 0.6% |
| gnn::full | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0914 ± 0.0649 | 0.1114 | 3.6% |
| gnn::full | UNSW-NB15 → ToN-IoT | **cross-dataset** | 1 | 0.1008 (one seed) | 0.1713 | 54.5% |
| xgboost::flow | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0817 ± 0.0093 | 0.0483 | 3.6% |
| xgboost::flow | UNSW-NB15 → ToN-IoT | **cross-dataset** | 3 | 0.0386 ± 0.0006 | 0.0062 | 54.5% |

- A direction with out-of-vocabulary above 50% is confounded: the model was given a degraded input as well as a different network. Rely on the other direction.
- A row marked "one seed" has no standard deviation. Do not attach one.

## 8. Explainability (over the evidence packs)

*Source: `results/metrics/evidence/evidence_NF-ToN-IoT-v2_summary.json`, generated 2026-10-03T05:27:09 UTC.*

60 evidence packs: 51 true attacks and 9 false positives (benign flows the detector flagged). Checkpoint: NF-ToN-IoT-v2_full_seed0.pt. The packs are a stratified sample of *flagged* flows, not of traffic.

### Share of each decision assigned to the neighbourhood channel

| Group | Packs | Median | 25th–75th percentile | Mean |
|---|---|---|---|---|
| all packs | 60 | 0.5782 | 0.3011–0.7714 | 0.5511 |
| true attacks | 51 | 0.5735 | 0.2894–0.7740 | 0.5480 |
| false positives | 9 | 0.6683 | 0.3224–0.7114 | 0.5691 |
| backdoor | 5 | 0.1317 | 0.1317–0.1318 | 0.1449 |
| ddos | 6 | 0.4556 | 0.4339–0.5433 | 0.4670 |
| dos | 5 | 0.9115 | 0.8759–0.9152 | 0.8927 |
| injection | 7 | 0.7047 | 0.6340–0.7172 | 0.6685 |
| mitm | 5 | 0.8591 | 0.8497–0.8609 | 0.8557 |
| password | 7 | 0.4509 | 0.4003–0.5092 | 0.4456 |
| ransomware | 1 | 0.1912 | 0.1912–0.1912 | 0.1912 |
| scanning | 8 | 0.2245 | 0.2042–0.2628 | 0.2913 |
| xss | 7 | 0.7653 | 0.7647–0.7740 | 0.7654 |

### Features most often among the top 5 attributed

| Feature | Share of packs | Times it raised suspicion | Times it lowered suspicion |
|---|---|---|---|
| L4_DST_PORT | 98% | 62 | 10 |
| PROTOCOL | 70% | 31 | 11 |
| L7_PROTO | 42% | 12 | 13 |
| TCP_WIN_MAX_IN | 32% | 2 | 17 |
| SHORTEST_FLOW_PKT | 23% | 1 | 13 |
| IN_BYTES | 18% | 10 | 1 |
| CLIENT_TCP_FLAGS | 17% | 5 | 5 |
| L4_SRC_PORT | 15% | 9 | 2 |
| DURATION_IN | 15% | 0 | 9 |
| NUM_PKTS_UP_TO_128_BYTES | 15% | 0 | 9 |

A feature and its encoded variants (a port and its port bucket) count as one feature per pack but as separate entries in the last two columns, so those can exceed the pack count.

### The attack family named in the pack

- Correct for 0.8235 of 51 true attacks in the sample
- False positives were named as: {'password': 5, 'scanning': 2, 'dos': 2}
- Ambiguous packs (confidence below 0.9): 3

### Influence of single neighbouring flows

**NOT YET MEASURED.** These packs were made before neighbour occlusion was recorded. To produce it: `python scripts/make_evidence.py`.
Write `[NUM?]` wherever a number from this section is needed.

## 9. Language-model comparison

Evidence packs: 60 (9 false positives). By true family: {'Benign': 9, 'backdoor': 5, 'ddos': 6, 'dos': 5, 'injection': 7, 'mitm': 5, 'password': 7, 'ransomware': 1, 'scanning': 8, 'xss': 7}.

**NOT YET MEASURED.** No language model has been run; only the template control exists. To produce it: `python scripts/run_llm_study.py`.
Write `[NUM?]` wherever a number from this section is needed.

*Source: every `results/reports/reports_*.json`, pooled by `scripts/summarise_llm_study.py`.*

| Model | Prompt | Reports | Failed | Groundedness | Fabricated-address reports | Groundedness on false positives | Names predicted class | Substitutes class | Attributed features cited | Cites none | Latency p50 / p95 (s) | Charged (USD) | List price (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 60 | 0 | 1.0000 | 0 | 1.0000 (n = 9) | 1.0000 | 0.0000 | 0.4783 | 0.0000 | 0.00 / 0.00 | 0.0000 | unknown |
| stub/template-v1 | v2 | 60 | 0 | 1.0000 | 0 | 1.0000 (n = 9) | 1.0000 | 0.0000 | 0.4783 | 0.0000 | 0.00 / 0.00 | 0.0000 | unknown |

- The stub is a template with no language model: the control arm.
- Groundedness measures "invented nothing", not "reasoned correctly".
- The class and feature checks are lexical: they match names and phrases, are blind to negation, and are generous about what counts as citing a feature.
- Latency is comparable within a hosting arm only.
- A cost of "unknown" is not zero.


## 10. Replay (the demonstration harness)

**NOT YET MEASURED.** Replay metrics at full scale. To produce it: `python scripts/replay.py`.
Write `[NUM?]` wherever a number from this section is needed.

## 11. Figures

- **`results/figures/results_01_topology_ablation.png`** — Topology ablation on NF-ToN-IoT-v2. Each dot is one seed (n = 3 per variant); the tick is the mean. The variants share architecture, optimiser, depth and width, so the gap between the first and third is the contribution of message passing. Trained with early stopping (superseded by the fixed-budget re-run).
- **`results/figures/results_02_gnn_vs_xgboost.png`** — The graph model against XGBoost on NF-ToN-IoT-v2, both reported at 4% prevalence. Left: ranking quality. Right: the operating point reached at the validation-chosen threshold. Each dot is one seed; the tick is the mean. The two panels have separate scales and must be read together: the graph model ranks better and operates worse. Graph model trained with early stopping (superseded by the fixed-budget re-run).
- **`results/figures/results_03_topology_gain_by_dataset.png`** — Gain in PR-AUC from message passing (full model minus the flow-features-only variant, matched by seed) on each dataset. Each dot is one seed; the tick is the mean; the line marks no gain. The gain is separated from zero on NF-ToN-IoT-v2 and not on NF-UNSW-NB15-v2, where a linear model already scores above 0.99.
- *zeroday: not drawn yet (results/metrics/zeroday/zeroday_NF-ToN-IoT-v2.json does not exist yet)*
- **`results/figures/results_05_cross_dataset_transfer.png`** — Cross-dataset transfer with no retraining. For each model and direction, the blue dot is the in-dataset control and the orange dot is the same model scored on the other dataset (bar: one standard deviation over seeds, where more than one was run). Every control is healthy and every transfer falls to the level of a random model. † More than half of this direction's categorical values fall outside the source vocabulary, so it measures a degraded input as well as a different network.
- **`results/figures/results_06_channel_weights.png`** — How the fusion layer divides each decision between a flow's own features and its neighbourhood, over 60 flagged flows. Dot: median; bar: interquartile range; n: evidence packs in the group. The sample is stratified by family, so the groups are small and the ordering is indicative.
- **`results/figures/results_07_top_features.png`** — The features that carry the detections: for each, the share of 60 evidence packs in which integrated gradients ranks it among the 5 most influential. A column and its encoded variants are counted once per pack.
- *confusion: not drawn yet (the results predate scoring of the attack-family head (re-run train_gnn.py))*
- *llm: not drawn yet (no language-model runs yet -- only the template control exists)*

## 12. Process

- Automated tests in the repository: 242
- Counts of logged decisions and defects are kept in the project notes (Decision Register, Defect Register), not in this file.

