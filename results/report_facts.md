# Report facts

*Exported 2026-10-10T09:42:06 UTC by `scripts/export_report_facts.py`. Every number below was read from a results file at export time. None was typed in.*

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

*Source: `results/metrics/baselines/baselines_NF-UNSW-NB15-v2.json`, generated 2026-10-10T09:42:05 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).

| Model :: features | PR-AUC | F1 | FPR at 95% recall |
|---|---|---|---|
| logreg::flow | 0.9870 ± 0.0003 (n = 3) | 0.9422 ± 0.0031 (n = 3) | 0.00273 ± 0.00003 (n = 3) |
| logreg::flow+host | 0.9887 ± 0.0002 (n = 3) | 0.9492 ± 0.0014 (n = 3) | 0.00232 ± 0.00003 (n = 3) |
| mlp::flow | 0.9890 ± 0.0007 (n = 3) | 0.9424 ± 0.0014 (n = 3) | 0.00270 ± 0.00006 (n = 3) |
| mlp::flow+host | 0.9888 ± 0.0004 (n = 3) | 0.9479 ± 0.0010 (n = 3) | 0.00230 ± 0.00016 (n = 3) |
| random_forest::flow | 0.9899 ± 0.0001 (n = 3) | 0.9498 ± 0.0006 (n = 3) | 0.00205 ± 0.00002 (n = 3) |
| random_forest::flow+host | 0.9743 ± 0.0009 (n = 3) | 0.9251 ± 0.0022 (n = 3) | 0.00438 ± 0.00024 (n = 3) |

**Do not quote:** xgboost::flow, xgboost::flow+host. These were run before the reporting path was fixed and are at native prevalence, not 4%. Re-run with `python scripts/train_baselines.py`.

## 3. Topology ablation on NF-ToN-IoT-v2

*Source: `results/metrics/gnn/gnn_NF-ToN-IoT-v2.json`, generated 2026-10-10T04:53:38 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).
Training: fixed budget of 40 epochs, best-validation checkpoint kept.

| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | Best epoch per seed |
|---|---|---|---|---|---|
| flow features only (channel 1) | 0.9804 ± 0.0009 (n = 3) | 0.9316 ± 0.0086 (n = 3) | 0.00396 ± 0.00061 (n = 3) | 0.9809, 0.9792, 0.9811 | 32, 40, 40 |
| neighbourhood only (channel 2) | 0.9729 ± 0.0019 (n = 3) | 0.9183 ± 0.0080 (n = 3) | 0.00440 ± 0.00010 (n = 3) | 0.9704, 0.9733, 0.9751 | 26, 30, 39 |
| both channels (full model) | 0.9922 ± 0.0011 (n = 3) | 0.9503 ± 0.0001 (n = 3) | 0.00226 ± 0.00012 (n = 3) | 0.9907, 0.9932, 0.9928 | 31, 23, 40 |

- **Topology gain** (full minus flow-features-only, PR-AUC): **+0.0118**
- Seed ranges: full 0.9907–0.9932; flow-features-only 0.9792–0.9811. **They do not overlap.**
- False-positive rate at 95% recall: 0.00396 → 0.00226 (-43%)
- Seeds per variant: 3. With so few, say whether the ranges overlap; do not use the word "significant".
- Mean share of the decision assigned to the neighbourhood channel (full model, seed 0, whole test split): 0.4645

### Attack-family head (full model)

Scored on the whole test split at native prevalence.

- Macro-F1 over families: 0.8629 ± 0.0245 (n = 3)
- Accuracy over all flows: 0.9808 ± 0.0012 (n = 3)
- Correct family among true attacks: 0.9740 ± 0.0010 (n = 3)
- **Correct family on true-positive detections** (the flows that become evidence packs): 0.9786 ± 0.0016 (n = 3)

## 4. The graph model against XGBoost

Same dataset, same prevalence, same reporting path.

| Model | PR-AUC (ranking) | F1 (operating point) |
|---|---|---|
| graph model (full) | 0.9922 ± 0.0011 (n = 3) | 0.9503 ± 0.0001 (n = 3) |
| xgboost::flow | 0.9860 ± 0.0005 (n = 3) | 0.9416 ± 0.0004 (n = 3) |
| xgboost::flow+host | 0.9809 ± 0.0011 (n = 3) | 0.9497 ± 0.0011 (n = 3) |

- Against the best XGBoost by PR-AUC (xgboost::flow): PR-AUC **+0.0062**
- Against the best XGBoost by F1 (xgboost::flow+host): F1 **+0.0006**
- **Both halves must be stated together.** State the sign of each difference as given above.

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

*Source: `results/metrics/zeroday/zeroday_NF-ToN-IoT-v2.json`, generated 2026-10-10T09:31:41 UTC.*

Each family was removed from training and validation entirely; recall is on that family in the test split, at 4% prevalence. Model variant: full.

| Held-out family | Test flows | Recall | PR-AUC | Seeds | Thin? | The family head calls it |
|---|---|---|---|---|---|---|
| scanning | 241,497 | 0.0005 ± 0.0002 (n = 3) | 0.0415 ± 0.0049 (n = 3) | 3 | no | Benign 98%, password 2% |
| xss | 162,337 | 0.2517 ± 0.0815 (n = 3) | 0.8965 ± 0.0207 (n = 3) | 3 | no | injection 57%, Benign 23% |
| password | 107,086 | 0.4667 ± 0.1724 (n = 3) | 0.8280 ± 0.0253 (n = 3) | 3 | no | injection 51%, Benign 32% |
| ddos | 92,066 | 0.7293 ± 0.0166 (n = 3) | 0.9669 ± 0.0056 (n = 3) | 3 | no | injection 63%, password 19% |
| dos | 77,328 | 0.0616 ± 0.0368 (n = 3) | 0.7512 ± 0.0317 (n = 3) | 3 | no | Benign 41%, xss 40% |
| injection | 61,648 | 0.8948 ± 0.0672 (n = 3) | 0.9853 ± 0.0075 (n = 3) | 3 | no | xss 65%, password 23% |
| backdoor | 4,498 | 0.0006 ± 0.0008 (n = 3) | 0.3130 ± 0.0954 (n = 3) | 3 | yes | Benign 99%, dos 0% |
| mitm | 897 | 0.6639 ± 0.1800 (n = 3) | 0.9491 ± 0.0382 (n = 3) | 3 | yes | dos 83%, xss 9% |

"Thin" means fewer than 20,000 held-out flows in the test split: quote that recall with its count.

- **Not measurable: ransomware** — only 3 held-out flows reach the test split (need 500); 2,554 were removed from train, so this family is concentrated in one split and cannot be held out here

Do not report the mean across families. Which families are caught and which are not is the finding.

## 7. Cross-dataset transfer (no retraining)

*Source: every `results/metrics/transfer/transfer_*.json`, pooled by `scripts/summarise_transfer.py`.*

PR-AUC of a random model at this prevalence: 0.04.

| Model | Trained on → tested on | Kind | Seeds | PR-AUC | F1 | Out-of-vocabulary |
|---|---|---|---|---|---|---|
| gnn::full | ToN-IoT → ToN-IoT | in-dataset control | 1 | 0.9736 (one seed) | 0.9178 | 2.1% |
| gnn::full | UNSW-NB15 → UNSW-NB15 | in-dataset control | 1 | 0.9954 (one seed) | 0.9706 | 0.6% |
| xgboost::flow | ToN-IoT → ToN-IoT | in-dataset control | 1 | 0.8828 (one seed) | 0.7015 | 2.1% |
| xgboost::flow | UNSW-NB15 → UNSW-NB15 | in-dataset control | 1 | 0.9848 (one seed) | 0.9524 | 0.6% |
| gnn::full | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0391 ± 0.0078 | 0.0305 | 3.6% |
| gnn::full | UNSW-NB15 → ToN-IoT | **cross-dataset** | 1 | 0.1008 (one seed) | 0.1713 | 54.5% |
| xgboost::flow | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0817 ± 0.0093 | 0.0483 | 3.6% |
| xgboost::flow | UNSW-NB15 → ToN-IoT | **cross-dataset** | 3 | 0.0386 ± 0.0006 | 0.0062 | 54.5% |

- A direction with out-of-vocabulary above 50% is confounded: the model was given a degraded input as well as a different network. Rely on the other direction.
- A row marked "one seed" has no standard deviation. Do not attach one.

## 8. Explainability (over the evidence packs)

*Source: `results/metrics/evidence/evidence_NF-ToN-IoT-v2_summary.json`, generated 2026-10-10T09:42:05 UTC.*

200 evidence packs: 170 true attacks and 30 false positives (benign flows the detector flagged). Checkpoint: NF-ToN-IoT-v2_full_seed0.pt. The packs are a stratified sample of *flagged* flows, not of traffic.

### Share of each decision assigned to the neighbourhood channel

| Group | Packs | Median | 25th–75th percentile | Mean |
|---|---|---|---|---|
| all packs | 200 | 0.6482 | 0.2469–0.8024 | 0.5452 |
| true attacks | 170 | 0.5729 | 0.2334–0.8027 | 0.5198 |
| false positives | 30 | 0.7125 | 0.6749–0.7861 | 0.6892 |
| backdoor | 22 | 0.0839 | 0.0825–0.0903 | 0.0863 |
| ddos | 21 | 0.4342 | 0.1526–0.4823 | 0.3784 |
| dos | 21 | 0.8109 | 0.7947–0.9231 | 0.8068 |
| injection | 21 | 0.7462 | 0.6971–0.7891 | 0.7055 |
| mitm | 21 | 0.8076 | 0.8034–0.8130 | 0.8292 |
| password | 22 | 0.2921 | 0.2407–0.4699 | 0.3650 |
| scanning | 21 | 0.2430 | 0.2174–0.2947 | 0.2583 |
| xss | 21 | 0.7260 | 0.6600–0.8388 | 0.7570 |

### Features most often among the top 5 attributed

| Feature | Share of packs | Times it raised suspicion | Times it lowered suspicion |
|---|---|---|---|
| L4_DST_PORT | 86% | 183 | 20 |
| PROTOCOL | 72% | 104 | 39 |
| L7_PROTO | 50% | 43 | 57 |
| TCP_WIN_MAX_IN | 28% | 9 | 48 |
| SHORTEST_FLOW_PKT | 28% | 4 | 51 |
| DNS_QUERY_TYPE | 25% | 14 | 37 |
| L4_SRC_PORT | 16% | 25 | 11 |
| CLIENT_TCP_FLAGS | 16% | 19 | 13 |
| IN_BYTES | 16% | 30 | 1 |
| NUM_PKTS_UP_TO_128_BYTES | 13% | 0 | 26 |

A feature and its encoded variants (a port and its port bucket) count as one feature per pack but as separate entries in the last two columns, so those can exceed the pack count.

### The attack family named in the pack

- Correct for 0.8588 of 170 true attacks in the sample
- False positives were named as: {'scanning': 21, 'password': 5, 'xss': 2, 'injection': 1, 'ddos': 1}
- Ambiguous packs (confidence below 0.9): 22

### Influence of single neighbouring flows

- Up to 50 neighbouring flows were removed one at a time per detection (median tested: 50; median sharing a host: 5554)
- Largest change in the detection score from removing one neighbour: median 1.00e-05, 95th percentile 2.81e-04, maximum 1.54e-02
- Packs with any neighbour above the 0.01 threshold: 1 of 200

## 9. Language-model comparison

Evidence packs: 200 (30 false positives). By true family: {'Benign': 30, 'backdoor': 22, 'ddos': 21, 'dos': 21, 'injection': 21, 'mitm': 21, 'password': 22, 'scanning': 21, 'xss': 21}.

**NOT YET MEASURED.** No language model has been run; only the template control exists. To produce it: `python scripts/run_llm_study.py`.
Write `[NUM?]` wherever a number from this section is needed.

*Source: every `results/reports/reports_*.json`, pooled by `scripts/summarise_llm_study.py`.*

| Model | Prompt | Reports | Failed | Groundedness | Fabricated-address reports | Groundedness on false positives | Names predicted class | Substitutes class | Attributed features cited | Cites none | Latency p50 / p95 (s) | Charged (USD) | List price (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.5245 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| stub/template-v1 | v2 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.5245 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |

- The stub is a template with no language model: the control arm.
- Groundedness measures "invented nothing", not "reasoned correctly".
- The class and feature checks are lexical: they match names and phrases, are blind to negation, and are generous about what counts as citing a feature.
- Latency is comparable within a hosting arm only.
- A cost of "unknown" is not zero.


## 10. Replay (the demonstration harness)

*Source: `results/metrics/replay/replay_NF-ToN-IoT-v2.json`, generated 2026-10-10T04:57:23 UTC.*

Offline replay of the held-out split. Not a real-time measurement.

- Windows replayed: 100 (1,000,000 flows)
- Alerts: 621,320 (621.32 per 1,000 flows)
- Precision 0.9987, recall 0.9478
- Windows in which the language model would be invoked: 93.0%
- Scoring time per window: median 18.06 ms, 95th percentile 26.96 ms

## 11. Figures

- **`results/figures/results_01_topology_ablation.png`** — Topology ablation on NF-ToN-IoT-v2. Each dot is one seed (n = 3 per variant); the tick is the mean. The variants share architecture, optimiser, depth and width, so the gap between the first and third is the contribution of message passing. Trained with fixed training budget.
- **`results/figures/results_02_gnn_vs_xgboost.png`** — The graph model against XGBoost on NF-ToN-IoT-v2, both reported at 4% prevalence. Left: ranking quality. Right: the operating point reached at the validation-chosen threshold. Each dot is one seed; the tick is the mean. The two panels have separate scales and must be read together: the graph model ranks better and operates worse. Graph model trained with fixed training budget.
- **`results/figures/results_03_topology_gain_by_dataset.png`** — Gain in PR-AUC from message passing (full model minus the flow-features-only variant, matched by seed) on each dataset. Each dot is one seed; the tick is the mean; the line marks no gain. The gain is separated from zero on NF-ToN-IoT-v2 and not on NF-UNSW-NB15-v2, where a linear model already scores above 0.99.
- **`results/figures/results_04_zero_day_recall.png`** — Leave-one-attack-out on NF-ToN-IoT-v2: recall on each attack family when that family was removed entirely from training and validation. Each dot is one seed; the tick is the mean. Families are ordered by the number of held-out flows in the test split, shown in brackets. Hollow dots mark families with fewer than 20,000 test flows, whose recall is correspondingly uncertain. Not measurable on this dataset: ransomware.
- **`results/figures/results_05_cross_dataset_transfer.png`** — Cross-dataset transfer with no retraining. For each model and direction, the blue dot is the in-dataset control and the orange dot is the same model scored on the other dataset (bar: one standard deviation over seeds, where more than one was run). Every control is healthy and every transfer falls to the level of a random model. † More than half of this direction's categorical values fall outside the source vocabulary, so it measures a degraded input as well as a different network.
- **`results/figures/results_06_channel_weights.png`** — How the fusion layer divides each decision between a flow's own features and its neighbourhood, over 200 flagged flows. Dot: median; bar: interquartile range; n: evidence packs in the group. The sample is stratified by family, so the groups are small and the ordering is indicative.
- **`results/figures/results_07_top_features.png`** — The features that carry the detections: for each, the share of 200 evidence packs in which integrated gradients ranks it among the 5 most influential. A column and its encoded variants are counted once per pack.
- **`results/figures/results_08_attack_family_confusion.png`** — The attack-family head on the NF-ToN-IoT-v2 test split, seed 0: each row shows how flows of one true family were named. Cells below 0.05 are left unlabelled. Macro-F1 0.846; on true-positive detections the named family is correct for 98.0%.
- *llm: not drawn yet (no language-model runs yet -- only the template control exists)*

## 12. Process

- Automated tests in the repository: 257
- Counts of logged decisions and defects are kept in the project notes (Decision Register, Defect Register), not in this file.

