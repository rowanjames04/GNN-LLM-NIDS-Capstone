# Report facts

*Exported 2026-10-17T17:04:04 UTC by `scripts/export_report_facts.py`. Every number below was read from a results file at export time. None was typed in.*

**For whoever drafts from this file (human or model):**

1. Use these numbers exactly as written. Do not round further, recompute, or combine them.
2. A section marked **NOT YET MEASURED** has no numbers. Write `[NUM?]`.
3. A section marked **SUPERSEDED** has numbers that will be replaced. Do not quote them in final text.
4. `0.9887 ± 0.0030 (n = 3)` means mean ± standard deviation over 3 seeds.
5. Every PR-AUC is at the stated prevalence. A random model scores the prevalence itself.

## 1. Datasets and preprocessing

### NF-ToN-IoT-v2 (primary)

*Source: `results/metrics/preprocess_NF-ToN-IoT-v2.json`, generated 2026-10-17T06:05:58 UTC.*

- Flows used: **5,910,000** of 16,940,496 in the source file (capped for memory; every attack family is represented)
- Attack families: backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss (9, plus Benign)
- Features: 24 continuous (13 log-transformed), 5 categorical, 4 presence indicators, 2 port buckets
- Dropped as shortcuts: MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN
- Dropped as uninformative: DNS_QUERY_ID, ICMP_IPV4_TYPE
- Split method: whole 10,000-flow windows assigned to splits at random (the capture is ordered by scenario, not by time); rows purged between splits: 0

| Split | Flows | Attack rate |
|---|---|---|
| train | 4,130,000 | 65.49% |
| val | 590,000 | 69.33% |
| test | 1,190,000 | 62.80% |

### NF-UNSW-NB15-v2 (secondary)

*Source: `results/metrics/preprocess.json`, generated 2026-10-17T06:06:07 UTC.*

- Flows used: **2,390,275** of 2,390,275 in the source file
- Attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms (9, plus Benign)
- Features: 24 continuous (16 log-transformed), 5 categorical, 4 presence indicators, 2 port buckets
- Dropped as shortcuts: MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN
- Dropped as uninformative: DNS_QUERY_ID, ICMP_IPV4_TYPE
- Split method: contiguous by position in the capture; rows purged between splits: 40,000

| Split | Flows | Attack rate |
|---|---|---|
| train | 1,663,192 | 2.62% |
| val | 219,027 | 6.71% |
| test | 468,056 | 7.33% |

## 2a. Data quality, duplicates, and the artefact comparison (D47, D48)

### NF-ToN-IoT-v2

*Source: `results/metrics/data_quality_NF-ToN-IoT-v2.json`, generated 2026-10-17T06:07:27 UTC.*

- Flow duration wrapped (32-bit counter overflow): 849,865 rows, 14.38%. Most affected: ddos 72.5%, injection 26.7%, password 22.9%, ransomware 10.8%.
- In every row that is not wrapped the duration is exactly 0, so the three duration columns carry no genuine timing. They and the four throughput columns computed from them were removed from the model's inputs (D47).
- Test flows identical to some training flow (all measured columns): 14.7%.
- Flows whose identical twin carries another label: 0.
- Flows touching an address outside the testbed: 7.14% of rows; attack rate 0.6438 when true against 0.654 when false.
- Flows touching 192.168.1.1: 13.23% of rows; attack rate 0.9293 when true against 0.6112 when false.
- Test flows whose model input matches a training flow (the D48 "novel only" subset excludes these): 24.1%.

### NF-UNSW-NB15-v2

*Source: `results/metrics/data_quality_NF-UNSW-NB15-v2.json`, generated 2026-10-17T06:07:44 UTC.*

- Flow duration wrapped (32-bit counter overflow): 144,204 rows, 6.03%. Most affected: Worms 17.1%, Exploits 16.2%, DoS 12.0%, Fuzzers 10.2%.
- In every row that is not wrapped the duration is exactly 0, so the three duration columns carry no genuine timing. They and the four throughput columns computed from them were removed from the model's inputs (D47).
- Test flows identical to some training flow (all measured columns): 22.3%.
- Flows whose identical twin carries another label: 14,503.
- Flows touching an address outside the testbed: 5.27% of rows; attack rate 0.7544 when true against 0.0 when false.
- Test flows whose model input matches a training flow (the D48 "novel only" subset excludes these): 23.8%.

### Scores on novel test flows only (D48)

PR-AUC at 4% prevalence (floor 0.04). "All" is the headline; "novel only" drops test flows whose model input matches a training flow.

| Dataset | Model | All test flows | Novel only |
|---|---|---|---|
| NF-ToN-IoT-v2 | GNN channel1_only | 0.9518 ± 0.0140 (n = 10) | 0.9294 ± 0.0172 (n = 10) |
| NF-ToN-IoT-v2 | GNN channel2_only | 0.9694 ± 0.0062 (n = 3) | 0.9665 ± 0.0088 (n = 3) |
| NF-ToN-IoT-v2 | GNN full | 0.9883 ± 0.0031 (n = 10) | 0.9833 ± 0.0052 (n = 10) |
| NF-ToN-IoT-v2 | logreg::flow | 0.9579 ± 0.0039 (n = 3) | 0.9372 ± 0.0062 (n = 3) |
| NF-ToN-IoT-v2 | logreg::flow+host | 0.9824 ± 0.0026 (n = 3) | 0.9748 ± 0.0044 (n = 3) |
| NF-ToN-IoT-v2 | mlp::flow | 0.9696 ± 0.0017 (n = 3) | 0.9554 ± 0.0027 (n = 3) |
| NF-ToN-IoT-v2 | mlp::flow+host | 0.9853 ± 0.0048 (n = 3) | 0.9813 ± 0.0060 (n = 3) |
| NF-ToN-IoT-v2 | random_forest::flow | 0.9537 ± 0.0017 (n = 3) | 0.9345 ± 0.0045 (n = 3) |
| NF-ToN-IoT-v2 | random_forest::flow+host | 0.9703 ± 0.0017 (n = 3) | 0.9579 ± 0.0024 (n = 3) |
| NF-ToN-IoT-v2 | xgboost::flow | 0.9743 ± 0.0004 (n = 3) | 0.9608 ± 0.0007 (n = 3) |
| NF-ToN-IoT-v2 | xgboost::flow+host | 0.9793 ± 0.0012 (n = 3) | 0.9718 ± 0.0016 (n = 3) |
| NF-UNSW-NB15-v2 | GNN channel1_only | 0.9884 ± 0.0002 (n = 3) | 0.9790 ± 0.0003 (n = 3) |
| NF-UNSW-NB15-v2 | GNN channel2_only | 0.9570 ± 0.0013 (n = 3) | 0.9225 ± 0.0035 (n = 3) |
| NF-UNSW-NB15-v2 | GNN full | 0.9892 ± 0.0000 (n = 3) | 0.9806 ± 0.0001 (n = 3) |
| NF-UNSW-NB15-v2 | logreg::flow | 0.9839 ± 0.0005 (n = 3) | 0.9697 ± 0.0007 (n = 3) |
| NF-UNSW-NB15-v2 | logreg::flow+host | 0.9839 ± 0.0005 (n = 3) | 0.9688 ± 0.0009 (n = 3) |
| NF-UNSW-NB15-v2 | mlp::flow | 0.9875 ± 0.0002 (n = 3) | 0.9766 ± 0.0005 (n = 3) |
| NF-UNSW-NB15-v2 | mlp::flow+host | 0.9866 ± 0.0007 (n = 3) | 0.9749 ± 0.0009 (n = 3) |
| NF-UNSW-NB15-v2 | random_forest::flow | 0.9850 ± 0.0014 (n = 3) | 0.9744 ± 0.0003 (n = 3) |
| NF-UNSW-NB15-v2 | random_forest::flow+host | 0.9809 ± 0.0014 (n = 3) | 0.9771 ± 0.0004 (n = 3) |
| NF-UNSW-NB15-v2 | xgboost::flow | 0.9890 ± 0.0001 (n = 3) | 0.9789 ± 0.0002 (n = 3) |
| NF-UNSW-NB15-v2 | xgboost::flow+host | 0.9901 ± 0.0002 (n = 3) | 0.9811 ± 0.0003 (n = 3) |

### What the duration artefact was worth (D47)

"With" is the archived run (`results/superseded/2026-10-10_duration_artefact/`), which saw the seven artefact columns; "without" is the current run. The rows are identical and in the same order, so the feature list is the only difference on NF-ToN-IoT-v2. On NF-UNSW-NB15-v2 the earlier preprocessing also lacked the outlier clip, so that comparison has two differences.

| Dataset | Model | PR-AUC with the artefact | PR-AUC without | Difference |
|---|---|---|---|---|
| NF-ToN-IoT-v2 | GNN channel1_only | 0.9804 | 0.9518 | -0.0286 |
| NF-ToN-IoT-v2 | GNN channel2_only | 0.9729 | 0.9694 | -0.0036 |
| NF-ToN-IoT-v2 | GNN full | 0.9922 | 0.9883 | -0.0039 |
| NF-ToN-IoT-v2 | logreg::flow | 0.9748 | 0.9579 | -0.0169 |
| NF-ToN-IoT-v2 | logreg::flow+host | 0.9833 | 0.9824 | -0.0009 |
| NF-ToN-IoT-v2 | mlp::flow | 0.9832 | 0.9696 | -0.0136 |
| NF-ToN-IoT-v2 | mlp::flow+host | 0.9833 | 0.9853 | +0.0020 |
| NF-ToN-IoT-v2 | random_forest::flow | 0.9710 | 0.9537 | -0.0173 |
| NF-ToN-IoT-v2 | random_forest::flow+host | 0.9753 | 0.9703 | -0.0050 |
| NF-ToN-IoT-v2 | xgboost::flow | 0.9860 | 0.9743 | -0.0117 |
| NF-ToN-IoT-v2 | xgboost::flow+host | 0.9809 | 0.9793 | -0.0015 |
| NF-UNSW-NB15-v2 | GNN channel1_only | 0.9903 | 0.9884 | -0.0019 |
| NF-UNSW-NB15-v2 | GNN full | 0.9914 | 0.9892 | -0.0022 |
| NF-UNSW-NB15-v2 | logreg::flow | 0.9870 | 0.9839 | -0.0031 |
| NF-UNSW-NB15-v2 | logreg::flow+host | 0.9887 | 0.9839 | -0.0049 |
| NF-UNSW-NB15-v2 | mlp::flow | 0.9890 | 0.9875 | -0.0015 |
| NF-UNSW-NB15-v2 | mlp::flow+host | 0.9888 | 0.9866 | -0.0022 |
| NF-UNSW-NB15-v2 | random_forest::flow | 0.9899 | 0.9850 | -0.0049 |
| NF-UNSW-NB15-v2 | random_forest::flow+host | 0.9743 | 0.9809 | +0.0065 |

Leave-one-attack-out recall on NF-ToN-IoT-v2, with and without the artefact:

| Held-out family | Recall with | Recall without | Difference |
|---|---|---|---|
| scanning | 0.0005 | 0.0007 | +0.0002 |
| xss | 0.2517 | 0.2447 | -0.0070 |
| password | 0.4667 | 0.5626 | +0.0959 |
| ddos | 0.7293 | 0.7910 | +0.0618 |
| dos | 0.0616 | 0.0431 | -0.0185 |
| injection | 0.8948 | 0.8078 | -0.0870 |
| backdoor | 0.0006 | 0.0000 | -0.0006 |
| mitm | 0.6639 | 0.6454 | -0.0185 |

## 2. Baselines (models that cannot see topology)

### NF-ToN-IoT-v2

*Source: `results/metrics/baselines/baselines_NF-ToN-IoT-v2.json`, generated 2026-10-17T06:41:02 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).

| Model :: features | PR-AUC | F1 | FPR at 95% recall |
|---|---|---|---|
| logreg::flow | 0.9579 ± 0.0039 (n = 3) | 0.8442 ± 0.0136 (n = 3) | 0.00977 ± 0.00141 (n = 3) |
| logreg::flow+host | 0.9824 ± 0.0026 (n = 3) | 0.9328 ± 0.0031 (n = 3) | 0.00367 ± 0.00029 (n = 3) |
| mlp::flow | 0.9696 ± 0.0017 (n = 3) | 0.8386 ± 0.0097 (n = 3) | 0.00819 ± 0.00041 (n = 3) |
| mlp::flow+host | 0.9853 ± 0.0049 (n = 3) | 0.9450 ± 0.0026 (n = 3) | 0.00219 ± 0.00017 (n = 3) |
| random_forest::flow | 0.9537 ± 0.0017 (n = 3) | 0.8341 ± 0.0018 (n = 3) | 0.00976 ± 0.00001 (n = 3) |
| random_forest::flow+host | 0.9703 ± 0.0017 (n = 3) | 0.8860 ± 0.0168 (n = 3) | 0.00314 ± 0.00036 (n = 3) |
| xgboost::flow | 0.9743 ± 0.0004 (n = 3) | 0.8936 ± 0.0006 (n = 3) | 0.00763 ± 0.00002 (n = 3) |
| xgboost::flow+host | 0.9793 ± 0.0012 (n = 3) | 0.9463 ± 0.0018 (n = 3) | 0.00272 ± 0.00004 (n = 3) |

### NF-UNSW-NB15-v2

*Source: `results/metrics/baselines/baselines_NF-UNSW-NB15-v2.json`, generated 2026-10-17T06:54:04 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).

| Model :: features | PR-AUC | F1 | FPR at 95% recall |
|---|---|---|---|
| logreg::flow | 0.9839 ± 0.0005 (n = 3) | 0.9388 ± 0.0043 (n = 3) | 0.00311 ± 0.00032 (n = 3) |
| logreg::flow+host | 0.9839 ± 0.0005 (n = 3) | 0.9452 ± 0.0025 (n = 3) | 0.00280 ± 0.00017 (n = 3) |
| mlp::flow | 0.9875 ± 0.0002 (n = 3) | 0.9441 ± 0.0022 (n = 3) | 0.00260 ± 0.00005 (n = 3) |
| mlp::flow+host | 0.9866 ± 0.0006 (n = 3) | 0.9451 ± 0.0012 (n = 3) | 0.00263 ± 0.00014 (n = 3) |
| random_forest::flow | 0.9850 ± 0.0014 (n = 3) | 0.9468 ± 0.0005 (n = 3) | 0.00238 ± 0.00006 (n = 3) |
| random_forest::flow+host | 0.9809 ± 0.0014 (n = 3) | 0.9298 ± 0.0025 (n = 3) | 0.00413 ± 0.00040 (n = 3) |
| xgboost::flow | 0.9890 ± 0.0001 (n = 3) | 0.9544 ± 0.0002 (n = 3) | 0.00197 ± 0.00001 (n = 3) |
| xgboost::flow+host | 0.9901 ± 0.0002 (n = 3) | 0.9503 ± 0.0012 (n = 3) | 0.00202 ± 0.00011 (n = 3) |

## 3. Topology ablation on NF-ToN-IoT-v2

*Source: `results/metrics/gnn/gnn_NF-ToN-IoT-v2.json`, generated 2026-10-17T15:21:54 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).
Training: fixed budget of 40 epochs, best-validation checkpoint kept.

| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | Best epoch per seed |
|---|---|---|---|---|---|
| flow features only (channel 1) | 0.9518 ± 0.0140 (n = 10) | 0.8404 ± 0.0146 (n = 10) | 0.01271 ± 0.00208 (n = 10) | 0.9509, 0.9651, 0.9522, 0.9587, 0.9519, 0.9485, 0.9625, 0.9129, 0.9620, 0.9532 | 16, 32, 19, 22, 22, 15, 31, 10, 36, 19 |
| neighbourhood only (channel 2) | 0.9694 ± 0.0062 (n = 3) | 0.9159 ± 0.0155 (n = 3) | 0.00447 ± 0.00055 (n = 3) | 0.9611, 0.9758, 0.9712 | 19, 32, 29 |
| both channels (full model) | 0.9883 ± 0.0031 (n = 10) | 0.9383 ± 0.0067 (n = 10) | 0.00278 ± 0.00051 (n = 10) | 0.9881, 0.9902, 0.9864, 0.9906, 0.9889, 0.9898, 0.9868, 0.9914, 0.9803, 0.9909 | 38, 32, 12, 38, 36, 39, 14, 40, 20, 40 |

- **Topology gain** (full minus flow-features-only, PR-AUC): **+0.0365**
- Seed ranges: full 0.9803–0.9914; flow-features-only 0.9129–0.9651. **They do not overlap.**
- False-positive rate at 95% recall: 0.01271 → 0.00278 (-78%)
- Seeds per variant: 10. With so few, say whether the ranges overlap; do not use the word "significant".
- Mean share of the decision assigned to the neighbourhood channel (full model, seed 0, whole test split): 0.6181

### Attack-family head (full model)

Scored on the whole test split at native prevalence.

- Macro-F1 over families: 0.8650 ± 0.0220 (n = 10)
- Accuracy over all flows: 0.9777 ± 0.0014 (n = 10)
- Correct family among true attacks: 0.9692 ± 0.0019 (n = 10)
- **Correct family on true-positive detections** (the flows that become evidence packs): 0.9746 ± 0.0022 (n = 10)

## 4. The graph model against XGBoost

Same dataset, same prevalence, same reporting path.

| Model | PR-AUC (ranking) | F1 (operating point) |
|---|---|---|
| graph model (full) | 0.9883 ± 0.0031 (n = 10) | 0.9383 ± 0.0067 (n = 10) |
| xgboost::flow | 0.9743 ± 0.0004 (n = 3) | 0.8936 ± 0.0006 (n = 3) |
| xgboost::flow+host | 0.9793 ± 0.0012 (n = 3) | 0.9463 ± 0.0018 (n = 3) |

- Against the best XGBoost by PR-AUC (xgboost::flow+host): PR-AUC **+0.0090**
- Against the best XGBoost by F1 (xgboost::flow+host): F1 **-0.0080**
- **Both halves must be stated together.** The graph model ranks better and operates worse.

## 5. Topology ablation on NF-UNSW-NB15-v2 (the saturated dataset)

*Source: `results/metrics/gnn/gnn_NF-UNSW-NB15-v2.json`, generated 2026-10-17T08:37:00 UTC.*

Reported at 4% prevalence (PR-AUC of a random model: 0.04).
Training: fixed budget of 40 epochs, best-validation checkpoint kept.

| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | Best epoch per seed |
|---|---|---|---|---|---|
| flow features only (channel 1) | 0.9884 ± 0.0002 (n = 3) | 0.9475 ± 0.0004 (n = 3) | 0.00241 ± 0.00006 (n = 3) | 0.9881, 0.9884, 0.9886 | 31, 40, 40 |
| neighbourhood only (channel 2) | 0.9570 ± 0.0013 (n = 3) | 0.9133 ± 0.0045 (n = 3) | 0.00545 ± 0.00037 (n = 3) | 0.9582, 0.9575, 0.9552 | 32, 15, 34 |
| both channels (full model) | 0.9892 ± 0.0000 (n = 3) | 0.9434 ± 0.0041 (n = 3) | 0.00259 ± 0.00023 (n = 3) | 0.9893, 0.9892, 0.9892 | 34, 33, 36 |

- **Topology gain** (full minus flow-features-only, PR-AUC): **+0.0008**
- Seed ranges: full 0.9892–0.9893; flow-features-only 0.9881–0.9886. **They do not overlap.**
- False-positive rate at 95% recall: 0.00241 → 0.00259 (+7%)
- Seeds per variant: 3. With so few, say whether the ranges overlap; do not use the word "significant".
- Mean share of the decision assigned to the neighbourhood channel (full model, seed 0, whole test split): 0.0749

### Attack-family head (full model)

Scored on the whole test split at native prevalence.

- Macro-F1 over families: 0.5491 ± 0.0206 (n = 3)
- Accuracy over all flows: 0.9765 ± 0.0004 (n = 3)
- Correct family among true attacks: 0.7217 ± 0.0169 (n = 3)
- **Correct family on true-positive detections** (the flows that become evidence packs): 0.7399 ± 0.0078 (n = 3)

## 6. Zero-day: leave-one-attack-out

*Source: `results/metrics/zeroday/zeroday_NF-ToN-IoT-v2.json`, generated 2026-10-17T13:22:38 UTC.*

Each family was removed from training and validation entirely; recall is on that family in the test split, at 4% prevalence. Model variant: full.

| Held-out family | Test flows | Recall | PR-AUC | Seeds | Thin? | The family head calls it |
|---|---|---|---|---|---|---|
| scanning | 241,497 | 0.0007 ± 0.0001 (n = 3) | 0.0465 ± 0.0028 (n = 3) | 3 | no | Benign 100%, password 0% |
| xss | 162,337 | 0.2447 ± 0.0984 (n = 3) | 0.8497 ± 0.0400 (n = 3) | 3 | no | Benign 50%, injection 33% |
| password | 107,086 | 0.5626 ± 0.1179 (n = 3) | 0.8071 ± 0.0178 (n = 3) | 3 | no | injection 66%, Benign 29% |
| ddos | 92,066 | 0.7910 ± 0.0500 (n = 3) | 0.9769 ± 0.0050 (n = 3) | 3 | no | injection 57%, password 32% |
| dos | 77,328 | 0.0431 ± 0.0549 (n = 3) | 0.7155 ± 0.0704 (n = 3) | 3 | no | xss 42%, Benign 33% |
| injection | 61,648 | 0.8078 ± 0.0995 (n = 3) | 0.9858 ± 0.0021 (n = 3) | 3 | no | xss 53%, password 33% |
| backdoor | 4,498 | 0.0000 ± 0.0000 (n = 3) | 0.3671 ± 0.0347 (n = 3) | 3 | yes | Benign 100%, dos 0% |
| mitm | 897 | 0.6454 ± 0.1916 (n = 3) | 0.9488 ± 0.0162 (n = 3) | 3 | yes | dos 89%, xss 6% |

"Thin" means fewer than 20,000 held-out flows in the test split: quote that recall with its count.

- **Not measurable: ransomware** — only 3 held-out flows reach the test split (need 500); 2,554 were removed from train, so this family is concentrated in one split and cannot be held out here

Do not report the mean across families. Which families are caught and which are not is the finding.

## 7. Cross-dataset transfer (no retraining)

*Source: every `results/metrics/transfer/transfer_*.json`, pooled by `scripts/summarise_transfer.py`.*

PR-AUC of a random model at this prevalence: 0.04.

| Model | Trained on → tested on | Kind | Seeds | PR-AUC | F1 | Out-of-vocabulary |
|---|---|---|---|---|---|---|
| gnn::full | ToN-IoT → ToN-IoT | in-dataset control | 1 | 0.9595 (one seed) | 0.8992 | 2.1% |
| gnn::full | UNSW-NB15 → UNSW-NB15 | in-dataset control | 1 | 0.9936 (one seed) | 0.9692 | 0.6% |
| xgboost::flow | ToN-IoT → ToN-IoT | in-dataset control | 3 | 0.9167 ± 0.0071 | 0.8442 | 2.1% |
| xgboost::flow | UNSW-NB15 → UNSW-NB15 | in-dataset control | 3 | 0.9847 ± 0.0003 | 0.9018 | 0.6% |
| gnn::full | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0541 ± 0.0270 | 0.0618 | 3.6% |
| gnn::full | UNSW-NB15 → ToN-IoT | **cross-dataset** | 3 | 0.1006 ± 0.0359 | 0.1967 | 54.5% |
| xgboost::flow | ToN-IoT → UNSW-NB15 | **cross-dataset** | 3 | 0.0633 ± 0.0136 | 0.0400 | 3.6% |
| xgboost::flow | UNSW-NB15 → ToN-IoT | **cross-dataset** | 3 | 0.0401 ± 0.0049 | 0.0052 | 54.5% |

- A direction with out-of-vocabulary above 50% is confounded: the model was given a degraded input as well as a different network. Rely on the other direction.
- A row marked "one seed" has no standard deviation. Do not attach one.

## 8. Explainability (over the evidence packs)

*Source: `results/metrics/evidence/evidence_NF-ToN-IoT-v2_summary.json`, generated 2026-10-17T15:21:55 UTC.*

200 evidence packs: 170 true attacks and 30 false positives (benign flows the detector flagged). Checkpoint: NF-ToN-IoT-v2_full_seed0.pt. The packs are a stratified sample of *flagged* flows, not of traffic.

### Share of each decision assigned to the neighbourhood channel

| Group | Packs | Median | 25th–75th percentile | Mean |
|---|---|---|---|---|
| all packs | 200 | 0.5271 | 0.2878–0.8219 | 0.5251 |
| true attacks | 170 | 0.4368 | 0.2460–0.8365 | 0.5195 |
| false positives | 30 | 0.6059 | 0.4225–0.6328 | 0.5564 |
| backdoor | 21 | 0.0818 | 0.0812–0.1095 | 0.0934 |
| ddos | 21 | 0.3048 | 0.2531–0.3388 | 0.2987 |
| dos | 21 | 0.8371 | 0.4194–0.8539 | 0.6990 |
| injection | 21 | 0.6665 | 0.6427–0.6982 | 0.6800 |
| mitm | 21 | 0.8419 | 0.8377–0.8498 | 0.8585 |
| password | 23 | 0.3914 | 0.3785–0.5910 | 0.4517 |
| scanning | 21 | 0.1685 | 0.1508–0.2409 | 0.2022 |
| xss | 21 | 0.8829 | 0.8203–0.9514 | 0.8793 |

### Features most often among the top 5 attributed

| Feature | Share of packs | Times it raised suspicion | Times it lowered suspicion |
|---|---|---|---|
| L4_DST_PORT | 56% | 119 | 30 |
| TCP_FLAGS | 46% | 28 | 63 |
| SHORTEST_FLOW_PKT | 44% | 41 | 48 |
| CLIENT_TCP_FLAGS | 42% | 30 | 55 |
| L7_PROTO | 36% | 33 | 40 |
| SERVER_TCP_FLAGS | 32% | 40 | 24 |
| IN_BYTES | 32% | 21 | 42 |
| DNS_QUERY_TYPE | 30% | 18 | 43 |
| NUM_PKTS_UP_TO_128_BYTES | 22% | 1 | 44 |
| PROTOCOL | 18% | 0 | 37 |

A feature and its encoded variants (a port and its port bucket) count as one feature per pack but as separate entries in the last two columns, so those can exceed the pack count.

### The attack family named in the pack

- Correct for 0.8765 of 170 true attacks in the sample
- False positives were named as: {'scanning': 23, 'ddos': 7}
- Ambiguous packs (confidence below 0.9): 9

### Influence of single neighbouring flows

- Up to 50 neighbouring flows were removed one at a time per detection (median tested: 50; median sharing a host: 5360)
- Largest change in the detection score from removing one neighbour: median 1.00e-05, 95th percentile 4.30e-04, maximum 1.98e-03
- Packs with any neighbour above the 0.01 threshold: 0 of 200

## 9. Language-model comparison

Evidence packs: 200 (30 false positives). By true family: {'Benign': 30, 'backdoor': 21, 'ddos': 21, 'dos': 21, 'injection': 21, 'mitm': 21, 'password': 23, 'scanning': 21, 'xss': 21}.

*Source: every `results/reports/reports_*.json`, pooled by `scripts/summarise_llm_study.py`.*

| Model | Prompt | Reports | Failed | Groundedness | Fabricated-address reports | Groundedness on false positives | Names predicted class | Substitutes class | Attributed features cited | Cites none | Latency p50 / p95 (s) | Charged (USD) | List price (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.7345 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| stub/template-v1 | v2 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.7345 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| ollama/gemma4:31b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.6062 | 0.0050 | 1.61 / 2.56 | unknown | 0.0584 |
| ollama/gpt-oss:120b | v3 | 200 | 0 | 0.9993 | 0 | 1.0000 (n = 30) | 0.9500 | 0.0000 | 0.6112 | 0.0050 | 2.15 / 3.16 | unknown | 0.1060 |
| ollama/gpt-oss:20b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 0.9800 | 0.0000 | 0.6095 | 0.0050 | 8.39 / 23.47 | unknown | 0.0869 |
| ollama/nemotron-3-nano:30b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 0.9950 | 0.0000 | 0.6480 | 0.0000 | 15.27 / 55.40 | unknown | 0.2023 |
| ollama/nemotron-3-super | v3 | 200 | 0 | 0.9997 | 0 | 1.0000 (n = 30) | 0.9850 | 0.0000 | 0.6610 | 0.0000 | 28.67 / 57.56 | unknown | 0.1588 |
| ollama/nemotron-3-ultra | v3 | 200 | 0 | 0.9996 | 0 | 0.9986 (n = 30) | 1.0000 | 0.0000 | 0.6610 | 0.0000 | 13.63 / 27.61 | unknown | 0.6576 |
| stub/template-v1 | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.4405 | 0.0200 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| ollama/gemma4:31b | v4 | 80 | 0 | 1.0000 | 0 | 1.0000 (n = 12) | 0.9875 | 0.0000 | 0.3725 | 0.0250 | 1.94 / 4.28 | unknown | 0.0229 |
| ollama/gpt-oss:120b | v4 | 80 | 0 | 1.0000 | 0 | 1.0000 (n = 12) | 0.8750 | 0.0000 | 0.6194 | 0.0000 | 4.09 / 5.65 | unknown | 0.0658 |
| ollama/gpt-oss:20b | v4 | 80 | 0 | 1.0000 | 0 | 1.0000 (n = 12) | 0.9625 | 0.0000 | 0.5550 | 0.0000 | 29.01 / 70.98 | unknown | 0.0950 |
| ollama/nemotron-3-nano:30b | v4 | 40 | 0 | 1.0000 | 0 | 1.0000 (n = 6) | 1.0000 | 0.0000 | 0.6763 | 0.0000 | 13.08 / 19.50 | unknown | 0.0549 |
| ollama/nemotron-3-super | v4 | 40 | 0 | 1.0000 | 0 | 1.0000 (n = 6) | 0.9750 | 0.0000 | 0.6700 | 0.0000 | 44.33 / 81.60 | unknown | 0.0495 |
| ollama/nemotron-3-ultra | v4 | 40 | 0 | 1.0000 | 0 | 1.0000 (n = 6) | 0.8250 | 0.0000 | 0.6475 | 0.0000 | 11.72 / 17.83 | unknown | 0.1662 |
| stub/template-v1 | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.0000 | 0.4405 | 0.0200 | 0.00 / 0.00 | 0.0000 | 0.0000 |

### Direction of reasons, and borderline detections (D49)

| Model | Prompt | Uncertainty conveyed (ambiguous category) | Reports with a checkable reason | Direction faithful | Direction errors | Borderline detections | Borderline conveyed |
|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 1.0000 | 133 | 1.0000 | 0 | 24 | 0.5833 |
| stub/template-v1 | v2 | 1.0000 | 133 | 1.0000 | 0 | 24 | 0.5833 |
| ollama/gemma4:31b | v3 | 1.0000 | 162 | 0.9691 | 5 | 24 | 1.0000 |
| ollama/gpt-oss:120b | v3 | 1.0000 | 146 | 0.9726 | 4 | 24 | 1.0000 |
| ollama/gpt-oss:20b | v3 | 1.0000 | 126 | 0.9683 | 4 | 24 | 1.0000 |
| ollama/nemotron-3-nano:30b | v3 | 1.0000 | 180 | 0.9722 | 5 | 24 | 1.0000 |
| ollama/nemotron-3-super | v3 | 1.0000 | 191 | 0.9686 | 6 | 24 | 1.0000 |
| ollama/nemotron-3-ultra | v3 | 1.0000 | 133 | 0.9699 | 5 | 24 | 1.0000 |
| stub/template-v1 | v3 | 1.0000 | 0 | n/a | 0 | 24 | 1.0000 |
| ollama/gemma4:31b | v4 | 0.8750 | 17 | 1.0000 | 0 | 14 | 1.0000 |
| ollama/gpt-oss:120b | v4 | 0.8750 | 54 | 0.9074 | 8 | 14 | 1.0000 |
| ollama/gpt-oss:20b | v4 | 1.0000 | 51 | 0.9412 | 3 | 14 | 1.0000 |
| ollama/nemotron-3-nano:30b | v4 | 1.0000 | 35 | 0.9143 | 3 | 11 | 1.0000 |
| ollama/nemotron-3-super | v4 | 1.0000 | 36 | 0.9167 | 3 | 11 | 1.0000 |
| ollama/nemotron-3-ultra | v4 | 0.7143 | 17 | 0.9412 | 1 | 11 | 1.0000 |
| stub/template-v1 | v4 | 1.0000 | 0 | n/a | 0 | 24 | 1.0000 |

- The stub is a template with no language model: the control arm.
- The direction check reads a reason's direction from clause-level wording and skips any clause it cannot read, so its error count is a lower bound.
- "Borderline" is decided by the facts layer by fixed rules: margin over the threshold below a tenth of the room above it, category confidence below 0.6, or a second category at 0.05 or more.
- Groundedness measures "invented nothing", not "reasoned correctly".
- The class and feature checks are lexical: they match names and phrases, are blind to negation, and are generous about what counts as citing a feature.
- Latency is response time from a hosted service; it is not model speed.
- A cost of "unknown" is not zero.

- gemma4:31b: digest `221b330d11a8`, 32682372656, BF16, GPU none, 5 session(s)
- gemma4:31b: digest `221b330d11a8`, 32682372656, BF16, GPU none, 2 session(s)
- gpt-oss:120b: digest `d98fe6ba01e6`, 116829156672, MXFP4, GPU none, 5 session(s)
- gpt-oss:120b: digest `d98fe6ba01e6`, 116829156672, MXFP4, GPU none, 2 session(s)
- gpt-oss:20b: digest `05afbac4bad6`, 20914757184, MXFP4, GPU none, 6 session(s)
- gpt-oss:20b: digest `05afbac4bad6`, 20914757184, MXFP4, GPU none, 2 session(s)
- nemotron-3-nano:30b: digest `a4b4f1851393`, 32000000000, FP8, GPU none, 5 session(s)
- nemotron-3-nano:30b: digest `a4b4f1851393`, 32000000000, FP8, GPU none, 1 session(s)
- nemotron-3-super: digest `da11955bb451`, 120000000000, NVFP4, GPU none, 5 session(s)
- nemotron-3-super: digest `da11955bb451`, 120000000000, NVFP4, GPU none, 1 session(s)
- nemotron-3-ultra: digest `4eecabff7a75`, 550000000000, , GPU none, 5 session(s)
- nemotron-3-ultra: digest `4eecabff7a75`, 550000000000, , GPU none, 1 session(s)

## 10. Replay (the demonstration harness)

*Source: `results/metrics/replay/replay_NF-ToN-IoT-v2.json`, generated 2026-10-17T08:41:01 UTC.*

Offline replay of the held-out split. Not a real-time measurement.

- Windows replayed: 100 (1,000,000 flows)
- Alerts: 618,727 (618.73 per 1,000 flows)
- Precision 0.9984, recall 0.9436
- Windows in which the language model would be invoked: 96.0%
- Scoring time per window: median 18.61 ms, 95th percentile 21.56 ms

## 11. Figures

- **`results/figures/results_01_topology_ablation.png`** — Topology ablation on NF-ToN-IoT-v2. Each dot is one seed (n = 10 per variant); the tick is the mean. The variants share architecture, optimiser, depth and width, so the gap between the first and third is the contribution of message passing. Trained with fixed training budget.
- **`results/figures/results_02_gnn_vs_xgboost.png`** — The graph model against XGBoost on NF-ToN-IoT-v2, both reported at 4% prevalence. Left: ranking quality. Right: the operating point reached at the validation-chosen threshold. Each dot is one seed; the tick is the mean. The two panels have separate scales and must be read together: the graph model ranks better and operates worse. Graph model trained with fixed training budget.
- **`results/figures/results_03_topology_gain_by_dataset.png`** — Gain in PR-AUC from message passing (full model minus the flow-features-only variant, matched by seed) on each dataset. Each dot is one seed; the tick is the mean; the line marks no gain. The gain is separated from zero on NF-ToN-IoT-v2 and not on NF-UNSW-NB15-v2, where a linear model already scores above 0.99.
- **`results/figures/results_04_zero_day_recall.png`** — Leave-one-attack-out on NF-ToN-IoT-v2: recall on each attack family when that family was removed entirely from training and validation. Each dot is one seed; the tick is the mean. Families are ordered by the number of held-out flows in the test split, shown in brackets. Hollow dots mark families with fewer than 20,000 test flows, whose recall is correspondingly uncertain. Not measurable on this dataset: ransomware.
- **`results/figures/results_05_cross_dataset_transfer.png`** — Cross-dataset transfer with no retraining. For each model and direction, the blue dot is the in-dataset control and the orange dot is the same model scored on the other dataset (bar: one standard deviation over seeds, where more than one was run). Every control is healthy and every transfer falls to the level of a random model. † More than half of this direction's categorical values fall outside the source vocabulary, so it measures a degraded input as well as a different network.
- **`results/figures/results_06_channel_weights.png`** — How the fusion layer divides each decision between a flow's own features and its neighbourhood, over 200 flagged flows. Dot: median; bar: interquartile range; n: evidence packs in the group. The sample is stratified by family, so the groups are small and the ordering is indicative.
- **`results/figures/results_07_top_features.png`** — The features that carry the detections: for each, the share of 200 evidence packs in which integrated gradients ranks it among the 5 most influential. A column and its encoded variants are counted once per pack.
- **`results/figures/results_08_attack_family_confusion.png`** — The attack-family head on the NF-ToN-IoT-v2 test split, seed 0: each row shows how flows of one true family were named. Cells below 0.05 are left unlabelled. Macro-F1 0.885; on true-positive detections the named family is correct for 97.7%.
- **`results/figures/results_09_llm_groundedness.png`** — Groundedness of each language model's reports on identical evidence packs and an identical prompt. Blue: all reports. Orange: the false-positive arm, where the detector was wrong. The line is the template control, which invents nothing by construction. n is the number of reports scored. Groundedness measures whether a report invented anything, not whether its reasoning is correct.

## 12. Process

- Automated tests in the repository: 286
- Counts of logged decisions and defects are kept in the project notes (Decision Register, Defect Register), not in this file.

