# Report facts

*Exported 2026-10-29T03:21:28 UTC by `scripts/export_report_facts.py`. Every number below was read from a results file at export time. None was typed in.*

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

### Flows per class and split

*Source: `results/metrics/class_balance.json`, generated 2026-10-29T02:57:21 UTC.*

**NF-ToN-IoT-v2**: 5,910,000 flows in the three splits; the largest attack family (scanning) is 561.6x the smallest (ransomware).

| Class | Train | Val | Test | Share of flows | Share of attacks |
|---|---|---|---|---|---|
| backdoor | 12,311 | 0 | 4,498 | 0.28% | 0.44% |
| Benign | 1,425,451 | 180,963 | 442,640 | 34.67% | n/a |
| ddos | 430,241 | 68,974 | 92,066 | 10.00% | 15.31% |
| dos | 282,291 | 33,056 | 77,328 | 6.64% | 10.17% |
| injection | 191,566 | 20,907 | 61,648 | 4.64% | 7.10% |
| mitm | 2,306 | 180 | 897 | 0.06% | 0.09% |
| password | 213,937 | 31,284 | 107,086 | 5.96% | 9.12% |
| ransomware | 2,554 | 0 | 3 | 0.04% | 0.07% |
| scanning | 1,067,523 | 127,108 | 241,497 | 24.30% | 37.20% |
| xss | 501,820 | 127,528 | 162,337 | 13.40% | 20.50% |

**NF-UNSW-NB15-v2**: 2,350,275 flows in the three splits; the largest attack family (Exploits) is 188.1x the smallest (Worms).

| Class | Train | Val | Test | Share of flows | Share of attacks |
|---|---|---|---|---|---|
| Analysis | 951 | 336 | 951 | 0.10% | 2.42% |
| Backdoor | 1,023 | 344 | 754 | 0.09% | 2.29% |
| Benign | 1,619,577 | 204,327 | 433,744 | 96.06% | n/a |
| DoS | 2,557 | 875 | 2,220 | 0.24% | 6.10% |
| Exploits | 14,005 | 5,156 | 11,494 | 1.30% | 33.10% |
| Fuzzers | 10,586 | 3,673 | 7,665 | 0.93% | 23.67% |
| Generic | 8,282 | 1,937 | 5,857 | 0.68% | 17.36% |
| Reconnaissance | 5,504 | 2,112 | 4,795 | 0.53% | 13.40% |
| Shellcode | 636 | 240 | 511 | 0.06% | 1.50% |
| Worms | 71 | 27 | 65 | 0.01% | 0.18% |

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

"With" is the archived run (`results/superseded/2026-10-10_duration_artefact/`), which saw the seven artefact columns; "without" is the current run. The rows are identical and in the same order, the training configuration is identical, and both sides are averaged over the same seeds (n), so the feature list is the only difference on NF-ToN-IoT-v2. On NF-UNSW-NB15-v2 the earlier preprocessing also lacked the outlier clip, so that comparison has two differences.

| Dataset | Model | n | PR-AUC with the artefact | PR-AUC without | Difference |
|---|---|---|---|---|---|
| NF-ToN-IoT-v2 | GNN channel1_only | 3 | 0.9804 | 0.9561 | -0.0243 |
| NF-ToN-IoT-v2 | GNN channel2_only | 3 | 0.9729 | 0.9694 | -0.0036 |
| NF-ToN-IoT-v2 | GNN full | 3 | 0.9922 | 0.9882 | -0.0040 |
| NF-ToN-IoT-v2 | logreg::flow | 3 | 0.9748 | 0.9579 | -0.0169 |
| NF-ToN-IoT-v2 | logreg::flow+host | 3 | 0.9833 | 0.9824 | -0.0009 |
| NF-ToN-IoT-v2 | mlp::flow | 3 | 0.9832 | 0.9696 | -0.0136 |
| NF-ToN-IoT-v2 | mlp::flow+host | 3 | 0.9833 | 0.9853 | +0.0020 |
| NF-ToN-IoT-v2 | random_forest::flow | 3 | 0.9710 | 0.9537 | -0.0173 |
| NF-ToN-IoT-v2 | random_forest::flow+host | 3 | 0.9753 | 0.9703 | -0.0050 |
| NF-ToN-IoT-v2 | xgboost::flow | 3 | 0.9860 | 0.9743 | -0.0117 |
| NF-ToN-IoT-v2 | xgboost::flow+host | 3 | 0.9809 | 0.9793 | -0.0015 |
| NF-UNSW-NB15-v2 | GNN channel1_only | 3 | 0.9903 | 0.9884 | -0.0019 |
| NF-UNSW-NB15-v2 | GNN full | 3 | 0.9914 | 0.9892 | -0.0022 |
| NF-UNSW-NB15-v2 | logreg::flow | 3 | 0.9870 | 0.9839 | -0.0031 |
| NF-UNSW-NB15-v2 | logreg::flow+host | 3 | 0.9887 | 0.9839 | -0.0048 |
| NF-UNSW-NB15-v2 | mlp::flow | 3 | 0.9890 | 0.9875 | -0.0015 |
| NF-UNSW-NB15-v2 | mlp::flow+host | 3 | 0.9888 | 0.9866 | -0.0022 |
| NF-UNSW-NB15-v2 | random_forest::flow | 3 | 0.9899 | 0.9850 | -0.0049 |
| NF-UNSW-NB15-v2 | random_forest::flow+host | 3 | 0.9743 | 0.9809 | +0.0065 |

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

## 4a. Are the seed-level differences larger than seed noise?

Mann-Whitney U, exact, two-sided on PR-AUC at 4% prevalence, per seed. With n seeds per side the smallest possible p is 2 / C(n1 + n2, n1), so three seeds against three cannot go below 0.1.

| Dataset | Comparison | Seeds | Mean difference (unrounded) | p | Smallest possible p |
|---|---|---|---|---|---|
| NF-ToN-IoT-v2 | full vs channel1_only | 10 vs 10 | +0.03656 | 1.1e-05 | 1.1e-05 |
| NF-UNSW-NB15-v2 | full vs channel1_only | 3 vs 3 | +0.00082 | 0.1 | 0.1 |
| NF-ToN-IoT-v2 | full vs xgboost::flow+host | 10 vs 3 | +0.00901 | 0.014 | 0.007 |

*Source: `results/metrics/significance.json`, generated 2026-10-26T05:54:27 UTC.*

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

*Source: `results/metrics/evidence/evidence_NF-ToN-IoT-v2_summary.json`, generated 2026-10-26T03:14:50 UTC.*

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
- mitm: 0 of 21 named correctly (named as {'dos': 21})
- Ambiguous packs (confidence below 0.9): 9

### Influence of single neighbouring flows

- Up to 50 neighbouring flows were removed one at a time per detection (median tested: 50; median sharing a host: 5360)
- Largest change in the detection score from removing one neighbour: median 1.00e-05, 95th percentile 4.30e-04, maximum 1.98e-03
- Packs with any neighbour above the 0.01 threshold: 0 of 200

### Deletion test of the stated reasons (D56)

*Source: `results/metrics/evidence/deletion_NF-ToN-IoT-v2.json`, generated 2026-10-29T03:00:21 UTC.*

Each pack's top 5 attributions are the reasons the facts layer passes on. Those that raised the score were set to the integrated-gradients baseline (0 in standardised units, the mean flow) and the flow rescored; the same number of other features was removed 20 times at random for comparison. Logit units. Rescored packs matched their stored score to within 5e-06.

- Median logit drop, stated reasons removed: **1.1994**; random features removed: **0.0421**
- Packs where the reasons cost more than random: 94.5% of 200
- Still flagged after removal: 66.5% (reasons) against 99.0% (random)
- Removing the reasons that lowered the score raised the logit by a median of 0.2752

| True class | Packs | Median drop (reasons) | Median drop (random) | Reasons cost more | Still flagged (reasons) |
|---|---|---|---|---|---|
| Benign | 30 | 0.95 | 0.0151 | 96.7% | 13.3% |
| backdoor | 21 | 4.4377 | 0.0822 | 100.0% | 42.9% |
| ddos | 21 | 2.5463 | 0.0619 | 100.0% | 71.4% |
| dos | 21 | 0.3033 | 0.0277 | 95.2% | 100.0% |
| injection | 21 | 1.4188 | 0.0722 | 100.0% | 85.7% |
| mitm | 21 | 0.2324 | 0.0167 | 90.5% | 95.2% |
| password | 23 | 2.0653 | 0.1343 | 100.0% | 78.3% |
| scanning | 21 | 2.5296 | 0.0538 | 100.0% | 47.6% |
| xss | 21 | 0.2758 | 0.0097 | 66.7% | 85.7% |

## 9. Language-model comparison

Evidence packs: 200 (30 false positives). By true family: {'Benign': 30, 'backdoor': 21, 'ddos': 21, 'dos': 21, 'injection': 21, 'mitm': 21, 'password': 23, 'scanning': 21, 'xss': 21}.

*Source: every `results/reports/reports_*.json`, pooled by `scripts/summarise_llm_study.py`.*

| Model | Prompt | Reports | Failed | Groundedness | Fabricated-address reports | Groundedness on false positives | Names predicted class | Class faithful | Substitutes class | Attributed features cited | Cites none | Latency p50 / p95 (s) | Charged (USD) | List price (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.7345 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| stub/template-v1 | v2 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.7345 | 0.0000 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| ollama/deepseek-v4-pro:0813 | v3 | 196 | 4 | 1.0000 | 0 | 1.0000 (n = 29) | 1.0000 | 1.0000 | 0.0000 | 0.6686 | 0.0000 | 7.43 / 12.78 | unknown | 1.3371 |
| ollama/gemma4:31b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.6062 | 0.0050 | 1.61 / 2.56 | unknown | 0.0584 |
| ollama/glm-5.3 | v3 | 200 | 0 | 0.9992 | 0 | 0.9965 (n = 30) | 1.0000 | 0.9950 | 0.0000 | 0.6120 | 0.0000 | 21.88 / 108.69 | unknown | 8.6262 |
| ollama/gpt-oss:120b | v3 | 200 | 0 | 0.9993 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9500 | 0.0000 | 0.6112 | 0.0050 | 2.15 / 3.16 | unknown | 0.1060 |
| ollama/gpt-oss:20b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9800 | 0.0000 | 0.6095 | 0.0050 | 8.39 / 23.47 | unknown | 0.0869 |
| ollama/mistral-large-4 | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.6878 | 0.0000 | 28.23 / 72.54 | unknown | 1.9323 |
| ollama/nemotron-3-nano:30b | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9950 | 0.0000 | 0.6480 | 0.0000 | 15.27 / 55.40 | unknown | 0.2023 |
| ollama/nemotron-3-super | v3 | 200 | 0 | 0.9997 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9850 | 0.0000 | 0.6610 | 0.0000 | 28.67 / 57.56 | unknown | 0.1588 |
| ollama/nemotron-3-ultra | v3 | 200 | 0 | 0.9996 | 0 | 0.9986 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.6610 | 0.0000 | 13.63 / 27.61 | unknown | 0.6576 |
| stub/template-v1 | v3 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.4405 | 0.0200 | 0.00 / 0.00 | 0.0000 | 0.0000 |
| ollama/deepseek-v4-pro:0813 | v4 | 25 | 15 | 1.0000 | 0 | 1.0000 (n = 2) | 1.0000 | 1.0000 | 0.0000 | 0.6640 | 0.0000 | 20.05 / 47.31 | unknown | 0.4473 |
| ollama/gemma4:31b | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9950 | 0.0000 | 0.3770 | 0.0150 | 2.04 / 3.57 | unknown | 0.0571 |
| ollama/glm-5.3 | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9650 | 0.0000 | 0.6308 | 0.0000 | 25.16 / 58.00 | unknown | 4.4009 |
| ollama/gpt-oss:120b | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.8700 | 0.0000 | 0.6145 | 0.0000 | 3.71 / 5.28 | unknown | 0.1669 |
| ollama/gpt-oss:20b | v4 | 200 | 0 | 0.9996 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9400 | 0.0000 | 0.5825 | 0.0050 | 28.49 / 82.46 | unknown | 0.2553 |
| ollama/mistral-large-4 | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9750 | 0.0000 | 0.5923 | 0.0000 | 25.08 / 46.76 | unknown | 1.4802 |
| ollama/nemotron-3-nano:30b | v4 | 199 | 1 | 0.9995 | 1 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.6525 | 0.0050 | 15.03 / 61.04 | unknown | 0.2668 |
| ollama/nemotron-3-super | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 0.9750 | 0.0000 | 0.6478 | 0.0050 | 42.57 / 96.37 | unknown | 0.2495 |
| ollama/nemotron-3-ultra | v4 | 200 | 0 | 0.9996 | 0 | 1.0000 (n = 30) | 1.0000 | 0.8700 | 0.0000 | 0.6458 | 0.0000 | 13.13 / 56.65 | unknown | 0.8611 |
| stub/template-v1 | v4 | 200 | 0 | 1.0000 | 0 | 1.0000 (n = 30) | 1.0000 | 1.0000 | 0.0000 | 0.4405 | 0.0200 | 0.00 / 0.00 | 0.0000 | 0.0000 |

### Prompt sensitivity: v3 against v4 on the same packs

Each model is compared with itself, on the packs it answered under both prompts (failed and degenerate reports left out on either side). deepseek-v4-pro:0813 was capped at 40 packs on v4 after runaway generations (D51), so its pairing is small.

| Model | Prompt | Packs | Groundedness | Direction faithful | Borderline conveyed | Uncertainty conveyed | Names predicted class | Mean words |
|---|---|---|---|---|---|---|---|---|
| deepseek-v4-pro:0813 | v3 | 25 | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 195 |
| deepseek-v4-pro:0813 | v4 | 25 | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 351 |
| gemma4:31b | v3 | 200 | 1.0000 | 0.9691 | 1.0000 | 1.0000 | 1.0000 | 193 |
| gemma4:31b | v4 | 200 | 1.0000 | 1.0000 | 1.0000 | 0.7778 | 1.0000 | 172 |
| glm-5.3 | v3 | 200 | 0.9992 | 0.9727 | 1.0000 | 1.0000 | 1.0000 | 262 |
| glm-5.3 | v4 | 200 | 1.0000 | 0.9673 | 1.0000 | 1.0000 | 1.0000 | 445 |
| gpt-oss:120b | v3 | 200 | 0.9993 | 0.9726 | 1.0000 | 1.0000 | 1.0000 | 241 |
| gpt-oss:120b | v4 | 200 | 1.0000 | 0.9365 | 1.0000 | 0.8889 | 1.0000 | 419 |
| gpt-oss:20b | v3 | 200 | 1.0000 | 0.9683 | 1.0000 | 1.0000 | 1.0000 | 234 |
| gpt-oss:20b | v4 | 200 | 0.9996 | 0.9552 | 1.0000 | 1.0000 | 1.0000 | 342 |
| mistral-large-4 | v3 | 200 | 1.0000 | 0.9695 | 1.0000 | 1.0000 | 1.0000 | 275 |
| mistral-large-4 | v4 | 200 | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 313 |
| nemotron-3-nano:30b | v3 | 199 | 1.0000 | 0.9721 | 1.0000 | 1.0000 | 1.0000 | 264 |
| nemotron-3-nano:30b | v4 | 199 | 0.9995 | 0.9532 | 1.0000 | 1.0000 | 1.0000 | 360 |
| nemotron-3-super | v3 | 200 | 0.9997 | 0.9686 | 1.0000 | 1.0000 | 1.0000 | 245 |
| nemotron-3-super | v4 | 200 | 1.0000 | 0.9639 | 0.9583 | 1.0000 | 1.0000 | 377 |
| nemotron-3-ultra | v3 | 200 | 0.9996 | 0.9699 | 1.0000 | 1.0000 | 1.0000 | 256 |
| nemotron-3-ultra | v4 | 200 | 0.9996 | 0.9643 | 1.0000 | 0.6667 | 1.0000 | 353 |

### Output beyond the note

The service counts a model's reasoning as output tokens; only the note is kept and scored. Plain English needs about 1.3-1.7 tokens a word, so a higher ratio is reasoning nobody reads. Medians over completed reports.

| Model | Prompt | Reports | Output tokens | Note words | Output tokens per note word |
|---|---|---|---|---|---|
| deepseek-v4-pro:0813 | v3 | 196 | 1,304 | 196 | 6.3 |
| gemma4:31b | v3 | 200 | 323 | 192 | 1.7 |
| glm-5.3 | v3 | 200 | 6,094 | 262 | 23.3 |
| gpt-oss:120b | v3 | 200 | 595 | 240 | 2.5 |
| gpt-oss:20b | v3 | 200 | 902 | 234 | 3.9 |
| mistral-large-4 | v3 | 200 | 3,518 | 279 | 13.0 |
| nemotron-3-nano:30b | v3 | 200 | 3,421 | 260 | 12.8 |
| nemotron-3-super | v3 | 200 | 1,191 | 242 | 5.0 |
| nemotron-3-ultra | v3 | 200 | 1,088 | 256 | 4.2 |
| deepseek-v4-pro:0813 | v4 | 25 | 4,550 | 353 | 11.3 |
| gemma4:31b | v4 | 200 | 380 | 172 | 2.2 |
| glm-5.3 | v4 | 200 | 4,438 | 434 | 10.2 |
| gpt-oss:120b | v4 | 200 | 1,150 | 416 | 2.8 |
| gpt-oss:20b | v4 | 200 | 3,260 | 303 | 11.2 |
| mistral-large-4 | v4 | 200 | 3,236 | 315 | 10.2 |
| nemotron-3-nano:30b | v4 | 199 | 4,959 | 368 | 13.6 |
| nemotron-3-super | v4 | 200 | 1,888 | 382 | 5.0 |
| nemotron-3-ultra | v4 | 200 | 1,394 | 355 | 3.9 |

List-price equivalent of every language-model call: $26.05 (v3 $14.21, v4 $11.84); $21.35 of it for reports that completed, the rest on failed or runaway generations. The calls ran first on Ollama's free allowance (D45) and then on a flat-fee subscription (D50); neither itemises per call, so what each call actually cost is unknown.

### End to end: detection outcome and translation together

*Source: `results/metrics/end_to_end_NF-ToN-IoT-v2.json`, generated 2026-10-29T03:02:00 UTC.*

Detection outcome of the 200 packs: attack right family 149; attack wrong family 21; false positive borderline 5; false positive clear 25. A note passes when it meets every check: completed, nothing ungrounded, no invented address, predicted family named, no family substituted, no direction error, borderline conveyed where borderline, uncertainty conveyed where the family is uncertain. Shares describe the stratified sample, not the alert stream.

| Model | Prompt | Completed | Passed every check | Correct and faithful (of 149) | Median latency (s) | Median words | List price per 1,000 notes (USD) |
|---|---|---|---|---|---|---|---|
| deepseek-v4-pro:0813 | v3 | 196 of 200 | 191 | 146 | 7.44 | 195 | 12.15 |
| gemma4:31b | v3 | 200 of 200 | 195 | 148 | 1.61 | 192 | 0.29 |
| glm-5.3 | v3 | 200 of 200 | 194 | 148 | 21.88 | 262 | 43.13 |
| gpt-oss:120b | v3 | 200 of 200 | 194 | 146 | 2.15 | 240 | 0.53 |
| gpt-oss:20b | v3 | 200 of 200 | 196 | 148 | 8.39 | 233 | 0.43 |
| mistral-large-4 | v3 | 200 of 200 | 194 | 148 | 28.23 | 279 | 9.66 |
| nemotron-3-nano:30b | v3 | 200 of 200 | 195 | 149 | 15.27 | 260 | 1.01 |
| nemotron-3-super | v3 | 200 of 200 | 193 | 147 | 28.67 | 242 | 0.79 |
| nemotron-3-ultra | v3 | 200 of 200 | 194 | 147 | 13.63 | 255 | 3.29 |
| template-v1 | v3 | 200 of 200 | 200 | 149 | 0.00 | 169 | 0.00 |
| deepseek-v4-pro:0813 | v4 | 25 of 40 | 24 | 21 | 20.05 | 353 | 163.85 |
| gemma4:31b | v4 | 200 of 200 | 198 | 149 | 2.04 | 171 | 0.29 |
| glm-5.3 | v4 | 200 of 200 | 195 | 148 | 25.16 | 433 | 22.00 |
| gpt-oss:120b | v4 | 200 of 200 | 191 | 147 | 3.71 | 416 | 0.83 |
| gpt-oss:20b | v4 | 200 of 200 | 192 | 146 | 28.49 | 303 | 1.28 |
| mistral-large-4 | v4 | 200 of 200 | 193 | 147 | 25.08 | 315 | 7.40 |
| nemotron-3-nano:30b | v4 | 199 of 200 | 190 | 144 | 15.03 | 368 | 1.38 |
| nemotron-3-super | v4 | 200 of 200 | 193 | 148 | 42.57 | 382 | 1.25 |
| nemotron-3-ultra | v4 | 200 of 200 | 192 | 144 | 13.13 | 355 | 4.31 |
| template-v1 | v4 | 200 of 200 | 200 | 149 | 0.00 | 169 | 0.00 |

### Direction of reasons, and borderline detections (D49)

| Model | Prompt | Uncertainty conveyed (ambiguous category) | Reports with a checkable reason | Direction faithful | Direction errors | Borderline detections | Borderline conveyed |
|---|---|---|---|---|---|---|---|
| stub/template-v1 | v1 | 1.0000 | 133 | 1.0000 | 0 | 24 | 0.5833 |
| stub/template-v1 | v2 | 1.0000 | 133 | 1.0000 | 0 | 24 | 0.5833 |
| ollama/deepseek-v4-pro:0813 | v3 | 1.0000 | 166 | 0.9699 | 5 | 24 | 1.0000 |
| ollama/gemma4:31b | v3 | 1.0000 | 162 | 0.9691 | 5 | 24 | 1.0000 |
| ollama/glm-5.3 | v3 | 1.0000 | 183 | 0.9727 | 5 | 24 | 1.0000 |
| ollama/gpt-oss:120b | v3 | 1.0000 | 146 | 0.9726 | 4 | 24 | 1.0000 |
| ollama/gpt-oss:20b | v3 | 1.0000 | 126 | 0.9683 | 4 | 24 | 1.0000 |
| ollama/mistral-large-4 | v3 | 1.0000 | 197 | 0.9695 | 7 | 24 | 1.0000 |
| ollama/nemotron-3-nano:30b | v3 | 1.0000 | 180 | 0.9722 | 5 | 24 | 1.0000 |
| ollama/nemotron-3-super | v3 | 1.0000 | 191 | 0.9686 | 6 | 24 | 1.0000 |
| ollama/nemotron-3-ultra | v3 | 1.0000 | 133 | 0.9699 | 5 | 24 | 1.0000 |
| stub/template-v1 | v3 | 1.0000 | 0 | n/a | 0 | 24 | 1.0000 |
| ollama/deepseek-v4-pro:0813 | v4 | 1.0000 | 22 | 0.9545 | 1 | 9 | 1.0000 |
| ollama/gemma4:31b | v4 | 0.7778 | 53 | 1.0000 | 0 | 24 | 1.0000 |
| ollama/glm-5.3 | v4 | 1.0000 | 153 | 0.9673 | 5 | 24 | 1.0000 |
| ollama/gpt-oss:120b | v4 | 0.8889 | 126 | 0.9365 | 12 | 24 | 1.0000 |
| ollama/gpt-oss:20b | v4 | 1.0000 | 134 | 0.9552 | 6 | 24 | 1.0000 |
| ollama/mistral-large-4 | v4 | 1.0000 | 154 | 0.9545 | 7 | 24 | 1.0000 |
| ollama/nemotron-3-nano:30b | v4 | 1.0000 | 171 | 0.9532 | 8 | 24 | 1.0000 |
| ollama/nemotron-3-super | v4 | 1.0000 | 166 | 0.9639 | 7 | 24 | 0.9583 |
| ollama/nemotron-3-ultra | v4 | 0.6667 | 84 | 0.9643 | 3 | 24 | 1.0000 |
| stub/template-v1 | v4 | 1.0000 | 0 | n/a | 0 | 24 | 1.0000 |

Reports that opened with the model's own working-out instead of the note (C38): gpt-oss:20b v4: 2; nemotron-3-nano:30b v4: 1.

Failed reports by model (empty after the output limit, network, or error) and degenerate reports (collapsed into repeated characters, C37; kept out of every quality mean): deepseek-v4-pro:0813 v3: 4 failed; deepseek-v4-pro:0813 v4: 15 failed; nemotron-3-nano:30b v4: 1 failed; nemotron-3-nano:30b v4: 1 degenerate.

- The stub is a template with no language model: the control arm.
- The direction check reads a reason's direction from clause-level wording and skips any clause it cannot read, so its error count is a lower bound.
- "Borderline" is decided by the facts layer by fixed rules: margin over the threshold below a tenth of the room above it, category confidence below 0.6, or a second category at 0.05 or more.
- Of the 30 false-positive packs, 5 are borderline by these rules (of 24 borderline packs in all). On the other 25 the detector was wrong with a clear margin, so the evidence a model reads gives it no reason to doubt the detection.
- Groundedness measures "invented nothing", not "reasoned correctly".
- The class and feature checks are lexical: they match names and phrases, are blind to negation, and are generous about what counts as citing a feature.
- "Class faithful" means the report names the predicted category and no category the detector gave below 0.05. It is lexical: "may be scanning" or naming injection beside cross-site scripting (a kind of injection) counts against it. "Substitutes class" -- another category named instead of the prediction -- is the failure the study looks for.
- Latency is response time from a hosted service; it is not model speed.
- A cost of "unknown" is not zero.

- deepseek-v4-pro:0813 (v3): digest `db976f1eacdd`, 1,650.5B parameters, FP8, hardware not disclosed by the hosted service, 5 session(s)
- deepseek-v4-pro:0813 (v4): digest `db976f1eacdd`, 1,650.5B parameters, FP8, hardware not disclosed by the hosted service, 2 session(s)
- gemma4:31b (v3): digest `221b330d11a8`, 32.7B parameters, BF16, hardware not disclosed by the hosted service, 5 session(s)
- gemma4:31b (v4): digest `221b330d11a8`, 32.7B parameters, BF16, hardware not disclosed by the hosted service, 5 session(s)
- glm-5.3 (v3): digest `632dfda18c6d`, 753.3B parameters, FP8, hardware not disclosed by the hosted service, 5 session(s)
- glm-5.3 (v4): digest `632dfda18c6d`, 753.3B parameters, FP8, hardware not disclosed by the hosted service, 5 session(s)
- gpt-oss:120b (v3): digest `d98fe6ba01e6`, 116.8B parameters, MXFP4, hardware not disclosed by the hosted service, 5 session(s)
- gpt-oss:120b (v4): digest `d98fe6ba01e6`, 116.8B parameters, MXFP4, hardware not disclosed by the hosted service, 5 session(s)
- gpt-oss:20b (v3): digest `05afbac4bad6`, 20.9B parameters, MXFP4, hardware not disclosed by the hosted service, 6 session(s)
- gpt-oss:20b (v4): digest `05afbac4bad6`, 20.9B parameters, MXFP4, hardware not disclosed by the hosted service, 5 session(s)
- mistral-large-4 (v3): digest `52309cacf736`, 1,050.0B parameters, quantisation not reported, hardware not disclosed by the hosted service, 5 session(s)
- mistral-large-4 (v4): digest `52309cacf736`, 1,050.0B parameters, quantisation not reported, hardware not disclosed by the hosted service, 5 session(s)
- nemotron-3-nano:30b (v3): digest `a4b4f1851393`, 32.0B parameters, FP8, hardware not disclosed by the hosted service, 5 session(s)
- nemotron-3-nano:30b (v4): digest `a4b4f1851393`, 32.0B parameters, FP8, hardware not disclosed by the hosted service, 7 session(s)
- nemotron-3-super (v3): digest `da11955bb451`, 120.0B parameters, NVFP4, hardware not disclosed by the hosted service, 5 session(s)
- nemotron-3-super (v4): digest `da11955bb451`, 120.0B parameters, NVFP4, hardware not disclosed by the hosted service, 5 session(s)
- nemotron-3-ultra (v3): digest `4eecabff7a75`, 550.0B parameters, quantisation not reported, hardware not disclosed by the hosted service, 5 session(s)
- nemotron-3-ultra (v4): digest `4eecabff7a75`, 550.0B parameters, quantisation not reported, hardware not disclosed by the hosted service, 5 session(s)

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
- **`results/figures/results_09_llm_groundedness.png`** — Groundedness of each language model's reports on identical evidence packs and an identical prompt. Blue: all reports. Orange: the false-positive arm, where the detector was wrong. The line is the template control, which invents nothing by construction. n is the number of reports scored. Groundedness measures whether a report invented anything, not whether its reasoning is correct. The axis is zoomed: the lowest point is 0.9965, so every model is within 0.0035 of the control.

## 12. Process

*Counted from the results files and the repository by `scripts/export_report_facts.py`.*

- Automated tests in the repository: 300
- Graph-model training runs behind the reported results (both datasets, all three variants, every seed): 32
- Leave-one-attack-out training runs: 24
- Baseline model fits (every model, feature set and seed, both datasets): 48
- Cross-dataset transfer evaluations, controls included: 20
- Evidence packs generated for the language-model study: 200
- Incident reports requested from language models: 3440, of which 3420 completed and were scored (20 failed); written by the template control: 800
- Result files archived rather than deleted when superseded: 43
- Commits in the project repository: 111, the first on 2026-07-27
- Every result file records the configuration that produced it and the version of the data it was computed on; a loader refuses a result whose data version does not match.
- Counts of logged decisions and defects are kept in the project notes (Decision Register, Defect Register), not in this file.

