# Statistics Report

## Success rate (mean, 95% bootstrap CI)

| controller              | tier   | condition_group   |   n |   success_rate |   ci95_low |   ci95_high |
|:------------------------|:-------|:------------------|----:|---------------:|-----------:|------------:|
| A_baseline              | dense  | clean             |  50 |          0.08  |      0.02  |       0.16  |
| A_baseline              | dense  | faulty            | 300 |          0.09  |      0.06  |       0.123 |
| A_baseline              | sparse | clean             |  50 |          0.2   |      0.1   |       0.32  |
| A_baseline              | sparse | faulty            | 300 |          0.15  |      0.11  |       0.193 |
| B_handtuned_flc         | dense  | clean             |  50 |          0.06  |      0     |       0.14  |
| B_handtuned_flc         | dense  | faulty            | 300 |          0.043 |      0.023 |       0.07  |
| B_handtuned_flc         | sparse | clean             |  50 |          0.1   |      0.02  |       0.2   |
| B_handtuned_flc         | sparse | faulty            | 300 |          0.097 |      0.063 |       0.13  |
| D_nsga2_flc             | dense  | clean             |  50 |          0.02  |      0     |       0.06  |
| D_nsga2_flc             | dense  | faulty            | 300 |          0.017 |      0.003 |       0.033 |
| D_nsga2_flc             | sparse | clean             |  50 |          0.02  |      0     |       0.06  |
| D_nsga2_flc             | sparse | faulty            | 300 |          0.05  |      0.027 |       0.077 |
| D_nsga2_flc_efficiency  | dense  | clean             |  50 |          0.06  |      0     |       0.14  |
| D_nsga2_flc_efficiency  | dense  | faulty            | 300 |          0.037 |      0.017 |       0.06  |
| D_nsga2_flc_efficiency  | sparse | clean             |  50 |          0.08  |      0.02  |       0.16  |
| D_nsga2_flc_efficiency  | sparse | faulty            | 300 |          0.07  |      0.043 |       0.1   |
| D_nsga2_flc_fault_aware | dense  | clean             |  50 |          0     |      0     |       0     |
| D_nsga2_flc_fault_aware | dense  | faulty            | 300 |          0.01  |      0     |       0.023 |
| D_nsga2_flc_fault_aware | sparse | clean             |  50 |          0.02  |      0     |       0.06  |
| D_nsga2_flc_fault_aware | sparse | faulty            | 300 |          0.037 |      0.017 |       0.06  |
| D_nsga2_flc_safety      | dense  | clean             |  50 |          0.08  |      0.02  |       0.16  |
| D_nsga2_flc_safety      | dense  | faulty            | 300 |          0.053 |      0.03  |       0.08  |
| D_nsga2_flc_safety      | sparse | clean             |  50 |          0.12  |      0.04  |       0.22  |
| D_nsga2_flc_safety      | sparse | faulty            | 300 |          0.077 |      0.05  |       0.107 |
| E_ann_imitator          | dense  | clean             |  50 |          0     |      0     |       0     |
| E_ann_imitator          | dense  | faulty            | 300 |          0.01  |      0     |       0.023 |
| E_ann_imitator          | sparse | clean             |  50 |          0     |      0     |       0     |
| E_ann_imitator          | sparse | faulty            | 300 |          0.017 |      0.003 |       0.033 |

## Pairwise Mann-Whitney U vs A_baseline (clean condition, per tier)

| comparison                            | tier   | condition   |   n_baseline |   n_other |    U |   p_value |   rank_biserial_effect_size | nan_free   |
|:--------------------------------------|:-------|:------------|-------------:|----------:|-----:|----------:|----------------------------:|:-----------|
| A_baseline vs B_handtuned_flc         | dense  | clean       |           50 |        50 | 1275 |    0.7023 |                       -0.02 | True       |
| A_baseline vs B_handtuned_flc         | sparse | clean       |           50 |        50 | 1375 |    0.1652 |                       -0.1  | True       |
| A_baseline vs D_nsga2_flc             | dense  | clean       |           50 |        50 | 1325 |    0.1737 |                       -0.06 | True       |
| A_baseline vs D_nsga2_flc             | sparse | clean       |           50 |        50 | 1475 |    0.0043 |                       -0.18 | True       |
| A_baseline vs D_nsga2_flc_efficiency  | dense  | clean       |           50 |        50 | 1275 |    0.7023 |                       -0.02 | True       |
| A_baseline vs D_nsga2_flc_efficiency  | sparse | clean       |           50 |        50 | 1400 |    0.0864 |                       -0.12 | True       |
| A_baseline vs D_nsga2_flc_fault_aware | dense  | clean       |           50 |        50 | 1350 |    0.0433 |                       -0.08 | True       |
| A_baseline vs D_nsga2_flc_fault_aware | sparse | clean       |           50 |        50 | 1475 |    0.0043 |                       -0.18 | True       |
| A_baseline vs D_nsga2_flc_safety      | dense  | clean       |           50 |        50 | 1250 |    1      |                        0    | True       |
| A_baseline vs D_nsga2_flc_safety      | sparse | clean       |           50 |        50 | 1350 |    0.2801 |                       -0.08 | True       |
| A_baseline vs E_ann_imitator          | dense  | clean       |           50 |        50 | 1350 |    0.0433 |                       -0.08 | True       |
| A_baseline vs E_ann_imitator          | sparse | clean       |           50 |        50 | 1500 |    0.0009 |                       -0.2  | True       |

## Clean vs faulty, per controller/tier (inverse-success-anomaly check)

| controller              | tier   |   n_clean |   n_faulty |   clean_success |   faulty_success | faulty_outperforms_clean   |    U |   p_value |   rank_biserial_effect_size | nan_free   |
|:------------------------|:-------|----------:|-----------:|----------------:|-----------------:|:---------------------------|-----:|----------:|----------------------------:|:-----------|
| A_baseline              | dense  |        50 |        300 |            0.08 |           0.09   | True                       | 7425 |    0.8192 |                      0.01   | True       |
| A_baseline              | sparse |        50 |        300 |            0.2  |           0.15   | False                      | 7875 |    0.3698 |                     -0.05   | True       |
| B_handtuned_flc         | dense  |        50 |        300 |            0.06 |           0.0433 | False                      | 7625 |    0.6034 |                     -0.0167 | True       |
| B_handtuned_flc         | sparse |        50 |        300 |            0.1  |           0.0967 | False                      | 7525 |    0.9425 |                     -0.0033 | True       |
| D_nsga2_flc             | dense  |        50 |        300 |            0.02 |           0.0167 | False                      | 7525 |    0.8693 |                     -0.0033 | True       |
| D_nsga2_flc             | sparse |        50 |        300 |            0.02 |           0.05   | True                       | 7275 |    0.3488 |                      0.03   | True       |
| D_nsga2_flc_efficiency  | dense  |        50 |        300 |            0.06 |           0.0367 | False                      | 7675 |    0.4376 |                     -0.0233 | True       |
| D_nsga2_flc_efficiency  | sparse |        50 |        300 |            0.08 |           0.07   | False                      | 7575 |    0.8009 |                     -0.01   | True       |
| D_nsga2_flc_fault_aware | dense  |        50 |        300 |            0    |           0.01   | True                       | 7425 |    0.4812 |                      0.01   | True       |
| D_nsga2_flc_fault_aware | sparse |        50 |        300 |            0.02 |           0.0367 | True                       | 7375 |    0.5509 |                      0.0167 | True       |
| D_nsga2_flc_safety      | dense  |        50 |        300 |            0.08 |           0.0533 | False                      | 7700 |    0.4538 |                     -0.0267 | True       |
| D_nsga2_flc_safety      | sparse |        50 |        300 |            0.12 |           0.0767 | False                      | 7825 |    0.3049 |                     -0.0433 | True       |
| E_ann_imitator          | dense  |        50 |        300 |            0    |           0.01   | True                       | 7425 |    0.4812 |                      0.01   | True       |
| E_ann_imitator          | sparse |        50 |        300 |            0    |           0.0167 | True                       | 7375 |    0.3605 |                      0.0167 | True       |
