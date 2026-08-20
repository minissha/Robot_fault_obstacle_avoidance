# Statistics Report

## Success rate (mean, 95% bootstrap CI)

| controller                  | tier   | condition_group   |   n |   success_rate |   ci95_low |   ci95_high |
|:----------------------------|:-------|:------------------|----:|---------------:|-----------:|------------:|
| A_baseline                  | dense  | clean             |  50 |          0.08  |      0.02  |       0.16  |
| A_baseline                  | dense  | faulty            | 300 |          0.09  |      0.06  |       0.123 |
| A_baseline                  | sparse | clean             |  50 |          0.2   |      0.1   |       0.32  |
| A_baseline                  | sparse | faulty            | 300 |          0.15  |      0.11  |       0.193 |
| B_handtuned_flc             | dense  | clean             |  50 |          0.06  |      0     |       0.14  |
| B_handtuned_flc             | dense  | faulty            | 300 |          0.043 |      0.023 |       0.07  |
| B_handtuned_flc             | sparse | clean             |  50 |          0.1   |      0.02  |       0.2   |
| B_handtuned_flc             | sparse | faulty            | 300 |          0.097 |      0.063 |       0.13  |
| B_handtuned_flc_fault_aware | dense  | clean             |  50 |          0.08  |      0.02  |       0.16  |
| B_handtuned_flc_fault_aware | dense  | faulty            | 300 |          0.05  |      0.027 |       0.073 |
| B_handtuned_flc_fault_aware | sparse | clean             |  50 |          0.14  |      0.04  |       0.24  |
| B_handtuned_flc_fault_aware | sparse | faulty            | 300 |          0.107 |      0.073 |       0.143 |

## Pairwise Mann-Whitney U vs A_baseline (clean condition, per tier)

| comparison                                | tier   | condition   |   n_baseline |   n_other |    U |   p_value |   rank_biserial_effect_size | nan_free   |
|:------------------------------------------|:-------|:------------|-------------:|----------:|-----:|----------:|----------------------------:|:-----------|
| A_baseline vs B_handtuned_flc             | dense  | clean       |           50 |        50 | 1275 |    0.7023 |                       -0.02 | True       |
| A_baseline vs B_handtuned_flc             | sparse | clean       |           50 |        50 | 1375 |    0.1652 |                       -0.1  | True       |
| A_baseline vs B_handtuned_flc_fault_aware | dense  | clean       |           50 |        50 | 1250 |    1      |                        0    | True       |
| A_baseline vs B_handtuned_flc_fault_aware | sparse | clean       |           50 |        50 | 1325 |    0.4299 |                       -0.06 | True       |

## Clean vs faulty, per controller/tier (inverse-success-anomaly check)

| controller                  | tier   |   n_clean |   n_faulty |   clean_success |   faulty_success | faulty_outperforms_clean   |    U |   p_value |   rank_biserial_effect_size | nan_free   |
|:----------------------------|:-------|----------:|-----------:|----------------:|-----------------:|:---------------------------|-----:|----------:|----------------------------:|:-----------|
| A_baseline                  | dense  |        50 |        300 |            0.08 |           0.09   | True                       | 7425 |    0.8192 |                      0.01   | True       |
| A_baseline                  | sparse |        50 |        300 |            0.2  |           0.15   | False                      | 7875 |    0.3698 |                     -0.05   | True       |
| B_handtuned_flc             | dense  |        50 |        300 |            0.06 |           0.0433 | False                      | 7625 |    0.6034 |                     -0.0167 | True       |
| B_handtuned_flc             | sparse |        50 |        300 |            0.1  |           0.0967 | False                      | 7525 |    0.9425 |                     -0.0033 | True       |
| B_handtuned_flc_fault_aware | dense  |        50 |        300 |            0.08 |           0.05   | False                      | 7725 |    0.3878 |                     -0.03   | True       |
| B_handtuned_flc_fault_aware | sparse |        50 |        300 |            0.14 |           0.1067 | False                      | 7750 |    0.4895 |                     -0.0333 | True       |
