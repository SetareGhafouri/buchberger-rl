Evaluation on 1000 held-out ideals from `3-20-10-weighted` (seed 987654321); checkpoint `epoch_0100.npz`.
Reference: best classical strategy = **degree**. Paired differences are reference minus method (positive = method is cheaper); CI is a 95% bootstrap interval; p-values from a two-sided Wilcoxon signed-rank test.

| method | mean_additions | std_additions | improvement_vs_ref_% | mean_paired_diff | ci95_low | ci95_high | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| random | 180.02 | 68.66 | -30.0 | -41.56 | -45.91 | -37.22 | 1.37e-67 |
| first | 176.71 | 66.47 | -27.6 | -38.25 | -42.68 | -33.91 | 1.75e-60 |
| degree | 138.45 | 48.62 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| normal | 139.79 | 48.73 | -1.0 | -1.34 | -2.93 | 0.25 | 0.119 |
| sugar | 156.39 | 56.79 | -13.0 | -17.93 | -21.37 | -14.51 | 7.99e-26 |
| agent (greedy) | 142.98 | 49.43 | -3.3 | -4.53 | -7.54 | -1.55 | 0.000441 |
| agent (sampled) | 141.1 | 50.29 | -1.9 | -2.64 | -5.69 | 0.44 | 0.0263 |
