Evaluation on 1000 held-out ideals from `3-20-10-weighted` (seed 987654321); checkpoint `best.npz`.
Reference: best classical strategy = **degree**. Paired differences are reference minus method (positive = method is cheaper); CI is a 95% bootstrap interval; p-values from a two-sided Wilcoxon signed-rank test.

| method | mean_additions | std_additions | improvement_vs_ref_% | mean_paired_diff | ci95_low | ci95_high | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| random | 180.02 | 68.66 | -30.0 | -41.56 | -45.91 | -37.22 | 1.37e-67 |
| first | 176.71 | 66.47 | -27.6 | -38.25 | -42.68 | -33.91 | 1.75e-60 |
| degree | 138.45 | 48.62 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| normal | 139.79 | 48.73 | -1.0 | -1.34 | -2.93 | 0.25 | 0.119 |
| sugar | 156.39 | 56.79 | -13.0 | -17.93 | -21.37 | -14.51 | 7.99e-26 |
| true_degree | 126.84 | 44.04 | 8.4 | 11.62 | 9.12 | 14.2 | 9.79e-22 |
| monomial_first | 138.03 | 49.29 | 0.3 | 0.42 | -0.09 | 1.06 | 7.76e-09 |
| agent (greedy) | 120.15 | 42.1 | 13.2 | 18.3 | 15.2 | 21.46 | 3.91e-29 |
| agent (sampled) | 121.59 | 45.67 | 12.2 | 16.86 | 13.63 | 20.12 | 6.19e-26 |
