Evaluation on 1000 held-out ideals from `3-20-4-weighted` (seed 987654321); checkpoint `best.npz`.
Reference: best classical strategy = **degree**. Paired differences are reference minus method (positive = method is cheaper); CI is a 95% bootstrap interval; p-values from a two-sided Wilcoxon signed-rank test.

| method | mean_additions | std_additions | improvement_vs_ref_% | mean_paired_diff | ci95_low | ci95_high | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| random | 198.08 | 95.95 | -27.6 | -42.8 | -47.99 | -37.57 | 2.89e-55 |
| first | 197.98 | 90.72 | -27.5 | -42.71 | -47.35 | -38.1 | 1.99e-67 |
| degree | 155.27 | 69.84 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| normal | 156.21 | 69.23 | -0.6 | -0.94 | -2.75 | 0.87 | 0.494 |
| sugar | 173.9 | 77.66 | -12.0 | -18.63 | -22.37 | -14.9 | 9.88e-27 |
| true_degree | 144.93 | 64.73 | 6.7 | 10.34 | 7.82 | 12.9 | 2.61e-21 |
| monomial_first | 155.71 | 70.03 | -0.3 | -0.44 | -0.81 | 0.01 | 7.96e-28 |
| agent (greedy) | 137.71 | 55.67 | 11.3 | 17.56 | 14.2 | 21.11 | 1.04e-23 |
| agent (sampled) | 142.18 | 58.11 | 8.4 | 13.1 | 9.73 | 16.59 | 4.54e-13 |
