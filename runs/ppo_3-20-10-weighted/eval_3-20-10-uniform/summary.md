Evaluation on 1000 held-out ideals from `3-20-10-uniform` (seed 987654321); checkpoint `best.npz`.
Reference: best classical strategy = **degree**. Paired differences are reference minus method (positive = method is cheaper); CI is a 95% bootstrap interval; p-values from a two-sided Wilcoxon signed-rank test.

| method | mean_additions | std_additions | improvement_vs_ref_% | mean_paired_diff | ci95_low | ci95_high | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| random | 306.14 | 97.17 | -60.5 | -115.38 | -121.47 | -109.26 | 7.18e-138 |
| first | 343.97 | 114.66 | -80.3 | -153.21 | -160.31 | -146.22 | 1.93e-154 |
| degree | 190.77 | 53.41 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| normal | 192.98 | 55.36 | -1.2 | -2.21 | -5.0 | 0.61 | 0.246 |
| sugar | 256.33 | 85.76 | -34.4 | -65.56 | -70.89 | -60.38 | 1.44e-92 |
| true_degree | 178.21 | 51.03 | 6.6 | 12.55 | 9.16 | 15.9 | 1e-12 |
| monomial_first | 191.51 | 56.31 | -0.4 | -0.74 | -1.83 | 0.43 | 1.42e-34 |
| agent (greedy) | 189.39 | 59.21 | 0.7 | 1.38 | -2.74 | 5.4 | 0.203 |
| agent (sampled) | 194.94 | 59.63 | -2.2 | -4.17 | -8.33 | -0.07 | 0.193 |
