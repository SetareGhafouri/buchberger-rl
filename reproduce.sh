#!/usr/bin/env bash
# Reproduce every number and figure in the README (about 2 hours on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")"

python -m unittest discover -s tests

python scripts/benchmark_strategies.py --n 1000 --out results/strategies

python scripts/train.py --config configs/ppo_3-20-10-weighted.yaml --out runs/ppo_3-20-10-weighted
for t in no_value k1 hidden4; do
  python scripts/train.py --config configs/ablation_$t.yaml --out runs/ablation_$t
done
# second seed for the two closest variants
python scripts/train.py --config configs/ppo_3-20-10-weighted.yaml --set run.seed=1 --set run.epochs=100 --out runs/full_seed1
python scripts/train.py --config configs/ablation_k1.yaml --set run.seed=1 --out runs/ablation_k1_seed1

python scripts/evaluate.py --run runs/ppo_3-20-10-weighted --n 1000
python scripts/evaluate.py --run runs/ppo_3-20-10-weighted --n 1000 --distribution 3-20-4-weighted --no-lookahead
python scripts/evaluate.py --run runs/ppo_3-20-10-weighted --n 1000 --checkpoint epoch_0100.npz --tag epoch100 --no-lookahead
for r in ablation_no_value ablation_k1 ablation_hidden4 full_seed1 ablation_k1_seed1; do
  python scripts/evaluate.py --run runs/$r --n 1000 --checkpoint epoch_0100.npz --no-lookahead
done

python scripts/analyze_policy.py --run runs/ppo_3-20-10-weighted --n 300 --out results
python scripts/plot_training.py --runs runs/ppo_3-20-10-weighted \
  --baselines runs/ppo_3-20-10-weighted/eval_3-20-10-weighted/summary.json \
  --out results/figures/training_curve.png
python scripts/plot_training.py --runs runs/ppo_3-20-10-weighted runs/ablation_no_value runs/ablation_k1 runs/ablation_hidden4 \
  --labels "full model" "no value baseline" "k = 1 (leading term only)" "4 hidden units" \
  --title "Ablations on 3-20-10-weighted (seed 0)" --show degree \
  --baselines runs/ppo_3-20-10-weighted/eval_3-20-10-weighted/summary.json \
  --out results/figures/ablations.png
