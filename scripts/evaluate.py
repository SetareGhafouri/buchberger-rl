#!/usr/bin/env python
"""Evaluate a trained agent against selection heuristics on held-out ideals.

    python scripts/evaluate.py --run runs/ppo_3-20-10-weighted --n 1000
    python scripts/evaluate.py --run runs/ppo_3-20-10-weighted --distribution 3-20-4-weighted

The evaluation ideals are drawn with a seed disjoint from all training seeds.
Writes ``eval_<distribution>/{per_ideal.csv, summary.md, summary.json}`` inside the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from buchberger_rl.agent.network import PairScorer  # noqa: E402
from buchberger_rl.config import ExperimentConfig  # noqa: E402
from buchberger_rl.evaluation import (  # noqa: E402
    agent_costs,
    best_strategy,
    compare,
    format_table,
    sample_ideals,
    strategy_costs,
)
from buchberger_rl.strategies import CLASSICAL, make_strategy  # noqa: E402

EVAL_SEED = 987_654_321


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.npz")
    ap.add_argument("--distribution", default=None, help="defaults to the training distribution")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--no-lookahead", action="store_true", help="skip true_degree / monomial_first")
    ap.add_argument("--tag", default=None, help="suffix for the output directory")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.run / "config.yaml")
    dist = args.distribution or cfg.env.distribution
    env_cfg = replace(cfg.env, distribution=dist)
    policy = PairScorer.load(args.run / "checkpoints" / args.checkpoint)
    ideals = sample_ideals(dist, args.n, args.seed, cfg.env.order, cfg.env.elimination)

    results = {"random": strategy_costs(make_strategy("random", seed=args.seed), ideals)}
    for name in CLASSICAL + (() if args.no_lookahead else ("true_degree", "monomial_first")):
        results[name] = strategy_costs(make_strategy(name), ideals, cfg.env.elimination)
        print(f"{name:15s} {results[name].mean():9.2f}")
    results["agent (greedy)"] = agent_costs(policy, ideals, env_cfg, greedy=True)
    results["agent (sampled)"] = agent_costs(policy, ideals, env_cfg, greedy=False, seed=args.seed)
    print(f"{'agent (greedy)':15s} {results['agent (greedy)'].mean():9.2f}")

    reference = best_strategy(results, CLASSICAL)
    rows = [c.as_row() for c in compare(results, reference)]

    out = args.run / f"eval_{dist}{'_' + args.tag if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_ideal.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ideal", *results])
        for i in range(len(ideals)):
            w.writerow([i, *(int(results[k][i]) for k in results)])
    header = (
        f"Evaluation on {args.n} held-out ideals from `{dist}` (seed {args.seed}); "
        f"checkpoint `{args.checkpoint}`.\nReference: best classical strategy = **{reference}**. "
        "Paired differences are reference minus method (positive = method is cheaper); "
        "CI is a 95% bootstrap interval; p-values from a two-sided Wilcoxon signed-rank test.\n\n"
    )
    (out / "summary.md").write_text(header + format_table(rows) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"distribution": dist, "n": args.n, "seed": args.seed, "reference": reference, "rows": rows}, indent=2))
    print("\n" + format_table(rows))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
