#!/usr/bin/env python
"""Benchmark hand-designed selection strategies across ideal distributions.

    python scripts/benchmark_strategies.py --distributions 3-20-10-weighted 3-20-10-uniform \
        --n 1000 --out results/strategies
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from buchberger_rl.evaluation import format_table, sample_ideals, strategy_costs  # noqa: E402
from buchberger_rl.strategies import STRATEGIES, make_strategy  # noqa: E402

DEFAULT = ["3-20-10-weighted", "3-20-4-weighted", "3-20-10-uniform", "3-10-10-maximum"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distributions", nargs="+", default=DEFAULT)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--elimination", default="gebauermoeller")
    ap.add_argument("--out", type=Path, default=Path("results/strategies"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dist in args.distributions:
        ideals = sample_ideals(dist, args.n, args.seed, elimination=args.elimination)
        for name in args.strategies:
            strat = make_strategy(name, seed=args.seed) if name == "random" else make_strategy(name)
            t0 = time.time()
            c = strategy_costs(strat, ideals, args.elimination)
            rows.append({
                "distribution": dist,
                "strategy": name,
                "mean_additions": round(float(c.mean()), 2),
                "std_additions": round(float(c.std()), 2),
                "median": float(sorted(c)[len(c) // 2]),
                "ms_per_ideal": round(1000 * (time.time() - t0) / len(c), 2),
            })
            print(f"{dist:18s} {name:15s} {rows[-1]['mean_additions']:9.2f} ± {rows[-1]['std_additions']:7.2f}")

    with (args.out / "strategies.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "strategies.md").write_text(format_table(rows) + "\n")
    print(f"wrote {args.out}/strategies.csv and strategies.md")


if __name__ == "__main__":
    main()
