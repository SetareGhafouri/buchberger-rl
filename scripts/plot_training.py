#!/usr/bin/env python
"""Plot training curves of one or more runs against classical baselines.

    python scripts/plot_training.py --runs runs/ppo_3-20-10-weighted \
        --baselines runs/ppo_3-20-10-weighted/eval_3-20-10-weighted/summary.json \
        --out results/figures/training_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import _plotstyle as style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402


def read_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["epoch"]) for r in rows]), np.array([float(r["mean_additions"]) for r in rows])


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    return np.array([x[max(0, i - w + 1): i + 1].mean() for i in range(len(x))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--baselines", type=Path, default=None, help="summary.json from evaluate.py")
    ap.add_argument("--show", nargs="+", default=["degree", "normal", "true_degree"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--title", default="Training on 3-20-10-weighted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = args.labels or [r.name for r in args.runs]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xmax = 0
    for i, (run, label) in enumerate(zip(args.runs, labels)):
        ep, y = read_log(run / "train_log.csv")
        xmax = max(xmax, ep.max())
        ax.plot(ep, y, color=style.SERIES[i], alpha=0.18, linewidth=1.0)
        ax.plot(ep, rolling(y, args.window), color=style.SERIES[i], label=label)

    if args.baselines:
        rows = {r["method"]: r for r in json.loads(args.baselines.read_text())["rows"]}
        lines = sorted((rows[n]["mean_additions"], n) for n in args.show if n in rows)
        for v, _ in lines:
            ax.axhline(v, color=style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        # spread labels vertically so that nearby baselines stay legible
        lo, hi = ax.get_ylim()
        gap = 0.045 * (hi - lo)
        positions = []
        for v, _ in lines:
            positions.append(max(v, positions[-1] + gap) if positions else v)
        for (v, name), y in zip(lines, positions):
            ax.annotate(f"{name} ({v:.1f})", xy=(xmax, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=style.TEXT_2, annotation_clip=False)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("epoch (100 episodes each)")
    ax.set_ylabel("mean polynomial additions per ideal")
    ax.set_title(args.title)
    if len(args.runs) > 1:
        ax.legend(loc="upper right")
    fig.text(0.01, 0.005, f"Faint lines: per-epoch means; solid: {args.window}-epoch rolling mean. "
             "Dashed: heuristics on 1,000 held-out ideals. Lower is better.",
             fontsize=7.5, color=style.TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 0.88, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
