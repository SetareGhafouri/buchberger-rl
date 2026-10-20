#!/usr/bin/env python
"""What has the agent learned? Relate its preference scores to interpretable pair features.

For every state visited by the greedy agent on held-out ideals, each candidate
pair ``(f, g)`` is described by

* ``lcm_degree``  -- total degree of lcm(LM f, LM g) (the quantity the degree strategy minimises);
* ``true_degree`` -- degree of LM(S(f, g)) (what the true-degree strategy minimises);
* ``monomial``    -- whether S(f, g) is a single term.

We report, averaged over states, the Spearman correlation between the agent's
scores and each feature, and how often the agent's choice coincides with the
choice of each heuristic. The same statistics for the untrained network (same
initialisation seed) serve as a control.

    python scripts/analyze_policy.py --run runs/ppo_3-20-10-weighted --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _plotstyle as style  # noqa: E402

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

from buchberger_rl.agent.network import PairScorer  # noqa: E402
from buchberger_rl.agent.ppo import make_env  # noqa: E402
from buchberger_rl.algebra import monomials as M  # noqa: E402
from buchberger_rl.algebra.reduction import spoly  # noqa: E402
from buchberger_rl.config import ExperimentConfig  # noqa: E402
from buchberger_rl.evaluation import sample_ideals  # noqa: E402

ANALYSIS_SEED = 55_555


def pair_table(state):
    rows = []
    for i, j in state.pairs:
        s = spoly(state.polys[i], state.polys[j])
        rows.append((
            M.degree(M.lcm(state.lms[i], state.lms[j])),
            -1 if s.is_zero() else M.degree(s.LM),
            float(len(s) <= 1),
        ))
    return np.array(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.run / "config.yaml")
    trained = PairScorer.load(args.run / "checkpoints" / args.checkpoint)
    untrained = PairScorer(trained.input_dim, cfg.ppo.hidden, seed=cfg.run.seed)
    env = make_env(cfg.env)
    ideals = sample_ideals(cfg.env.distribution, args.n, ANALYSIS_SEED, cfg.env.order, cfg.env.elimination)

    names = ["lcm_degree", "true_degree", "monomial"]
    corr = {m: {k: [] for k in names} for m in ("trained", "untrained")}
    agree = {m: {"degree": [], "true_degree": [], "monomial_first": []} for m in ("trained", "untrained")}
    by_gap = {m: {} for m in ("trained", "untrained")}  # lcm-degree gap above state minimum -> z-scores

    for F in ideals:
        obs = env.reset(F)
        done = env.state.done
        while not done:
            if len(env.state.pairs) >= 2:
                T = pair_table(env.state)
                for tag, net in (("trained", trained), ("untrained", untrained)):
                    s = net.scores(obs)
                    for k, name in enumerate(names):
                        if np.ptp(T[:, k]) > 0 and np.ptp(s) > 0:
                            corr[tag][name].append(sps.spearmanr(s, T[:, k]).statistic)
                    a = int(np.argmax(s))
                    agree[tag]["degree"].append(T[a, 0] == T[:, 0].min())
                    agree[tag]["true_degree"].append(T[a, 1] == T[:, 1].min())
                    if T[:, 2].max() > 0:
                        agree[tag]["monomial_first"].append(T[a, 2] == 1)
                    z = (s - s.mean()) / (s.std() + 1e-12)
                    for gap, zz in zip(T[:, 0] - T[:, 0].min(), z):
                        by_gap[tag].setdefault(int(gap), []).append(zz)
            a = int(np.argmax(trained.scores(obs)))
            obs, _, done, _ = env.step(a)

    report = {
        tag: {
            "spearman_score_vs": {k: round(float(np.mean(v)), 3) for k, v in corr[tag].items()},
            "agreement_with": {k: round(float(np.mean(v)), 3) for k, v in agree[tag].items()},
        }
        for tag in corr
    }
    report["states_analyzed"] = len(agree["trained"]["degree"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "policy_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Figure: mean within-state standardized score vs. lcm-degree gap.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for tag, color in (("untrained", style.MUTED), ("trained", style.SERIES[0])):
        gaps = sorted(g for g, v in by_gap[tag].items() if len(v) >= 50 and g <= 15)
        means = [np.mean(by_gap[tag][g]) for g in gaps]
        ax.plot(gaps, means, color=color, marker="o", markersize=4,
                label=f"{tag} network" + (" (control)" if tag == "untrained" else ""))
    ax.axhline(0, color=style.MUTED, linewidth=0.8)
    ax.set_xlabel("deg lcm(LM f, LM g) minus the smallest such degree in the state")
    ax.set_ylabel("mean standardized preference score")
    ax.set_title("Preference score vs. pair degree")
    ax.legend(loc="upper right")
    fig.tight_layout()
    (args.out / "figures").mkdir(exist_ok=True)
    fig.savefig(args.out / "figures" / "preference_vs_degree.png")
    print(f"wrote {args.out / 'figures' / 'preference_vs_degree.png'}")


if __name__ == "__main__":
    main()
