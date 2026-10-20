#!/usr/bin/env python
"""Train a pair-selection agent with PPO.

Examples
--------
    python scripts/train.py --config configs/ppo_3-20-10-weighted.yaml
    python scripts/train.py --config configs/ppo_3-20-10-weighted.yaml \
        --set run.epochs=50 --set ppo.value=none --out runs/no_value
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from buchberger_rl.agent.ppo import PPOTrainer  # noqa: E402
from buchberger_rl.config import ExperimentConfig  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config entry (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: runs/<name>)")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config, parse_overrides(args.set))
    out = args.out or Path("runs") / cfg.name
    print(f"training '{cfg.name}' on {cfg.env.distribution} -> {out}")
    best = PPOTrainer(cfg, out).train()
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
