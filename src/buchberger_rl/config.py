"""Experiment configuration (YAML <-> dataclasses)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class EnvConfig:
    distribution: str = "3-20-10-weighted"   # n-d-s-<degrees>, see buchberger_rl.ideals
    order: str = "grevlex"
    elimination: str = "gebauermoeller"
    k: int = 2                               # leading terms per polynomial in the features
    feature_scale: float = 0.05              # exponents are multiplied by this before the network


@dataclass
class PPOConfig:
    hidden: tuple[int, ...] = (128,)
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.97
    clip_eps: float = 0.2
    target_kl: float = 0.01                  # early-stop policy updates when KL > 1.5 * target
    policy_updates: int = 40                 # max minibatch gradient steps per epoch
    batch_size: int = 512                    # states per minibatch
    entropy_coef: float = 0.0
    value: str = "degree"                    # "degree" (rollout baseline) or "none"
    normalize_advantages: bool = True


@dataclass
class RunConfig:
    epochs: int = 200
    episodes_per_epoch: int = 100
    workers: int = 2
    seed: int = 0
    checkpoint_every: int = 25
    smoothing: int = 10                      # window for selecting the best checkpoint


@dataclass
class ExperimentConfig:
    name: str = "ppo"
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for key, value in (overrides or {}).items():
            section, _, name = key.partition(".")
            raw.setdefault(section, {})[name] = value
        return cls(
            name=raw.get("name", cls.name),
            env=_build(EnvConfig, raw.get("env", {})),
            ppo=_build(PPOConfig, raw.get("ppo", {})),
            run=_build(RunConfig, raw.get("run", {})),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ppo"]["hidden"] = list(self.ppo.hidden)
        return d

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    if "hidden" in values:
        values = {**values, "hidden": tuple(values["hidden"])}
    return cls(**values)
