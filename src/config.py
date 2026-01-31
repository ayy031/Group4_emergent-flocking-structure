# src/config.py
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SimConfig:
    """Default simulation parameters we settled on after some trial runs."""
    N: int = 300
    steps: int = 1500
    dt: float = 1.0
    box_size: float = 1.0
    save_every: int = 30

    attraction: float = 0.10
    interaction_range: float = 0.30
    noise: float = 0.03
    repulsion: float = 0.02
    repulsion_radius: float = 0.05


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metric computation parameters.
    eps: clustering distance threshold
    burn_frac: what fraction of trajectory to discard as transient
    """
    eps: float = 0.06
    bins: int = 20
    min_size: int = 3
    burn_frac: float = 0.6


def cfg_dict(cfg):
    return asdict(cfg)