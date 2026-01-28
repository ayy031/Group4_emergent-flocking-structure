# src/experiments.py
from __future__ import annotations

# Allow running this file directly (python src/experiments.py) by ensuring
# the project root is on sys.path so `import src.*` works.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional
import time

import numpy as np
import pandas as pd

from src.config import SimConfig, MetricsConfig
from src.baseline_clustering_2d import run_simulation
from src.metrics import (
    nearest_neighbor_distance,
    largest_cluster_fraction,
    density_variance_grid,
    number_of_clusters,
)


def _ensure_history(sim_out: Any) -> np.ndarray:
    """
    Normalize run_simulation output to a history array of shape (T, N, 2).
    If run_simulation returns (history, ...), we take the first element.
    """
    if isinstance(sim_out, (tuple, list)):
        sim_out = sim_out[0]
    history = np.asarray(sim_out)
    if history.ndim != 3 or history.shape[-1] != 2:
        raise ValueError(f"Expected history with shape (T, N, 2), got {history.shape}")
    return history


def frames_after_burn(history: np.ndarray, burn_frac: float) -> np.ndarray:
    T = history.shape[0]
    start = int(np.floor(T * burn_frac))
    return history[start:]


def metrics_on_positions(pos: np.ndarray, sim: SimConfig, met: MetricsConfig) -> Dict[str, float]:
    """Compute metrics on ONE frame (positions pos, shape (N,2))."""
    # Call metric functions using a positional first argument for maximum compatibility.
    return {
        "nn": float(nearest_neighbor_distance(pos, box_size=sim.box_size)),
        "densvar": float(density_variance_grid(pos, box_size=sim.box_size, bins=met.bins)),
        "lcf": float(largest_cluster_fraction(pos, eps=met.eps, box_size=sim.box_size)),
        "nclusters": float(number_of_clusters(pos, eps=met.eps, box_size=sim.box_size)),
    }


def metrics_timeavg(history: np.ndarray, sim: SimConfig, met: MetricsConfig) -> Dict[str, float]:
    """
    Burn-in + time-average metrics; includes within-run LCF fluctuation.
    Returns:
      nn_mean, densvar_mean, lcf_mean, nclusters_mean, lcf_time_std
    """
    frames = frames_after_burn(history, met.burn_frac)
    if frames.shape[0] == 0:
        raise ValueError(
            "No frames left after burn-in. Reduce burn_frac or increase steps/save_every."
        )

    nn_list: List[float] = []
    dens_list: List[float] = []
    lcf_list: List[float] = []
    ncl_list: List[float] = []

    for pos in frames:
        m = metrics_on_positions(pos, sim, met)
        nn_list.append(m["nn"])
        dens_list.append(m["densvar"])
        lcf_list.append(m["lcf"])
        ncl_list.append(m["nclusters"])

    return {
        "nn_mean": float(np.mean(nn_list)),
        "densvar_mean": float(np.mean(dens_list)),
        "lcf_mean": float(np.mean(lcf_list)),
        "nclusters_mean": float(np.mean(ncl_list)),
        "lcf_time_std": float(np.std(lcf_list)),
    }


def run_one(
    sim_base: SimConfig,
    met: MetricsConfig,
    seed: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run one simulation and return a single row dict:
      seed + overrides + time-avg metrics + runtime_sec
    """
    overrides = overrides or {}

    sim_dict = asdict(sim_base)
    sim_dict.update(overrides)
    sim = SimConfig(**sim_dict)

    t0 = time.time()
    out = run_simulation(
        N=sim.N,
        steps=sim.steps,
        seed=seed,
        save_every=sim.save_every,
        box_size=sim.box_size,
        dt=sim.dt,
        attraction=sim.attraction,
        interaction_range=sim.interaction_range,
        noise=sim.noise,
        repulsion=sim.repulsion,
        repulsion_radius=sim.repulsion_radius,
    )
    history = _ensure_history(out)
    metrics = metrics_timeavg(history, sim, met)
    dt_run = time.time() - t0

    row: Dict[str, Any] = {
        "seed": int(seed),
        **{k: float(v) for k, v in overrides.items()},
        **metrics,
        "runtime_sec": float(dt_run),
    }
    return row


def run_sweep(
    sim_base: SimConfig,
    met: MetricsConfig,
    sweep_name: str,
    sweep_values: Iterable[float],
    seeds: Iterable[int],
) -> pd.DataFrame:
    """
    Sweep one parameter over sweep_values; repeat over seeds.
    Returns a raw dataframe with one row per (value, seed).
    """
    rows: List[Dict[str, Any]] = []
    for v in sweep_values:
        for s in seeds:
            rows.append(
                run_one(sim_base, met, seed=int(s), overrides={sweep_name: float(v)})
            )
    return pd.DataFrame(rows)


def summarize(df_raw: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Group by `by` and compute mean/std across seeds for all metric columns.
    Output columns look like: lcf_mean_mean, lcf_mean_std, ...
    """
    metric_cols = [c for c in df_raw.columns if c not in ["seed", by]]
    g = df_raw.groupby(by)[metric_cols].agg(["mean", "std"]).reset_index()
    g.columns = [by] + [f"{a}_{b}" for a, b in g.columns[1:]]  # flatten MultiIndex
    return g


if __name__ == "__main__":
    print("[experiments] __main__ starting...", flush=True)

    # Fast smoke test (should finish quickly)
    sim = SimConfig(N=80, steps=400, save_every=10)
    met = MetricsConfig()

    row = run_one(sim, met, seed=0, overrides={"attraction": sim.attraction})
    print("[experiments] smoke test OK. Keys:", sorted(row.keys()), flush=True)