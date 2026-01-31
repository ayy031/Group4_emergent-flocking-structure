# src/experiments.py
from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import from src/ when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import asdict
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


def _ensure_history(sim_out):
    """Convert simulation output to standard array format."""
    if isinstance(sim_out, (tuple, list)):
        sim_out = sim_out[0]
    history = np.asarray(sim_out)
    if history.ndim != 3 or history.shape[-1] != 2:
        raise ValueError(f"Expected history with shape (T, N, 2), got {history.shape}")
    return history


def frames_after_burn(history, burn_frac):
    """Discard initial transient - only keep frames after burn-in."""
    T = history.shape[0]
    start = int(np.floor(T * burn_frac))
    return history[start:]


def metrics_on_positions(pos, sim, met):
    """Compute all metrics for a single frame."""
    return {
        "nn": float(nearest_neighbor_distance(pos, box_size=sim.box_size)),
        "densvar": float(density_variance_grid(pos, box_size=sim.box_size, bins=met.bins)),
        "lcf": float(largest_cluster_fraction(pos, eps=met.eps, box_size=sim.box_size)),
        "nclusters": float(number_of_clusters(pos, eps=met.eps, box_size=sim.box_size)),
    }


def metrics_timeavg(history, sim, met):
    """
    Main metric computation: burn-in, then time-average.
    
    We throw out the first 60% of frames (system settling), then average
    metrics over remaining frames. Also track LCF fluctuations.
    """
    frames = frames_after_burn(history, met.burn_frac)
    if frames.shape[0] == 0:
        raise ValueError("No frames after burn-in. Reduce burn_frac or increase steps.")

    nn_list, dens_list, lcf_list, ncl_list = [], [], [], []

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
        "lcf_time_std": float(np.std(lcf_list)),  # within-run fluctuation
    }


def run_one(sim_base, met, seed, overrides=None):
    """
    Single simulation run with metrics.
    Takes base config, applies overrides, runs sim, returns results.
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

    row = {
        "seed": int(seed),
        **{k: float(v) for k, v in overrides.items()},
        **metrics,
        "runtime_sec": float(dt_run),
    }
    return row


def run_sweep(sim_base, met, sweep_name, sweep_values, seeds):
    """
    Parameter sweep: vary one parameter, repeat across seeds.
    Returns raw dataframe (one row per run).
    """
    rows = []
    for v in sweep_values:
        for s in seeds:
            rows.append(run_one(sim_base, met, seed=int(s), overrides={sweep_name: float(v)}))
    return pd.DataFrame(rows)


def summarize(df_raw, by):
    """
    Aggregate across seeds: compute mean and std for each parameter value.
    """
    metric_cols = [c for c in df_raw.columns if c not in ["seed", by]]
    g = df_raw.groupby(by)[metric_cols].agg(["mean", "std"]).reset_index()
    g.columns = [by] + [f"{a}_{b}" for a, b in g.columns[1:]]
    return g


if __name__ == "__main__":
    print("[experiments] smoke test starting...")

    sim = SimConfig(N=80, steps=400, save_every=10)
    met = MetricsConfig()

    row = run_one(sim, met, seed=0, overrides={"attraction": sim.attraction})
    print("[experiments] smoke test OK. Keys:", sorted(row.keys()))