import numpy as np
from typing import Optional


def initialize_particles(N: int, box_size: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random initial positions in a 2D periodic box."""
    return rng.random((N, 2)) * box_size


def _pairwise_diffs_pbc(positions: np.ndarray, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise displacements and distances under minimum-image PBC."""
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= box_size * np.round(diffs / box_size)
    dists = np.linalg.norm(diffs, axis=2)
    return diffs, dists


def step(
    positions: np.ndarray,
    rng: np.random.Generator,
    *,
    attraction: float = 0.01,
    repulsion: float = 0.02,
    noise: float = 0.01,
    box_size: float = 1.0,
    dt: float = 1.0,
    interaction_range: float = 0.6,
    repulsion_radius: float = 0.05,
    softening: float = 1e-6,
) -> np.ndarray:
    """
    One update step (overdamped dynamics) in a 2D periodic box:
      x <- x + dt*(attraction*F_att + repulsion*F_rep) + noise*eta

    - F_att: local attraction toward neighbors within interaction_range
             computed correctly under PBC via minimum-image displacements.
    - F_rep: short-range repulsion within repulsion_radius.
    """
    diffs, dists = _pairwise_diffs_pbc(positions, box_size)

    # --- Attraction: mean vector pointing from i to its neighbors (under PBC) ---
    # diffs[i,j] = x_i - x_j  => vector from i to j is -diffs[i,j]
    att_mask = (dists > 0) & (dists < interaction_range)
    counts = att_mask.sum(axis=1)

    mean_to_neighbors = np.zeros_like(positions)
    if np.any(counts > 0):
        sum_to_neighbors = np.sum((-diffs) * att_mask[:, :, None], axis=1)
        mean_to_neighbors = sum_to_neighbors / (counts[:, None] + 1e-12)

    F_att = mean_to_neighbors

    # --- Repulsion: push away from close neighbors ---
    rep_mask = (dists > 0) & (dists < repulsion_radius)
    rep_dir = diffs / (dists[:, :, None] + softening)
    rep_weight = (repulsion_radius - dists)
    F_rep = np.sum(rep_dir * rep_weight[:, :, None] * rep_mask[:, :, None], axis=1)

    # update
    positions = positions + dt * (attraction * F_att + repulsion * F_rep)
    positions = positions + noise * rng.normal(size=positions.shape)
    return positions % box_size


def run_simulation(
    *,
    N: int = 200,
    steps: int = 1000,
    box_size: float = 1.0,
    attraction: float = 0.01,
    repulsion: float = 0.02,
    noise: float = 0.01,
    dt: float = 1.0,
    interaction_range: float = 0.6,
    repulsion_radius: float = 0.05,
    seed: Optional[int] = None,
    save_every: int = 1,
) -> np.ndarray:
    """Run the simulation and return an array of saved configurations."""
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)
    positions = initialize_particles(N, box_size, rng)

    history = []
    for t in range(steps):
        positions = step(
            positions,
            rng=rng,
            attraction=attraction,
            repulsion=repulsion,
            noise=noise,
            box_size=box_size,
            dt=dt,
            interaction_range=interaction_range,
            repulsion_radius=repulsion_radius,
        )
        if (t + 1) % save_every == 0:
            history.append(positions.copy())

    return np.asarray(history)