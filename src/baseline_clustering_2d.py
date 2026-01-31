import numpy as np


def initialize_particles(N, box_size, rng):
    """Start particles randomly distributed in the box."""
    return rng.random((N, 2)) * box_size


def _pairwise_diffs_pbc(positions, box_size):
    """
    All pairwise distances with periodic BC.
    Returns displacement vectors and distances.
    """
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= box_size * np.round(diffs / box_size)  # minimum image convention
    dists = np.linalg.norm(diffs, axis=2)
    return diffs, dists


def step(
    positions,
    rng,
    *,
    attraction=0.01,
    repulsion=0.02,
    noise=0.01,
    box_size=1.0,
    dt=1.0,
    interaction_range=0.6,
    repulsion_radius=0.05,
    softening=1e-6,
):
    """
    Alternative model: overdamped dynamics with attraction/repulsion.
    
    Unlike the velocity-based Vicsek model, here we directly update positions
    based on forces from neighbors. Attraction pulls toward nearby particles,
    repulsion prevents collapse. No explicit velocities - just positions.
    """
    diffs, dists = _pairwise_diffs_pbc(positions, box_size)

    # Attraction: move toward average position of neighbors
    att_mask = (dists > 0) & (dists < interaction_range)
    counts = att_mask.sum(axis=1)

    mean_to_neighbors = np.zeros_like(positions)
    if np.any(counts > 0):
        sum_to_neighbors = np.sum((-diffs) * att_mask[:, :, None], axis=1)
        mean_to_neighbors = sum_to_neighbors / (counts[:, None] + 1e-12)

    F_att = mean_to_neighbors

    # Repulsion: push away if too close
    rep_mask = (dists > 0) & (dists < repulsion_radius)
    rep_dir = diffs / (dists[:, :, None] + softening)
    rep_weight = (repulsion_radius - dists)
    F_rep = np.sum(rep_dir * rep_weight[:, :, None] * rep_mask[:, :, None], axis=1)

    # Update positions (overdamped: position responds directly to forces)
    positions = positions + dt * (attraction * F_att + repulsion * F_rep)
    positions = positions + noise * rng.normal(size=positions.shape)
    return positions % box_size


def run_simulation(
    *,
    N=200,
    steps=1000,
    box_size=1.0,
    attraction=0.01,
    repulsion=0.02,
    noise=0.01,
    dt=1.0,
    interaction_range=0.6,
    repulsion_radius=0.05,
    seed=None,
    save_every=1,
):
    """Run simulation and return saved trajectory."""
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