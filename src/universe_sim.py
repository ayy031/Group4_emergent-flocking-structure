import numpy as np

def initialize_particles(N, box_size):
    """
    Initialize N particles randomly in a 2D box.
    """
    return np.random.rand(N, 2) * box_size


def step(positions, attraction=0.01, repulsion=0.01, noise=0.01, box_size=1.0):
    center = np.mean(positions, axis=0)
    direction = center - positions

    # attraction
    positions += attraction * direction

    # repulsion (short-range)
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mask = (dists > 0) & (dists < 0.05)
    repulsion_vec = np.sum(diffs * mask[:, :, None], axis=1)
    positions += repulsion * repulsion_vec

    # noise
    positions += noise * np.random.randn(*positions.shape)

    return positions % box_size


def run_simulation(N=200, steps=100, box_size=1.0, attraction=0.01, noise=0.01):
    positions = initialize_particles(N, box_size)

    history = []
    for _ in range(steps):
        positions = step(positions, attraction=attraction, noise=noise, box_size=box_size)
        history.append(positions.copy())

    return np.array(history)