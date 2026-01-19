import numpy as np


def nearest_neighbor_distance(positions, box_size):
    """
    Compute average nearest-neighbour distance under periodic boundary conditions.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Particle positions.
    box_size : float
        Size of the simulation box (assumed square).

    Returns
    -------
    float
        Mean nearest-neighbour distance.
    """
    N = positions.shape[0]
    distances = []

    for i in range(N):
        diff = positions - positions[i]
        diff -= box_size * np.round(diff / box_size)  # periodic BC
        dists = np.linalg.norm(diff, axis=1)
        dists = dists[dists > 0]  # remove self-distance
        distances.append(np.min(dists))

    return np.mean(distances)


def largest_cluster_fraction(positions, eps, box_size):
    """
    Compute fraction of particles in the largest cluster.
    Simple distance-based clustering.

    Parameters
    ----------
    positions : np.ndarray
    eps : float
        Distance threshold to define neighbours.
    box_size : float

    Returns
    -------
    float
        Size of largest cluster divided by N.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        cluster = []

        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster.append(j)

            diff = positions - positions[j]
            diff -= box_size * np.round(diff / box_size)
            dists = np.linalg.norm(diff, axis=1)

            neighbors = np.where(dists < eps)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)

        clusters.append(cluster)

    largest = max(len(c) for c in clusters)
    return largest / N