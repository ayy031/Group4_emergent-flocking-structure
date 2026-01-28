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
    
    Two particles are considered connected if their distance under
    periodic boundary conditions is smaller than eps.
    A cluster is defined as a connected component.
   
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


def density_variance_grid(positions: np.ndarray, box_size: float, bins: int = 20, normalized: bool = True) -> float:
    """
    Measure spatial density inhomogeneity using a grid.
    Works for both 2D and 3D positions.

    Parameters
    ----------
    positions : np.ndarray, shape (N, D)
        Particle positions (D = 2 or 3).
    box_size : float
        Size of the simulation box.
    bins : int
        Number of bins per dimension.
    normalized : bool
        Whether to normalize variance by mean^2.

    Returns
    -------
    float
        Density variance (higher = more clustered).
    """
    if positions.size == 0:
        return float("nan")

    dim = positions.shape[1]  # 2 or 3
    ranges = [[0, box_size]] * dim

    H, _ = np.histogramdd(
        positions,
        bins=[bins] * dim,
        range=ranges
    )

    mean = np.mean(H)
    var = np.var(H)

    if normalized:
        return var / (mean**2 + 1e-12)
    return float(var)


def number_of_clusters(positions: np.ndarray, eps: float, box_size: float, min_size: int = 1) -> int:

    """
    Count number of clusters under a simple distance-threshold connectivity rule (PBC).

    Two particles are connected if their distance (under periodic BC) is < eps.
    A cluster is a connected component in this graph.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
    eps : float
        Neighbour threshold distance.
    box_size : float
    min_size : int
        Only count clusters with size >= min_size (useful to ignore singletons).

    Returns
    -------
    int
        Number of clusters.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    n_clusters = 0

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        cluster_size = 0

        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster_size += 1

            diff = positions - positions[j]
            diff -= box_size * np.round(diff / box_size)
            dists = np.linalg.norm(diff, axis=1)

            neighbors = np.where(dists < eps)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)

        if cluster_size >= min_size:
            n_clusters += 1

    return n_clusters


def polarization_time_avg(positions, box_size=1.0, K=50):
    """
    Polarization measures how aligned the agents are in their direction of motion.

    - Polarization ≈ 0 → motion is random, no collective order
    - Polarization ≈ 1 → motion is highly aligned, strong flocking
    
    Polarization is the standard order parameter for Vicsek-type flocking models.

    Mathematically, polarization is the average of all unit vectors.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
    box_size : float
    K: int
        Number of timesteps to average order parameter.

    Returns
    -------
    float
        Level of order between 0 and 1

    """
    T = positions.shape[0] # number of steps
    phis = []
    for t in range(T-K+1, T):
        v = positions[t] - positions[t-1] # estimate velocity by subtracting previous positions
        v -= box_size * np.round(v / box_size)  # periodic boundary correction

        vhat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12) # normalize velocity

        phi = np.linalg.norm(np.sum(vhat, axis=0)) / vhat.shape[0] # compute polarization
        phis.append(phi)

    return float(np.mean(phis))

def global_density(positions, box_size):
    N = positions.shape[0]
    d = positions.shape[1]
    volume = box_size ** d
    return N / volume


def local_density(positions, r, box_size, K=50):
    """
    This measures how crowded the flock is around each agent, then averages.
    """
    T, N, _ = positions.shape
    densities = []

    for t in range(T-K+1, T):
        pos = positions[t]

        # pairwise displacement
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        # count neighbours within r (exclude self)
        neigh = (dist > 0) & (dist < r)
        counts = np.sum(neigh, axis=1)

        # density = neighbours / area of circle
        d = pos.shape[1]
        if d == 2:
            denom = np.pi * r * r
        elif d == 3:
         denom = (4.0/3.0) * np.pi * r**3
        else:
            denom = r**d
        rho = counts / denom
        
        densities.append(np.mean(rho))

    return float(np.mean(densities))



