import numpy as np

def nearest_neighbor_distance(positions, box_size):
    """
    Average distance to nearest neighbor (with periodic boundaries).
    Lower values = more clustering.
    """
    N = positions.shape[0]
    min_dists = []

    for i in range(N):
        diff = positions - positions[i]
        diff -= box_size * np.round(diff / box_size)
        dists = np.linalg.norm(diff, axis=1)
        dists = dists[dists > 0]  # exclude self
        min_dists.append(np.min(dists))

    return np.mean(min_dists)


def largest_cluster_fraction(positions, eps, box_size):
    """
    Fraction of particles in the biggest cluster.
    
    We use distance-based connectivity: two particles are in the same
    cluster if they're closer than eps (accounting for periodic boundaries).
    Then we find connected components using a simple DFS.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    max_size = 0

    for i in range(N):
        if visited[i]:
            continue

        # DFS to find this cluster
        stack = [i]
        size = 0

        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            size += 1

            # Find neighbors of j
            diff = positions - positions[j]
            diff -= box_size * np.round(diff / box_size)
            dists = np.linalg.norm(diff, axis=1)

            neighbors = np.where(dists < eps)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)

        max_size = max(max_size, size)

    return max_size / N


def density_variance_grid(positions, box_size, bins=20, normalized=True):
    """
    How non-uniform is the spatial distribution?
    
    We divide space into a grid and count particles per cell. High variance
    means clustering (some cells have many, others have few). Works in 2D or 3D.
    """
    if positions.size == 0:
        return float("nan")

    dim = positions.shape[1]
    H, _ = np.histogramdd(
        positions,
        bins=[bins] * dim,
        range=[[0, box_size]] * dim
    )

    var = np.var(H)
    if normalized:
        mean = np.mean(H)
        return var / (mean**2 + 1e-12)
    return float(var)


def number_of_clusters(positions, eps, box_size, min_size=1):
    """Count clusters (ignoring singletons if min_size > 1)."""
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    count = 0

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
            count += 1

    return count


def polarization_time_avg(positions, box_size=1.0, K=50):
    """
    Standard order parameter for flocking: how aligned are the velocities?
    
    We estimate velocity from consecutive position differences, then compute
    the norm of the average velocity vector. Close to 1 = aligned, close to 0 = random.
    """
    T = positions.shape[0]
    phis = []
    
    for t in range(T-K+1, T):
        # Estimate velocity from position change
        v = positions[t] - positions[t-1]
        v -= box_size * np.round(v / box_size)
        vhat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        
        # Polarization = |<v>| / N
        phi = np.linalg.norm(np.sum(vhat, axis=0)) / vhat.shape[0]
        phis.append(phi)

    return float(np.mean(phis))


def global_density(positions, box_size):
    """Just N / volume. Trivial but sometimes useful."""
    N = positions.shape[0]
    d = positions.shape[1]
    volume = box_size ** d
    return N / volume


def local_density(positions, r, box_size, K=50):
    """
    Average local crowding around each particle.
    For each particle, count neighbors within radius r, normalize by the
    volume of that ball, then average over particles and time.
    """
    T, N, _ = positions.shape
    densities = []

    for t in range(T-K+1, T):
        pos = positions[t]
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        neigh = (dist > 0) & (dist < r)
        counts = np.sum(neigh, axis=1)

        d = pos.shape[1]
        if d == 2:
            vol = np.pi * r * r
        elif d == 3:
            vol = (4.0/3.0) * np.pi * r**3
        else:
            vol = r**d  # crude fallback
        
        rho = counts / vol
        densities.append(np.mean(rho))

    return float(np.mean(densities))



