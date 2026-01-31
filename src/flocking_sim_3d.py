import numpy as np

def run_simulation(
    N=200,
    steps=400,
    box_size=1.0,
    align=1.0,
    cohesion=0.5,               
    noise=0.05,
    R=0.15,
    speed=0.03,
    repulsion_radius=0.05,
    repulsion_strength=1.0,
    dt=1.0,
    seed=None,
    save_every=1,
    softening=1e-6,
):
    """
    3D flocking with alignment + cohesion + short-range repulsion.
    
    Basically the 2D model but extended to 3D, and we added:
    - Cohesion: explicit attraction toward center-of-mass of neighbors
    - Repulsion: push away from very close neighbors (prevents collapse)
    """
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)
    pos = rng.random((N, 3)) * box_size

    # Random initial velocities (will be normalized)
    vel = rng.normal(size=(N, 3))
    vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)

    history = []

    for t in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        neigh = (dist > 0) & (dist < R)
        count = np.sum(neigh, axis=1)
        
        v_sum = np.sum(vel[None, :, :] * neigh[:, :, None], axis=1)
        v_avg = v_sum / (count[:, None] + 1e-9)

        # Alignment steering
        steer = v_avg - vel
        steer[count == 0] = 0.0

        # Repulsion from very close neighbors
        if repulsion_strength != 0.0 and repulsion_radius > 0.0:
            rep_mask = (dist > 0) & (dist < repulsion_radius)
            rep_dir = np.zeros_like(diff)
            rep_dir[rep_mask] = diff[rep_mask] / (dist[rep_mask, None] + softening)
            rep_weight = np.zeros_like(dist)
            rep_weight[rep_mask] = (repulsion_radius - dist[rep_mask]) / repulsion_radius
            F_rep = np.sum(rep_dir * rep_weight[:, :, None], axis=1)
        else:
            F_rep = np.zeros_like(pos)

        # Cohesion: attraction toward neighbor COM
        if cohesion != 0.0:
            sum_to_neighbors = np.sum((-diff) * neigh[:, :, None], axis=1)
            F_coh = sum_to_neighbors / (count[:, None] + 1e-9)
            F_coh[count == 0] = 0.0
        else:
            F_coh = np.zeros_like(pos)

        # Update and normalize
        vel = vel + dt * (align * steer + cohesion * F_coh + repulsion_strength * F_rep)
        vel = vel + noise * rng.normal(size=vel.shape)
        vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12) * speed

        pos = (pos + dt * vel) % box_size

        if (t + 1) % save_every == 0:
            history.append(pos.copy())

    return np.asarray(history)
