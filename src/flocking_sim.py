import numpy as np

def run_simulation(
    N=200,
    steps=400,
    box_size=1.0,
    align=1.0,
    noise=0.05,
    R=0.15,
    speed=0.03,
    dt=1.0,
    seed=None,
    save_every=1,
):
    """
    Basic 2D flocking simulation (Vicsek-style model).
    
    Particles move at constant speed but adjust direction based on neighbors.
    The key is the alignment rule: each particle turns toward the average
    direction of nearby particles, plus some noise.
    """
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2)) * box_size
    
    # Start with random directions
    ang = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vel = np.column_stack((np.cos(ang), np.sin(ang)))

    history = []

    for t in range(steps):
        # Compute all pairwise distances with periodic boundaries
        # (wrap-around so particles on opposite edges can interact)
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        # Find who's within interaction range
        neigh = (dist > 0) & (dist < R)
        count = np.sum(neigh, axis=1)
        
        # Average velocity of neighbors
        v_sum = np.sum(vel[None, :, :] * neigh[:, :, None], axis=1)
        v_avg = v_sum / (count[:, None] + 1e-9)

        # Steering: turn toward average (if we have neighbors)
        steer = v_avg - vel
        steer[count == 0] = 0.0

        # Update: alignment + noise, then normalize to constant speed
        vel = vel + dt * align * steer + noise * rng.normal(size=vel.shape)
        vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12) * speed

        # Move and wrap around boundaries
        pos = (pos + dt * vel) % box_size

        if (t + 1) % save_every == 0:
            history.append(pos.copy())

    return np.asarray(history)
