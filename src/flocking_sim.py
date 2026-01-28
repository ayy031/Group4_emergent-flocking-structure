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
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)

    # positions uniform in box
    pos = rng.random((N, 2)) * box_size

    # random velocity directions
    ang = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vel = np.column_stack((np.cos(ang), np.sin(ang)))  # unit vectors

    history = []

    for t in range(steps):
        # pairwise displacement (this follows same structure as baseline_clustering_2d.py)
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        # neighbors within R
        neigh = (dist > 0) & (dist < R)

        # average neighbor velocity
        count = np.sum(neigh, axis=1)
        v_sum = np.sum(vel[None, :, :] * neigh[:, :, None], axis=1)
        v_avg = v_sum / (count[:, None] + 1e-9)

        # steer toward average
        #steer towards 0 if no neighbors
        steer = v_avg - vel
        steer[count == 0] = 0.0

        # update velocity and noise with new values
        vel = vel + dt * (align * steer)
        vel = vel + noise * rng.normal(size=vel.shape)

        # normalize to the constant speed
        vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)
        vel = speed * vel

        # move and wrap
        pos = (pos + dt * vel) % box_size

        if (t + 1) % save_every == 0:
            history.append(pos.copy())

    return np.asarray(history)
