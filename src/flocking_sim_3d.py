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
    
    use_predator=False,
    predator_strength=5.0,         
    #defaults to 3*R
    predator_radius=None,
    #defaults to 1 * speed 
    predator_speed=None,           
    
    dt=1.0,
    seed=None,
    save_every=1,
    softening=1e-6,
):
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)

    # positions uniform in 3D box
    pos = rng.random((N, 3)) * box_size

    # random velocity directions in 3D
    vel = rng.normal(size=(N, 3))
    vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)

    # predator setup 
    if use_predator:
        if predator_radius is None:
            predator_radius = 3.0 * R
        if predator_speed is None:
            predator_speed = 1.0 * speed

        pred_pos = rng.random(3) * box_size
        pred_vel = rng.normal(size=3)
        pred_vel = pred_vel / (np.linalg.norm(pred_vel) + 1e-12)

    history = []

    for t in range(steps):

        # pairwise displacement with periodic boundaries
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        # neighbors within R
        neigh = (dist > 0) & (dist < R)

        # average neighbor velocity
        count = np.sum(neigh, axis=1)
        v_sum = np.sum(vel[None, :, :] * neigh[:, :, None], axis=1)
        v_avg = v_sum / (count[:, None] + 1e-9)

        # steering
        steer = v_avg - vel
        steer[count == 0] = 0.0

        # repulsion
        if repulsion_strength != 0.0 and repulsion_radius > 0.0:
            rep_mask = (dist > 0) & (dist < repulsion_radius)
            # compute only where needed to avoid nan*0 issues
            rep_dir = np.zeros_like(diff)
            rep_dir[rep_mask] = diff[rep_mask] / (dist[rep_mask, None] + softening)

            rep_weight = np.zeros_like(dist)
            rep_weight[rep_mask] = (repulsion_radius - dist[rep_mask]) / repulsion_radius

            F_rep = np.sum(rep_dir * rep_weight[:, :, None], axis=1)
        else:
            F_rep = np.zeros_like(pos)

        # cohesion 
        if cohesion != 0.0:
            sum_to_neighbors = np.sum((-diff) * neigh[:, :, None], axis=1)
            F_coh = sum_to_neighbors / (count[:, None] + 1e-9)
            F_coh[count == 0] = 0.0
        else:
            F_coh = np.zeros_like(pos)

        # predator avoidance 
        F_pred = np.zeros_like(pos)
        if use_predator:
            dp = pos - pred_pos[None, :]
            dp -= box_size * np.round(dp / box_size)
            d = np.linalg.norm(dp, axis=1)

            mask = (d > 0) & (d < predator_radius)
            F_pred[mask] = dp[mask] / (d[mask, None]**2 + 1e-12)

        # update velocity
        vel = vel + dt * (
            align * steer
            + cohesion * F_coh
            + repulsion_strength * F_rep
            + predator_strength * F_pred
        )
        vel = vel + noise * rng.normal(size=vel.shape)

        # normalize to constant speed
        vnorm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel / (vnorm + 1e-12)
        vel = speed * vel

        # move and wrap
        pos = (pos + dt * vel) % box_size

        # predator motion 
        if use_predator:
            com = pos.mean(axis=0)
            to_com = com - pred_pos
            to_com -= box_size * np.round(to_com / box_size)
            to_com = to_com / (np.linalg.norm(to_com) + 1e-12)

            pred_vel = pred_vel + 0.5 * to_com + 0.05 * rng.normal(size=3)
            pnorm = np.linalg.norm(pred_vel)
            pred_vel = pred_vel / (pnorm + 1e-12)
            pred_pos = (pred_pos + dt * predator_speed * pred_vel) % box_size

        if (t + 1) % save_every == 0:
            history.append(pos.copy())

    return np.asarray(history)
