### Emergent Flocking Structures

You know how birds suddenly form these really organized flocks, or fish schools seem to move together like they're one organism? We wanted to figure out if that kind of collective behavior could emerge just from simple local rules. Like, what if each bird only pays attention to its neighbors, tries to match their direction, but there's some randomness thrown in? Can you get those sudden transitions from chaos to order?

Turns out you can. With just a few basic rules (align with neighbors + random noise), you get phase transitions where the system suddenly switches from complete disorder to organized collective motion. This project is basically mapping out when and why those transitions happen.

We built agent-based models where particles interact with their neighbors, then systematically varied alignment strength, noise level, interaction range, and density. Ran hundreds of simulations to find the phase boundaries between disordered and ordered states. The notebooks folder has all the parameter sweeps and phase diagrams.

The main question was: how do alignment, noise, and interaction range work together to determine that phase transition? And once we find the boundaries, are they robust across different random starts, or does everything depend on getting lucky with the initial conditions?


**What's in here**

The src folder has three simulation engines. flocking_sim.py is the basic 2D Vicsek model with velocity-based flocking. We extended it to 3D in flocking_sim_3d.py because we got curious about how things change in higher dimensions. There's also baseline_clustering_2d.py which uses attraction/repulsion forces instead of velocity alignment, just to compare different mechanisms.

The metrics.py file has all our measurement functions for cluster sizes, nearest neighbor distances, polarization, etc. The experiments.py framework handles running parameter sweeps across multiple random seeds. We put default parameters in config.py after doing some initial trial runs to find reasonable values.


**How to use this**

Basic example:

```
from src.flocking_sim import run_simulation
from src.metrics import largest_cluster_fraction

history = run_simulation(N=200, steps=500, align=1.0, noise=0.05, R=0.15)
final_frame = history[-1]
lcf = largest_cluster_fraction(final_frame, eps=0.06, box_size=1.0)
print(f"Largest cluster: {lcf*100:.1f}% of particles")
```

We track LCF (largest cluster fraction) to see what fraction of particles are in the biggest cluster. High values mean everyone's grouped together. Nearest neighbor distance tells us about local crowding - lower means more clustering. Polarization checks if velocities are aligned, which is the standard order parameter for Vicsek-type models.

The main result is there's a pretty clear phase transition. Crank up alignment or reduce noise past certain thresholds and the system suddenly snaps into a flocking state. The exact location depends on interaction range too, which makes intuitive sense since it's harder to coordinate if you can't see your neighbors. Check notebooks/2d_phase_diagram.ipynb for the full analysis and phase diagrams.
