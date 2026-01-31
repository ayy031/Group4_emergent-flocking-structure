#Emergence of Flocking Structure from Local Interaction Rules

This project studies how collective flocking behavior can emerge from very simple local rules.

In nature, birds, fish, and other animals often move together as a group, even though each individual only reacts to nearby neighbors. We wanted to see whether this kind of organized motion can appear in a simple simulation where agents follow basic rules like alignment and noise, without any global control.

To study this, we built a Vicsek-style agent-based model and explored how different parameters affect the transition from random motion to organized flocking.


##Project Overview

In our model, each agent:
	•	moves at a constant speed
	•	looks at nearby neighbors within a certain radius
	•	tries to align its direction with them
	•	is affected by random noise
	•	moves inside a periodic box (wrap-around boundaries)

Even with these simple rules, the system can show:
	•	formation of large clusters
	•	collective motion
	•	sharp transitions between disorder and order

We start with a 2D model because it is fast and easy to analyze, and then extend it to 3D to study more realistic flocking behavior.


##What We Study

The main questions we explore are:
	•	Under what conditions does flocking appear?
	•	How do noise, alignment strength, and interaction range affect the system?
	•	Is the transition from disorder to order stable?
	•	How does flocking change when going from 2D to 3D?
	•	Do large flocks show consistent geometric structure?


##Model Description

At each time step, every agent:
	1.	Finds all neighbors within a distance R
	2.	Computes the average direction of those neighbors
	3.	Turns toward that direction (alignment)
	4.	Adds random noise
	5.	Updates its position while keeping a fixed speed

In the 3D version, we also include:
	•	short-range repulsion (to prevent overlap)
	•	a weak cohesion term (to keep the group together)

These additions help stabilize the flock and make the behavior more realistic.


##Parameters

The main parameters used in the simulations are:
	•	noise – how random the motion is
	•	align – how strongly agents follow neighbors
	•	R – interaction radius
	•	N – number of agents
	•	dt – time step size
	•	speed – movement speed

All simulations use periodic boundary conditions.


##Measurements

We use several metrics to analyze the system:

Largest Cluster Fraction (LCF)

Measures how many agents belong to the largest connected group.
	•	High LCF → strong flocking
	•	Low LCF → disordered state

Nearest Neighbor Distance (NN)

Shows how tightly agents are packed.

Polarization

Measures how aligned the velocities are across the system.

3D Shape Measures

In 3D, we use PCA to measure:
	•	flock thickness
	•	shape ratios
	•	volume and density

All measurements are taken after the system reaches a steady state.


##Main Results
	•	Flocking appears only when noise is low.
	•	Noise is the most important parameter controlling behavior.
	•	Alignment helps, but only when noise is already small.
	•	In 3D, flocks are more stable but require a larger interaction radius.
	•	Large flocks show consistent geometric structure as size increases.

Overall, the system shows a clear phase transition from disorder to order.


##Summary

This project shows that:
	•	Complex collective behavior can emerge from very simple rules
	•	No global control is needed for flocking to appear
	•	Noise plays a key role in destroying or enabling order
	•	3D flocks show more stable structure than 2D ones

The results support the idea that large-scale organization in nature can arise purely from local interactions.

