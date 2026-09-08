# Source ledger

## Morphology and engine sources

- FlyGym / NeuroMechFly v2.1.0, revision
  `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`, Apache-2.0. The imported
  morphology is the simplified, at-most-2000-face micro-CT mesh of one adult
  female. Chreatures uses the author's `l_wing.stl` and Y reflection for the
  right wing. The source rig gives each wing mass `2.5e-6` and the whole compiled
  body mass `0.00102531` in unnamed model mass units.
- Wang-Chen et al., “NeuroMechFly 2.0, a framework for simulating embodied
  sensorimotor control in adult Drosophila,” Nature Methods (2024),
  <https://doi.org/10.1038/s41592-024-02497-y>. This is the author model and
  body-mechanics scope; it does not supply a validated flight-aerodynamics model
  for the imported wing.
- MuJoCo fluid-force documentation,
  <https://mujoco.readthedocs.io/en/3.1.1/computation/fluid.html>. Its ellipsoid
  model combines added mass, quadratic and viscous drag, Magnus and Kutta lift.
  It motivated a stateless physical load interface, but the first crate uses the
  actual mesh-derived blade planform rather than replacing the wing by an
  ellipsoid.
- Vaxenburg et al., “Whole-body physics simulation of fruit fly locomotion,”
  Nature (2025), <https://doi.org/10.1038/s41586-025-09029-4>. Its separate
  female body has measured total mass 0.983 mg and 8 micrograms per wing. It uses
  a tuned MuJoCo ellipsoid fluid model with `fluidcoef=[1,0.5,1.5,1.7,1]`.
  Those values are useful context but are not copied into this different body.

## Aerodynamic basis

- Dickinson, Lehmann and Sane, “Wing rotation and the aerodynamic basis of
  insect flight,” Science 284 (1999),
  <https://doi.org/10.1126/science.284.5422.1954>. Dynamically scaled fruit-fly
  experiments separated translational, rotational and wake-capture mechanisms
  and measured the empirical translational coefficient curves used here.
- Sane and Dickinson, “The aerodynamic effects of wing rotation and a revised
  quasi-steady model of flapping flight,” Journal of Experimental Biology 205
  (2002), <https://doi.org/10.1242/jeb.205.8.1087>. This establishes why a
  translation-only first kernel is incomplete, especially near stroke reversal.
- Sane, “The aerodynamics of insect flight,” Journal of Experimental Biology 206
  (2003), <https://doi.org/10.1242/jeb.00663>. This reviews the limits of the
  quasi-steady assumption and the importance of unsteady mechanisms.
- Nakata et al., “A semi-empirical model of the aerodynamics of manoeuvring
  insect flight,” Royal Society Open Science 8 (2021),
  <https://doi.org/10.1098/rsos.201552>. It restates the Dickinson curves in
  radians as `CL=0.225+1.58 sin(2.13 alpha-0.14)` and
  `CD=1.92-1.55 cos(2.04 alpha-0.17)` and explains their fitted nature.

Scry schema was read on 2026-09-08 as required. The bounded academic query then
returned HTTP 429 because the shared research capacity was exhausted, so no
Scry discovery record is treated as evidence. Primary publisher papers, official
MuJoCo documentation and pinned author assets above are the evidence sources.
