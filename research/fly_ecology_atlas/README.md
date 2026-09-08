# Physical developmental response atlas

This research lane runs the current MuJoCo fly habitat and native ecological
growth at 100 Hz for eight simulated seconds per world. Two flies remain present
under explicitly diagnostic zero MOTOR92 commands. No CNS is evaluated or replaced
by this atlas. Every colony branch uses the production local proposal, measured
clearance, finite biomass/ATP allocation and atomic physical commit.

The frozen design has 24 deterministic maximin Latin-hypercube inherited-gene
settings and three physical illumination/clearance layouts. The last six entire
gene settings are held out. All four varied genes remain inside the native declared
range: branch angle, lateral probability, phototropism and contact avoidance.
Other chemistry, developmental costs and directional persistence stay fixed.
Layout changes move the actual screen or an existing physical obstacle; light and
clearance are measured by the host, never injected as invented sensor values.

`prepare.py` freezes host/Wasm bytes and scene variants. `campaign.py --pilot`
executes the first actual world once, retaining it as a campaign member. Use its
complete wall time to choose independent workers. The approved campaign starts
with four workers and has up to 120 minutes after the photon/metabolic coupling
and clearance scratch-model reuse are frozen. The campaign preserves failed runs and never retries an unknown
physical mutation. A budget stops new launches, while already-running bounded
worlds finish coherently.

`fit.py` requires all requested outcome receipts and fits the installed native
`gamfit` library. It compares joint and additive models with mean baselines,
reports whole-setting and whole-layout leave-out errors, and preserves constant
or rejected targets. Native artifacts are serialized, reloaded, and their
predictions compared. The six heldout settings never tune model selection or
proposals. Selection uses training leave-setting-out evidence and asks for distinct
canopy/route-permeability outcomes, not one global winner.

Each diversity proposal runs again through `run.mjs` on the fourth, previously
unused physical layout. `confirm.py` records predicted and actual outcomes without
converting mere execution into a successful prediction or promoted genotype.

All generated worlds, raw receipts, native models and checkpoints belong under
`~/paperbin/chreatures/integration/fly-ecology-atlas/`, not large repository assets.
Run Python orchestration as modules from the repository root; native GAM fitting
uses `integrations/.venv/bin/python`.

The initial no-photon run under `seed20260910` is excluded from fitting and retained
only as a throughput record. Final worlds use a distinct plan and exact frozen host
and native module hashes. Route conductance is calculated from actual measured
openness, declared base openness, geometry and pool diffusivity using the native
transport equation; it is not a claim that concentration-driven flux is nonzero.
Captured photon energy and actual construction expenditure accompany the geometry.

The response surface predicts a gene setting averaged across the three training
layouts. Whole-layout leave-out tests environmental transfer; the unused fourth
layout is an actual physical confirmation, not an extra fitting row. Constant
route outcomes remain explicitly constant and receive no invented fitted surface.
