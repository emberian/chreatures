# Third-party body assets

`assets/neuromechfly-2.1.0-ca65a510-ypr/` contains source and generated material
from FlyGym / NeuroMechFly v2:

- upstream: <https://github.com/NeLy-EPFL/flygym>
- version: 2.1.0
- exact revision: `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`
- upstream copyright: 2023-2026 The NeuroMechFly v2 Authors
- license: Apache License 2.0
- retained license: `author-source/FlyGym-LICENSE`

The `model/*.stl` files are byte-identical copies of the upstream bundled
`simplified_max2000faces` meshes. The author exporter mirrors the left-side mesh
assets through signed mesh scale for the right side. `model/model.xml` is a
generated composition of those assets and author model code. Chreatures changes
the composition by selecting 84 of the 126 author joint axes for position-servo
control, enabling the six author adhesion actuators and six full per-leg contact
sensors, and including the author eye cameras. These modifications remain
described separately from the source morphology in `schema.json`.

Please cite:

- Wang-Chen et al., “NeuroMechFly 2.0, a framework for simulating embodied
  sensorimotor control in adult Drosophila,” Nature Methods (2024),
  <https://doi.org/10.1038/s41592-024-02497-y>.
- Lobato-Rios et al., “NeuroMechFly, a neuromechanical model of adult
  Drosophila melanogaster,” Nature Methods (2022),
  <https://doi.org/10.1038/s41592-022-01466-7>.

The morphology is based on a micro-CT scan of one adult female fly and contains
author-adjusted parts, including antennae. It is not the body of the individual
male CNS specimen used by Chreatures. The author axes, fitted rigid-body masses,
passive spring/damping values, position servos, contact law, and adhesion model
do not establish measured muscle or neuromuscular-junction identity.

The same 39 mesh bytes and `FlyGym-LICENSE` are copied beside generated scene
XML under `native/browser-world/fixtures/fly-ecology/` and
`native/fly-body/scenes/training-4/` so MuJoCo's virtual filesystem can load each
scene as a self-contained bundle. Those copies retain the same Apache-2.0 terms
and SHA-256 receipts in each `physics.json`.

`assets/author-step-bank-v1/author-source/` retains the exact Apache-2.0
FlyGym files used to build the research-only trajectory bank, including the
source `single_steps_flybody.npz`, its metadata, the upstream ball-walking clip,
and the relevant implementation modules. Their hashes are recorded in the
bank manifest. The sampled bank remains an author-derived diagnostic asset and
is not part of Chreatures' learned control implementation.
