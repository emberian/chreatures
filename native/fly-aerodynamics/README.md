# fly-aerodynamics

This crate computes quasi-steady aerodynamic loads from the actual articulated
NeuroMechFly wing mesh and current wing kinematics. It is a mechanics kernel,
not a flight controller. It never creates a wingbeat or modifies a motor target.

The checked-in geometry is regenerated with:

```sh
python3 tools/extract_neuromechfly_wing.py \
  --stl ../fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/model/l_wing.stl \
  --model-xml ../fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml \
  --body-schema ../fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json \
  --json assets/neuromechfly-v2.1.0-ca65a510-wing-planform.json \
  --rust src/neuromechfly_geometry.rs
cargo fmt --manifest-path Cargo.toml
```

At every MuJoCo physics step, obtain each wing body's current world pose,
origin linear velocity, and angular velocity. Convert them to SI with a declared
`ModelUnitScale`, pair them with a world-space local airflow velocity, and call
`evaluate_into`. Apply each returned force at its returned world point. An
equivalent host may apply the summed force at the wing origin and the returned
torque about that origin. If a host uses MuJoCo `xfrc_applied`, whose wrench is
at the body's center of mass, it must shift the torque with
`WingLoad::torque_about_world_point`. Calling `mj_applyFT` for each element at
its returned point is the less error-prone reference path.

The imported model's `NEUROMECHFLY_MM_GRAM_INFERRED` scale interprets one length
unit as 1 mm, one time unit as 1 s, and one mass unit as 1 g. The length and time
units are explicit in FlyGym; the mass unit is an engineering inference from the
author's numerical total mass and remains configurable.

See [DESIGN.md](DESIGN.md) for equations and limits and [SOURCES.md](SOURCES.md)
for the evidence ledger.
