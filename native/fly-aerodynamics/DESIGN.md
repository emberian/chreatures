# Native fly-wing aerodynamic load design

This crate evaluates external aerodynamic loads from the actual articulated
wing poses and velocities. It does not generate a wingbeat, alter a controller,
or add a force based on a desired behavior. A motionless wing in still air has
exactly zero load. Wind on a stationary wing and a moving wing in still air are
the same relative-flow calculation.

## Frozen first model

The first implementation is a stateless translational blade-element model. The
author `l_wing.stl` is reduced offline to a 24-strip convex planform. Every strip
retains its area, chord and center in the author wing-body frame. The right wing
uses the author model's exact Y reflection. At each physics sample, the kernel:

1. Converts the wing origin velocity, angular velocity, pose and airflow to SI.
2. Computes each strip-center velocity `v + omega cross r` relative to air.
3. Removes the radial/spanwise component, computes angle of attack in the local
   chord-normal plane, and evaluates the Dickinson fruit-fly empirical
   translational lift and drag coefficient curves.
4. Returns the per-strip world force and point plus the summed force, torque
   about the wing-body origin, aerodynamic power and Reynolds-number range.

The kernel runs at the MuJoCo physics cadence (currently 0.0001 s). Evaluating
it from the 100 Hz controller packet would alias wing motion and is unsupported.
It has no phase state, oscillator, gait, lift bonus or target altitude.

## Rust integration API

```rust
pub struct ModelUnitScale {
    pub meters_per_length_unit: f64,
    pub kilograms_per_mass_unit: f64,
    pub seconds_per_time_unit: f64,
}

pub struct AeroConfig {
    pub air_density_kg_m3: f64,
    pub kinematic_viscosity_m2_s: f64,
    pub lift_scale: f64,
    pub drag_scale: f64,
    pub minimum_speed_m_s: f64,
}

pub struct WingKinematics {
    pub origin_world_m: Vec3,
    pub rotation_world_from_local: Mat3,
    pub linear_velocity_world_m_s: Vec3,
    pub angular_velocity_world_rad_s: Vec3,
    pub airflow_world_m_s: Vec3,
}

pub fn evaluate_into(
    config: &AeroConfig,
    geometry: &WingGeometry,
    kinematics: &WingKinematics,
    element_loads: &mut [ElementLoad],
) -> Result<WingLoad, AeroError>;
```

The hot path allocates no memory. `ElementLoad` contains a force and application
point in SI world coordinates. `WingLoad` contains summed lift, drag, total
force, root torque, signed mechanical power, maximum Reynolds number and active
strip count. A unit helper converts force and torque into declared MuJoCo model
units; it never assumes SI silently.

For the imported NeuroMechFly fixture the proposed engineering conversion is
`1 model length = 1 mm` and `1 model mass = 1 g`. It makes the author's total
mass `0.00102531` equal 1.02531 mg and each `2.5e-6` wing equal 2.5 micrograms.
FlyGym does not explicitly name its mass base unit, so this conversion remains a
declared calibration. Under it, 1 model force unit is 1e-6 N and 1 model torque
unit is 1e-9 N m.

The scale type also declares seconds per time unit, rather than baking time into
the conversion. The imported fixture sets it to one. Force conversion uses
`M L / T^2`, torque uses `M L^2 / T^2`, linear velocity uses `L / T`, and
angular velocity uses `1 / T`.

## Scope limits

The empirical coefficient curves came from a dynamically scaled fruit-fly wing,
not this individual micro-CT mesh. The convex planform omits corrugation and
flexibility. The first kernel omits rotational circulation, added mass, wake
capture, leading-edge-vortex history, wing-wing interaction, body aerodynamics
and ground effect. `lift_scale=drag_scale=1` means uncalibrated use of the cited
coefficient curves; it is not a hover fit. Reynolds diagnostics expose when the
operating point moves away from the experimental regime.

## Native MuJoCo host handoff

The host resolves `l_wing` and `r_wing` body IDs from each resident's compiled
`segments69` records once. At every 0.0001 s substep, after clearing the prior
external-force buffers and before `mj_step`:

1. Read each wing's `xpos`, `xmat`, and world-frame `mj_objectVelocity` result.
   MuJoCo orders that six-vector as angular then linear velocity. Use the whole
   body velocity; three servo joint rates omit parent translation and rotation.
2. Convert model positions and velocities with the declared `ModelUnitScale`.
   Convert the fixture's world airflow from mm/s through the same scale.
3. Evaluate the matching left/right `WingGeometry` into a resident-owned
   24-element scratch array.
4. Convert every force and point back to declared model units and call
   `mj_applyFT` at that point for the wing body, accumulating into
   `qfrc_applied`. This preserves the actual moment arm. If the host instead
   uses `xfrc_applied`, shift the returned root torque to MuJoCo's body center of
   mass with `WingLoad::torque_about_world_point` before writing the world-frame
   force/torque six-vector.

The kernel has no snapshot state. The host preserves the configuration and
geometry derivation hashes in its fixture/runtime identity. Aerodynamic forces
enter cognition through their physical effect on existing body sensors and
joint loads; the coefficient, wind vector, and computed force are not controller
inputs.
