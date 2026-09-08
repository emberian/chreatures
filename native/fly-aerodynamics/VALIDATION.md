# Validation receipt

Executed on 2026-09-08 with `rustc 1.98.0-nightly (91fe22da8 2026-06-21)`:

```text
cargo test --manifest-path native/fly-aerodynamics/Cargo.toml -- --nocapture
1 passed; 0 failed
```

The joined invariant check covers the generated author geometry and load kernel:

- a motionless wing in still air produces exact zero force and torque;
- adding the same world translation to wing velocity and airflow leaves loads
  unchanged;
- drag opposes relative translational motion;
- all 24 author-mesh blade strips produce finite load and Reynolds values;
- right-wing strip data are the exact author Y reflection of the left;
- the declared mm/gram/s inference converts 1 micro-newton to 1 model force
  unit and 1 nanonewton-metre to 1 model torque unit.

The diagnostic case was a left wing translating at `(0.75, 0, 0.15) m/s` in
still air with its body frame aligned to world. It produced total force
`(-0.783719, 0.058542, 0.862171) micro-newtons`, root torque
`(0.996852, 0.036821, 0.903644) nanonewton-metres`, and maximum strip Reynolds
number `54.897`. This is a numerical mechanics check, not evidence of hover,
flight stability, biological force calibration, or adequate wing actuation.
