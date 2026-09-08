# Wing-derived near-field air disturbance

Status: native module implemented; browser-world integration pending root-owned
runtime wiring. This mechanism is an engineering transduction from actual MuJoCo
wing motion. It is not validated aerodynamics, a lift model, or a courtship-song
controller.

## Evidence that constrains the interface

The fly antenna is a near-field particle-velocity receiver, and a courting male
produces sound by moving an extended wing. Early simultaneous wing and acoustic
measurements relate the waveform to the asymmetric wing stroke rather than to an
independent sound generator ([Bennet-Clark and Ewing 1968](https://doi.org/10.1242/jeb.49.1.117)).
The implementation consequently has no oscillator: zero relative wing motion
produces zero endogenous source.

The two antennae receive a directional acoustic flow, and both source angle and
separation affect their particle velocities ([Morley et al. 2012](https://doi.org/10.1242/jeb.068940)).
Near-field particle velocity is strongly distance-dependent; Drosophila song
recording work describes the idealized point-source velocity falloff as inverse
cube ([Arthur et al. 2013](https://doi.org/10.1186/1471-2202-14-11)). This motivates
a directional, softened dipole at each actual wing centroid. It does not validate
the chosen gain or replace fluid simulation.

Johnston's-organ populations distinguish slow or sustained wind-like antennal
deflection from oscillatory near-field sound ([Yorozu et al. 2009](https://doi.org/10.1038/nature07843)).
Fast auditory populations respond in the roughly 100--350 Hz courtship-song range
and adapt to intensity statistics ([Clemens et al. 2018](https://doi.org/10.1038/s41467-017-02453-9)).
The runtime therefore retains bilateral antenna-local airflow vectors and a
separate spectral history. It does not attach behavioral meaning, sender identity,
or a courtship label to either signal.

## Engineering transduction

At each physical sample, the source point is a declared centroid on the actual
left or right wing body. Its relative velocity is

```text
v_rel = v_wing(point) - [v_root + omega_root x (point - x_root)].
```

This subtraction makes common rigid translation and rotation exactly silent. A
first-order high-pass removes static relative motion and a first-order low-pass
conditions the sampled signal. For displacement `r` from wing centroid to antenna,
unit vector `n`, effective radius `a`, geometry source gain `g`, and global
engineering scale `G`, the particle-velocity proxy is

```text
u(r) = G g a^3 / (|r|^3 + a^3) * [3 n (n dot v_rel) - v_rel].
```

All wings contribute, including the listener's own wings. The current vector is
rotated into each actual antenna frame. A persisted Hann-window ring estimates
RMS at the 16 fixed BODY807 band centers; output is divided by a declared reference
particle velocity. At the current 1000 Hz physical-packet rate, the low-pass and
reporting ceiling is `0.4 * sample_hz = 400 Hz`, so only centers 40 through
365.844 Hz can receive endogenous energy. Channels centered at 467.843 through
1600 Hz are exactly zero for this source. Analytic external visitor tones remain
a separate 40--1600 Hz mechanism.

The filters operate after MuJoCo kinematics have been sampled. They cannot prevent
motion above the 500 Hz Nyquist limit from aliasing into a 1000 Hz packet, recover
that motion, or establish that a measured high-frequency wingbeat was resolved.
`sample_hz` is part of configuration and state cadence so a future dedicated
10 kHz wing packet can use the same interface with a higher derived band ceiling.

For the current imported morphology, `l_wing.stl` SHA-256 is
`142f77f15008fd75e1490b546bb859cf98c056121f1abd65acea2052c4e7764c` and
the MJCF scale is `(1000,1000,1000)`. An area-weighted triangle centroid after
that scale is `(-0.1255656, 1.1510138, 0.1189536)` mm in the left wing body;
the right mesh is the same file with negative Y scale, giving
`(-0.1255656,-1.1510138,0.1189536)` mm. These are the initial source-point
offsets. Because both sides use the same mirrored mesh, both initial geometry
source gains are `1.0`, normalized to this wing's area-times-span proxy. That
normalization does not calibrate acoustic efficiency.

## Native API and state

[`fly_acoustics.rs`](../../native/browser-world/src/fly_acoustics.rs) exposes:

- `ResidentKinematics`: root and two wing rigid motions, actual local wing
  centroids, two geometry-derived source gains, and two world antennal anchors
  with rotations. It contains no resident identifier.
- `FlyAcoustics::push_sample(time_s, residents)`: accepts each strictly periodic
  physical sample and updates dipoles, filters, and antenna rings without a
  spectral pass. The 100 Hz host pushes the current ten 1 kHz packet samples.
- `FlyAcoustics::frame()`: returns bilateral antenna-local endogenous airflow in
  mm/s plus 16 effective BODY acoustic amplitudes once per control tick.
  Supported-band Hann sine/cosine kernels are precomputed at construction and
  deterministically rebuilt on restore; `frame` performs no trigonometry.
- `FlyAcoustics::ingest_sample(...)` remains a probe convenience that performs
  one push followed by one frame. It is not the production packet hot path.
- `AcousticSnapshot`: last sample time, every per-wing filter, and every
  per-antenna spectral ring with cursor and count. `restore` validates all shapes
  against the configured cadence and window duration.

Root integration should add the airflow vectors only to BODY807 channels 40:46
and the effective bands only to channels 46:62. These channels then follow the
existing BODY807-to-CNS mask. They must not enter a controller, predictor, memory,
or action scorer through a parallel identity or physics-state input.

The focused native Cargo check on 2026-09-08 used browser-world's actual Serde
`float_roundtrip` setting and passed one joined scenario. It established exact
zero output under common rigid translation plus rotation, invariant antenna-local
output after a global 90-degree rotation, a 174.938 Hz kinematic source peaking
at BODY band 6 while every unsupported band remained zero, exact JSON snapshot
round-trip, and bit-identical continuation after restore. The scenario pushes ten
1 kHz samples between spectral frames. This checks the declared transformation,
hot path, and state semantics; it is not an acoustic calibration or a fly-behavior
result.

## Limits

- Source gains and the global coupling scale are engineered. A useful geometry
  convention is wing planform area times span divided by a frozen reference, but
  this has not been calibrated to measured Drosophila particle velocities.
- The dipole has no enclosure reflections, substrate coupling, wake advection,
  occlusion, viscosity, wing flexibility, thrust, or lift.
- Fixed MuJoCo position servos are not anatomical wing muscles. The module makes
  their resulting body motion perceptible; it does not claim biological song
  production or supplied courtship behavior.
- Spectral bands are effective features shared by the existing auditory mask;
  they do not assign frequency tuning to individual MaleCNS auditory cells.
