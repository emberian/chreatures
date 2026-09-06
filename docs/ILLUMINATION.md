# Physical illumination cycle

Living Reef uses one current native solar law rather than a timed resource
refill. `SolarCycle` in `native/world-kernels/src/illumination.rs` owns the
simulation clock and emits a sun position, normalized direction, and direct and
diffuse irradiance. The host applies that state to one predeclared MuJoCo
directional light. Resident retinal illumination and the native developmental
light sampler therefore receive the same direction and flux.

The birth-v5 `illumination_cycle` object records its units explicitly:
`period_seconds` is simulated seconds per orbit; `phase_offset_cycles` is in
turns, with dawn at 0, zenith at 0.25, dusk at 0.5, and midnight at 0.75;
`path_azimuth_degrees` rotates the east-west orbit in the world frame;
`twilight_degrees` is the sub-horizon transition angle; `orbit_radius_m` and
`center_m` determine the displayed source position. `peak_irradiance` is a
bounded normalized flux and `diffuse_fraction` partitions it into hemispheric
and directional contributions. `color` affects rendered light, not chemistry.
The referenced light must be attached to a static, identity-oriented physical
entity and declare `directional: true`.

For solar altitude `h = sin(2*pi*phase)`, the Rust law maps
`(h + sin(twilight))/(1 + sin(twilight))` through a clamped cubic smoothstep.
This gives a continuous dawn and dusk with zero endpoint slope. Direct light is
ray-tested toward the moving sun at every leaf, founder surface, and active bud.
Blocked rays retain only the sampling profile's declared transmission. Diffuse
light uses the configured weighted hemisphere rays. Thus terraces, arches, and
subsequently built structure create changing light niches as the sun moves;
local authored lamps remain independent finite-distance sources.

The cycle changes no material pool. Phototrophic reactions may convert the
sampled external photon flux under the existing common chemistry, whose energy
ledger records used photons, heat, stored energy, and work. The biosphere-v5
snapshot stores the native clock and cycle configuration hash; restoring the
biosphere reapplies the derived light state to both MuJoCo and the native
sampler before continuation.
