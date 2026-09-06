# Mobile phototrophy and mixotrophy

Current birth-v5 worlds give every resident two inherited continuous metabolic
traits: `phototrophic_absorptivity` and `dorsal_capture_fraction`, each in
`[0,1]`. Zero is an ordinary trait value. These traits do not select a role or
controller. Every resident retains the same movement, feeding, gut digestion,
and respiration mechanisms; some fresh genotypes additionally express a small
amount of the shared `carbon_fixation` enzyme in their body compartment.

The capture surface is physical. At each model revision Biosphere resolves the
resident's actual MuJoCo thorax ellipsoid. Its maximum projected dorsal area is
`pi * semiaxis_x * semiaxis_y`; the inherited capture fraction can only reduce
that area. The sample point and outward normal are attached to the thorax body,
so rolling, shelter, built structure, and moving solar direction change
capture. `mobile_phototrophy.light_sampling` pins the weighted diffuse
hemisphere and blocked transmission. The shared
`photon_flux_per_square_meter_second` records synthetic photon-supply units per
square meter per simulated second.

`LightEnvironment` evaluates all colony and mobile surfaces in one native
MuJoCo ray batch. For mobile surfaces it transforms the dorsal normal, applies
`max(0, normal dot direction)` to direct solar and point-light flux, and weights
diffuse rays by incidence and physical visibility. Rust then computes supplied
photons from timestep, flux density, physical area, absorptivity, and incident
irradiance. Direct and diffuse solar terms partition one irradiance value and
are never added twice at full exposure.

Supplied photons are placed only in the resident's existing private body row.
The unchanged common-chemistry reactor decides actual use from carbon-fixation
enzyme activity, substrate saturation, and its photon cost. No pool is refilled.
The normal metabolic ledger records external photon use, stored chemical
energy, heat, and work. Biosphere reports supplied and used photons, effective
capture area, body material mass, and body stored energy for inspection; none
of these values enter the resident observation or identify another organism.
The biosphere-v6 snapshot binds the immutable profile and traits by hash while
the existing web and solar clock remain the only mutable chemical and lighting
state.
