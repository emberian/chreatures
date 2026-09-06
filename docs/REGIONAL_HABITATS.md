# Inherited regional habitats

The current habitat family is a native, cold-build regional grammar. It replaces
the three fixed eight-platform nursery layouts. An environment genome carries a
bounded set of physical parameters: world dimensions, region and lane counts,
elevation span, graph loop density, shelter and underpass fractions, landmark
density, finite founder-resource scale, and movable construction-piece count.
Four supplied archetype ranges bias that same grammar toward terraced deltas,
vaulted courts, braided ridges, or sheltered basins. They are parameter ranges,
not separate runtime behavior scripts.

`HabitatFamily.initial_genome` samples one immutable founder genome.
`mutate_genome` applies the configured bounded perturbation recipe and records the
parent genome hash, operator, seed, recipe hash, profile hash, and archive epoch.
Dimensions and every scalar or integer remain inside both global capacity and the
selected archetype range. There is no load path for the former nursery-family
format.

The generator constructs a connected graph of 10–40 broad platforms. Lane
backbones, cross-lane links, nearest-component repair, and bounded extra loops
create multiple routes. Adjacent elevations are relaxed before geometry is
emitted so every box ramp respects the declared rise/run limit. Elevated
platforms stand on narrow supports, leaving real underpasses. Repair and loop
ramps must also clear every non-endpoint platform, preventing a shortcut from
cutting through an unrelated region. Roofs and supports
form optically and chemically occluding sheltered volumes. Landmarks are ordinary
colored geoms. Their analyst annotations are never delivered to a controller.

World dimensions vary from 10–24 m by 7–16 m, with configurable depth. The
generator replaces the template ground and walls at those dimensions. A second,
wider catchment basin sits below the habitat and has physical perimeter walls.
Escaped finite packets therefore fall, collide, and settle without clamping,
teleportation, or chemistry deletion.

Existing passive mechanisms remain ordinary MuJoCo assemblies. The pressure
lift and linked gate move as one cluster; balance, resonant bell, hinged leaf,
and physical lights are placed as coherent clusters at different regions.
Movable balls and blocks are retained or cloned from the same declarative
physical templates up to the genome's explicit bound. Their positions and joint
states remain part of the world snapshot.

The biosphere output is coupled to the generated physics. All existing colonies
and active finite material packets are placed on generated platforms. A bounded
resource-scale allele multiplies only the declared founder pools of the finite
packets; it is an initial-condition budget recorded in the resource hash, never a
runtime refill. Colony chemistry and reaction laws remain the common chemistry.

Resident count is an environment/campaign parameter from 1 through 32. The cold
composer supplies a resident bundle containing each physical body, somatic
traits, exchange bounds, and exact founder compartments for body, gut,
structure, gland, and brood. The native generator assigns five private rows per
resident immediately after fixed colony rows, shifts every preserved material
and emitter row consistently, and rejects caller-provided row indices. Broad
platform-local spawn slots are pairwise clearance checked. Spawn regions are
reserved before mechanisms, shelters, landmarks, movable objects, colonies, or
finite packets are placed.

Generation returns three documents. Habitat and biosphere JSON are the only
runtime inputs. The third is analyst-only metadata containing the designer graph,
spawn audit, resource budget, complete environment genome, and immutable record:

```text
sha256, parents[0..2], variation, topology_sha256, resource_sha256,
profile_sha256, epoch
```

The record matches the population archive boundary. Policy observations remain
body-local rays, chemistry, contact, sound, and private physiology; region IDs,
resource placement, graph connectivity, ancestry, and world coordinates are not
sensory features.

Current sources are:

- `data/habitat-families/regional-v1.json`: bounds, archetypes, mutation recipe,
  physical cluster definitions, and initial training genomes.
- `data/habitat-families/regional-residents-v1.json`: the 32-resident capacity
  bundle of somatic-v3 founders at the five-compartment boundary. A campaign
  selects one fixed prefix from 1 through 32 residents.
- `data/training/regional-environment-schedule-v1.json`: explicit current
  training and held-out environment founder seeds.
- `chreatures/habitat_family.py`: the thin hash-checking host boundary.
