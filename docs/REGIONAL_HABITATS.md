# Inherited regional habitats

The current habitat family is a native, cold-build regional grammar. It replaces
the three fixed eight-platform nursery layouts. An environment genome carries a
bounded set of physical parameters: world dimensions, region and lane counts,
elevation span, graph loop density, shelter and underpass fractions, landmark
density, finite founder-resource scale, and movable construction-piece count.
The same genome carries bounded multipliers for Growth-v4 guidance, transport,
and local light/nutrient/support/competition responses. Generation applies the
vector to every colony grammar before hashing the biosphere, so descendants can
inherit different local construction laws without runtime scripts or named roles.
Four supplied archetype ranges bias that same grammar toward terraced deltas,
vaulted courts, braided ridges, or sheltered basins. They are parameter ranges,
not separate runtime behavior scripts.

`HabitatFamily.initial_genome` samples one immutable founder genome.
`mutate_genome` applies the configured bounded perturbation recipe and records the
parent genome hash, registered parent environment-record hash, operator, seed,
recipe hash, profile hash, and archive epoch. Genome `parents` and
`environment_parents` remain separate SHA domains; the generated record copies
only registered environment parents.
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
sha256, genome_sha256, genome_parents[0..2], parents[0..2], variation,
topology_sha256, resource_sha256, profile_sha256, epoch, descriptors,
generation_cost
```

The five normalized descriptors are `regional_scale`, `elevation_relief`,
`resource_density`, `renewal_rate`, and `connectivity`. They derive from actual
generated dimensions and node elevations, finite material elemental equivalents,
declared colony capture area and photon flux, and graph cycle rank. The structured
generation cost retains raw physical-geom, region, edge, movable, and compartment
counts plus their mean normalized cost against limits declared in the family
configuration. These fields are archive and challenge-scheduling evidence only.

The record matches the population archive boundary. Policy observations remain
body-local rays, chemistry, contact, sound, and private physiology; region IDs,
resource placement, graph connectivity, ancestry, and world coordinates are not
sensory features.

Generator v3 computes ramp orientations with closed-form half angles using
basic arithmetic and square roots. It avoids platform-specific `atan2` and
`sin_cos` results, which changed environment hashes by a few final quaternion
bits between macOS and Linux in the first reciprocal campaign. Generated source
identities remain strict. Earlier profiles and failed campaign inputs retain
their original v2 generator rather than being relabeled.

Current sources are:

- `data/habitat-families/regional-v3.json`: bounds, archetypes, mutation recipe,
  physical cluster definitions, and initial training genomes.
- `data/habitat-families/regional-residents-v2.json`: the 32-resident capacity
  bundle of somatic-v3 founders at the five-compartment boundary. A campaign
  selects one fixed prefix from 1 through 32 residents. Gut founders carry
  bounded, lineage-varying fermentation, fermentate-respiration, and detritus-
  hydrolysis enzyme allocations. The fermenting founders' initial gut reserve
  is transferred from their own body founder pool, so generation does not add
  chemical material.
- `data/training/regional-environment-schedule-v1.json`: explicit current
  training and held-out environment founder seeds.
- `chreatures/habitat_family.py`: the thin hash-checking host boundary.

Materialize one founder environment headlessly with:

```sh
PROFILE_SHA256="$(shasum -a 256 docs/development/POPULATION_WAVE.md | awk '{print $1}')"
uv run python scripts/generate_regional_family.py \
  --output runs/regional-founder initial \
  --archetype terraced-delta --seed 20260906 --residents 8 \
  --profile-sha256 "$PROFILE_SHA256"
```

An inherited environment uses the same command and grammar:

```sh
uv run python scripts/generate_regional_family.py \
  --output runs/regional-child mutate \
  --parent-genome runs/regional-founder/environment.genome.json \
  --parent-analyst runs/regional-founder/analyst.json \
  --variation-seed 20260907
```

Each output directory contains the genome, concrete habitat, concrete birth-v6
biosphere, analyst-only graph, and a manifest hashing all inputs and outputs.
