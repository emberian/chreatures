# Terrarium garden

The terrarium garden is a reusable nursery habitat assembled from the existing
MuJoCo entity, ecology, and acoustics contracts. It is independent of the
running hollow-garden lives.

| Layer | File |
| --- | --- |
| Physical scene | `data/habitats/terrarium-garden.json` |
| Physical sound transducers | `data/components/terrarium-play.json` |
| Finite resource production | `data/ecology/terrarium-orchard.json` |

## Physical circulation

The northern route climbs from the western floor over two broad ramps. The
first rises to a 0.28 m middle terrace; the second reaches a 0.62 m upper
terrace. Ramp faces overlap their adjoining platforms and their low ends are
buried slightly into the preceding surface, avoiding isolated ledges at each
transition. A separate 4.2 m south-facing ramp returns from the upper terrace
to the eastern floor. Low edge rails leave wide entrances and exits rather
than enclosing the decks.

The upper terrace stands on four narrow supports. Its 0.38 m underside remains
open as a shaded route, so the same structure can be crossed above, passed
under, or circled on the broad southern floor. An independent masonry arch,
garden screen, and movable leaf gate create shorter occluded passages without
partitioning the habitat into rooms. The open floor around the balance plank
and bell frame is deliberately generous for turning and object transport.

These are ordinary compound boxes and hinges. The scene contains no waypoint,
route label, navigation trigger, or scripted transport. Terrain appears to a
resident only through physical contact, proprioception, local illumination,
odor, sound, and occluded retinal rays.

## Shared physical consequences

Residents and the outside hand can roll two balls, move and stack blocks,
carry finite food bodies, tilt the balance plank, swing the arch gate, and set
the hanging bell in motion. `terrarium-play.json` binds five finite-energy
transducers to those mechanisms:

- the violet ball and arch block sound from contact work;
- the balance plank and leaf gate sound from contact and extracted hinge work;
- the hanging bell sounds from bounded opposing hinge torque.

Their source positions follow the physical bodies. Direct geometry attenuates
sound, so an action can be audible on one side of the terrace and muted below
or behind it. Acoustic events contain the struck entity and bounded mechanical
measurements, without an actor identity. Organism observations contain only
the three received tone amplitudes.

## Resource cycle

Five edible bodies are initially distributed across the floor and terraces.
Three are renewable producers in `terrarium-orchard.json`: a sun-exposed berry
on the upper deck, nectar beneath that deck, and a seed near the garden screen.
Each food body is movable. Its light sample and visual growth follow its actual
pose, so carrying it into shade changes later production.

Each producer draws from its own finite material and energy reservoir. Ambient
material inflow, photon flux, maintenance, conversion loss, turnover, and
recycling are declared and recorded by the existing ecology ledgers. The two
remaining foods are finite transportable opportunities. Resource machinery
changes quantities; it does not make a resident approach or consume anything.

## Loading a nursery

Pass all three files when creating a new `Habitat3D`:

```python
import json
from pathlib import Path

from chreatures.runtime3d import Habitat3D

spec = json.loads(Path("data/habitats/terrarium-garden.json").read_text())
habitat = Habitat3D(
    seed=17,
    spec=spec,
    resources="data/ecology/terrarium-orchard.json",
    acoustics="data/components/terrarium-play.json",
    brain_url="http://127.0.0.1:18769",
)
```

Use a dedicated neural authority and checkpoint path for the new cohort. The
spec does not alter default habitat selection or any saved world.

## Construction check and limits

A focused local check constructed the final scene as an
`ArticulatedSensoriumWorld`, attached its ecology and acoustics layers, and let
all residents and movable objects settle for 5 simulated seconds. MuJoCo state
and senses stayed finite. Resource mass and energy residuals were below
`2e-14`; acoustic energy and mechanical residuals were below `3e-16`.

A separate physical traversal used only the articulated body's existing joint
torques and a small external heading correction. Mica began on the western
floor, reached the middle surface at `(3.25, 1.52, 0.35)`, and reached the upper
terrace at `(8.75, 2.10, 0.69)`. No pose, height, or velocity was assigned after
construction. The check establishes connected contact geometry; the steering
controller was an evaluation instrument and is not part of the habitat.

The return ramp is steeper than the two ascending ramps, and its complete
descent was not exercised in this focused check. Sparse rails reduce accidental
falls without preventing them. Free food can roll off a deck or be displaced
from favorable light, which is an intended persistent consequence rather than
automatic replenishment or repositioning.
