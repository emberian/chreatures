# Finite visitor material supply

The original browser food tools send an `add` command for a physical preset.
That path creates a legacy scalar `food` component. Fresh chemical reef specs
deliberately contain no `berry` or `nectar` preset, so those commands are
rejected. A legacy food object would also remain outside the shared metabolic
web: mobile physiology acquires chemistry only from entity rows registered by
`MaterialObjects`.

`VisitorMaterialSupply` provides an opt-in chemical-world path. A supply choice
binds one finite, inert source compartment in the world's existing
`MetabolicWeb`, a fixed chemical mixture per offering, and a bounded set of
dormant `MaterialObjects` slots. The command

```json
{
  "op": "offer_material",
  "material": "reserve-fruit",
  "x": 2.4,
  "y": 1.8,
  "z": 0.25
}
```

transfers a portion from that source row into the first available slot and
spawns its ordinary free rigid body at the requested position. An active slot
cannot be teleported. Once contact-mediated acquisition empties a slot,
`MaterialObjects` removes its geometry and the slot can be used again. When all
slots are occupied, or the remaining source mixture is below its declared
minimum fraction, the offering is rejected before state changes.

The outside reserve is present in the shared web at birth, so it is finite and
included in the world's initial elemental and chemical-energy totals. Offering
does not mint matter. `accounting()` derives the amount crossing the
outside-to-habitat boundary as `initial source - current source`; the transfer
receipt reports its elemental composition and chemical energy. Later packet,
gut, body, deposit, and root transactions continue to use the same web.

The current Living Reef birth includes two finite choices in
`data/visitors/living-reef.json`: reserve fruit and a fibrous pulp with mineral,
reserve, tough tissue, and detritus. Each choice owns two dormant ordinary
`MaterialObjects` slots. These four slots are disjoint from mobile egestion and
colony exudation slots. The two inert source compartments and four empty slot
compartments are part of the shared web at birth; an offering transfers existing
inventory and creates no recurring source.

Enable this supply only when creating the fresh world:

```sh
uv run chreatures \
  --habitat data/habitats/living-reef.json \
  --biosphere data/biosphere/living-reef.json \
  --visitor-materials data/visitors/living-reef.json \
  --resident-artifact /path/to/rich-developmental-resident.npz
```

The whole-world checkpoint then owns the remaining depot inventory and active
offering slots. Restoring that checkpoint does not reread the launch file.

Choice and slot names are environment-side bindings. Resident observations do
not receive either name. They receive the packet's physical rays and contacts,
plus color and odor computed from its current chemical pools through the usual
material surface coefficients. No resident identifier appears in an offering
command.

The visitor-facing `view()` reports `offer_count` and a `choices` mapping. Each
choice contains `pools`, `remaining_resources`, `remaining_portions`,
`available_slots`, and an authoritative `available` boolean. `remaining_portions`
is the limiting source-to-recipe ratio and can exceed one; `available` also
requires a free physical slot.

The visitor-supply experiment used a retired birth-v2/v3 habitat builder. Its
exact source remains available at
[`867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/build_chemical_habitat.py).
It supplied three mixtures with distinct chemistry-derived sensory surfaces:

- `reserve-fruit`: directly absorbable reserve chemistry.
- `soft-fruit`: reserve plus digestible soft tissue.
- `detrital-cake`: reserve, mineral, and digestible detritus.

Runtime integration is intentionally optional. Construct
`VisitorMaterialSupply` only when a new world was given a visitor-supply spec,
dispatch `offer_material` to its `command()` method, and include its snapshot
beside the Biosphere snapshot. Existing checkpoints with no supply state keep
their prior commands and chemistry.

The focused physical check is historical. Its exact source is
[`scripts/probe_visitor_materials.py` from `867cdb8`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/probe_visitor_materials.py).

It places a reserve packet at a measured resident mouth point, advances real
MuJoCo contact and SomaticPhysiology acquisition, checks that packet inventory
falls while gut inventory rises, and verifies zero elemental and stored-energy
residual for the offering transfer. It then crosses canonical JSON restore and
compares a three-step continuation exactly. The probe creates a fresh local
non-neural world and does not load or contact a live runtime.
