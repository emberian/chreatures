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

The default chemical habitat builder is unchanged. To create a new compatible
set of habitat, Biosphere, and visitor-supply specifications:

```bash
.venv/bin/python scripts/build_chemical_habitat.py \
  --visitor-supply \
  --output runs/visitor-chemical-birth
```

Add `--recycling` if the new world should also contain the existing egestion,
root uptake, and colony exudation mechanisms. The builder currently supplies
three mixtures with distinct chemistry-derived sensory surfaces:

- `reserve-fruit`: directly absorbable reserve chemistry.
- `soft-fruit`: reserve plus digestible soft tissue.
- `detrital-cake`: reserve, mineral, and digestible detritus.

Runtime integration is intentionally optional. Construct
`VisitorMaterialSupply` only when a new world was given a visitor-supply spec,
dispatch `offer_material` to its `command()` method, and include its snapshot
beside the Biosphere snapshot. Existing checkpoints with no supply state keep
their prior commands and chemistry.

The focused physical check is:

```bash
.venv/bin/python scripts/probe_visitor_materials.py
```

It places a reserve packet at a measured resident mouth point, advances real
MuJoCo contact and SomaticPhysiology acquisition, checks that packet inventory
falls while gut inventory rises, and verifies zero elemental and stored-energy
residual for the offering transfer. It then crosses canonical JSON restore and
compares a three-step continuation exactly. The probe creates a fresh local
non-neural world and does not load or contact a live runtime.
