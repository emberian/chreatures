# Seeded nursery habitat families

`HabitatFamily` is a native, build-time habitat compiler. It starts from the
exact Living Reef habitat and biosphere byte strings pinned in
`data/habitat-families/nursery-v1.json`. A different template hash is rejected.
Generation never runs inside a resident timestep.

The current grammar has three parameterized topology families:

- `courtyard-ring` makes a loop around a central court, with two cross routes.
- `tiered-shelf` makes two rising lanes connected at every elevation.
- `braided-passages` alternates two lanes and provides crossovers at the middle.

Each node is a stable slab with sparse supports. Elevated slabs therefore make
real underpasses instead of solid plinths. Each graph edge compiles to a wide
oriented box ramp between the two slab surfaces. The native validator requires
a connected graph, bounded platforms, minimum ramp width, maximum rise/run,
declared underpass clearance, and separated resident spawns. Seed variation is
limited to small horizontal and elevation offsets and the constraints are
checked again after variation.

The compiler removes only the terrain entities named in the family config. It
retains the six resident definitions and their inherited body and metabolic
traits, the 1024-ray retinal profile, chemistry compartments, finite material
rows, acoustic components, passive assemblies, and all stable binding IDs.
It then relocates the twelve colony attachment frames, twelve active material
packets, movable play objects, passive mechanisms, two local lights, canopies,
and colored landmarks onto the generated graph. Different seeds consequently
change traversable elevations, underpasses, occlusion, illumination exposure,
and physical resource encounters without changing a resident controller.

Generate the pinned six-member training family with a freshly built native
extension:

```console
.venv/bin/python native/world-kernels/build_extension.py
.venv/bin/python scripts/generate_nursery_family.py \
  --output runs/nursery-family-v1
```

Use one or more `--variant FAMILY:SEED` arguments to generate a selected set.
The output directory contains a habitat, biosphere birth config, and analyst
record for each variant plus `manifest.json`. The manifest hashes the config,
both source templates, the Rust generator source, native extension, and every
output. Existing nonempty output directories are rejected unless `--replace`
is explicit.

The analyst record contains the designer graph, slopes, widths, feature nodes,
spawn placements, and resource placements. It is a separate artifact marked
`runtime_visible: false`; no graph node, family name, colony ID, or resource ID
is added to sensory observations. Residents encounter only geometry, light,
sound, odor, and contact through the normal world mechanisms.

## Headless construction check

A `courtyard-ring:20260911` artifact was built with the current native engine
and run headlessly for 4 simulated seconds with six articulated residents,
twelve colonies, finite material objects, passive acoustics, native solar/light
sampling, and a 1024-ray retina per resident. It compiled 266 initial geoms;
resource-funded development reached 43 parts and model revision 5 (327 geoms).
All residents remained supported at z=0.2082–0.7701 m, physical contacts caused
104 acoustic events, and chemistry conservation residuals were below
`7.2e-15` for elements and `4.0e-14` for energy.

The graph checks establish geometric continuity and conservative slope and
clearance bounds. They do not prove that every inherited morphology or every
dynamic gate state can traverse every route. Those are empirical morphology
and controller questions for headless cohort evaluation.
