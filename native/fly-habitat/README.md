# Fly habitat compiler

`fly-habitat` creates a deterministic, parameterized physical/material plan in
fly-scale millimetres. One plan binds collision geometry, eight-pool material
regions and routes, finite movable packets, colony attachment surfaces, and
resident spawn poses. Terrain IDs and topology remain host configuration and
observer provenance; they are never policy inputs.

```sh
cargo run --release --manifest-path native/fly-habitat/Cargo.toml -- \
  --output /path/to/habitat-plan.json --seed 20260908 --residents 4 \
  --width-mm 80 --depth-mm 60 --height-mm 30
```

The default plan has 48 touching soil tiles, four connected sloping stem and
branch networks, 32 elevated leaf surfaces, bark shelters with open cavities,
ramps, moist niches, movable grains and elevated food pods, and four colony
attachment leaves. Every material region names physical geometry and the route
graph is validated as connected. The estimate includes the 69 imported author
segment geoms plus two adhesion helper geoms per resident and must remain
between 300 and 600.

The planner does not compute paths, tasks, targets, or motor commands. Shelter
means a physical cavity made from collision/occlusion geometry; it is not a
claim of validated radiometry or a label delivered to a resident.
