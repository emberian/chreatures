# Finite resource growth

Resource production can now join the running 3D loop. The optional `Ecology`
layer advances after contact ingestion and before the chemical field updates.
Growth therefore changes the physical food quantity and the next scent source.
Moving a producer, shading it, or consuming it changes subsequent opportunities.
The mechanism has no object-seeking policy or named preference inside it.

Start a **new world** with an explicit resource configuration:

```sh
uv run chreatures --port 8769 --brain-url http://127.0.0.1:18769 \
  --checkpoint runs/orchard-garden.json \
  --resources data/ecology/portable-orchard.json
```

This example requires its own neural authority at the chosen forwarded port.
Existing checkpoints preserve their original ecology, regardless of command-line
defaults. Do not point two independently advancing worlds at one ordered neural
authority. Worlds that exchange physical interactions belong under one authority.

The portable orchard binds two existing food bodies to finite resource pools.
Each producer captures light, spends energy and material to grow, pays maintenance,
and loses some biomass through turnover. Recycling returns a declared fraction
of turnover to available material. Ambient material inflow and photon flux are
explicit external inputs. This is a synthetic model, not calibrated plant biology.

All transfers enter mass and energy ledgers. Contact consumption leaves the
producer system; its downstream physiological conversion remains the body's
engineered digestion rule. The ledger covers resource production and removal,
not a claim of whole-universe energy conservation. Gentle visual growth follows
the physical component without repeatedly allocating rendering geometry.

The configuration supplies rates and capacities, rather than a timer that fills
empty food. Producers can exhaust their pools. Physical occlusion reduces their
light capture; the focused module run measured illumination falling from 0.487
to 0.055 under shade. Consumption and growth ran together with ledger residuals
around 1e-14. World/resource JSON snapshots continued exactly in the same runtime.

The whole-world manifest now also preserves resource configuration, pools,
oscillating light state, ledger, growth scale, clock and random state. Snapshot
restoration binds these to the restored physical world before any advancement.

The separate diffusion layer carries local fictional chemical concentrations
around solid geometry; see [chemical fields](FIELDS.md). Production, transport,
reception and behavior are separate mechanisms whose effects can be investigated.
