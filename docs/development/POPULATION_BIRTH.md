# Reproducible population birth export

`scripts/export_population_birth.py` creates a current population-v6 cold birth
without starting or mutating a brain or world service. It selects one world from
an authenticated campaign assignment, regenerates the profile-pinned base habitat
and biosphere through the existing family generator, and compiles the selected
genomes' canonical MaleCNS neural phenotypes through
`compile_population_phenotypes`. Runtime construction applies each genome once
through `compose_population_birth`; the exported environment remains unmodified
so inherited traits cannot be applied twice.

The output is built in a sibling staging directory and renamed only after every
artifact validates. An existing output is never overwritten. It contains
`habitat.json`, `biosphere.json`, `resident-birth.json`, `receipt.json`, and one
compact NPZ per resident under `neural/`. Each NPZ contains exactly the seven
current float32 gain arrays plus authenticated metadata. The receipt pins the
profile, assignment and selected world, controller, graph, ports, neural recipe,
compatibility group, files, and immutable candidate order.

The bulk graph remains external. `--service-phenotype-root` records where the
operator will place the compact NPZ files on the neural service host; copying
those files is deliberately an operator step.

The current controller NPZ is also an explicit external input. The command
strictly requires format `chreatures-native-developmental-resident-population-v6`,
metadata version 6, and execution
`developmental-resident-native-population-v6`. Native execution v6 is
independent of the body-facing organism-interface v4 contract, which remains
4,459 observations, 12 physiology fields, and 12 actions. New controller weights
are exported from current v5 inherited policy weights together with an
authenticated recurrent-v3 predictor using
`scripts/export_developmental_resident.py`. The fitted native-v6 controller is
available in the [research release](https://github.com/emberian/chreatures/releases/tag/reciprocal-v6-research-20260906).
Its file SHA-256 is
`00bbd8580baa0f5f016b5c69548210f86412bc54ced32e737793fcedbd0019c4`.
This download contains inherited weights and fitted laws; it contains no personal
memory, neural activity, bulk connectome or complete birth bundle. There is no
implicit conversion of an old controller.

The prior population-v4 controller artifact
`92a1f264e91dd0d3ce156e7e289837d82c1770bf50afb92ef785dfcb66fd6356`
remains historical evidence in Git source revision
`74831ab5fe912c24b57e252e2c8263940657e7a6`. It records the frozen earlier
campaign and is not accepted by the current export command.

```sh
python scripts/export_population_birth.py \
  --profile /tank/chreatures/campaigns/v1/campaign/profile.json \
  --assignments /tank/chreatures/campaigns/v1/campaign/plans/plan-0000/batch-0000.json \
  --world-index 0 \
  --resident-artifact /path/to/developmental-resident-population-v6.npz \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v2-maps.npz \
  --neural-recipe data/ports/neural-variant-canonical-v1.json \
  --service-phenotype-root /srv/chreatures/births/batch-0000-world-0000/neural \
  --output /tank/chreatures/births/batch-0000-world-0000
```

This is a founder artifact, not a continuation. It never copies neural state,
episodic memory, RNG state, personal learning, or an adult snapshot.
