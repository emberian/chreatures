# Reproducible population birth export

`scripts/export_population_birth.py` creates a current population-v4 cold birth
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

The current v4 controller NPZ is also an explicit external input. The cold
conversion that produced the campaign's `92a1f264...6356` artifact was retained
with its research export receipt; this command authenticates those bytes and the
candidate controller pins but does not recreate the controller. New controller
weights should be exported from their current training checkpoint with
`scripts/export_developmental_resident.py`. There is no implicit v3 conversion
or bundled downloadable controller in this workflow.

```sh
python scripts/export_population_birth.py \
  --profile /tank/chreatures/campaigns/v1/campaign/profile.json \
  --assignments /tank/chreatures/campaigns/v1/campaign/plans/plan-0000/batch-0000.json \
  --world-index 0 \
  --resident-artifact data/genomes/developmental-resident-population-v4.npz \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v2-maps.npz \
  --neural-recipe data/ports/neural-variant-canonical-v1.json \
  --service-phenotype-root /srv/chreatures/births/batch-0000-world-0000/neural \
  --output /tank/chreatures/births/batch-0000-world-0000
```

This is a founder artifact, not a continuation. It never copies neural state,
episodic memory, RNG state, personal learning, or an adult snapshot.
