# Batched 3D development

`chreatures/development.py` couples independent MuJoCo habitats to one
in-process full MaleCNS sparse circuit. Each habitat contains the three crawler
bodies from `hollow-garden.json`; eight worlds therefore train 24 residents in
one GPU batch while CPU threads advance eight separate physical worlds.

This is a circuit and sensorimotor development experiment. It does not train a
task oracle and it is not evidence of recovered fly behavior. Policies receive
only the 48 annotation-derived neural readouts, local physiology, and prior
bodily outcomes. Object IDs, positions, kinds, resource colors, and goal
coordinates never enter `AdaptiveOrgan`.

## Developmental process

The first phase creates a simple garden with boundaries, four edible physical
objects, and two movable balls. Each body starts adequately fed and one resource
is placed nearby at a varied angle and distance. The second phase keeps the
same neural and cognitive residents but moves them into the full hollow garden:
ramps, raised ground, an arch, seesaw, pendulum, movable objects, resonators,
shade, food and social signals. World seeds vary body heading and benign object
placement. They do not create deprivation trials.

Every 50 ms step performs this causal sequence:

1. sense all bodies in each real MuJoCo world;
2. encode the exact 16 local sensory channels and step all full MaleCNS states
   in one GPU sparse batch;
3. normalize each resident's 48 readouts with its own online statistics;
4. let its private `AdaptiveOrgan` produce eight continuous body coordinates;
5. advance each physical world and return ingestion, contact, distance and
   effort for the next learning update.

The organ's actor and critic learn from change in bodily homeostatic potential,
hunger-weighted ingestion, improvement in action-conditioned prediction, and a
small effort cost. The intrinsic term uses learning progress rather than raw
surprise. Its forward model learns the feature change conditioned on the actual
action. Context and selectively stored transitions remain private to each
resident throughout development.

## Reproducible run

Use an isolated ROCm environment and keep bulk output on `/tank/chreatures`:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
CHREATURES_MALECNS_DIR=/tank/chreatures/data/malecns/derived \
PYTHONPATH=/tank/chreatures/development-src \
PYTORCH_KERNEL_CACHE_PATH=/tank/chreatures/cache/pytorch-development \
/tank/chreatures/envs/rocm-dev/bin/python \
  /tank/chreatures/development-src/scripts/develop3d.py \
  --graph /tank/chreatures/data/malecns/derived \
  --output /tank/chreatures/runs/development/initial-8x3 \
  --worlds 8 --simple-steps 768 --rich-steps 1280 \
  --checkpoint-every 512 --record-every 1 --workers 8 \
  --seed 20260905 --inheritance-seed 7301
```

This process constructs a separate in-process `RemoteBrain`; it never calls or
changes the live adult service on port 8765. Capacity equals `worlds * 3`, so no
unused persistent residents are allocated.

Resume from a complete checkpoint with the identical run arguments plus:

```sh
--resume /tank/chreatures/runs/development/initial-8x3/checkpoints/development-step-000512.json.gz
```

The compressed development checkpoint records the exact MuJoCo integration and
mutable component state, world RNGs, full private neural rates/adaptation/
support, organ actor/critic/model/context/eligibility, personal memory and RNG,
feature normalizers, prior bodily outcomes, phase, and step. The paired neural
NPZ has its own SHA-256 receipt and graph compatibility checks.

## Outputs and inheritance boundary

`run.json` records the command, environment, source hashes, graph manifest,
PyTorch/ROCm versions, and device. `development.jsonl` is an append-only raw
step log. `trajectory.npz` contains resident-aligned features, actions,
physiology, outcomes, activity and learning diagnostics. `summary.json` reports
throughput and early/late aggregates. The `checkpoints/` directory holds exact
continuations.

`egg.npz` is deliberately smaller in scope. It averages the developed context,
encoder, actor, critic and action-conditioned model arrays plus feature
normalization statistics across the nursery. `apply_egg` imports only the
inherited organ arrays and leaves a recipient's personal context, physical and
neural state, random stream and autobiographical memory untouched.
`experienced_episodes.json.gz` stores the private training records separately
for research inspection. Its contents are not silently copied into an egg.

`egg-manifest.json` states this boundary and hashes every export. Individual
organ checkpoints remain the evidence for variation among residents; the egg
is an ensemble inheritance proposal rather than a claim that averaging is a
biological mechanism.
