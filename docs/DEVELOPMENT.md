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

## Initial measured baseline

The command above completed on hbox as PID `3433244`. It used the 165,122
neuron, 25,563,197 edge MaleCNS artifact with dataset SHA-256
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.
The runtime was PyTorch `2.9.1+rocm6.3`, HIP `6.3.42134-a9a80e791`, SciPy
`1.18.1`, and MuJoCo `3.12.0` on gfx1030. The live adult service on port 8765
was not addressed by the runner.

| Measurement | Result |
| --- | ---: |
| Curriculum | 768 simple + 1,280 rich steps at 50 ms |
| Population | 8 worlds × 3 residents = 24 |
| Total experience | 49,152 resident-steps; 102.4 simulated seconds/resident |
| Wall time | 293.17 s |
| Sustained end-to-end throughput | 6.99 world-batches/s; 167.66 resident-steps/s |
| GPU allocation after run | 257,306,112 bytes |
| Neural activity mean | 0.022386 |
| Physical travel summed across residents | 233.82 m |
| Contact-positive resident-steps | 1,602 across 16 residents |
| Ingestion | 59 events, total 0.17691, across 11 residents |
| Selective personal memories | 205 total; 1–32/resident, mean 8.54 |
| Trajectory | 19,106,493 bytes, SHA-256 `e0c5ade7c74d78684c91a0f00c6302a0fc21ecc9a700b542592a67a18f6aa0e9` |
| Egg | 43,718 bytes, SHA-256 `34ed3170f94b657d8a7f56313e773349568626d168d70ec2adffbe271f798bc4` |

All residents began from the same inherited arrays but sampled separate motor
streams in varied worlds. Their final individual L2 parameter changes ranged
from `0.00193–0.00952` for the actor, `0.00081–0.00400` for the critic, and
`0.00049–0.10909` for the action-conditioned model. Ensemble egg changes were
`0.000952`, `0.000848`, and `0.01935`, respectively. These nonzero changes and
the stored eligibility/memory traces establish that actual online updates ran;
they do not establish a learned behavioral skill.

The simple-phase mean prediction error fell from `1.04e-6` in its first 256
steps to `5.86e-8` in its last 256. Entering the richer habitat produced new
contacts and less predictable feature changes; error rose to `6.09e-5` in the
last 256 steps. A future held-out comparison is required to separate curriculum
difficulty, habituation, resource depletion, and policy learning. The current
baseline therefore preserves the raw evidence and makes no improvement claim.

The final checkpoint cold-restored all 24 residents, eight rich physical worlds,
full neural state, organs and memories in 6.24 seconds. Neural clocks restored
to 102.4 seconds and rich-world clocks to 64.0 seconds. A separate import audit
confirmed that applying the egg changes inherited arrays while preserving a
recipient's personal time, context, memory and state.
