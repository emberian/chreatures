# Population and ecological development wave

This is the shared implementation contract for the September 6 population wave.
It replaces the previous timed build window. Implementation proceeds across the
coupled mechanisms before the main integration campaign. Existing live worlds
retain their immutable deployments.

## Organism interface v4

The current organism has twelve continuous action channels, in this exact order:
`thrust, yaw, gaze_pitch, posture, grip, signal_low, signal_mid, signal_high,
eat, release, secrete, allocate`. Indices 0–3 are signed [-1,1]; indices 4–11
are rectified [0,1]. The twelve delivered actions are also the efference-copy
history. There is no separate automatic eating command. Release transfers finite
gut material through physical contact or a material bolus. Secretion spends a
real gland budget on anonymous local chemical emission. Allocation directs a
bounded share of available material/energy into inherited developmental organs,
including structural growth and reproductive tissue. These are physical drives,
not commands naming a target, role, behavior, or successful outcome.

The sensory vector is `rich rays 4096 + canonical channels 351 + physiology 12`
(4459 columns). Physiology order is `energy, gut, fatigue, speed, turn,
neural_support, structural_integrity, development_fraction, gland_fill,
brood_fill, reproductive_maturity, exchange_load`. Speed and turn are signed;
other normalized channels are in [0,1]. New quantities must come from actual
private body state. The existing six physiology channels retain their meanings.
The body front therefore takes 363 columns. Neural readout remains 384 columns;
the visual front remains the existing 1024 body-local rays in this wave. Larger
worlds do not grant a global view or world coordinates to the controller.

Snapshots/artifacts change version. Old weights may enter through a single
explicit cold conversion which extends tensors and records the inherited source;
the production runtime accepts only the current shape. A frozen H1 predictor
may remain a named inherited organ with its original six-state observation and
eight-motor-plus-oral projection. Its score cannot claim to predict the new organ
effects. No legacy engine fallback is added.

### Learning to use the new organs

The organism interface remains v4; the current controller artifact is v5. The
first cold v4 inheritance exposed a concrete coupling limitation: its six new
body-input columns were zero while the body encoder was frozen. Population
training also froze the zero new-actuator output weights, so a learned policy
adapter could not make eating, release, secretion or allocation depend on context.
In the actual frozen training run these axes appeared only about 40–63 times
each across 143,360 transitions. Those histories remain preserved as v4 evidence.

The successor adds a trainable, bias-free 12-to-128 physiology projection before
the worker GRU, plus trainable active/magnitude projections for the four new
actuators after the candidate policy adapter. These are shared inherited organs
optimized in designated training populations and frozen into each exported
artifact. They are distinct from resident-private lifetime learning. Achieved-goal
encoding stays frozen; new physiology can influence action and recurrent state
without changing the coordinate system of remembered sensory goals.

The single cold converter records an explicit new-axis exploration probability,
default 0.05, with uniform positive magnitude bins. It specifies initial motor
exploration rather than a successful eating, secretion or reproductive sequence.
Current trainers and exporters accept only the current controller format. The
[native parity receipt](POPULATION_V5_NATIVE.md) checks the new pathways with
nonzero parameters; it is mechanism evidence, not learned behavior.

## Population, inheritance, and lives

A candidate is an immutable genome artifact: parent genome hashes (one or two),
variation seed/recipe, graph and base-controller hashes, body parameters,
metabolic/developmental allocation, neural modulation and controller adaptation.
Genome recombination never combines learned private states or world checkpoints.
The existing MaleCNS graph is the default anatomical ancestor, and existing body,
metabolic, and circuit-blueprint mechanisms supply the actual phenotype.

Large logical populations execute in hardware-sized groups. Shared graph/base
weights are immutable; candidate adapters, personal learning, neural state, body,
RNG and history are private. Candidate grouping is by compatible graph/ports and
controller interface, not by pretending the candidates have identical parameters.
Resident counts are a world parameter, not a hard-coded six. Capacity exhaustion
is explicit; no resident is evicted to hide the cost of a birth.

The native search service owns deterministic variation, a bounded multi-member
quality-diversity archive, selection, environment mutations and durable search
state. It consumes complete physical episode results; a failure is a retained
terminal result. Descriptor and quality recipes are versioned and report their
underlying physical components. A changed learned descriptor creates a new
archive epoch. A separate environment archive preserves topology/resource
ancestry; probe-policy panels used for environment difficulty/novelty are frozen
within an epoch. Direct transfers precede any candidate fine-tuning.

## Physical ecology and development

The environment generator creates connected regions with variable scale,
elevation, ramps, cavities, exposed and sheltered surfaces, resource/light
gradients, movable finite material, and sites that organisms can physically alter.
Regional designer graphs remain analyst-only. Geometry and resource variation
are inherited environmental parameters; no resource is teleported to a creature.

All mobile organisms and constructed tissue use the existing conserved chemistry.
Photosynthetic, digestive, structural and reproductive allocation differ through
heritable budgets, not species-role scripts. Growth, secretions, exchange and
brood maturation debit material/energy. Contact and local fields mediate social
effects. No caregiver/organism identity is a privileged sensory channel.

An in-world reproductive event produces a funded embryo/egg with a genome and
development state. Hatching is a committed topology/population transition that
creates a new life and fresh private controller/neural state. Deferred or blocked
hatching remains physical brood state, not a silently dropped birth. The initial
campaign can also instantiate fresh evaluation lives directly from genomes; this
is experimental initialization and is recorded separately from embodied birth.

The current in-world rule is clonal and asexual: an offspring inherits its
parent's immutable candidate genome and constitutive body, enzyme, capacity, and
allocation template. It receives no parent memory, controller state, neural
state, or optimizer state. Population search varies genomes only between cold
evaluation lives and is not consulted by a hatch inside a running world.

The joined birth check moved 0.01 conserved material and 0.01 ATP from a funded
parent brood row into a fresh newborn body row. It appended one physical body and
five private chemistry rows, retained the existing metabolic program and clock,
and left every old chemistry row byte-identical except the debited parent brood
row. The chemical commit then synchronized all parent and newborn public body
readouts after physical topology adoption; a version 7 snapshot restored exactly
and completed a subsequent three-resident physical and chemical step. This is
mechanism evidence for the transaction, not evidence of reproductive success in
a population campaign.

## Runtime and evidence boundaries

Rust owns recurring physiology, lifecycle, population search, memory and numeric
kernels. Python hosts MuJoCo bindings, artifact I/O and Torch/ROCm optimization.
The campaign coordinator assembles current components and schedules isolated jobs;
it does not implement another evolutionary algorithm in Python.

Native GAM laws participate in declared internal mechanisms with operating-domain
and stability bounds. Population fits characterize phenotype/history/environment
effects using whole lives/worlds as units. New fitted laws become inherited
artifacts by an explicit new candidate birth, never an invisible resident update.

Universal Weave records candidate/environment ancestry, births, checkpoints,
completed and failed evaluations, archive decisions, transfers and fitted laws.
Genetic parentage, environment parentage and continuation of a life are different
edge roles. Arrays/checkpoints remain external content-addressed blobs. Publishing
an archive or viewing a branch never advances or merges a life.

The joined campaign follows the development wave: compile/interface checks during
construction, then full physical episodes with heterogeneous candidates and richer
worlds, actual AMD/Metal costs, complete coupled restoration, and retained failures.
Useful ecology, inheritance and learning are empirical results, not properties
conferred by this architecture document.

## Material packet performance receipt

On an Apple M2 running macOS 26.6.1, Python 3.12.14 and MuJoCo 3.12.0, an
eight-resident `terraced-delta` region (seed 2026090602, 130 chemistry rows) was
advanced for three warm-up and twenty measured 0.05-second steps. Each resident
requested release 0.1, secretion 0.2 and allocation 0.3. The original compiled
clearance path spent 56.62 ms per biosphere step on average (138.76 ms p95).
The native conservative packet-overlap path spent 6.01 ms on average (11.18 ms
p95). The retained comparison disables only the conservative preflight for its
forced-compiled control, so both arms use the current chemistry and authoritative
topology transaction. Historical implementations remain available from repository
Git pins. The command was:

```text
PYTHONPATH=/tmp/chreatures-material-overlap-native2:/Users/ember/dev/chreatures .venv/bin/python research/performance/material_packet.py
```

The final source hashes were `bf0317e541e03fac5e0175f98129ae2fa400bfa9e9b14d68de8b855a54625b2e`
for `chreatures/material_objects.py`,
`ea8d86447a4857f0ddfb6e987fd9bc39a0726701a8b023cfd6a6d185a7a51c2f`
for `native/world-kernels/src/material_overlap.rs`, and
`65b5e8a3d15bc39aa4e7a107e0c4b1ae9e838465dd6a02f5cf107332471fa8a2`
for the native module registry. The test extension SHA-256 was
`e713c58a207ba12ae3b5d02b787a99af7a5b9b122893c980361167413ab85866`;
the retained comparison script SHA-256 is
`29d21aaada125c4cb0ab2869941c695c5de199411f5d00c4bf55ddc685b5b879`.

A 90-step optimized-versus-forced-compiled comparison was bit-identical for
both the complete biosphere snapshot
(`89c2acfcf2ff30ba1b23032c4e60795a2daf183e56385fb5f43e7ec38cfec7b0`)
and physical-world snapshot
(`a5fe7dd996a5a2c9f3d2fc9ca55826b0e1a22f3ac6af2b84fae0898ea7556f17`).
An overlapping request with an empty donor retained receiver limiters, reported
the donor limitation and created no boundary. Two funded edges targeting the
same obstructed dormant entity were both blocked and matched the compiled path
exactly, including snapshot SHA-256
`bbbe2040c2ed47c5286678b52588598e94ac3205c0f437fb3b1cc837da76614d`.
