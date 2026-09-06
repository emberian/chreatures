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
