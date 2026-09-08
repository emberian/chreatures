# Chreatures

**A world to grow into.** An open artificial-life experiment in inherited bodies, personal histories, measured fly wiring, and ecologies that change their own habitat.

Inspired by *Creatures* and the changing societies and environments of *Children of Time*, we are building organisms whose sensations, chemistry, movement, memories and surroundings have consequences for one another. The goal is a world worth inhabiting, with instruments for discovering how its inhabitants develop.

[Public field guide and observatory](https://emberian.github.io/chreatures/) · [Source and data notices](NOTICE.md) · [Development log](docs/CYCLE_LOG.md)

## The current build

**MaleCNS V4 runs in articulated fly bodies inside a layered physical habitat.**
The full MaleCNS v1.0 graph supplies 165,122 neurons and 25,563,197 directed edges
representing 124,025,046 synapses. We advance neural activity, adaptation, local
support, release availability and three modulatory fields on that graph. Their
dynamics are explicit modeling assumptions; fitted response curves do not replace
its recurrence.

The body starts from the author’s micro-CT-derived NeuroMechFly morphology:
69 segments, 126 articulated axes, legs, antennae, proboscis, wings and halteres.
It is a female body paired explicitly with a different male CNS specimen.
Actual physical rays supply RGB at 1,771 bilateral retinal sites, of which 1,486
have connectome anchors. Another 807 body channels carry joint state, loads,
contact, airflow, acoustic bands and internal chemistry. Antennal collision
proxies now transmit reciprocal material contact through those same channels.
Their shape and compliance are engineered approximations derived from the meshes.

Learned interfaces route physical senses into anatomically matched afferents.
Only post-recurrence activity from 815 annotated motor neurons supplies the 92
body outputs: 84 neutral-relative joint servos, six foot adhesion channels and
two oral drives. These are effective actuators, not identified muscles. Private
Rust memory reads CNS activity and sends twelve signed context currents back
through 1,314 descending neurons. It cannot command the body directly.

The selected two-fly habitat contains 156 finite material regions, 190 connecting
routes, bark shelters, stems, elevated leaves, ramps and movable resource packets.
Material-funded branching creates physical geometry; light and clearance affect
where it can grow. Visitors can place a movable object on a chosen surface, push
it, send tones or put Bad Apple on a physical screen. The inspector shows actual
retinal inputs, all seven neural fields, body signals and motor outputs.

[Open the live garden](https://emberian.github.io/chreatures/live.html) ·
[Try Bad Apple live](https://emberian.github.io/chreatures/live.html?stimulus=bad-apple) ·
[Fly morphology and mechanics](docs/development/FLY_BODY_ANATOMY.md) ·
[Antennal touch](docs/development/FLY_ANTENNA_TOUCH.md) ·
[Download the selected initialized V4 artifacts](https://github.com/emberian/chreatures/releases/tag/live-cns-v4-initialized-20260908)

The complete antenna-enabled CNS/body/private-memory run passed 128 physical
ticks, exact whole-life restoration and insertion/pushing of an object into the
changing world. Its M2 mean complete tick was 337.7 ms for 10 ms of model time
under concurrent load. This is an execution measurement, not a real-time or
browser UI performance claim. [Joined receipt](research/fly_embodiment/antenna-habitat-joined-receipt.json).

### What training has established

**The selected public model is initialized and untrained.** The trained children
remain research artifacts: offline motor prediction improved, but actual physical
rollouts toppled sooner and used more effort. The disagreement is recorded rather
than hidden by promoting the trained model. Torch and WebGPU agree numerically;
this is a learned-control problem, not merely a deployment discrepancy.

Twelve recovery worlds supplied 49,152 resident transitions. A new fit on
persvati adds a neutral cold-state prior through the existing learned motor
decoder; hbox concurrently collects supported author-step histories and unchanged
child probes. Failed recovery and useful supported transitions remain distinct.
The imported walking references presume a supported fly and do not supply
self-righting. [Training implementation and evidence](research/fly_learning/README.md).

Native **GAM** fits physical and neural mechanisms and proposes experiments that
are replayed in the actual simulator. Earlier proposals missed their held-out
confirmation targets; those failures remain visible. A new native ecology
campaign varies inherited branching parameters across physical layouts and uses
measured route apertures. **Universal Weave** connects the actual episodes, fits,
controls and contradictory outcomes; its archive is separate from private memory.

Lifetime learning currently changes private context memories and bounded
consequence estimates. Lifetime plasticity of the anatomical synapses, structural
connectome evolution, reliable locomotion and durable social competence remain
work ahead. The [Pages release process](docs/PAGES.md) pins source, body, native
runtime and model independently. Old lives retain their frozen engines and state.

### Earlier generations remain distinct

[V3](https://github.com/emberian/chreatures/releases/tag/live-cns-v3-research-20260908)
introduced the anatomical sensory/motor and context-feedback boundary using a
simpler body. Its training and GAM experiments remain in
[their own receipts](docs/receipts/anatomical-cns-v3/).

The earlier V2 film-versus-blank experiment changed neural activity and body
trajectories over 30 model seconds. Those measurements belong to its older body
and controller, not the current V4 life.
[Earlier screen evidence](https://emberian.github.io/chreatures/evidence.html#physical-screen-response-v2).
The ecological v8 world below likewise remains an explicitly recorded baseline;
its direct sensory routes are not a compatibility mechanism in the current runtime.

## The completed ecological v8 baseline

The **ecological specialization wave** adds connected finite material stores,
physical outlets, inherited metabolic acclimation, and private recall of executed
motor sequences. Construction can obstruct the material routes. Enzyme changes
have inherited time constants, expression budgets and ATP costs; an offspring
inherits response rules rather than an adult's current expression. Native v8
residents compare four local plans and up to four remembered action sequences
through the recurrent predictor, then act for one physical tick and reconsider.
The joined eight-founder physical world reached tick 10,006 and was paused
with a complete checkpoint. Its public recording contains short sequence
continuations but no completed eight-tick sequences. Per-tick reselection
interrupts them too frequently; learned termination is part of the replacement.
These mechanisms are not evidence of ecological competence. [Current wave](docs/development/ECOLOGICAL_SPECIALIZATION_WAVE.md) · [Earlier shutdown record](docs/development/ECOLOGICAL_SPECIALIZATION_HANDOFF.md).

The preceding reciprocal ecology wave completed **160 research lives** and
327,680 physical transitions. Its recording remains available as an archived observatory selection; the
default now shows the completed v8 ecological world. Its longer private
continuation exposed a material lifecycle fault: a loose packet escaped the
finite terrain and kept falling. The new regional accounting transfers a
packet's complete contents into a regional store and retires its physical body
at a declared exit face. The old failed life remains preserved under its frozen
engine. [Previous wave](docs/development/RECIPROCAL_ECOLOGY_WAVE.md).

The earlier **v4 organism and population wave** exposes twelve explicit actions—thrust, yaw, gaze pitch, posture, grip, three signal bands, eating, release, secretion and allocation—against twelve measured physical channels spanning movement, energy, digestion, fatigue, neural support, structure, development, gland and brood stores, reproductive maturity and exchange load. Native cohort execution keeps learning, memory and recurrent state private to each life. Immutable candidate genomes can inherit full-MaleCNS interface gains and completed GAM law fits without inheriting that private state.

Regional grammars generate connected physical habitats with variable elevation, cavities, ramps, finite resources and growing material. Clonal births debit a parent's actual brood stores and commit a new body with fresh private state. A native quality-diversity search evaluates genome–environment pairs and retains a bounded archive of varied candidates, including terminal failure records. The first campaign has closed **80 candidate lives across ten environments** on hbox: 16 completed their allotted run, while two shared engine faults ended the other 64. Twelve evaluations received archive admission; nine members remain after subsequent replacement. The project has executed separate 32-resident training populations on hbox and persvati and eight-founder interactive worlds on M2. The [population observatory](https://emberian.github.io/chreatures/population.html) publishes actual genomes, environments and recorded lives. Archive retention does not establish evolutionary improvement or ecological adaptation.

The earlier **Living Reef** supplied the constructed-world substrate: articulated bodies, growing colonies, terraces, ramps, underpasses, a coupled gate, acoustic mechanisms and movable materials. The current regional family extends that substrate; the ten campaign environments contain 15–35 platforms and 22–52 connecting structures each. Colonies build real collision geometry, changing light, contact, passage and chemical transport. A native solar cycle moves illumination through the landscape and supplies energy to phototrophic chemistry.

Residents acquire finite material through mouth contact, digest it into usable reserves, spend energy on activity, and return material through physical deposits. Colonies can release accumulated reserves into consumable packets. These mechanisms share conserved synthetic chemistry. Their combination provides an ecological substrate; a self-sustaining food web, reproduction and evolved social organization remain goals.

Some inherited bodies also capture light through a dorsal surface. Its area is bounded by the physical thorax; orientation and occlusion change the available photons. The same carbon-fixation chemistry converts them subject to finite substrates and enzyme activity. Absorptivity varies continuously down to zero, while all bodies retain feeding and movement. [Mobile mixotrophy](docs/MOBILE_PHOTOTROPHY.md).

The current neural substrate is the **MaleCNS v1.0 brain and ventral nerve cord**: **165,122 traced neurons**, **25,563,197 directed edges** and **124,025,046 synapses** represented by those edges. Anatomical wiring constrains the recurrent network. The rate dynamics, chemical rules, bodies and sensory/motor interfaces include explicit engineering assumptions. This is a synthetic species project, not a recovered fly.

The recorded v8 control loop combined the following mechanisms. Its direct
sensory paths are being removed from the replacement; this list describes the
frozen recording, not the intended CNS-only data flow:

- **Body-bound vision:** 1,024 native collision rays, divided between an 8×32 peripheral field and a 24×32 central field. Each supplies RGB and proximity. Bodies, constructed surfaces and movable objects can occlude them.
- **Measured recurrence:** 351 sensory channels enter the full connectome; 384 named population readouts reach the resident's goal selector. These are population summaries, not recordings of 384 individual neurons.
- **Spatial perception and working memory:** a native convolutional visual front, body-state encoder and persistent private GRU process the 4,459-column current observation at each 50 ms physical tick.
- **Experienced goals:** each individual retains a private reservoir of four-frame sensory encounters. A learned manager selects among those memories; the motor controller attempts to approach the selected sensory state. The current Rust controller also learns private, physiology-dependent goal preferences from the actual bodily return of completed attempts.
- **Personal consequence learning:** inherited GAM predictions and private bounded residual learning estimate movement, energy-cost and fatigue consequences of motor proposals. Actual delivered actions and their subsequent physical outcomes supply the updates.
- **Action-conditioned forecasts:** the fitted three-member recurrent ensemble predicts sensory and physiological changes across up to eight twelve-axis actions. The native planner compares eight-tick local hypotheses and recalled sequences over their actual stored lengths, delivers one physical tick, and replans from actual sensations. Its contribution to selection is bounded; predicted states never enter experienced memory. Older published lives retain their pinned one-step ensemble.
- **Private sequence memory:** observed attempts, attainment and context build sparse transitions between remembered encounters. Lifetime and recent memory share a bounded store, so old residents can continue acquiring new experience. Past succession can bias a proposal, but does not establish present reachability.

Eating is an explicit current action. Physical mouth contact, available material and digestive chemistry determine its consequences. Remembered goals are previously experienced states, not guarantees of present reachability. The first v4 cold inheritance conservatively initializes new action heads and extends the sensory interface; that initialization does not supply competence with the new organs. Useful navigation, durable learned habits and reciprocal interaction are still being developed.

The historical **Torch v5 policy** trains pathways from all twelve physiology channels into recurrent state and from the policy into the four new organ actions. Achieved-goal encoding stays fixed. Its completed hbox lineage reached 160 PPO updates and 655,360 resident transitions. The native-v8 export combines those weights with the fitted recurrent predictor and private acquired action sequences. The preceding update-20 recording remains available as evidence of that earlier deployment; initialized exploration and nonzero actuator weights alone do not establish competence. [Organ implementation](docs/development/POPULATION_V5_NATIVE.md) · [Current export identity](data/training/rich-recurrent-v3/fit-export-receipt.json).

## GAM × Universal Weave

[**SauersML/gam**](https://github.com/SauersML/gam) supplied fitted mechanisms in the recorded v8 controller. Native GAM fits compress experienced nonlinear body responses into small, immutable consequence models. The frozen v8 Rust engine evaluates these models while comparing motor proposals with the bodily component of a remembered sensory goal. Direct physiology-based candidate scoring is being removed from the replacement. GAM remains useful for physical law fitting and external analysis; future control-facing models must consume CNS-derived state. Each resident learns its own bounded corrections from its own experience; shared inherited predictions remain unchanged. Out-of-domain candidates retain the underlying actor's support without receiving a GAM refinement. This is an explicitly engineered control layer, not a happiness measure or proof of causal understanding. [Implementation and fitted data](docs/GAM_MECHANISMS.md).

[**transkatgirl/universal-weave**](https://github.com/transkatgirl/universal-weave) connects recorded development, model artifacts, snapshots, experiments and competing explanations. Its native adapter supplies stable event identities, multi-parent evidence records and deterministic serialization. The scientific archive is separate from the incomplete, private memory available to an organism. [Native integration](docs/LIBRARIES.md).

Both upstream libraries have executed against actual project data. Their artifacts preserve sources, versions and limits.

A new native **genotype-by-environment GAM atlas** uses the 160 completed lives
to separate pre-run inherited/environmental predictors from descriptive models
that also use realized actions. Grouped validation and holdout showed only a
small energy-prediction gain; contact, work and allocation did not pass both.
The atlas retains those failures, support limits and alternative fits. It can
rank supported experiments under an explicit selection policy, and rejects
new ecological-v7 mechanisms until they have observations of their own.
[Executed atlas and limits](docs/development/GENOTYPE_ENVIRONMENT_ATLAS.md).

The newest GAM bank fits energy change, fatigue change and effort from 384,000
transitions across 16 completed lives. Whole-life held-out errors improve over
training-mean baselines; 95.93% of final held-out transitions lie inside all three
declared domains. This is a compact inherited predictor of recorded responses,
with private lifetime learning kept separate. It does not establish causal
regulation or genotype–environment transfer. [Fit and support](docs/GAM_POPULATION_RESPONSE.md).

## What has run

The reciprocal population campaign completed **163,840 resident transitions**
across ten environments and 80 lives. It took **986 wall seconds**, including
startup, checkpoints and the first result-assembly failure; the final report was
reconstructed from preserved data without replay. Private goal associations
received 15,237 updates, while 1,003 evicted-memory receipts correctly received
no attribution. There were only **five mouth-contact ticks**: these results
establish coupled execution and recorded learning updates, not learned feeding
or a sustained food web. The search retained 62 entries in 19 archive cells
and selected a separate challenge wave. [Per-life results and costs](data/development/population-v6-principal-wave.receipt.json).

That challenge wave also completed: another **80 lives and 163,840 transitions**
in **969.03 seconds**, with 16 mouth-contact ticks and 15,162 private learning
updates. These 80 transfer assignments used ten selected genomes, so their difference from the first wave
is not a matched estimate of improvement. Across both waves, 160 completed life
records now inform the search, with 74 archive entries in 21 cells and no pending
assignments. [Challenge outcomes and selection scope](data/development/population-v6-challenge-wave.receipt.json).

The [default observatory](https://emberian.github.io/chreatures/living.html)
now replays **360 v8 frames from eight residents**, covering model seconds
265.05–352.65. Its 64,843 recorded events include regional material transport,
contact, growth, signals and eight outside light/tone stimuli. The recording
contains 179 continuation selections across all residents, zero complete
sequences, and 6,164 observed interruption increments. Native Universal Weave
preserves its evidence as 64,954 nodes and 134,994 links, with exact
serialization/reload equality. The [public projection](site/assets/ecological-specialization-evidence.json)
contains hashes and counts without private checkpoint contents.

The previous reciprocal v6 recording remains selectable as an archive. Its
independent restore check produced byte-identical world and neural snapshots
after one step; that result belongs to the frozen v6 engine.
[Previous restoration scope](data/development/reciprocal-v6-research-continuation.receipt.json).

The longer private continuation later paused at tick 9,170: a free material
packet escaped the finite terrain and eventually exceeded the field coordinate
sanity bound. Its coherent tick-9,148 checkpoint is retained. That engine lacks the new complete packet-exit transaction. The recording and
completed campaigns precede the failure; the current physical join exercises
finite outlet activation, exit transfer and retirement. [Longer-run stop](data/development/reciprocal-v6-long-continuation-stop.receipt.json).

The new predictor corpus contains **393,216 actual transitions**, collected in
six eight-resident worlds at **229.06 resident transitions/s**, including sealing
and checkpoints. Its three-member recurrent fit took **399.79 seconds** on
persvati's AMD Radeon 890M. In the final held-out source-world slot, four-tick
goal-code RMS was **0.23464**, compared with **0.55048** for persistence. Native
and Torch inference differed by at most `1.76e-6` in the retained numerical
comparison. These are prediction results from frozen-v5 source dynamics;
transfer to the new chemistry and improved physical control remain unestablished.
[Fit, split and numerical scope](docs/RICH_PREDICTION.md).

The earlier regional observatory replays **240 actual frames from eight v4 residents**, covering model seconds 51.65–118.50. The source recording contains delivered actions, body measurements, real geometry and private learning diagnostics. Its simulation ran on the M2 with the full MaleCNS graph. This is a recorded episode, not a live connection to the private world, and it does not establish successful feeding, reproduction or social learning. [Watch the regional world](https://emberian.github.io/chreatures/living.html?recording=regional-wave).

The earlier **trained v5 research world** also contains eight
residents, using the update-20 controller and the two additional population GAM
laws. Its 240 recorded frames cover model seconds 40.35–100.20 after an outside
material offering and light/sound sequence. During that interval, 7,694 committed
transitions were within the new bank's fitted domain and 1,882 were outside;
private goal learning continued and all four new actuator channels executed.
This establishes joined execution, without a demonstrated behavioral benefit.
[Watch the trained organs](https://emberian.github.io/chreatures/living.html?recording=trained-organs)
· [Run identity and capacity](docs/development/POPULATION_V5_RUN.md).

The first full population campaign recorded **1,033,824 resident transitions**. Two 32-life cohorts stopped at ticks 9,967 and 10,340 when a depleted ATP payment exceeded the available amount by one floating-point rounding unit. The remaining 16 lives reached tick 24,000, or 20 model minutes. Their completed runs did not demonstrate sustained feeding or regulation. The native repair conservatively partitions actual available ATP and commits cohort payments atomically; it does not grant energy. Native search retains the affected lives as engine failures with no archive quality. The [Universal Weave evidence](integrations/artifacts/population-wave-v1) preserves all 80 terminal histories and the shared causes.

Two new native GAM fits use the first cohort's 315,392-transition durable prefix. On 108,416 held-out transitions spanning eleven whole candidate/environment units, energy-change RMSE is `2.507e-5` versus `3.468e-5` for a training-mean baseline; effort RMSE is `0.03269` versus `0.13016`. The archive retains the failed fitting attempts and actual response surfaces. These are predictive associations from a censored run with one shared founder, not causal laws or lineage generalization. [Population response models](docs/GAM_POPULATION_RESPONSE.md).

The completed rich developmental lineage ran **491,520 resident transitions and 160 additional PPO updates** in four six-resident worlds on an AMD RX 6750 XT, at **224.76 resident transitions per wall second**. It inherited 40 earlier updates. Ninety of 96 resident episodes included mouth-material contact, and the run transferred 28.35 units of conserved material into bodies. Every episode still lost energy. These are actual physical outcomes, without a matched baseline establishing improvement. [Full-transition analysis](docs/RICH_DEVELOPMENT_ANALYSIS.md).

The earlier **196,608-transition rich play corpus** includes raw retinal observations, full-circuit readouts, delivered actions and physical outcomes. Native GAM fits on that corpus supply the embedded body-consequence bank. The newer 393,216-transition corpus supplies the current recurrent predictor; older recorded lives retain their original one-step forecasts. [Fitted mechanisms](docs/GAM_MECHANISMS.md).

The [earlier predictive courtyard recording](https://emberian.github.io/chreatures/living.html) preserves 240 actual frames of body-bound retinal inputs, remembered goals, forecasts, GAM updates and delivered controls. That life later paused on a physical-source error during an incomplete tick; its last complete checkpoint remains preserved. The published segment precedes the failure. A previous saved reef life ended with depleted reserves.

The current regional campaign uses the native resident controller, including private consequence and goal learning, and records actual actions and physical consequences. The earlier nursery families supplied courtyards, tiered shelves and braided passages with disjoint layout seeds. Superseded Python controllers and training entry points have been removed; their research records retain Git references. [Population campaign](docs/development/POPULATION_CAMPAIGN.md) · [Earlier nursery families](docs/NURSERY_FAMILIES.md).

## Build and run

The full graph and training checkpoints are acquired separately; bulk arrays stay outside Git. Python hosts the application boundary and Torch/ROCm research training. Recurring vision, resident cognition, motor kernels, chemical transport, growth bookkeeping and model inference use Rust or the native physics engine.

```sh
uv sync --extra dev
uv run python native/world-kernels/build_extension.py
uv run python native/cognitive-core/build_extension.py
```

The research release [`cns-optic-v1-research-20260907`](https://github.com/emberian/chreatures/releases/tag/cns-optic-v1-research-20260907) pins its exact source and supplies the fitted `CHCNS1` service, a source-bound initialized resident, its immutable sequence-control head and provenance receipts. The service received 128 updates of procedural optical pretraining. Held-out optic prediction MSE did not improve, and the resident motor controller is initialized but untrained; the release establishes an executable CNS-only path, not behavioral competence. The prior v8 life remains reproducible only from its [archival source pin](https://github.com/emberian/chreatures/tree/5a030ddaab4addd2de69ad36c1def197314a5307).

Build the native Metal service and start a dedicated empty instance. Capacity 32 leaves room for offspring when the founding cohort is smaller:

```sh
cargo build --release --manifest-path native/metal-brain/Cargo.toml \
  --bin metal-brain-server

RELEASE_DIR=/path/to/cns-optic-v1-research-20260907
RESIDENT="$RELEASE_DIR/cns-resident.npz"
uv run python scripts/serve_metal.py \
  --artifact "$RELEASE_DIR/cns-service-fitted-v1.bin" \
  --binary native/metal-brain/target/release/metal-brain-server \
  --capacity 32 --kernel simd \
  --snapshot-dir /path/to/cns-snapshots \
  --pid-file /path/to/cns-service.pid \
  --bind 127.0.0.1 --port 18790
```

The packaged resident already binds that exact service identity. To reproduce it from the fitted service, write to new paths because artifact publication refuses overwrites:

```sh
uv run python scripts/export_developmental_resident.py \
  --cns-service "$RELEASE_DIR/cns-service-fitted-v1.bin" \
  --output /path/to/reproduced/cns-resident.npz \
  --sequence-control-output /path/to/reproduced/sequence-control.npz \
  --seed 20260907
```

Materialize one physical world and its CNS-only birth manifest from a population profile and founder assignments. Candidate body and metabolic loci remain inherited; retired neural-gain and policy-adapter loci are not active controller inputs:

```sh
uv run python scripts/export_population_birth.py \
  --profile /path/to/profile.json \
  --assignments /path/to/founder-assignments.json \
  --world-index 0 \
  --resident-artifact "$RESIDENT" \
  --output /path/to/cns-birth
```

Start the world against the same dedicated service:

```sh
uv run chreatures --port 8790 \
  --brain-url http://127.0.0.1:18790 \
  --body articulated --ecology diffusion --physics-backend vectorized \
  --resident-artifact "$RESIDENT" \
  --population-birth /path/to/cns-birth/resident-birth.json \
  --habitat /path/to/cns-birth/habitat.json \
  --biosphere /path/to/cns-birth/biosphere.json \
  --checkpoint runs/new-cns-life.json
```

Use a fresh checkpoint path and a dedicated empty neural service for a new birth. The example ports are placeholders; choose unused local ports.

The local interface lets people manipulate physical objects, offer finite resources, and make light, sound and gesture stimuli. Its inspector shows actual retinal inputs, population readouts, motor actions, remembered goals and physical state. GitHub Pages offers both the live local Wasm/WebGPU habitat and clearly labeled archived recordings. The live page computes on the visitor’s device; it does not connect to a private running world. The [Pages release process](docs/PAGES.md) selects one authenticated fly-world, resident runtime and browser model bundle; the current initialized model is labeled without a competence claim.

Whole-world checkpoints preserve neural and physical state, private memories and learning, RNG, pending actions, chemical pools, constructed topology and the solar clock. Restore checks the pinned source, native binaries and runtime before touching the remote neural state. Deploy lives from immutable source directories. An ambiguous distributed mutation pauses the world. Current development deliberately breaks obsolete interfaces and checkpoint formats; old engines belong in Git history rather than parallel compatibility paths.

## Working together

The project is developed through human direction and parallel coding agents. We build substantial coupled capabilities, then examine their behavior in joined worlds. Body competence, memory, development, social interaction and ecology advance together.

The initial 2D habitat used a **female FlyWire v783 subset of 6,789 neurons**. It was not MaleCNS. That engine lives in the [compact archive](https://github.com/emberian/chreatures/tree/archive/compact-2d-flywire-v1). Further directions include [inherited bodies](docs/BODY_INHERITANCE.md), [diversifying neural blueprints](docs/CIRCUIT_BLUEPRINT.md), [constructed ecologies](docs/ECOLOGICAL_COMMONS.md) and [social organization](docs/SOCIAL_ECOLOGIES.md).

Original Chreatures code is **AGPL-3.0-or-later**. [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) distinguish original code from separately licensed scientific data, pretrained models and vendored libraries. MaleCNS attribution is recorded in [its manifest](data/malecns/manifest.json). The earlier female FlyWire extract retains the restrictions documented in [its source ledger](docs/CONNECTOME.md). GAM is AGPL-3.0-or-later; Universal Weave is Unlicense; Three.js is MIT.
