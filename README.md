# Chreatures

**A world to grow into.** An open artificial-life experiment in inherited bodies, personal histories, measured fly wiring, and ecologies that change their own habitat.

Inspired by *Creatures* and the changing societies and environments of *Children of Time*, we are building organisms whose sensations, chemistry, movement, memories and surroundings have consequences for one another. The goal is a world worth inhabiting, with instruments for discovering how its inhabitants develop.

[Public field guide and observatory](https://emberian.github.io/chreatures/) · [Source and data notices](NOTICE.md) · [Development log](docs/CYCLE_LOG.md)

## The current build

The **ecological specialization wave** adds connected finite material stores,
physical outlets, inherited metabolic acclimation, and private recall of executed
motor sequences. Construction can obstruct the material routes. Enzyme changes
have inherited time constants, expression budgets and ATP costs; an offspring
inherits response rules rather than an adult's current expression. Native v7
residents compare four local plans and up to four remembered action sequences
through the recurrent predictor, then act for one physical tick and reconsider.
The new long research runs are queued while development is quiescent at the
user’s request; these mechanisms are not yet evidence of ecological competence. [Current wave](docs/development/ECOLOGICAL_SPECIALIZATION_WAVE.md) · [Resume handoff](docs/development/ECOLOGICAL_SPECIALIZATION_HANDOFF.md).

The preceding reciprocal ecology wave completed **160 research lives** and
327,680 physical transitions. Its recording remains the default observatory
until the new coupled world produces its own evidence. Its longer private
continuation exposed a material lifecycle fault: a loose packet escaped the
finite terrain and kept falling. The new regional accounting transfers a
packet's complete contents into a regional store and retires its physical body
at a declared exit face. The old failed life remains preserved under its frozen
engine. [Previous wave](docs/development/RECIPROCAL_ECOLOGY_WAVE.md).

The current **v4 organism and population wave** exposes twelve explicit actions—thrust, yaw, gaze pitch, posture, grip, three signal bands, eating, release, secretion and allocation—against twelve measured physical channels spanning movement, energy, digestion, fatigue, neural support, structure, development, gland and brood stores, reproductive maturity and exchange load. Native cohort execution keeps learning, memory and recurrent state private to each life. Immutable candidate genomes can inherit full-MaleCNS interface gains and completed GAM law fits without inheriting that private state.

Regional grammars generate connected physical habitats with variable elevation, cavities, ramps, finite resources and growing material. Clonal births debit a parent's actual brood stores and commit a new body with fresh private state. A native quality-diversity search evaluates genome–environment pairs and retains a bounded archive of varied candidates, including terminal failure records. The first campaign has closed **80 candidate lives across ten environments** on hbox: 16 completed their allotted run, while two shared engine faults ended the other 64. Twelve evaluations received archive admission; nine members remain after subsequent replacement. The project has executed separate 32-resident training populations on hbox and persvati and eight-founder interactive worlds on M2. The [population observatory](https://emberian.github.io/chreatures/population.html) publishes actual genomes, environments and recorded lives. Archive retention does not establish evolutionary improvement or ecological adaptation.

The earlier **Living Reef** supplied the constructed-world substrate: articulated bodies, growing colonies, terraces, ramps, underpasses, a coupled gate, acoustic mechanisms and movable materials. The current regional family extends that substrate; the ten campaign environments contain 15–35 platforms and 22–52 connecting structures each. Colonies build real collision geometry, changing light, contact, passage and chemical transport. A native solar cycle moves illumination through the landscape and supplies energy to phototrophic chemistry.

Residents acquire finite material through mouth contact, digest it into usable reserves, spend energy on activity, and return material through physical deposits. Colonies can release accumulated reserves into consumable packets. These mechanisms share conserved synthetic chemistry. Their combination provides an ecological substrate; a self-sustaining food web, reproduction and evolved social organization remain goals.

Some inherited bodies also capture light through a dorsal surface. Its area is bounded by the physical thorax; orientation and occlusion change the available photons. The same carbon-fixation chemistry converts them subject to finite substrates and enzyme activity. Absorptivity varies continuously down to zero, while all bodies retain feeding and movement. [Mobile mixotrophy](docs/MOBILE_PHOTOTROPHY.md).

The current neural substrate is the **MaleCNS v1.0 brain and ventral nerve cord**: **165,122 traced neurons**, **25,563,197 directed edges** and **124,025,046 synapses** represented by those edges. Anatomical wiring constrains the recurrent network. The rate dynamics, chemical rules, bodies and sensory/motor interfaces include explicit engineering assumptions. This is a synthetic species project, not a recovered fly.

A resident's current control loop combines:

- **Body-bound vision:** 1,024 native collision rays, divided between an 8×32 peripheral field and a 24×32 central field. Each supplies RGB and proximity. Bodies, constructed surfaces and movable objects can occlude them.
- **Measured recurrence:** 351 sensory channels enter the full connectome; 384 named population readouts reach the resident's goal selector. These are population summaries, not recordings of 384 individual neurons.
- **Spatial perception and working memory:** a native convolutional visual front, body-state encoder and persistent private GRU process the 4,459-column current observation at each 50 ms physical tick.
- **Experienced goals:** each individual retains a private reservoir of four-frame sensory encounters. A learned manager selects among those memories; the motor controller attempts to approach the selected sensory state. The current Rust controller also learns private, physiology-dependent goal preferences from the actual bodily return of completed attempts.
- **Personal consequence learning:** inherited GAM predictions and private bounded residual learning estimate movement, energy-cost and fatigue consequences of motor proposals. Actual delivered actions and their subsequent physical outcomes supply the updates.
- **Action-conditioned forecasts:** the fitted three-member recurrent ensemble predicts sensory and physiological changes across up to eight twelve-axis actions. The native planner compares eight-tick local hypotheses and recalled sequences over their actual stored lengths, delivers one physical tick, and replans from actual sensations. Its contribution to selection is bounded; predicted states never enter experienced memory. Older published lives retain their pinned one-step ensemble.
- **Private sequence memory:** observed attempts, attainment and context build sparse transitions between remembered encounters. Lifetime and recent memory share a bounded store, so old residents can continue acquiring new experience. Past succession can bias a proposal, but does not establish present reachability.

Eating is an explicit current action. Physical mouth contact, available material and digestive chemistry determine its consequences. Remembered goals are previously experienced states, not guarantees of present reachability. The first v4 cold inheritance conservatively initializes new action heads and extends the sensory interface; that initialization does not supply competence with the new organs. Useful navigation, durable learned habits and reciprocal interaction are still being developed.

The inherited **Torch v5 policy** trains pathways from all twelve physiology channels into recurrent state and from the policy into the four new organ actions. Achieved-goal encoding stays fixed. Its completed hbox lineage reached 160 PPO updates and 655,360 resident transitions. The native-v7 export combines those weights with the fitted recurrent predictor and private acquired action sequences. The preceding update-20 recording remains available as evidence of that earlier deployment; initialized exploration and nonzero actuator weights alone do not establish competence. [Organ implementation](docs/development/POPULATION_V5_NATIVE.md) · [Current export identity](data/training/rich-recurrent-v3/fit-export-receipt.json).

## GAM × Universal Weave

[**SauersML/gam**](https://github.com/SauersML/gam) is part of the mechanism, not just a plotting dependency. Native GAM fits compress experienced nonlinear body responses into small, immutable consequence models. Rust evaluates these models while the resident compares motor proposals with the bodily component of a remembered sensory goal. Each resident learns its own bounded corrections from its own experience; shared inherited predictions remain unchanged. Out-of-domain candidates retain the underlying actor's support without receiving a GAM refinement. This is an explicitly engineered control layer, not a happiness measure or proof of causal understanding. [Implementation and fitted data](docs/GAM_MECHANISMS.md).

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
replays **360 frames from eight residents**, covering model seconds 79.05–200.65.
It records direct retinal input, neural population readouts, private-memory
summaries, forecasts and physical consequences. This is an explicit research
copy of a coherent tick-118 checkpoint under the fixed engine. Two independent
restores of its later tick-4095 checkpoint produced byte-identical world and
neural snapshots after one step. [Recording, branch and restoration scope](data/development/reciprocal-v6-research-continuation.receipt.json).

The longer private continuation later paused at tick 9,170: a free material
packet escaped the finite terrain and eventually exceeded the field coordinate
sanity bound. Its coherent tick-9,148 checkpoint is retained. Region exit and
material accounting remain a physical lifecycle gap; the recording and completed
campaigns precede this failure. [Longer-run stop](data/development/reciprocal-v6-long-continuation-stop.receipt.json).

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

Acquire the [MaleCNS graph](docs/MALECNS.md), build the current [retinal-v2 port bundle](docs/NEURAL_PORTS.md), and start a dedicated [AMD neural service](docs/REMOTE_BRAIN.md) or [Apple Metal service](docs/METAL_BRAIN.md). Reserve 32 neural slots for an ecology world so its founders leave room for offspring. New worlds require the **same graph and port identities** as the resident artifact. Existing frozen lives keep their loaded engine and service.

Current worlds require an ecological-v7 native controller, matching v3 candidate genomes and a birth manifest referencing their compiled neural phenotypes. The body interface remains v4. The **[previous v6 controller and fit receipts](https://github.com/emberian/chreatures/releases/tag/reciprocal-v6-research-20260906)** remain an archived release for that engine. Export the current v5 policy weights with a recurrent-v3 predictor using:

```sh
python scripts/export_developmental_resident.py --help
```

Follow the [population birth export guide](docs/development/POPULATION_BIRTH.md) to materialize a pinned campaign world and its heterogeneous founders. Environment generation by itself does not create the required candidate genomes or neural artifacts. With those outputs and a dedicated empty neural service, a new world uses:

```sh
uv run chreatures --port 8790 \
  --brain-url http://127.0.0.1:18790 \
  --body articulated --ecology diffusion --physics-backend vectorized \
  --resident-artifact /path/to/developmental-resident-population-v7.npz \
  --population-birth /path/to/export/resident-birth.json \
  --habitat /path/to/export/habitat.json \
  --biosphere /path/to/export/biosphere.json \
  --checkpoint runs/new-regional-life.json
```

Use a fresh checkpoint path and a dedicated empty neural service for a new birth. The example ports are placeholders; choose unused local ports.

The local interface lets people manipulate physical objects, offer finite resources, and make light, sound and gesture stimuli. Its inspector shows actual retinal inputs, population readouts, motor actions, remembered goals and physical state. Public GitHub Pages contain recordings, never an undisclosed connection to a private running world.

Whole-world checkpoints preserve neural and physical state, private memories and learning, RNG, pending actions, chemical pools, constructed topology and the solar clock. Restore checks the pinned source, native binaries and runtime before touching the remote neural state. Deploy lives from immutable source directories. An ambiguous distributed mutation pauses the world. Current development deliberately breaks obsolete interfaces and checkpoint formats; old engines belong in Git history rather than parallel compatibility paths.

## Working together

The project is developed through human direction and parallel coding agents. We build substantial coupled capabilities, then examine their behavior in joined worlds. Body competence, memory, development, social interaction and ecology advance together.

The initial 2D habitat used a **female FlyWire v783 subset of 6,789 neurons**. It was not MaleCNS. That engine lives in the [compact archive](https://github.com/emberian/chreatures/tree/archive/compact-2d-flywire-v1). Further directions include [inherited bodies](docs/BODY_INHERITANCE.md), [diversifying neural blueprints](docs/CIRCUIT_BLUEPRINT.md), [constructed ecologies](docs/ECOLOGICAL_COMMONS.md) and [social organization](docs/SOCIAL_ECOLOGIES.md).

Original Chreatures code is **AGPL-3.0-or-later**. [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) distinguish original code from separately licensed scientific data, pretrained models and vendored libraries. MaleCNS attribution is recorded in [its manifest](data/malecns/manifest.json). The earlier female FlyWire extract retains the restrictions documented in [its source ledger](docs/CONNECTOME.md). GAM is AGPL-3.0-or-later; Universal Weave is Unlicense; Three.js is MIT.
