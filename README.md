# Chreatures

**A world to grow into.** An open artificial-life experiment in inherited bodies, personal histories, measured fly wiring, and ecologies that change their own habitat.

Inspired by *Creatures* and the changing societies and environments of *Children of Time*, we are building organisms whose sensations, chemistry, movement, memories and surroundings have consequences for one another. The goal is a world worth inhabiting, with instruments for discovering how its inhabitants develop.

[Public field guide and observatory](https://emberian.github.io/chreatures/) · [Source and data notices](NOTICE.md) · [Development log](docs/CYCLE_LOG.md)

## The current build

The current **v4 organism and population wave** exposes twelve explicit actions—thrust, yaw, gaze pitch, posture, grip, three signal bands, eating, release, secretion and allocation—against twelve measured physical channels spanning movement, energy, digestion, fatigue, neural support, structure, development, gland and brood stores, reproductive maturity and exchange load. Native cohort execution keeps learning, memory and recurrent state private to each life. Immutable candidate genomes can inherit full-MaleCNS interface gains and completed GAM law fits without inheriting that private state.

Regional grammars generate connected physical habitats with variable elevation, cavities, ramps, finite resources and growing material. Clonal births debit a parent's actual brood stores and commit a new body with fresh private state. A native quality-diversity search evaluates genome–environment pairs and retains a bounded archive of varied candidates, including terminal failure records. The first campaign launched **80 candidate lives across ten environments** on hbox. Its first 32-life batch stopped at tick 9,967 on a shared engine error; 48 assignments remain in progress. A separate 32-resident training population runs on persvati, and eight founders inhabit an interactive M2 world. The [population observatory](https://emberian.github.io/chreatures/population.html) publishes actual genomes, environments and recorded lives. Evolutionary improvement and ecological adaptation remain unestablished.

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
- **Action-conditioned forecasts:** an inherited three-member neural ensemble predicts the next sensory change under candidate actions. Its contribution to goal-directed action selection is bounded; predicted states never enter the resident’s experienced-memory reservoir.

Eating is an explicit current action. Physical mouth contact, available material and digestive chemistry determine its consequences. Remembered goals are previously experienced states, not guarantees of present reachability. The first v4 cold inheritance conservatively initializes new action heads and extends the sensory interface; that initialization does not supply competence with the new organs. Useful navigation, durable learned habits and reciprocal interaction are still being developed.

The current **controller v5** adds trainable pathways from all twelve physiology channels into recurrent state and from the policy into the four new organ actions. Achieved-goal encoding stays fixed. This repairs a limitation of the first frozen v4 inheritance, whose new sensory columns and actuator weights could not learn in the population training configuration. The public regional recording remains that earlier, explicitly pinned v4 life. [Current organ implementation](docs/development/POPULATION_V5_NATIVE.md).

## GAM × Universal Weave

[**SauersML/gam**](https://github.com/SauersML/gam) is part of the mechanism, not just a plotting dependency. Native GAM fits compress experienced nonlinear body responses into small, immutable consequence models. Rust evaluates these models while the resident compares motor proposals with the bodily component of a remembered sensory goal. Each resident learns its own bounded corrections from its own experience; shared inherited predictions remain unchanged. Out-of-domain candidates retain the underlying actor's support without receiving a GAM refinement. This is an explicitly engineered control layer, not a happiness measure or proof of causal understanding. [Implementation and fitted data](docs/GAM_MECHANISMS.md).

[**transkatgirl/universal-weave**](https://github.com/transkatgirl/universal-weave) connects recorded development, model artifacts, snapshots, experiments and competing explanations. Its native adapter supplies stable event identities, multi-parent evidence records and deterministic serialization. The scientific archive is separate from the incomplete, private memory available to an organism. [Native integration](docs/LIBRARIES.md).

Both upstream libraries have executed against actual project data. Their artifacts preserve sources, versions and limits.

## What has run

The current regional observatory replays **240 actual frames from eight v4 residents**, covering model seconds 51.65–118.50. The source recording contains delivered actions, body measurements, real geometry and private learning diagnostics. Its simulation ran on the M2 with the full MaleCNS graph. This is a recorded episode, not a live connection to the private world, and it does not establish successful feeding, reproduction or social learning. [Watch the regional world](https://emberian.github.io/chreatures/living.html?recording=regional-wave).

The first population batch recorded **318,944 resident transitions** before an insufficient-ATP work payment stopped the cohort. Its last complete coupled checkpoint is tick 9,600. Native search retains the 32 affected lives as engine failures with no archive quality; these are not 32 independent organism failures. The actual [Universal Weave evidence](integrations/artifacts/population-wave-v1) contains 287 nodes and 601 edges linking founders, environments, evaluations, checkpoints and the shared failure.

The completed rich developmental lineage ran **491,520 resident transitions and 160 additional PPO updates** in four six-resident worlds on an AMD RX 6750 XT, at **224.76 resident transitions per wall second**. It inherited 40 earlier updates. Ninety of 96 resident episodes included mouth-material contact, and the run transferred 28.35 units of conserved material into bodies. Every episode still lost energy. These are actual physical outcomes, without a matched baseline establishing improvement. [Full-transition analysis](docs/RICH_DEVELOPMENT_ANALYSIS.md).

The **196,608-transition rich play corpus** includes raw retinal observations, full-circuit readouts, delivered actions and physical outcomes. Native GAM fits on that corpus supply the current consequence bank. A separate learned sensory ensemble supplies the installed one-step forecasts; longer action-suffix predictors have also been fitted and remain research artifacts. [Fitted mechanisms](docs/GAM_MECHANISMS.md).

The [earlier predictive courtyard recording](https://emberian.github.io/chreatures/living.html) preserves 240 actual frames of body-bound retinal inputs, remembered goals, forecasts, GAM updates and delivered controls. That life later paused on a physical-source error during an incomplete tick; its last complete checkpoint remains preserved. The published segment precedes the failure. A previous saved reef life ended with depleted reserves.

The current regional campaign uses the native resident controller, including private consequence and goal learning, and records actual actions and physical consequences. The earlier nursery families supplied courtyards, tiered shelves and braided passages with disjoint layout seeds. Superseded Python controllers and training entry points have been removed; their research records retain Git references. [Population campaign](docs/development/POPULATION_CAMPAIGN.md) · [Earlier nursery families](docs/NURSERY_FAMILIES.md).

## Build and run

The full graph and training checkpoints are acquired separately; bulk arrays stay outside Git. Python hosts the application boundary and Torch/ROCm research training. Recurring vision, resident cognition, motor kernels, chemical transport, growth bookkeeping and model inference use Rust or the native physics engine.

```sh
uv sync --extra dev
uv run python native/world-kernels/build_extension.py
uv run python native/cognitive-core/build_extension.py
```

Acquire the [MaleCNS graph](docs/MALECNS.md), build the current [retinal-v2 port bundle](docs/NEURAL_PORTS.md), and start a dedicated [AMD neural service](docs/REMOTE_BRAIN.md) or [Apple Metal service](docs/METAL_BRAIN.md) with sufficient cohort capacity. New worlds require the **same graph and port identities** as the resident artifact. Existing frozen lives keep their loaded engine and service.

Current worlds require a population-v5 native controller, matching candidate genomes and a birth manifest referencing their compiled neural phenotypes. The body interface remains v4. The tracked `developmental-resident-rich-grandchild-update160-v3.npz` is an ancestor artifact; it cannot directly launch the current runtime. Export a current training checkpoint with:

```sh
python scripts/export_developmental_resident.py --help
```

Follow the [population birth export guide](docs/development/POPULATION_BIRTH.md) to materialize a pinned campaign world and its heterogeneous founders. Environment generation by itself does not create the required candidate genomes or neural artifacts. With those outputs and a dedicated empty neural service, a new world uses:

```sh
uv run chreatures --port 8790 \
  --brain-url http://127.0.0.1:18790 \
  --body articulated --ecology diffusion --physics-backend vectorized \
  --resident-artifact /path/to/developmental-resident-population-v5.npz \
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
