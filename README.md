# Chreatures

**A world to grow into.** An open artificial-life experiment in inherited bodies, personal histories, measured fly wiring, and ecologies that change their own habitat.

Inspired by *Creatures* and the changing societies and environments of *Children of Time*, we are building organisms whose sensations, chemistry, movement, memories and surroundings have consequences for one another. The goal is a world worth inhabiting, with instruments for discovering how its inhabitants develop.

[Public field guide and observatory](https://emberian.github.io/chreatures/) · [Source and data notices](NOTICE.md) · [Development log](docs/CYCLE_LOG.md)

## The current build

**Living Reef** has six varied articulated bodies and twelve growing colonies in a physical world of terraces, ramps, underpasses, a coupled gate, acoustic mechanisms and movable materials. Colonies build real collision geometry. Changes to that geometry affect light, contact, passage and chemical transport. A native solar cycle moves illumination through the landscape and supplies energy to phototrophic chemistry.

Residents acquire finite material through mouth contact, digest it into usable reserves, spend energy on activity, and return material through physical deposits. Colonies can release accumulated reserves into consumable packets. These mechanisms share conserved synthetic chemistry. Their combination provides an ecological substrate; a self-sustaining food web, reproduction and evolved social organization remain goals.

Some inherited bodies also capture light through a dorsal surface. Its area is bounded by the physical thorax; orientation and occlusion change the available photons. The same carbon-fixation chemistry converts them subject to finite substrates and enzyme activity. Absorptivity varies continuously down to zero, while all bodies retain feeding and movement. [Mobile mixotrophy](docs/MOBILE_PHOTOTROPHY.md).

The current neural substrate is the **MaleCNS v1.0 brain and ventral nerve cord**: **165,122 traced neurons**, **25,563,197 directed edges** and **124,025,046 synapses** represented by those edges. Anatomical wiring constrains the recurrent network. The rate dynamics, chemical rules, bodies and sensory/motor interfaces include explicit engineering assumptions. This is a synthetic species project, not a recovered fly.

A resident's current control loop combines:

- **Body-bound vision:** 1,024 native collision rays, divided between an 8×32 peripheral field and a 24×32 central field. Each supplies RGB and proximity. Bodies, constructed surfaces and movable objects can occlude them.
- **Measured recurrence:** 351 sensory channels enter the full connectome; 384 named population readouts reach the resident's goal selector. These are population summaries, not recordings of 384 individual neurons.
- **Spatial perception and working memory:** a native convolutional visual front, body-state encoder and persistent private GRU process 4,453 sensory and physiological values at each 50 ms physical tick.
- **Experienced goals:** each individual retains a private reservoir of four-frame sensory encounters. A learned manager selects among those memories; the motor controller attempts to approach the selected sensory state.
- **Personal consequence learning:** inherited GAM predictions and private bounded residual learning estimate movement, energy-cost and fatigue consequences of motor proposals. Actual delivered actions and their subsequent physical outcomes supply the updates.

The oral command currently follows an engineered physiological law. Remembered goals are previously experienced states, not guarantees of present reachability. Useful navigation, durable learned habits and reciprocal interaction are still being developed.

## GAM × Universal Weave

[**SauersML/gam**](https://github.com/SauersML/gam) is part of the mechanism, not just a plotting dependency. Native GAM fits compress experienced nonlinear body responses into small, immutable consequence models. Rust evaluates these models while the resident compares motor proposals with the bodily component of a remembered sensory goal. Each resident learns its own bounded corrections from its own experience; shared inherited predictions remain unchanged. Out-of-domain candidates retain the underlying actor's support without receiving a GAM refinement. This is an explicitly engineered control layer, not a happiness measure or proof of causal understanding. [Implementation and fitted data](docs/GAM_MECHANISMS.md).

[**transkatgirl/universal-weave**](https://github.com/transkatgirl/universal-weave) connects recorded development, model artifacts, snapshots, experiments and competing explanations. Its native adapter supplies stable event identities, multi-parent evidence records and deterministic serialization. The scientific archive is separate from the incomplete, private memory available to an organism. [Native integration](docs/LIBRARIES.md).

Both upstream libraries have executed against actual project data. Their artifacts preserve sources, versions and limits.

## What has run

The earlier joined neural/controller experiment completed **245,760 resident transitions** and **160 PPO updates** in four three-resident chemical worlds on an AMD RX 6750 XT. It trained a continuous worker and a slower achieved-goal manager alongside the full MaleCNS circuit. These are execution and training results; they do not establish improved physical skill. [Recorded run](research/sensorimotor_skills/ONLINE_DEVELOPMENT.md).

GAM consequence models were fitted to **192,000 recorded physical transitions**, holding out complete physical worlds. Native interpolation, operating-domain checks and artifact loading have run. The larger rich-world collection has now completed **196,608 transitions** across six bodies in four worlds, including raw retinal observations, full-circuit readouts, delivered actions and physical outcomes. The new private learner, rich visual controller and their complete stochastic snapshots have executed in isolation; learned current-life deployment is the next integration step.

The new reef's physical and chemical runs have exercised growth, exudation, recycling, acoustic contacts and changing solar exposure. The coupled collection exposed and fixed two integration defects: construction invalidating a cached retinal model, and physical emitters crossing the finite chemical grid boundary. Outside-domain emission is now accounted separately rather than placed in an unrelated boundary cell.

Earlier results, including unsuccessful behavioral comparisons, remain in the research records. A good action-likelihood fit is not treated as evidence that a requested goal can be achieved with the body.

## Build and run

The full graph and training checkpoints are acquired separately; bulk arrays stay outside Git. Python hosts the application boundary and Torch/ROCm research training. Recurring vision, resident cognition, motor kernels, chemical transport, growth bookkeeping and model inference use Rust or the native physics engine.

```sh
uv sync --extra dev
uv run python native/world-kernels/build_extension.py
uv run python native/cognitive-core/build_extension.py
```

Acquire the [MaleCNS graph](docs/MALECNS.md), build the current [retinal-v2 port bundle](docs/NEURAL_PORTS.md), and start a dedicated [AMD neural service](docs/REMOTE_BRAIN.md) or [Apple Metal service](docs/METAL_BRAIN.md) with sufficient cohort capacity. New worlds require the **same graph and port identities** as the resident artifact. Existing frozen lives keep their loaded engine and service.

The current resident artifact is exported from a self-contained rich developmental training checkpoint, including the fitted GAM law bank:

```sh
python scripts/export_sensorimotor_worker.py --help
```

Then start a fresh world using that artifact:

```sh
uv run chreatures --port 8777 \
  --brain-url http://127.0.0.1:18777 \
  --resident-artifact /path/to/rich-developmental-resident.npz \
  --habitat data/habitats/living-reef.json \
  --biosphere data/biosphere/living-reef.json \
  --visitor-materials data/visitors/living-reef.json \
  --checkpoint runs/living-reef.json
```

The local interface lets people manipulate physical objects, offer finite resources, and make light, sound and gesture stimuli. Its inspector shows actual retinal inputs, population readouts, motor actions, remembered goals and physical state. Public GitHub Pages contain recordings, never an undisclosed connection to a private running world.

Whole-world checkpoints preserve neural and physical state, private memories and learning, RNG, pending actions, chemical pools, constructed topology and the solar clock. Restore checks the pinned source, native binaries and runtime before touching the remote neural state. Deploy lives from immutable source directories. An ambiguous distributed mutation pauses the world. Current development deliberately breaks obsolete interfaces and checkpoint formats; old engines belong in Git history rather than parallel compatibility paths.

## Working together

The project is developed through human direction and parallel coding agents. We build substantial coupled capabilities, then examine their behavior in joined worlds. Body competence, memory, development, social interaction and ecology advance together.

The initial 2D habitat used a **female FlyWire v783 subset of 6,789 neurons**. It was not MaleCNS. That engine lives in the [compact archive](https://github.com/emberian/chreatures/tree/archive/compact-2d-flywire-v1). Further directions include [inherited bodies](docs/BODY_INHERITANCE.md), [diversifying neural blueprints](docs/CIRCUIT_BLUEPRINT.md), [constructed ecologies](docs/ECOLOGICAL_COMMONS.md) and [social organization](docs/SOCIAL_ECOLOGIES.md).

Original Chreatures code is **AGPL-3.0-or-later**. [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) distinguish original code from separately licensed scientific data, pretrained models and vendored libraries. MaleCNS attribution is recorded in [its manifest](data/malecns/manifest.json). The earlier female FlyWire extract retains the restrictions documented in [its source ledger](docs/CONNECTOME.md). GAM is AGPL-3.0-or-later; Universal Weave is Unlicense; Three.js is MIT.
