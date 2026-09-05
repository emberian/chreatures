# Chreatures

**A world to grow into.** Connectome-rooted artificial life, physical bodies, personal memories, and people on the other side of the glass.

Chreatures is an ambitious work in progress inspired by *Creatures*: organisms whose surroundings, physiology, learning, and encounters have consequences for one another. We want lives that become different through experience, and instruments that let us investigate why.

![Articulated residents in the live MaleCNS habitat](docs/assets/articulated-garden.png)

*The articulated garden, running real physics, full MaleCNS neural state, and chemical transport. Six-legged bodies explore an elevated walk, sheltered passages and movable objects.*

## What exists today

The live 3D loop couples MuJoCo bodies to **165,122 traced MaleCNS neurons**, **25,563,197 directed connections**, and **124,025,046 measured synapses**. Each resident has private neural activity, adaptation and support state, contextual/episodic memory, an online predictive model, and an adaptive actor–critic motor interface.

The garden contains ramps, an elevated walk, an underpass, rolling and stackable objects, edible resources, a seesaw and a resonant pendulum. Objects have independently specified geometry, material and sensory properties. Residents see an occluded retinal field from their bodies, sample scent and contact, and can move, look, grip and signal. A human can physically hold and release objects, offer resources, place lights and make sounds. The browser renders the authoritative simulation state.

The newer live world runs six-legged bodies with twelve physical hinges, conservative 3D chemical transport around solid geometry, and a 351-channel sensory interface with 384 anatomical population readouts. Its retina rotates with the complete body orientation, including climbing and inversion. Batched developmental worlds run on AMD GPUs. Older saved residents retain their original bodies and interfaces; see the [cycle log](docs/CYCLE_LOG.md).

**This is a synthetic species experiment, not a recovered fly.** Anatomy is measured; rate dynamics, physiological laws, body mechanics and sensory/motor mappings contain explicit modeling assumptions. Personal learning mechanisms are running, but durable individuality, useful learned manipulation and reciprocal social skills remain research targets.

## GAM × Universal Weave

Two projects are central to where this is going:

- **[SauersML/gam](https://github.com/SauersML/gam)** — Rust-backed statistical modeling for fitted physiological response laws, nonlinear effects of state and upbringing, event histories, representation geometry and intervention analysis. The native integration now fits real 3D developmental telemetry with entire worlds held out, compares against persistence baselines, and saves/reloads immutable model artifacts. The broader mechanism and geometry work is ahead.
- **[transkatgirl/universal-weave](https://github.com/transkatgirl/universal-weave)** — branching histories and evidence records. Our Rust adapter imports real journal events and multi-parent comparisons into a native independent weave, preserving event identity and artifact ancestry through serialization. The [3D observatory](docs/OBSERVATORY.md) keeps adult snapshots, research cohorts, fits and claims distinct from residents’ own memories.

Both native libraries have been executed, not merely named in an architecture diagram. See [the integration contracts and receipts](docs/LIBRARIES.md). Their warnings and limitations are retained with their results.

## Try the compact reference

The smallest runnable version needs no GPU or model service:

```sh
git clone https://github.com/emberian/chreatures.git
cd chreatures
uv sync --extra dev
uv run chreatures
```

Open **http://127.0.0.1:8765**. Place food, toys and shelter; drag objects; use **A / S / D** to sing, **Space** to pause, and the inspection panel to see local sensory and neural state. Whole-world state saves under `runs/` and resumes on restart.

This compact **2D reference uses a female FlyWire v783 subset of 6,789 neurons**, not MaleCNS. Its attraction/avoidance and association rules are explicit bootstrap mechanisms. It remains useful for reproducible development while the larger world grows. [Data selection and provenance](docs/CONNECTOME.md).

## Run the full 3D habitat

The full graph is downloaded separately; large arrays and checkpoints do not belong in Git. The neural service needs PyTorch with a working accelerator backend, while the habitat/browser can run on another machine.

1. Follow [MaleCNS acquisition](docs/MALECNS.md), the [persistent neural service instructions](docs/REMOTE_BRAIN.md), and [rich retinal port setup](docs/NEURAL_PORTS.md).
2. Reach that service over localhost or an SSH forward, then run:

```sh
uv run chreatures3d --port 8768 --brain-url http://127.0.0.1:18767 \
  --checkpoint runs/articulated-garden.json
```

Open **http://127.0.0.1:8768**. The example assumes the rich neural service is forwarded to local port 18767. A 3D checkpoint contains physics, personal cognitive state and the checksum of a server-side neural snapshot. Preserve both parts. Restore checks anatomy and sensory-interface identity; a failed distributed step pauses instead of blindly advancing again.

Actual circuit workloads have run on an AMD RX 6750 XT and Radeon 890M. [GPU results](docs/GPU_NURSERY.md), [developmental training](docs/DEVELOPMENT.md), [3D direction](docs/THREE_DIMENSIONS.md), [articulated body](docs/ARTICULATED.md), [chemical fields](docs/FIELDS.md), [retinal ports](docs/NEURAL_PORTS.md).

The next learning stage uses a shared predictive actor–critic with private recurrent state, articulated environments in persistent worker processes, and the full neural graph on every physical tick. [Learning protocol](docs/LEARNING.md) and [measured sparse-loop optimization](docs/FAST_CIRCUIT.md). Training is running; useful learned behavior must still earn its claim in held-out environments and neural controls.

## Building wide and deep

We are building with parallel agents and human direction. Working commits are intentionally frequent. The priorities are physical possibilities, consequential personal learning, credible biological contribution and a world worth spending time in.

Python currently connects the scientific ecosystem; MuJoCo and PyTorch do the heavy numerical work. Rust is already present through GAM and the Weave adapter, and PyO3 is an option where profiling identifies a useful native boundary. We choose implementation languages by the work they improve.

Useful contributions include better embodied learning, controllable bodies, combinable physical environments, grounded perception, memory mechanisms and experiments that distinguish the proposed explanation from a simpler one. An LLM may become a bounded perceptual or semantic organ; it should not silently supply all behavior while the rest of the organism is decorative.

## Attribution and reuse

- **MaleCNS v1.0:** [Janelia release](https://male-cns.janelia.org/download/), with source hashes, filtering and license attribution in [the local manifest](data/malecns/manifest.json).
- **FlyWire / published brain model:** [the source ledger](docs/CONNECTOME.md). The compact extracted data retains the conservative **CC BY-NC 4.0** public-release restriction documented there; it is not relicensed by this repository.
- **GAM:** AGPL-3.0-or-later. **Universal Weave:** Unlicense. **Three.js:** MIT, with its license beside the vendored renderer.

Scientific source data, upstream libraries and original project code have distinct licensing scopes. Keep their notices and provenance with reused artifacts.
