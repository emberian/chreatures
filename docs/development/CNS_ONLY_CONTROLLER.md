# Mandatory CNS route: current implementation contract

Root approved this breaking architecture on 2026-09-07. This is a specification
and implementation wave, not evidence of pretrained competence. The paused v8
world at tick 10,006 remains an authenticated historical baseline. Its direct
sensory controller, predictors and personal outcome updates do not meet this
contract. No old checkpoint is reinterpreted as a new life.

## One information boundary

`optic RGB + body-local senses -> learned afferent currents -> full MaleCNS
recurrence -> learned CNS latent -> private recurrent control -> motor command`.

The neural service accepts float32 **[B,5356]** logically. Its native process
wire is channel-major **[5356,capacity]**. The first 5,313 entries are 1,771
bilateral atlas sites times RGB, in `site_side_hex` order, RGB fastest. The last
43 entries are the old canonical nonretinal 31 entries (indices 320:351 in
retinal-v2) followed by the twelve `organism_interface.PHYSIOLOGY_NAMES`, in that
order. This retains measured body-local quantities as afferent inputs. There is
no optical proximity component, direct foveal image or raw body controller path.
The geometric interpretation of eye rays remains an explicit model, not measured
receptive-field geometry.

The service returns **Z[B,512]**, computed inside the CNS service, and separate
observer-only diagnostics. The controller accepts only Z, an efference copy of
its command **[B,12]**, tick/reset/ack scheduling metadata and private RNG. Body
IDs route state; they are not numerical features. No raw observations, actual
physiology, reward, outcome labels, event archive, world positions or capability
flags enter controller inference, memory or private online learning. Raw values
may be supervised targets or offline RL rewards. Those training targets are not
additional deployed inputs. Command receipts contain command/tick/ack, not the
body's measured motion or metabolic consequence.

## Anatomical injection and learned adapters

The immutable graph is the pinned 165,122-neuron, 25,563,197-edge MaleCNS graph.
Its signed incoming-count-normalized CSR is retained. Every neuron still runs.

The optic atlas contains 4,107 receptor graph rows and 4,669 receptor/site entries.
For each supported receptor, normalize its measured outgoing synapse counts into
same-side annotated sites to sum to one. Mix the site's RGB using a learned
softmax over three spectral logits for its receptor type (ten sorted actual type
names). For the resulting scalar x, the injected current is

`sigmoid(optic.bias[type] + softplus(optic.gain_raw[type]) * (2*x - 1))`.

The 171 unsupported receptors receive exactly zero injected current, including
zero tonic current. No nearest-coordinate or cross-side fallback is allowed.
Supported receptors have a learned tonic response through the sigmoid. This is
a modeled transduction rule, not fitted fly spectral physiology.

Body targets are exactly the **11,233** ascending graph-order rows annotated
`cb_sensory` or `vnc_sensory`. Uncertain `*_tbc`, intrinsic and ascending projection
classes are excluded. The 43 inputs are standardized by immutable fitted mean
and positive scale, clipped to [-8,8], then mapped by **43 -> 128 tanh -> 11233
sigmoid**. These learned mixed-modality afferents are an engineered body adapter;
we do not claim a measured correspondence between each artificial body variable
and a particular fly sensory neuron. No body-role rules choose actions.

All 4,107 receptor rows plus the 11,233 body rows are masked from the policy
readout. Consequently an adapter cannot pass directly from an injected neuron
to the decoder: sensory influence must cross an actual retained graph edge.
Unsupported receptors remain masked. Observer views may display every neuron.

## CNS dynamics and readout

There are 11,752 type labels in the exact atlas order. Five trainable raw arrays
are tied by that index. Their effective values are:

```
bias = .5*tanh(bias_raw)
tau = .025 + .475*sigmoid(tau_raw)
source,target,excitability = .5 + sigmoid(corresponding_raw)
```

For each physical dt, execute two Jacobi substeps, using the previous substep's
rates for all recurrent sources:

```
alpha = min(1, dt/(2*tau))
rec = target * (W_fixed @ (source * rate))
q = relu(tanh(bias + excitability*(drive + .92*rec) - .10*adaptation))
rate_next = rate + alpha*(q*support - rate)
```

After the second substep, update adaptation by `dt/5*(rate-adaptation)` and support
by `dt*(.024*(1-support)-.003*rate)`, clipped to [.65,1]. Initial effective defaults
are bias .005, tau .16, and gains 1; these are explicit untrained model defaults.
The receptor transmitter signs remain the imported modeled signs. Strong
inhibitory receptor drive can silence rectified downstream units; learned tonic
biases and time constants must address this through task optimization. An image
at the input alone does not demonstrate usable downstream computation.

The learned dense readout is `Z=tanh(W_readout*(rate*mask)+bias_readout)` with
weights **[512,165122]**, bias **[512]**, no raw skip and no fixed population means.
This is 84,542,464 weights, approximately 338 MB float32. Parameter/gradient/two
Adam moment storage is approximately 1.35 GB before other model and activation
memory. Metal uses MPS GEMM over neuron-major resident tiles; Torch uses GPU GEMM.
The Rust cognitive module offers a matching Accelerate/BLAS reference path,
not an all-neuron HTTP transfer in the deployed loop.

## Downstream temporal control

The core GRU consumes `concat(Z512,previous_command12)` (524), has hidden 256,
and uses PyTorch GRUCell r,z,n gate order. Weights are [768,524] and [768,256],
with two [768] biases. A shared goal encoder maps `concat(Z512,h256)` to 128
through tanh, then L2 normalizes with epsilon 1e-8. Memories contain CNS-derived
keys, latent/motor sequences and generation/tick metadata. Context, goal
selection, acquired suffixes, online progress and termination use these values.

The canonical sequence/predictor context is
`concat(Z512,h256,current_goal128,previous_command12)` = **908**. Three predictors
initialize hidden 256 from that context and recurrently consume proposed command
12 for eight ticks, predicting CNS latent deltas 512. Predictor hidden is not
interchangeable with core hidden: predicting a future goal requires rolling the
core on predicted latents or explicitly retaining current core context. The
native sequence owner and Torch owner freeze their detailed head packing
together. The old raw-668 sequence state and raw-1560 predictor are superseded.

## Training and private plasticity

Current `fast_circuit.py` and `tiled_circuit.py` are inference implementations:
they disable autograd and update tensors in place. Training needs a functional
version of these same equations. Use frozen W and a retained transpose CSR so
the backward operation is `W.T @ grad`, without edge gradients or dense N*N
allocation. Train afferents, type dynamics and readout jointly with the temporal
controller. Start with B8/T32 truncated sequences, checkpoint recurrence groups
of four ticks, preserve detached numerical state at chunk boundaries, and measure
actual ROCm memory/throughput. hbox currently has Torch 2.9.1+rocm6.3. General
PyTorch documentation is not an executed ROCm backward measurement.

Supervised reconstruction/forecast heads may target optical motion and body
variables to prevent a useless latent; those heads and targets do not introduce
a controller skip. RL rewards can use actual outcomes in the offline optimizer.
Online private updates may use only CNS-derived prediction error, achieved goal
alignment and actual command continuity. Existing raw-physiology GAM updates
must be removed, not hidden behind a new actor API. Old raw trajectories do not
contain the new atlas samples or automatically authenticate new recurrent state.

Neural Assemblies supplies useful principles: bounded coactivity plasticity,
homeostasis and recurrence, not a ready-to-import MaleCNS controller. Its local
`core/brain.py` explicitly documents a recurrence-disabled fast path and failures
when enabling it; `compute/plasticity.py` contains multiplicative updates over
dense submatrices. Do not replace real graph edges with random assembly areas or
copy an unbounded multiplicative rule. A later optional private residual should
modify a bounded, authenticated set of existing edges and derive modulation
from CNS activity. The existing actual KC/MBON/DAN bridge is the natural starting
point; its synthetic-modulator option is incompatible with this boundary.
Private plasticity, if enabled, needs its eligibility, bounded efficacy and
modulator state in the same coherent checkpoint. It is not silently enabled by
the first artifact.

## Artifacts, protocol and ownership

`chreatures/cns_adapter_contract.py` is the authoritative tensor order and binary
writer. `chreatures-cns-service-v1` is service-only. Its file begins with eight
bytes `CHCNS1\0\0`, little-endian u32 JSON length, canonical UTF-8 metadata, then
all `ARRAY_SPECS` tensors as contiguous little-endian bytes, with no padding or
trailing data. Metadata binds graph, atlas, readout mask, every tensor checksum,
dimensions, parameter order, explicit training status and internal adapter hash.
The internal adapter hash covers only format, graph/atlas/mask hashes, dimensions,
parameter order and tensor hashes. This all-integer/string projection has the
same canonical JSON representation in Rust and Python; floating provenance and
training labels remain authenticated by the whole-file SHA.
The resident artifact binds both service file SHA and internal adapter SHA.

Native process step is `{dt,active_mask,sensory,selected_neuron_indices?}`;
sensory is flattened channel-major [5356,capacity]. Reply `latent` is flattened
[512,capacity]; optional `selected_rates` is an observer channel. Existing ordered
host request receipts remain mandatory. No old phenotype gain stack is applied:
the new artifact's parameters are authoritative. Snapshots use a new incompatible
`CNSSTATE1` envelope with exact artifact/graph/atlas/mask identities, per-slot
time, all rate/adaptation/support values, active life identities and private
controller/memory/RNG state in the enclosing world checkpoint.

Owners: root host/runtime and final integration; cognitive_integration shared
adapter contract, native reference and CNS service export; its Metal deputy
native/metal-brain serving; learning_pipeline functional Torch model/training;
native_sequence_control CNS-only developmental/core; population_launch resident
export and artifact dependency binding; optic/world owner physical atlas sampling.
One joined real GPU/native/controller scenario follows the integrated build.
It must establish gradients, graph-edge dependence, private restore continuity
and stimulus-to-CNS-to-body behavior. A local movie displays measured neural
activity and actual body motion, not a copied input masquerading as brain output.

## Executed adapter and serving boundary

The initialized service is sealed outside the source tree at
`/Users/ember/paperbin/chreatures/integration/cns-only-v1-export/cns-service-initialized-untrained.bin`
(550,191,061 bytes; file SHA-256
`5c243ee93951ee5a22c1a5a7ec589643198cc0d43487a04e98a2c0f849d3f407`;
internal adapter `46307c8d930851b8a8f2b8314fd380e4b10cc2871fad6a514befa08a1bf8bacf`).
Its actual T4/B2 Torch fixture uses black, white, colored stripe and shifted stripe
stimuli with finite body variation. Native/Accelerate afferent/readout parity gave
maximum absolute errors 1.86e-8 and 1.67e-7 respectively. Altering every injected
afferent rate left the readout exactly unchanged; all 171 unsupported receptors
received exactly zero current. The receipt is `native-adapter-boundary.receipt.json`
in the same integration directory and binds the extension used for that pass.

The full Apple M2 Max Metal/MPS recurrence against that fixture gave maximum
absolute errors 2.10e-7 for latent, 8.94e-8 for rates, 2.79e-9 for adaptation and
5.96e-8 for support. Capacity-two GPU ticks took 421–469 ms. Snapshot inspection
and full restore reproduced captured rates byte-for-byte. GPU command failure
poisons the service before publishing successful state/time; a full-capacity
reset or full restore is required to recover. MPS matrix extents use checked
arithmetic, and snapshot values must satisfy finite physical bounds.

The first optical optimization completed 128 updates / 32,768 resident ticks in
528.12 seconds. It used procedural bars, checkers and occluders; Bad Apple was
excluded and body inputs were fixed zero during this optical phase. Held-out
aggregate loss improved from 0.0875755 to 0.0856562, while current and future optic
prediction losses slightly worsened (0.045188→0.045460 and 0.045173→0.045352).
The identical-input fixture showed stronger stimulus-dependent latent changes.
These are optimization and stimulus-response measurements, with no demonstrated
held-out visual-prediction or behavioral competence. Trained parameters, service
and fixture have separate immutable identities; neither overwrites initialization.
Native parity is an analytical execution boundary, not an actual physical-world
outcome. The world/video integration supplies that separate observation.

The distinct fitted service is sealed in the same integration directory as
`cns-service-fitted-v1.bin` (550,191,319 bytes; file SHA-256
`395120785a56a5f9bfe0f70edf744e1251e6141656f799ff1b5963d3e0217da9`;
internal adapter `cfedd0ed4286b9ea5a2cde36a0b66ee4443dfb7a459cde70851b95cc8230aab4`).
`fitted-relay.receipt.json` binds its verified transfer and the identical-input
fitted Torch fixture. Parameter status is `trained`, which denotes executed
optimization rather than a competence threshold.

## Reproduce the current CNS-only execution

The research release tag `cns-optic-v1-research-20260907` resolves to its exact
source commit. The resident metadata binds that source even though the release
uses compact canonical filenames. The fitted service is unchanged across the
source-only runtime correction:

| Artifact | File SHA-256 | Internal identity |
| --- | --- | --- |
| `cns-service-fitted-v1.bin` | `395120785a56a5f9bfe0f70edf744e1251e6141656f799ff1b5963d3e0217da9` | adapter `cfedd0ed4286b9ea5a2cde36a0b66ee4443dfb7a459cde70851b95cc8230aab4` |
| `cns-resident.npz` | See the release notes | initialized resident bound to the release source and fitted service |
| `sequence-control.npz` | See the release notes | initialized immutable head, version 0 |

Build the native modules from that source and start a dedicated Metal service:

```sh
git checkout cns-optic-v1-research-20260907
uv sync --extra dev
uv run python native/world-kernels/build_extension.py
uv run python native/cognitive-core/build_extension.py
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

The release resident is a deterministic initialized controller bound to the
fitted service. Re-exporting it is reproducible and requires new output paths:

```sh
uv run python scripts/export_developmental_resident.py \
  --cns-service "$RELEASE_DIR/cns-service-fitted-v1.bin" \
  --output /path/to/reproduced/cns-resident.npz \
  --sequence-control-output /path/to/reproduced/sequence-control.npz \
  --seed 20260907
```

Create a physical birth bundle from an authenticated profile and assignment
bank, then start the Habitat against that same service:

```sh
uv run python scripts/export_population_birth.py \
  --profile /path/to/profile.json \
  --assignments /path/to/founder-assignments.json \
  --world-index 0 \
  --resident-artifact "$RESIDENT" \
  --output /path/to/cns-birth

uv run chreatures --port 8790 \
  --brain-url http://127.0.0.1:18790 \
  --body articulated --ecology diffusion --physics-backend vectorized \
  --resident-artifact "$RESIDENT" \
  --population-birth /path/to/cns-birth/resident-birth.json \
  --habitat /path/to/cns-birth/habitat.json \
  --biosphere /path/to/cns-birth/biosphere.json \
  --checkpoint /path/to/new-cns-life.json
```

Use an empty service and new checkpoint/output paths for a fresh life. The
service's optical adapter completed training, while held-out current and future
optic prediction MSE changed from `0.045188` to `0.045460` and `0.045173` to
`0.045352`. The resident core and sequence-control head remain explicitly
initialized and untrained, so these artifacts make no motor, feeding or policy
competence claim.

## Consequential sources

The connectome-constrained fly visual model supports optimizing unknown cellular
parameters while keeping measured topology, and warns against equating wiring
alone with function. It changes this design toward trainable type dynamics and
task objectives, rather than a fixed reservoir plus powerful raw sensory actor.
[Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3)
and [author dynamics implementation](https://github.com/TuragaLab/flyvis/blob/main/flyvis/network/dynamics.py).

Assembly calculus motivates recurrent coactivation and plasticity but assumes
different connectivity/activity rules; it does not validate the current organism.
[Papadimitriou et al., PNAS 2020](https://doi.org/10.1073/pnas.2001893117).
Sparse-dense backward is documented in [PyTorch sparse.mm](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm);
our installed ROCm execution still requires the joined measurement.

Scry discovery was attempted after reading its current schema; the bounded HN
query was admission-limited and then timed out with patient admission. The
primary sources above were read directly. No search result is counted as an
implementation result.
