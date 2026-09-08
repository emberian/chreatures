# MaleCNS embodiment-driven CNS V4

The current trainable/runtime artifact is CHCNS4. It binds the fixed
NeuroMechFly-derived morphology, BODY807 channel schema, all 11,798 annotated
nonvisual sensory rows, M92 actuator schema, centered motor calibration, and the
quantized graph identity. The V3 material below records the completed synthetic
twelve-hinge research epoch; it is not a loader or compatibility path.

`anatomy.py` produces the immutable anatomical supports for the V3 coupled CNS
wave. It reads the canonical MaleCNS v1.0 CSR and row-aligned neuron metadata;
it does not contact, advance, or mutate a neural service.

The artifact contains the frozen dense structural masks `atlas.body_mask`
`[11233,110]` and `atlas.motor_mask` `[34,815]`, plus exact body, descending,
and motor row indices. `graph.channel` labels every neuron as unknown, fast, DA,
OA, or 5HT. `graph.weight` recomputes every canonical edge from measured counts:
fast signed efficacy is normalized by all incoming counts at its target; each
modulator is positive and normalized by same-family incoming counts. Unknown
transmitter edges remain zero. These labels establish anatomical support, not
receptor sign or kinetics.

All sensory supports first select a declared annotation modality. Side and leg
entry nerve are preserved where available. Chemical identity, body axes,
directions, interoceptive signals, and auditory frequency are learned within the
appropriate support because the annotations do not measure those functions. All
16 frequency bands therefore share the typed auditory support. Interoception,
shade, and fatigue mappings are explicitly marked proxies. MANC motor labels
remain dataset annotations; mapping those labels onto the synthetic fixture's
joints is an engineered assumption rather than a measured muscle homology.

The physical fixture order is read from `site/live/fixtures/garden.xml`: LF, LM,
LH, RF, RM, RH, each hip then knee. BODY110 stores all twelve joint angles in
that order, then all velocities, then all loads. Foot contacts use LF through RH;
fatigue repeats the twelve-joint order. MOTOR34 uses the same joint order, each
positive then negative direction, followed by its ten named ancillary outputs.

Run on the bulk host:

```sh
python research/anatomical_cns/anatomy.py \
  /tank/chreatures/data/malecns/derived \
  /tank/chreatures/data/malecns/v3/anatomical-cns-v3-r2.npz
```

The adjacent `.manifest.json` pins the graph, dimensions, per-channel rules and
counts, fast/modulator edge and synapse counts, artifact size, and checksum.
Loading the NPZ must use `allow_pickle=False`.
The exporter refuses to overwrite either an existing artifact or its manifest.

`anatomical-cns-v3-r2.npz` is the current artifact. The un-suffixed first export
and its retained manifest are historical evidence of the rejected ordering and
must not be loaded: it interleaved joint families and imposed unsupported
functional hash partitions.

## Source and interpretation

- MaleCNS connectome v1.0, adult male *Drosophila* brain and VNC, CC BY 4.0.
- Dataset DOI: `10.1016/j.cell.2026.08.015`.
- Canonical dataset SHA-256:
  `48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.
- Derived metadata SHA-256: `0e6706229b93cbdcab48905504bd3b8f6075534488c65ad2c8f844668036ea9a`.
- Canonical CSR orientation is postsynaptic row with presynaptic `indices`.

The selector design follows the repository's source audit in
`docs/development/ANATOMICAL_MOTOR_BOOTSTRAP.md`. In the inspected metadata,
there are 815 annotated motor neurons, 1,314 descending neurons, 115 auditory
cells (114 sensory plus one annotation-missing cell), and 425 chordotonal cells
(403 VNC sensory plus 22 sensory ascending). The sensory exporter deliberately
excludes the annotation-missing auditory cell because the CNS input contract
requires a typed afferent.

## Torch recurrence and service export

`model.py` is the differentiable full-graph V4 reference used for ROCm training.
It accepts retina `[B,1771,3]`, BODY807, and the context actually delivered
to the CNS `[B,12]`. A 10 ms step performs two 5 ms Jacobi substeps and returns
latent `[B,512]`, MOTOR92, and seven
private `[165122,B]` fields: rate, adaptation, support, release availability, and
the dopamine, octopamine, and serotonin target traces. The body and motor weights
are multiplied by the exported structural masks on every forward pass. Context is
injected only at the 1,314 descending rows, and both latent and motor values are
computed after recurrent activity. Raw observations are training targets only.

The motor decoder operates on actual post-recurrence motor-neuron state:
`z=(rate-reference_rate)/rate_scale`, followed by a signed masked `92×815`
linear map. Outputs 0:84 use tanh and outputs 84:92 use sigmoid. Reference and
scale remain explicit frozen float32 tensors; they are never folded into a large
bias. This preserves below-reference activity and avoids the numerical
cancellation that rejected the earlier V3 centered-decoder candidate.

`export.py` is the sole offline bridge. It reads a sealed CHCNS3 research seed,
the final fly neural atlas and schemas, plus the measured normalized R2 graph.
It rounds that measured graph once to canonical IEEE binary16 bits and writes a
fresh CHCNS4 artifact. Torch, Rust/Metal, and WebGPU decode those same bits to
float32. No V3 runtime loader or private neural state migration remains.

```sh
python -m research.anatomical_cns.export \
  --v3-service /path/to/sealed-v3.bin \
  --anatomy-v3 /tank/chreatures/data/malecns/v3/anatomical-cns-v3-r2.npz \
  --fly-atlas research/fly_embodiment/fly-body-neural-atlas-v1.npz \
  --morphology-schema native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json \
  --sensory-schema research/fly_embodiment/body807-channel-schema.json \
  --actuator-schema research/fly_embodiment/motor92-channel-schema.json \
  --output /tank/chreatures/runs/malecns-v4/seed/initialized-cns-v4.bin
```

The anatomy file must be the corrected current export for the epoch. Its SHA is
bound into the service metadata; a prior selector draft must not be substituted.

## Actual physical bootstrap curriculum

`native/webgpu-probe/collect-anatomical.mjs` runs the epoch-2 MuJoCo/Wasm world
directly. Each fresh process records one 512-transition, three-resident world.
Episodes 0–5 are training worlds and 6–7 are held-out worlds with distinct world,
variation and layout identities. The curriculum covers pose hold, turning,
stopping, reaching and bounded antagonist babbling. Four actual acoustic sources
at 80, 200, 500 and 1250 Hz accompany the first four physical bouts. The sound
reaches the CNS only through BODY110 acoustic afferents.

Each strict NPZ contains actual pre-action `optic_rgb[513,3,5313]` and
`body[513,3,110]`, exact zero bootstrap `delivered_context[512,3,12]`, and the
physically delivered `delivered_motor[512,3,34]`. The target-only members are
`teacher_joint_target[512,3,12]`, `root_pose[513,3,12]` (XYZ plus a 3×3 root
rotation), and `skill_id[512,3]`. Reset and terminal arrays preserve chronology.
No world geometry is retained. Privileged bout time, joint targets and root pose
may select demonstrations or measure results, but never enter the recurrent CNS.
Old abstract 12-axis action records are not context values.

Run every episode in a fresh Node process; MuJoCo's Emscripten factory is not
reinitialized in a long-lived process:

```sh
for episode in 0 1 2 3 4 5 6 7; do
  node native/webgpu-probe/collect-anatomical.mjs \
    --world native/browser-world \
    --output /tank/chreatures/runs/anatomical-cns-v3/physical-source \
    --episode "$episode"
done

python -m research.anatomical_cns.train seal-corpus \
  /tank/chreatures/runs/anatomical-cns-v3/physical-source \
  /tank/chreatures/runs/anatomical-cns-v3/physical-corpus
```

`data.py` authenticates exact members, shapes, bounds, source identities and the
six/two whole-world split. It rejects nonzero bootstrap context, renamed old
actions, gaps in teacher bouts, and any declaration that privileged fields are
model inputs.

## ROCm training

`train.py` loads one current CHCNS3 service and runs the full 165,122-neuron
sparse recurrence. Training updates the structurally masked BODY110 and MOTOR34
interfaces plus recurrent, release and family-modulation gains. Retina, context,
latent readout, baseline and other type kinetics stay pinned for this bootstrap.
Body normalization is fit on the six training worlds only and exported with the
body interface.

The primary demonstration loss fits the physically delivered MOTOR34 command.
Training-only heads also predict the next actual BODY110 and fixed regional
retinal changes, root-pose change, and the teacher joint target from the current
CNS latent and actually delivered teacher motor. These heads force temporal physical information through
the full graph and are omitted from the service artifact. Future senses and pose
are targets only; they are never recurrent inputs. Short actual preceding
history initializes each truncated sequence. Five skills are sampled evenly,
while unsuccessful transitions remain in the corpus.

```sh
/home/ember/kaxsim/.venv7/bin/python -m research.anatomical_cns.train train \
  --corpus /home/ember/chreatures-data/anatomical-cns-v3/physical-corpus \
  --service /home/ember/chreatures-data/anatomical-cns-v3/initialized-cns-v3.bin \
  --run /home/ember/chreatures-data/anatomical-cns-v3/bootstrap-run-01 \
  --device cuda --updates 320 --sequence 4 --burn-in 8 \
  --motor-learning-rate 3e-3 --body-learning-rate 1e-3 \
  --dynamics-learning-rate 3e-4 --head-learning-rate 1e-3
```

Optimizer snapshots are portable CPU tensors and include the complete model,
training-only heads, optimizer, RNG states, recipe and immutable input hashes.
Held-out worlds are evaluated before and after training and never select a
checkpoint. The resulting service is labeled a physical motor bootstrap. A
separate actual learned closed-loop assay is required before claiming useful
autonomous competence; an initialized context organ remains distinct from this
zero-context motor curriculum.

Motor and body interfaces use larger learning rates than the selected type-level
dynamics because their initialized raw scales would otherwise barely move in a
few hundred full-graph updates. Receipts report output saturation and held-out
motor variation by skill. Fixed collection bout order confounds tone with body
state and time, so those offline differences are not evidence of a learned tone
mapping; the headless assay uses a separate counterbalanced intervention.

If the fitted motor output remains nearly constant, diagnose the route before
changing the ABI. `diagnose-motor` replays whole held-out worlds and separates
the temporal spread of the 815 motor-neuron rates from tonic decoder drive,
bias, and activity-dependent preactivation. `calibrate-motor` is a bounded
same-contract follow-up: it fits the existing structurally masked positive
34×815 decoder on standardized actual motor-neuron rates from training worlds,
then folds feature centering and scaling algebraically into the existing
positive weights and bias. The replay and candidate service use graph weights
rounded through IEEE binary16 exactly as the WebGPU deployment pack does; this
prevents calibration from amplifying a precision difference it never saw. Rate
standard deviations have declared floors and caps, and coefficients on the
standardized features are bounded. Folded effective weights may therefore be
large when measured neural variation is small; their realized range is reported
and is accepted only after fresh GPU parity. The report includes exact float32 centered-versus-folded
cancellation error and held-out worlds before writing a new research candidate.

```sh
python -m research.anatomical_cns.train diagnose-motor \
  --corpus /path/to/physical-corpus \
  --service /path/to/initialized.bin --service /path/to/bootstrap.bin \
  --output /new/path/motor-activity.json --device cuda

python -m research.anatomical_cns.train calibrate-motor \
  --corpus /path/to/physical-corpus --service /path/to/bootstrap.bin \
  --output-service /new/path/bootstrap-calibrated.bin \
  --output-report /new/path/bootstrap-calibrated.json --device cuda
```

Calibration reads only CNS motor-neuron rates and teacher MOTOR34 targets. It
does not receive BODY110, retina, root pose, skill names or world geometry as a
decoder input. Offline improvement still does not establish physical competence.
The calibrated artifact additionally requires fresh Torch/WebGPU MOTOR34 parity;
small-weight parity from the initialized service does not cover this candidate.
If folding fails its float32 numerical gate, the trainer writes a structured
rejection and a compressed research-only centered state containing the fitted
coefficients, intercept, rate mean and rate scale before it raises. It never
writes a runtime service for that rejected fit.
