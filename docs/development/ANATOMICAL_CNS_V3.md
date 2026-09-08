# Anatomical CNS V3: coupled implementation contract

Approved integration wave 2026-09-07 following the anatomical motor audit.
Breaking current engine; prior sealed V2 artifacts/lives remain historical.
Root owns this contract and integration. This document specifies implementation,
not evidence of fitted biological physiology or learned motor competence.

## Information and physical interfaces

Per 50 ms physical tick: actual retina/body + previously selected context12 ->
CNS recurrence -> latent512 and motor34 -> physical actuators. Private cognitive
memory reads latent512 and its own delivered context12, chooses next context12,
and receives delivery acknowledgement only after that context actually enters
CNS recurrence. It no longer outputs physical actions. No raw sensory, task ID,
world coordinates or teacher pose enters the cognitive organ.

BODY110, contiguous ranges (exclusive stop): odor[0:6], root linear opponents
[6:12], angular[12:18], contact normal[18:24], count[24], coarse touch[25:27],
16 acoustic log-frequency bands[27:43], overhead shade[43], interoception[44:56],
12 normalized joint angles[56:68], joint velocities[68:80], loads[80:92], six
foot contact values[92:98], twelve actuator-fatigue signals[98:110]. Joint order
is fixture leg order, each hip then knee. Acoustic bins logarithmic 40..1600 Hz,
Gaussian log2 frequency tuning sigma .35 octaves, amplitude envelope attenuation
and finite propagation. Silence is zeros, not a cue ID. These are effective
synthetic mechanosensory features, not waveform-level auditory physiology.

MOTOR34: 24 nonnegative antagonist activation values (joint-major positive then
negative direction), followed by gaze_pitch,posture,grip,signal_low,signal_mid,
signal_high,eat,release,secrete,allocate. Only indices24,25 signed[-1,1]; others
[0,1]. Synthetic effective actuator mechanics use activation dynamics, joint
length/velocity dependence and fatigue, never a supplied gait oscillator. Physics
passive limits/damping remain explicit. Body/sound/muscle state is persistent.

Context12: all signed[-1,1], no action names or physical actuation semantics.
Learned bounded currents enter 1,314 annotated descending cells. All motor34
outputs depend exclusively on annotated motor-cell state after recurrence.
Memory remains a synthetic learned organ; it cannot directly command the body.

## Artifact tensors (one current V3 contract)

Base counts N165122/E25563197/T11752; retina1771RGB/receptors4107/10types/
4669siteedges. Body rows11233 retained for this epoch with explicit omissions
recorded; context rows1314; motor rows815. Ancillary outputs use constrained
motor/efferent mapping assumptions recorded by exporter, never false muscle IDs.

Reuse graph.crow/col; graph.weight now signed normalized fast efficacy for fast
sources, positive within-family incoming-normalized counts for modulatory sources,
zero for unknown. graph.channel[N] u32:0unknown,1fast,2DA,3OA,4HT. The topology
is unchanged. Export must recompute from measured counts and annotate this change;
never infer missing modulator weights from V2 zero values.

Keep atlas.receptor_rows/type/ptr, site_indices/weight, neuron_type. Add
atlas.body_mask[11233,110] f32 0/1, atlas.context_rows[1314]u32,
atlas.motor_rows[815]u32, atlas.motor_mask[34,815]f32 0/1. atlas.body_rows unchanged.
Masks use actual modality/side/nerve/segment annotation, with missing/unresolved
selectors reported. Masks are structural and enforced at runtime and training.

Keep optic.spectral_logits[10,3]/gain_raw[10]/bias[10]. Replace dense mixed
body MLP with body.mean/scale[110], body.weight[11233,110],body.bias[11233].
body_current=sigmoid((body.weight*body_mask)@standardized_body+body.bias).
Standardized input clipped[-8,8]. This is a learned tuning curve per afferent;
no modality is available outside its mask.

Keep V2 dynamics.baseline_raw/recurrent_gain_raw/tau_raw/adaptation_gain_raw/
adaptation_tau_raw[T]. Add dynamics.release_tau_raw[T], release_use_raw[T],
mod_gain_raw[T,3], mod_adaptation_raw[T,3], modulation_tau_raw[3].
Keep afferent.neutral_drive[N] computed at RGB.5/body.mean. Add
context.weight[1314,12],context.bias[1314]; context_current=.15*tanh(weight@c+bias)
minus .15*tanh(bias), guaranteeing zero context produces zero current.
Keep readout.projection.weight[64,N],output.weight[512,64],output.bias[512].
Mask retinal/body/context injected rows from this global readout.
Add motor.weight_raw[34,815],motor.bias[34].
Motor preactivation=(softplus(motor.weight_raw)*motor_mask)@r_motor+motor.bias;
activation=sigmoid(preactivation), signed transforms for24/25 only. Never decode
context/premotor/raw input directly as a motor action.

## Exact private dynamics

State[N,B]: rate r, adaptation a, support s, release availability q,
three signed target modulatory traces mDA,mOA,mHT. Initially r=r0,a=0,s=1,q=1,m=0.
Traces are deviations from tonic modulation, not absolute transmitter concentration.
V2 bounds r0=.05+.4sigmoid, g=.5+1.5sigmoid, tau=.02+.23sigmoid,
k=.5sigmoid, ta=.25+4.75sigmoid, h=min(r0,1-r0).
Release tr=.05+1.95sigmoid(raw), use=.01+.49sigmoid(raw).
Modulator family tm=.1+4.9sigmoid(raw), initial [.5,1,2]seconds.

Two Jacobi substeps, delta=dt/2. At each substep, using previous r,q,m,a,s:

```
x = r-r0
fast_i = sum graph.weight_ij*x_j*q_j for channel_j==1
mod_input_ik = sum graph.weight_ij*x_j for channel_j==k+2
m_next = m + (-expm1(-delta/tm))*(mod_input-m)
mg = sum_k .5*tanh(mod_gain_raw[type,k])*m_next_k/h
ma = sum_k .5*tanh(mod_adaptation_raw[type,k])*m_next_k/h
g_eff=g*exp(.5*tanh(mg)); k_eff=k*(1+.5*tanh(ma))
u = sensory_current-neutral_drive+context_current+g_eff*fast-k_eff*a
target=r0+s*h*tanh(u/h)
r_next=r+(-expm1(-delta/tau))*(target-r)
```

After both substeps with final x:
```
a += (-expm1(-dt/ta))*(x-a)
s = clip(s+dt*(.024*(1-s)-.003*abs(x)/h),.65,1)
q = clip(q+dt*((1-q)/tr-use*abs(x)/h*q),.2,1)
```

q is a neuron-shared effective release resource, not a measured per-bouton pool.
Modulatory signs/strengths are learned type-level receptor-response assumptions,
not a universal reward or mood scalar. Unknown-transmitter edges stay explicitly
unresolved. No arbitrary topology edits. Neutral input/context and birth state
are a fixed point. Baseline/adaptation/support/release/family traces and context
pending delivery belong in complete checkpoints.

## Initialization and learning

V2 readout/optic/type dynamics may initialize a NEW V3 artifact through an explicit
one-way export, not a V2 loader in production. Body/motor/context interfaces are
new and must be labeled initialized. Body weights mask-structured tuning, context
small learned projection, motor paired bias matched for supported posture; no
claim of biological muscle correspondence from side/neuromere alone. New release
tr=.5,use=.1; mod gain/adaptation raw=.2 (small synthetic shared priors, fit later).

Torch implements these same equations with differentiable sparse full-graph
propagation. Teacher motor/pose labels are target-only. Current body measurements
are afferent inputs; future body measurements are prediction targets. Collect body110,
retina1771x3, delivered context12 and motor34, reset/time, and T+1 observations.
Use whole worlds for splits; retain unsuccessful transitions. Train sensory and
motor interfaces plus selected dynamics on actual physical teacher/babbling
sequences, temporal sensory prediction, then autonomous goal-conditioned rollouts.
Tone4 pose-hold/release is one sensory interaction curriculum alongside body
repertoire and material/social actions. No old 12-axis policy record may silently
be relabeled a 12-value neural-context action.

Runtime ownership: Rust native dynamics/body/cognition, WebGPU full-graph parallel
implementation and thinJS transport; Python remains Torch training/export/research.
One integrated native/WebGPU/Torch parity plus actual physical training/response
campaign follows the build batch. No repeated whole-suite microvalidation loop.

Headless Node/Dawn owners must strongly retain the object returned by `create()`
for the entire GPU lifetime, as required by the upstream
[node-webgpu lifetime contract](https://github.com/dawn-gpu/node-webgpu#lifetime)
and tracked in [Chromium issue 387965810](https://issues.chromium.org/issues/387965810).
An lldb diagnosis on Node 26/Dawn 0.6 reached
`dawn::native::InstanceBase::ProcessEvents` at `std::mutex::lock` after successful
ticks when that owner was collected. This is a headless-host lifetime rule, not
a browser CNS mechanism.

## Executed learning boundary

The first physical bootstrap fits shared interface and cell-type parameters; it
does not rewrite anatomical edges or learn personal CNS synapses during a life.
An individual retains its own seven neural state fields, physiology, private
cognitive state, acquired context sequences and random state. Shared parameter
training and lifetime acquisition are separate mechanisms. Structural evolution
and local Hebbian changes to measured wiring remain subsequent implementation
work, rather than being implied by the current parameter gradients.

The 512-update ROCm bootstrap reduced whole-world-heldout motor imitation error
from 0.07140 to 0.04997 in 92.3 seconds. Its motor output remains nearly constant
across curriculum activities. The training-only future-sense and pose probes
improved from random initialization but still slightly underperformed persistence.
These results establish trainable anatomical interfaces, not learned tone poses
or useful autonomous motor control. See the source-bound
[training receipt](../receipts/anatomical-cns-v3/physical-bootstrap-training.json).
