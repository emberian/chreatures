# CNS dynamics V2: operating-point-relative full MaleCNS

Root approved this breaking model on 2026-09-07. The old v1 artifacts/recordings
remain historical evidence; they are not V2 residents. This file freezes the
mathematical interface before the joined native/browser/training integration.
The topology remains the pinned 165,122 neuron / 25,563,197 edge MaleCNS v1.0,
with signed synapse-count / incoming-synapse-count CSR weights. No synthetic
edges, random replacement areas, or sensory readout skips are introduced.

## Exact frozen equations

All five raw dynamics arrays have shape `[11752]`, indexed by the existing
`atlas.neuron_type`. Effective per-neuron values are gathered by type:

| Exact tensor | Effective parameter | Initialization |
| --- | --- | --- |
| `dynamics.baseline_raw` | `r0 = .05 + .4*sigmoid(raw)` | r0=.2 |
| `dynamics.recurrent_gain_raw` | `g = .5 + 1.5*sigmoid(raw)` | g=.9 before supervised fitting |
| `dynamics.tau_raw` | `tau = .02 + .23*sigmoid(raw)` seconds | tau=.08 s |
| `dynamics.adaptation_gain_raw` | `k = .5*sigmoid(raw)` | k=.15 |
| `dynamics.adaptation_tau_raw` | `ta = .25 + 4.75*sigmoid(raw)` seconds | ta=1.5 s |

`h=min(r0,1-r0)`. Private state remains **rate r, signed adaptation a, support s**,
all `[165122,capacity]`, plus time/identity/receipts. Initialize `r=r0,a=0,s=1`.
Signed deviation `x=r-r0` is derived; it is not a second independent state.

Afferent topology and spectral/body interfaces retain their anatomical contracts.
The immutable `[165122]` tensor `afferent.neutral_drive` is recomputed from each
artifact's learned adapter at optical RGB=.5 and body input=`body.mean`
(standardized zero), then authenticated with every other array. Unsupported
receptors have zero raw and neutral drive; no replacement mapping is permitted.

Hold afferent drive and adaptation/support fixed over **two Jacobi substeps**:

```
x = r - r0
rec = W_fixed @ x
u = (afferent_drive - afferent.neutral_drive) + g*rec - k*a
target = r0 + s*h*tanh(u/h)
alpha = -expm1(-dt/(2*tau))
r_next = r + alpha*(target-r)
```

After both substeps, using the final `x=r-r0`:

```
a_next = a + (-expm1(-dt/ta))*(x-a)
s_next = clip(s + dt*(.024*(1-s) - .003*abs(x)/h), .65, 1)
```

Neutral input and neutral initialization are an exact mathematical fixed point.
Support reduces deviations around the operating point; it never drags the tonic
baseline toward zero. Adaptation tracks signed deviations and can represent both
post-stimulus rebound and habituation. It is not a physiological observation.
Baseline rates are normalized model activity, not a claim of measured spike Hz;
this graded representation is especially consequential for photoreceptors.

The redundant old source/target/excitability gain product and additive bias are
removed. Bounds ensure finite rate range and numerical updates, **not** a proof
that high-gain recurrence has useful or globally contractive dynamics. Strongly
saturated, stimulus-independent, unstable or uninformative regimes are rejected.

## Factorized learned readout and training boundary

All injected rows remain masked, including all 4,107 receptor rows (171 without
site support) and 11,233 body rows. Let `m` be that authenticated binary mask.
Exact tensor names and shapes:

- `readout.projection.weight`: float32 `[64,165122]`, V.
- `readout.output.weight`: float32 `[512,64]`, U.
- `readout.output.bias`: float32 `[512]`, b.

```
H = V @ (m * (r-r0))
Z = tanh(U @ H + b)
```

There is **no intermediate activation or projection bias**, and no fixed
population means. Every unmasked CNS neuron has learned readout weights.
This is a new rank-64 constrained learned map; it is not claimed equivalent to
the old unconstrained dense readout. SVD initialization is an optional explicit
one-way initialization, never a production compatibility path. Root owns the
new artifact envelope and browser packing. The research loader may read old
sealed arrays as a fixed anatomical/afferent seed, but cannot deploy v1 state as
v2 state. The readout may never use raw drive, neutral-adjusted drive, world
coordinates, body targets, labels or the sensory reconstruction head.

## Diagnosis, calibration and meaningful objective

The sealed fitted-v1 body output bias is approximately -2.95, giving neutral
body current approximately .05, **not .5**. The source initializer agrees. A
sigmoid's default midpoint was not the actual trained artifact operating point.
The v1 zero birth, very small bias, inhibitory receptor signs and rectification
remain a plausible combined reason for poor inhibitory information transmission;
the paired clip/blank audit quantifies that rather than diagnosing from a movie.

V2 preserves receptor sign and measured graph paths. Its positive interior
operating point permits inhibitory deviations without permanently silencing a
near-zero downstream unit. Centered recurrence absorbs the otherwise unknown
net tonic synaptic current into the chosen operating point. That is an explicit
modeling assumption, not a recovered biological current. Five per-type parameters
are still underdetermined by one visual movie: start with shared priors, fit only
identifiable gain/time-constant deviations, and retain tightly regularized
baseline/adaptation until multistimulus temporal data justify changing them.
Do not simultaneously free edge strength, source gain, target gain and neural
excitability when observations constrain their product.

The first joined research path compares the archived v1 response and multiple
V2 gains on the actual full graph. Independent whole clips vary grating phase,
direction/frequency and body-local channel 0. Masked neural features predict
current sensory features and 0.2-second sensory changes. A held-out frequency and
phase split prevents adjacent time samples from becoming validation data.
Future-change error is compared with persistence (zero predicted change), not
just an untrained head; activity amplitude is a diagnostic, never the objective.
The raw phase/body values are offline targets only. This is controlled excitation
and temporal information calibration, not learned movement or ecological skill.

`research/dynamics_v2/characterize.py` uses the full fixed CSR and a research-only
functional SciPy reference. Its fitted ridge head diagnoses response information
and is not a resident controller. `research/dynamics_v2/train.py` is a Torch/ROCm
functional gradient path over that same full graph, using its transpose for
backward, a factorized masked Z512, and training-only sensory heads. New recurring
resident/native/browser numerics remain Rust/GPU work. Neutral, saturated, reset,
restore and graph-edge ablation are joined integration checks, not evidence
that a trained temporal probe is a behaving organism.

## Consequential sources and boundaries

The [FlyVis author implementation](https://github.com/TuragaLab/flyvis/blob/main/flyvis/network/dynamics.py)
uses passive point neurons with instantaneous graded synaptic release, typed
unknown parameters, signed synapse-count strengths and activity initialized at
bias. This supports using graded operating points and fitting unknown dynamics
rather than treating a graph as an already functioning brain. Our bounded,
centered recurrence is a new model family, not a reproduction of FlyVis.
The [author project](https://github.com/TuragaLab/flyvis) and
[connectome-constrained visual modeling paper](https://doi.org/10.1038/s41586-024-07939-3)
provide the relevant task-constrained precedent. Anatomy alone does not identify
our normalized currents, time constants, graded baseline or metabolic support.

Scry discovery used authenticated `/v1/scry/schema`, then the bounded supported
HN token query for `flyvis`; record `147076ab-438c-410c-84d0-23c464995123` returned
zero records. This is discovery only; the author code was read directly. No
credential is stored here. The local neural-assemblies `core/brain.py` recurrence
notes describe collapse under repeated multiplicative strengthening and the need
for normalization, including limits of initial normalization. That changes the
plan toward bounded, explicitly normalized plasticity with task evidence rather
than copying recurrence or k-winner mechanisms onto MaleCNS. Private Hebbian
residuals and delays remain later identifiable extensions, not silently enabled
by this operating-point correction.

## Executed first full-graph regime audit

The completed receipt is at
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-regimes-20260907/receipt.json`.
Every sealed source tensor was checksummed before the sparse graph was built.
Each regime ran four 160-tick streams over all 25,563,197 edges, followed by a
ridge temporal probe fitted on two whole streams and evaluated on two others.
The 384 probe rows include strongly reached actual postsynaptic neurons and a
seeded broad CNS sample, with every injected row excluded.

| Model | Visual sine delta skill | Visual cosine delta skill | Body delta skill |
| --- | ---: | ---: | ---: |
| Sealed fitted v1 | -.654 | -.865 | .125 |
| V2 gain .9 | -.131 | -.088 | .373 |
| V2 gain 1.3 | -.282 | -.087 | -2.091 |
| V2 gain 1.8 | -.611 | -.048 | .904 |

Skill is `1 - heldout MSE / persistence MSE`; negative means **worse than
persistence**. No candidate establishes successful visual temporal prediction.
Higher gain increased neural deviation RMS but did not consistently improve
information. Gain .9 is the initial fitting choice because it improved both
visual probe errors and the body error over v1 without requiring expansive
instantaneous recurrence. The gain prior may move during supervised fitting;
there is no declaration that .9 is a recovered physiological constant.
Neutral V2 deviation was exactly zero and none of the sampled regimes saturated
above absolute deviation .19. Four parameter settings do not justify a GAM
response surface or parameter-interaction claim; a larger crossed design would
be required before using GAM to choose physiological gains.

The actual clip/blank arrays gave nonafferent clip-minus-blank RMS .0002719,
whereas blank-from-initial RMS was .0006969 (2.56 times larger). About 64.7% of
nonafferent blank samples were below 1e-6. This supports the specific concern
about operating-point transients and silent units; it does not say the clip had
no effect. Nonafferent per-frame temporal RMS was 2.02e-5 for the clip and
7.40e-6 for blank. The immutable fitted artifact's neutral body current was
.049746. The broader paired-runtime causal report remains the integration
owner's authenticated comparison; this physiology summary uses its saved arrays.

## First executed ROCm fit and canonical V2 artifact

The first actual full-graph fit completed 64 updates in 136.82 seconds on the
hbox RX 6750 XT using Torch 2.9.1+ROCm6.3. Supervisory sysfs observations showed
99% GPU use during training and about 3.06 GB VRAM total, returning to the
preexisting 1.06 GB idle baseline after completion. No live service was changed.
Type gains changed from .9 to min/mean/max .83966/.93079/1.05017; type time
constants became .06154/.07605/.10057 s. Thus the trained readout and actual
full-graph recurrent dynamics both changed; this was not a visualization filter.

The first head was instantaneous linear on Z512 and the held-out frequency band
(.4–.7 Hz) lay below the training band (.9–1.5 Hz). Current visual sine/cosine
skills improved to +.237/+.589 against the zero predictor. Future-change skills
were **−1.637/−1.646** and body future-change skill **−.415**, all worse than
persistence. That artifact is suitable for exact V2 integration/parity, **not**
a successful temporal predictor or trained creature controller. Subsequent
research uses a temporal training head that consumes Z alone and a crossed
frequency/direction design. It publishes new immutable parameter identities.

The first frozen export lives under
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-fitted-20260907`:

| File | SHA-256 |
| --- | --- |
| `canonical-v2-parameters.npz` | `ccaa43d8450b0c3312db10f7cb9963ed786495cc95556a6364f250aba39357a9` |
| `torch-v2-fixture.npz` | `eaac249554b7aa0d918ef6bdcf7f9a33b1da1319b031a347dd7db9a049ed8c8d` |
| `cns-service-v2-temporal-probe.bin` | `09bb8d18375639ffd3a2d3b9ed526a283acde69b45ff592b60697d3a49ee5def` |

The current CHCNS2 writer produced the 255,083,593-byte full service with adapter
identity `97cc5701b3f934dd0c1e6fbf8ce6dd8cd873780a38eac96994514020fa9bcccb`.
Its provenance embeds the actual training receipt and the explicit lack of a
trained controller. Neutral drive was independently recomputed from the frozen
afferent weights before writing. The B2/T8 Torch fixture has sensory `[8,2,5356]`,
rate/adaptation/support `[8,165122,2]`, latent `[8,2,512]`, dt=.05 and exact declared
initial state. Root/native/browser owners use that new fixture for their joined
execution check; this document does not claim their not-yet-reported results.

## Crossed temporal-response characterization

A second, broader experiment ran 27 combinations of gain `[.7,1.05,1.4]`, tau
`[.04,.08,.16]` s and adaptation gain `[.05,.15,.35]`. Each used all neurons/edges
for 12 streams of 128 ticks, followed by 20 neutral ticks. Eight training streams
covered both motion directions at four frequencies; four whole held-out streams
used intermediate frequencies and fresh phases. Body oscillation phase was
independent of visual phase. The same masked-neuron temporal ridge probe and
exact same stimuli were also run through the archived fitted v1 model.

The strongest measured V2 setting was gain=1.4, tau=.04 s, adaptation gain=.05.
Its mean three-channel future-delta skill was **.8214**, versus **.5914** for the
matched archived v1 baseline. Mean held-out MSE fell from **.38415 to .17532**
(54.4%); persistence MSE was .84076. This establishes improved temporal probe
information for this controlled stimulus family, not behavior or unique
physiological identification. This comparison uses a broader training design
than the earlier two-stream sweep, so absolute scores cannot be attributed to
the equations alone across the two experiments. The within-experiment v1/V2
comparison keeps the stimulus and decoder protocol fixed.

No configuration saturated past absolute deviation .19. The post-neutral RMS
ratio measures residual activity after one second; it is **not** treated as
memory utility. A system can retain an unhelpful offset. Raw records are at
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-crossed-20260907/receipt.json`,
with `v1-baseline.receipt.json` alongside. The native GAM response fit uses these
crossed records to distinguish gain/time/adaptation interactions, predicts held
out parameter configurations, and proposes only interpolated candidates for an
actual confirmation run. It does not treat its own fitted optimum as execution.

## Bounded circuit-learning extension contract

The present physiology correction does not enable private edge learning. A
later actual-connectome residual can maintain eligibility only on an explicitly
selected authenticated existing-edge set, with bounded pre/post deviation
products, bounded log-efficacy, fixed source sign, and explicit postsynaptic
renormalization. A candidate update family is `e <- rho*e + (1-rho)*pre*post`,
then `log_efficacy <- clip(log_efficacy + eta*modulator*e, -c, c)`; this is a
proposed computational family, not a measured fly STDP rule. Normalized pre/post
variables and the modulator must be computed from CNS state. Actual DAN/MBON/KC
annotations provide a better initial bounded anatomical scope than random
assembly areas. Any modulatory readout remains inside the CNS; a raw reward or
body signal cannot enter the private update through a separate route.
Eligibility, efficacy, normalization state and delay queues belong to each
resident's coherent snapshot. Evolution or outer-loop task fitting may select
bounded circuit parameters, but cannot silently change an existing creature's
anatomical graph or feed target labels into its life. Before adding more degrees
of freedom, multistimulus perturbations must distinguish time constants,
adaptation, baseline and recurrent gain; the current visual/body probes do not
identify individual edge physiology or biological homeostatic set points.

## Stronger temporal fit, canonical replay and graph dependence

The second fit completed 192 updates in 397.12 seconds on the same ROCm GPU.
The recurrence/type gains and rank-64 Z512 readout were initialized from the
first fitted artifact, while the supervised head now includes a GRU128 consuming
**only Z512** for future changes. Training frequencies came from `.35–.75` or
`.9–1.6` Hz; whole held-out streams used `.75–.9` Hz with independent optical
and body phases. No loss term rewarded activity variance.

| Held-out target | Current-value skill | Four-tick future-change skill |
| --- | ---: | ---: |
| Visual sine feature | .9931 | .9728 |
| Visual cosine feature | .9751 | .9189 |
| Body-local channel 0 | .6803 | .1203 |

Current skill uses the zero predictor; future skill uses persistence. The
controlled visual temporal task is substantially learned, while body prediction
is weaker. These are three supervised sensory features in procedural stimuli,
not proof of vision in a natural environment, movement, memory or social skill.
The temporal prediction head is a training/evaluation organ and is not smuggled
into the sensory interface or labeled a trained resident controller.

The new immutable export is under
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-temporal-fit-20260907`:

| File | SHA-256 |
| --- | --- |
| `canonical-v2-parameters.npz` | `10e792a150641c80263f20ecfeac377ababd05cf6cec7de9f9a44d7f2e8bc4a3` |
| `torch-v2-fixture.npz` | `280bd335ce519a73d5d331ff397d76e179b42c9b3cedb2f16d6931404e768229` |
| `cns-service-v2-temporal-trained.bin` | `ca66b6e5c07a09d55fc7e29552f504bd86b3807a6298425b9b933edf2cde9378` |
| `canonical-trained-parameters.npz` | `e37fea1082e274a467997db731a833581a037e1a95a7ed6f630e5f7bc86a4f59` |

The strong service is 255,084,953 bytes with adapter identity
`6875314d15f46806abb0fab04802856d4427d3a6ef29b19b12c70bb79706806d`.
It does not overwrite the first fit. The last listed parameter artifact is the
current canonical metadata-bearing export, including all afferents and all five
type fields; its parameter identity is
`368a1e8f3b84ce804c19066a4faa92911ff5e853468cac4cf8248a981548553a`.

The current `research/sensorimotor_skills/cns_adapter.py` now owns the sole
canonical Torch V2 recurrence and factorized readout. The research training head
wraps that model through `step_from_drive` and `readout_from_state`; it no longer
maintains its own training recurrence. The main `scripts/train_cns_adapter.py`
uses V2 parameters, joint procedural optic/body inputs, four-tick forecasting,
correct five-type/readout gradient reporting, and zero coefficients for the
old activity variance/covariance objective. Its standard parameter exports use
V2 metadata. Isolated SciPy/crossed-sweep equations remain numerical research
references, and old V1 equations exist only in that explicit archival probe.

A joined actual-fullgraph ROCm audit replayed the trained model through this
canonical implementation, reproducing the reported metrics to roughly 1e-7.
Zeroing every recurrent graph edge in that isolated process collapsed future
visual skills from `.9728/.9189` to `−.00555/.00371`, and body skill from `.1203`
to `−.00354`. This establishes that the fitted task information depends on
retained CNS graph transmission; the supervised temporal head and scheduling
alone did not recover it. This ablation never touched the immutable artifact
or any live service.

The same joined audit ran the new canonical production training loss over an
8-tick B2 sequence with both optical and 43-channel body variation. Every
optical/body parameter group, all five type arrays and both learned readout
layers had finite, nonzero gradients. The newly initialized broad sensory heads
had poor prediction scores in this startup check; it is a gradient/execution
check, not a claim that those new heads were trained. Detailed values and the
export receipt are in `canonical-audit.receipt.json` beside the strong artifact.

The native owner separately reported strong-artifact Metal rate error 1.19e-7,
Z512 error 5.76e-6 and exact private snapshot replay. A declared 1e-5 absolute
latent tolerance accounts for float32 rate/reduction differences amplified by
the learned factorized projection. This is not bitwise Torch/Metal parity.

The native GAM's previously unmeasured candidate was also run through the full
graph. At gain1.4/tau.04/adaptation.0875, observed skill was .815785 versus the
GAM's .817397 prediction (absolute error .001612, below its .0127603 leave-out
RMSE). It did not improve the measured .821376 optimum. The result confirms one
useful interpolated response prediction, not recovered biological physiology.
