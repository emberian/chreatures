# Optic input anatomy audit and proposed CNS-only sensory boundary

Executed 2026-09-07: read pinned source anatomy, derived CSR, and retinal-v2
bundle; extracted a compact anatomy atlas. No resident, circuit service or world
was stepped or replaced. The proposed input adapter below is not implemented.
Latest user steering requires all policy-relevant information to traverse CNS
activity and trainable CNS-derived adapters. Frozen v8 is a legacy baseline;
the former learned-sequence state668/raw-input contract is superseded.

## What v8 actually supplies to the graph

`data/ports/retinal-v2.json` declares 351 total physical scalar channels. Of those,
320 are **80 spatial bins × RGB/proximity**, not 320 anatomical columns. Native
sensorium measures 1,024 rays: peripheral 8×32 plus foveal 24×32, each with four
components. Only the peripheral samples are area-pooled to 5×16 for neural drive.
The pooled elevation groups are [0:2], [2:3], [3:5], [5:6], [6:8]; adjacent
azimuth samples are paired. The foveal image never enters the v2 neural map.

The retained sparse input map has 48,258 retinal nonzeros, each weight 0.8,
contacting **23,720 different real neurons**. Thus small input bandwidth does not
mean only 80 neurons run. The 320 ports target these exact types:

| Engineered component | Target types |
| --- | --- |
| red | L1, L3, L5, C2 |
| green | L2, C3, Mi1, Mi4, Mi9 |
| blue | T1, Tm1, Tm2, Tm4, Tm9, Tm20 |
| proximity | all 15 types above |

These are intrinsic optic cells; **none of the 4,107 R1–R8-class photoreceptors
are direct retinal-v2 targets**. The color/cohort assignment is not measured
spectral physiology, and proximity is an engineered range signal.

Spatial routing uses real `assignedOlHex1`/`assignedOlHex2`, but bins their ranges
1..36 and 1..39 into the engineered 16×5 raster. 72 of 320 component ports hit
empty rectangular corners and use nearest measured coordinates in the cohort.
The bin mapping is not a measured angular receptive-field map.

**Retinal routing has no hemisphere filter.** The same camera bin can directly
stimulate both sides: the full target set has 10,453 L and 13,267 R neurons.
For example, `retina/e00/a00/red` targets 61 cells (18 L, 43 R), including actual
bodyIDs 33343, 41274, 48056, 49131, 49635, 52226, 55965 and 61905. This is a
specific side/hemifield defect, not merely low image resolution. Source side
labels exist and should govern separate body-eye projections.

The learned v8 controller also receives all 4,096 rich retinal scalar values
directly, along with the 351 physical values and 12 physiology values. Therefore
its detailed visual competence cannot be attributed solely to the connectome.
The 384 graph outputs are aggregate readouts: visual 160, mushroom-body 96,
navigation 80, efferent 48. They are stratified primarily by type/side/region and
population-averaged, not a spatial optic activity image. Both the input and
readout are bottlenecks; a neuron-count-sized display does not fix either.

Concrete code locations (line numbers may move during concurrent integration):

- `chreatures/neural_ports.py:222`: reads exact body IDs and assigned hex fields;
  `:294` builds input assignments, `:318` handles nearest fallback;
  `:491` constructs aggregate readout strata.
- `native/world-kernels/src/sensorium.rs:175`: returns coarse and rich sensorium.
- `chreatures/runtime3d.py:496`: concatenates rich retina, physical source and
  physiology for direct controller observation; `:743` separately steps the
  remote graph and `:766` supplies both paths to the resident.
- `native/cognitive-core/src/developmental.rs`, `encode_frames`: directly encodes
  peripheral/foveal tensors. `refine_candidates`: GAM/predictor scoring operates
  in addition to graph recurrence, not as its numerical replacement.

## What runs on the full graph

The pinned induced Traced graph is 165,122 neurons, 25,563,197 directed edges and
124,025,046 synapses; no top-k subset is introduced by retinal-v2. CSR rows are
postsynaptic cells and entries are real presynaptic indices. The runtime matrix
uses measured counts, divided by retained incoming count, and a modeled source
transmitter sign (`chreatures/malecns.py:395`). This is real graph propagation
with simplified model dynamics, not a detailed biophysical fly emulation.

`chreatures/remote_brain.py:212` advances private per-neuron rates, adaptation
and support using sparse recurrent multiplication over the graph, twice per
physical step. `chreatures/fast_circuit.py:68` implements the corresponding GPU
CSR row traversal; `:455` contains the readable rate equations. The fullgraph
training circuit is constructed in `chreatures/training_cohort.py:861`.
This audit inspected files and artifacts; it did not independently query or
change the current live process's backend flags.

The common rate target is rectified tanh of tonic 0.005 + sensory drive + scaled
signed recurrence − 0.10 adaptation. Those equations, gains, signs, normalization
and neuron support dynamics are explicit modeling choices. The separate GAMs
estimate candidate physical consequences; they do not stand in for the 25.6M-edge
matrix multiply. A displayed optic-lobe movie alone would not establish either
physiological realism or behavioral causal dependence.

## Actual available spatial anatomy

The pinned annotation Feather contains `somaLocation`, `tosomaLocation` and the
two hex fields, but the existing derived `neurons.npz` discards all four fields.
The new atlas retains them, aligned to the exact graph bodyID order.

| Source annotation | Traced cells with data |
| --- | ---: |
| Both assigned hex coordinates | 23,720 |
| Soma location, all graph | 140,024 |
| Soma location, hex-annotated cells | 18,320 |
| To-soma location, all graph | 995 |
| To-soma location, hex-annotated cells | 170 |

All 23,720 hex-annotated cells are the 15 intrinsic types already used by v2.
There are 892 distinct coordinate pairs if sides are collapsed, but **1,771
side-specific sites: 879 L and 892 R**. This is the available annotated lattice,
not a claim of complete eye coverage. L1 covers 875 distinct L and 892 R sites.
There is no assigned hex on any of the 4,107 photoreceptors.

All non-null location entries are integer triples. The Feather metadata does
not state coordinate units. The official download page documents MaleCNS SWC
skeletons in 8nm EM voxel units, synapse points in the same units, and precomputed
skeletons in 1nm units. As a cross-check, body10851/Tm2 soma annotation
`[75176,23526,20258]` is 115.83 raw units from the nearest same-cell official SWC
point `[75072,23488,20224]`, about0.93µm at8nm/unit. Coordinate magnitudes and this
comparison support interpreting soma triples in the same 8nm voxel frame;
**this remains a documented inference, not explicit annotation-field metadata**.

For an immediate anatomical activity view, use actual soma positions where
present and the measured side-specific hex lattice for a separate column view.
Do not mix missing soma locations with invented 3D coordinates. To-soma points
are a separately named fallback, not a soma centroid. A soma is not a receptive
field center. Full neurite geometry requires the separately published skeletons;
actual synapse positions are not included in the currently held CSR graph.

## Executed connectivity-derived photoreceptor mapping

For each actual photoreceptor, sum its measured outgoing synapse counts into
same-side, hex-annotated intrinsic cells, grouping by side+hex. Preserve the
distribution over sites rather than silently forcing the largest one.

This directly anchors **3,936/4,107 photoreceptors (95.84%)** with 196,357 measured
synapses, represented by 4,669 receptor/site entries. No cross-side synapses
were present in this selected relation. The remaining171 receptors stay explicitly
unmapped. There is no nearest-bodyID or nearest-soma assignment.
The retained evidence touches1,486/1,771 sites (691L,795R);285 sites have no direct
photoreceptor-to-anchor evidence in this Traced graph. Rendering all sites does
not manufacture a photoreceptor connection for the unsupported ones. Preserve
their mask, and add evidence-based geometry/connectivity mapping in a later
version if available rather than silently filling them.

| Photoreceptor type | All | With direct column evidence |
| --- | ---: | ---: |
| R1-R6 | 1,394 | 1,386 |
| R7R8_unclear | 85 | 34 |
| R7_unclear | 404 | 350 |
| R7d | 82 | 80 |
| R7p | 332 | 296 |
| R7y | 481 | 468 |
| R8_unclear | 442 | 436 |
| R8d | 76 | 75 |
| R8p | 330 | 330 |
| R8y | 481 | 481 |

Median largest-site synapse fraction is1.0 within every listed class; a minority
have broad or ambiguous evidence (minimum fraction0.333 among R7_unclear).
For example, body246994/R8y/L has counts42 at(5,9),7 at(4,9),7 at(6,9).
This is an anatomical projection-derived column estimate, not a measured optical
receptive field. The source photoreceptor rows have no `receptorType` annotation.
Do not infer complete retinal sampling, wavelength tuning or exact ommatidial
identity from these labels alone.

## Proposed high-bandwidth neural input contract

1. Use atlas `site_side_hex[1771,3]` order, sorted by side code1=L,2=R, then hex1,
   hex2. Render body-attached **separate left/right eyes** at those 1,771 sampling
   sites. Native output `optic_rgb[B,1771,3]` is 5,313 optical scalar values.
   The eye placement and hex→ray angular transform are versioned engineered body
   optics. Calibrate orientation/overlap with controlled hemifield stimuli;
   do not treat orthogonal raster binning or soma coordinates as eye angles.
2. Preserve exact observed sites instead of rectangular empty-corner fill.
   Project site intensities to `photoreceptor_graph_rows[4107]` using the atlas
   same-side sparse synapse-count distribution. Normalize each supported row;
   unsupported rows have zero visual drive and an explicit mask. Other CNS
   recurrence remains intact for those cells.
3. A trainable optical adapter supplies spectral mixture, gain/tonic level and
   causal temporal response per photoreceptor class, with optional bounded local
   adjustments to graph-derived site weights. Initialize from the measured
   projection distribution and declare all spectral/temporal assumptions.
   Raw RGB is allowed only at this sensory-to-CNS boundary. It cannot be supplied
   to a motor selector, memory predictor or goal manager around the graph.
   Optical depth/proximity is not a photoreceptor color channel; other physical
   senses require their own explicit afferent adapters under root's CNS contract.
4. Produce graph drive indexed by actual neuron rows, evolve the full recurrent
   graph, and use trainable CNS-derived readouts for all downstream control and
   memory. Input bandwidth is independent of the eventual learned observation
   embedding dimension. A compact CNS embedding is acceptable only after the
   1,771-site stimulus has actually influenced CNS activity; reusing the old
   351-column input bottleneck is not this implementation.
5. Expose an observation-only activity stream with actual per-neuron rates and
   bodyIDs, plus side/type/hex aggregations built from the atlas. Static anatomy
   is not a policy input. The stimulus→neural activity→creature response video
   must synchronize stimulus frame, acknowledged neural state and physical action
   by resident/tick. A stimulus movie painted directly onto a hex display is not
   evidence of propagation.

Substantive dynamics caveat: current source-sign mapping assigns histamine−1
(`scripts/acquire_malecns.py:269`). Rerouting input from L/Mi/Tm cells to actual
photoreceptors can therefore suppress downstream activity under the simplistic
small-tonic rectified rate model. Trainable tonic/type dynamics and the optical
adapter must be assessed with contrast steps, flicker and moving spatial stimuli
in the joined graph+body loop. Greater ray count alone is not sufficient.
No future training result or behavioral sufficiency is claimed here.

## Reproducible artifacts and sources

- `scripts/audit_optic_anatomy.py` verifies the pinned annotation hash, reads the
  existing graph without modifying it, and produces the atlas plus receipt.
- `data/ports/optic-anatomy-audit-v1.npz` (1,924,043bytes) retains graph-aligned
  bodyIDs, actual positions/masks, type names, side/hex membership and the sparse
  photoreceptor/site evidence. Missing coordinates/site are−1. Side0 means
  unresolved. It contains anatomy, not activity or a trained sensory adapter.
- `data/ports/optic-anatomy-audit-v1.receipt.json` records source/artifact hashes,
  coverage and per-type mapping concentration. Matching copies are under
  `hbox:/tank/chreatures/data/ports/`.
- Extraction used isolated pyarrow25.0.1 under
  `/tank/chreatures/scratch/optic-audit-pyarrow`; no shared environment changed.

Primary source: [official MaleCNS downloads](https://male-cns.janelia.org/download/)
documents the flat annotations, real connectivity, synapse geometry and coordinate
conventions of published skeleton formats. The exact pinned annotation URL is
recorded in `scripts/acquire_malecns.py`. Same-cell coordinate cross-check:
[official body10851 SWC](https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/10851.swc).
Direct source reads supplement the artifact audit; no screenshot was available
to this lane, so no claim is made about the implementation behind the user's
Bad Apple visualization.
