# MaleCNS mushroom-body plasticity

This module is a bounded research model of associative history on one measured
MaleCNS mushroom-body scaffold. It packages every directed Kenyon cell (KC) to
MBON11 edge in the bilateral γ1pedc compartment and keeps the measured synapse
counts immutable. Eligibility traces, modulation state, and learned efficacy
deviations are private runtime state.

An optional full-graph bridge maps those same edges into the canonical native
MaleCNS recurrence. It reads actual KC rates from the 165,122-neuron state and
injects only the learned recurrent-current difference at the two MBON11 rows.
The immutable native graph continues to supply every baseline edge.

The package does not assign food, reward, pleasure, punishment, or motor meaning
to the model. Its `modulator` argument is a synthetic, dimensionless experimental
pulse. PPL101 neuron identities provide compartment provenance; a pulse is not a
claim that either PPL101 neuron fired in a particular world event.

## Measured substrate

The selection was made from the full curated MaleCNS v1.0 graph with dataset
SHA-256
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`:

| Population | Selection | Body IDs | Measured bundle contents |
| --- | --- | --- | --- |
| KCs | `class == Kenyon_Cell` | 3,623 connected IDs, explicitly stored | Sources of all retained edges |
| left MBON | `type == MBON11` | 10704 | 2,048 edges, 16,863 synapses |
| right MBON | `type == MBON11` | 11402 | 2,136 edges, 24,597 synapses |
| left DAN reference | `type == PPL101` | 11900 | `PPL101(y1ped)_L` |
| right DAN reference | `type == PPL101` | 11327 | `PPL101(y1ped)_R` |

The bundle contains **4,184 directed KC→MBON11 connections, 41,460 measured
synapses, and 3,623 unique connected KCs**. The source annotations contain 4,064
curated KCs; the other 441 have no retained KC→MBON11 edge. The bundle preserves
cross-side connections. It does not manufacture bilateral segregation or
retinotopy.

MaleCNS also measures the following context around the two PPL101 and two MBON11
cells. These connections are recorded in bundle metadata and are outside the
plastic edge set:

| Direction | Connections | Synapses |
| --- | ---: | ---: |
| PPL101 → KC | 4,936 | 11,482 |
| KC → PPL101 | 6,168 | 24,068 |
| PPL101 → MBON11 | 4 | 2,311 |
| MBON11 → PPL101 | 4 | 205 |

The PPL101 cells carry ground-truth dopamine annotations and the MBON11 cells
carry ground-truth GABA annotations in MaleCNS. These labels are provenance.
They do not determine receptor physiology, synaptic sign, cotransmission, or the
plasticity equation used here.

The compartment choice and forward cue/modulator ordering are inspired by Hige
et al.'s experiment: PPL1-γ1pedc activation shortly after an odor produced
long-lasting, odor-specific suppression of the γ1pedc MBON response, while the
reported backward pairing did not. The experiment also found that MBON spiking
was not required for induction. See [Hige et al., Neuron 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/),
DOI `10.1016/j.neuron.2015.11.003`. The compartmental organization follows
[Aso et al., eLife 2014](https://elifesciences.org/articles/04577), DOI
`10.7554/eLife.04577`.

## Artifact and provenance

The compact bundle and its receipt are committed at:

- `data/mushroom/gamma1pedc-kc-mbon11-v1.npz`
- `data/mushroom/gamma1pedc-kc-mbon11-v1.json`

The full-graph bridge and receipt are committed at:

- `data/mushroom/gamma1pedc-fullgraph-bridge-v1.npz`
- `data/mushroom/gamma1pedc-fullgraph-bridge-v1.json`

The NPZ is 50,191 bytes with SHA-256
`27bbc1adde8efe26d9f62e8e5674d4a3c9e787ab200b498f89ddc19c9101bdff`.
Its content-level substrate hash is
`25b4597b6ac3448dde77d8010e58ef654ca8eb447b2389eb4cd05bd01f9d35b2`.
It stores local indices and body IDs for every source, target, and reference DAN;
each edge repeats its source and target identities so downstream use can audit
the exact mapping.

The bridge NPZ is 48,680 bytes with SHA-256
`ac7e686369b582d4035818408795f3b7d7ff2af974384805920319110b8bda05`.
Its content hash is
`7611dd524dcc8fab908d944ccba735c2c3a731d64d82d7d77f6308a1191a85a7`.
It selects 3,627 full-graph state values in the fixed order 3,623 connected KCs,
PPL101 left/right, and MBON11 left/right. The exact MBON local indices are 655
and 1306.

The original full incoming rows contain 19,135 synapses for left MBON11 and
27,730 for right MBON11. The KC subset contributes 16,863 and 24,597 of those
synapses. Consequently, the subset-normalized exploratory response described
below is unsuitable as an additive current in the complete circuit. The bridge
stores each original coefficient using the complete row denominator. A build
against the canonical matrix compared all 4,184 float32 coefficients bit for
bit and found maximum absolute delta zero. Its validation receipt is
`/tank/chreatures/runs/mushroom-plasticity/bridge-build-validation/bridge-build-validation.json`,
SHA-256 `e83ecb9b5ee53bfaa9afc6091e2812bd36f245a9701d03d1143c029531ce525f`.

The official source file hashes embedded in the artifact are:

| Official MaleCNS v1.0 source | SHA-256 |
| --- | --- |
| annotations | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| connectivity | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| neurotransmitters | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |

The MaleCNS dataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution:
“Sexual dimorphism in the complete Drosophila male central nervous system
connectome,” *Cell* (2026), DOI `10.1016/j.cell.2026.08.015`. The official project
and download pages are [male-cns.janelia.org](https://male-cns.janelia.org/) and
[the MaleCNS download page](https://male-cns.janelia.org/download/).

## Engineered dynamics

For edge `e` into target `j`, the immutable baseline is its measured synapse
count divided by the total retained KC synapse count into that target:

```text
b_e = synapse_count_e / sum_{q -> j}(synapse_count_q)
w_e = b_e * (1 + d_e)
y_j = sum_{e -> j}(w_e * KC_activity_source(e))
```

`d_e` is a private efficacy deviation in `[-maximum_depression, 0]`. The default
rule decays old eligibility and modulation traces, injects the current pulse,
applies depression using the old eligibility, then makes current KC activity
eligible:

```text
eligibility *= exp(-dt / eligibility_tau)
modulation *= exp(-dt / modulation_tau)
modulation = max(modulation, pulse)
d_e = clip(d_e - depression_rate * dt * eligibility_e * modulation_target(e),
           -maximum_depression, 0)
eligibility_e = max(eligibility_e, KC_activity_source(e))
```

This update ordering implements cue-before-modulator dependence. The equations,
normalization, default constants, synthetic cue masks, and pulse are engineered.
They have not been fitted as MaleCNS physiological parameters. The measured
counts remain available as `substrate.synapse_counts`; the normalized immutable
baseline is `substrate.baseline_weights`; learned state lives in
`model.efficacy_deviation`.

### Full-graph correction

For bridge edge `e`, the stored original recurrent coefficient is:

```text
a_e = float32(synapse_count_e)
      / float32(all retained incoming synapses at target(e))
      * float32(runtime source sign_e)
```

This is exactly the corresponding value in
`MaleCNSGraph.matrix(normalized=True, signed=True, dtype=float32)`. The runtime
source sign is an existing engineered whole-graph transform and is not evidence
about receptors at the KC→MBON synapse.

After native step `t`, the bridge gathers actual post-step KC rates and prepares:

```text
correction_j(t + 1) = sum_{e -> j}(a_e * d_e(t) * actual_KC_rate_source(e,t))
```

The next native step evaluates:

```text
activation_j = 0.005 + sensory_j
             + recurrent_gain * (original_recurrent_j + correction_j)
             - 0.10 * adaptation_j
```

Thus `d_e == 0` gives an exact zero correction. The baseline term is never added
again, and the bridge never substitutes its KC-subset normalization for the
complete incoming-row normalization. The one-step lag is deterministic and part
of the bridge and snapshot identity.

## API

The input is ordered by `substrate.kc_body_ids`. The two response values and a
two-element modulator are ordered left then right by the stored MBON11/PPL101
metadata.

```python
import numpy as np

from chreatures.mushroom_plasticity import (
    MushroomBodySubstrate,
    MushroomFullGraphBridgeSpec,
    MushroomPlasticity,
    WholeBrainMushroomCohort,
)

substrate = MushroomBodySubstrate.load(
    "data/mushroom/gamma1pedc-kc-mbon11-v1.npz",
    expected_sha256="27bbc1adde8efe26d9f62e8e5674d4a3c9e787ab200b498f89ddc19c9101bdff",
)
model = MushroomPlasticity(substrate)

kc_activity = np.zeros(substrate.kc_count, dtype=np.float64)
kc_activity[:20] = 1.0
cue = model.step(kc_activity, modulator=0.0, dt=0.2)
pulse = model.step(np.zeros_like(kc_activity), modulator=[1.0, 1.0], dt=0.1)

bridge_spec = MushroomFullGraphBridgeSpec.load(
    "data/mushroom/gamma1pedc-fullgraph-bridge-v1.npz",
    expected_sha256="ac7e686369b582d4035818408795f3b7d7ff2af974384805920319110b8bda05",
)
cohort = WholeBrainMushroomCohort(substrate, bridge_spec, capacity=3)
```

`step(input_KC_activity, modulator)` is the narrow causal interface. It returns
the two current MBON responses plus trace summaries. It has no motor output
mapping. A whole-brain runtime can supply measured-scaffold activity without a
projection:

```python
kc_activity = full_graph_rates[substrate.kc_neuron_indices]
result = model.step(kc_activity, explicit_experimental_modulator)
```

`save_snapshot` and `load_snapshot` preserve all private state, configuration,
time, update count, plasticity-enabled flag, and substrate hash. Snapshot loading
rejects a different substrate. `plasticity_enabled=False` supplies a frozen
anatomical response using the same measured edges.

`WholeBrainMushroomCohort` exposes `selected_neuron_indices`,
`selected_body_ids`, `target_neuron_indices`, `target_body_ids`, and
`pending_correction`. Its native-facing arrays are neuron-major:
`selected_rates[S, capacity]`, `modulator[2, capacity]`, and
`pending_correction[2, capacity]`. Only active slots advance. `snapshot()` uses
exact base64-encoded array bytes and `restore()` rejects changes to graph,
substrate, bridge, configuration, capacity, lag, or modulator mode.

The default `synthetic` mode requires an explicit pulse on every active step.
The optional `actual_ppl101_rate` mode takes the two gathered PPL101 rates through
an engineered identity mapping and rejects a synthetic pulse. That mode means
only that measured-neuron state drives the engineered plasticity rule; it does
not establish dopamine release, receptor effects, valence, or learning in the
animal.

## Probe result

The probe was run once on hbox against the complete graph artifact:

```bash
PYTHONPATH=/tank/chreatures/envs/python-packages /usr/bin/python3 \
  scripts/probe_mushroom_plasticity.py
```

Its full receipt is at
`/tank/chreatures/runs/mushroom-plasticity/gamma1pedc-v1/probe.json`, SHA-256
`06fc995ca0528459d5fb1268ef60a1f7548a9eb7af61b4ab69d88fabfbefaa23`.
It used two disjoint deterministic masks of 180 connected KCs and 20
counterbalanced presentations. Pairing cue A reduced its mean response retention
to `0.55875155`; unpaired B retained `0.99897405`. Reversing the history reduced
B to `0.55875155`; unpaired A retained `0.99897358`. Each history changed 414 of
4,184 efficacy deviations. The frozen model's response delta was exactly zero.
Saving, restoring, and replaying the next cue/pulse sequence produced exact zero
state and response deltas.

The symbolic two-cue uniform control reached paired retention `0.55875155` and
unpaired retention `1.0`. Its similar learning is expected from the deliberately
simple common update rule. The probe establishes cue-specific causal history and
exact persistence on real neuron identities and measured KC→MBON11 weights. It
does not establish anatomical superiority, biological parameter values, or a
behavioral meaning for the synthetic modulator.

## Joined full-graph assay

`scripts/assay_mushroom_bridge.py --build-bridge` constructs the bridge from the
memory-mapped canonical graph and compares every selected coefficient against
the original sparse matrix. Without that flag, the script launches isolated
plastic and frozen Metal circuits. It never connects to or mutates a live
service.

The joined assay uses three equal-state residents: one receives pulses after cue
A, one after cue B, and one receives no modulation. Cues A and B are explicit
engineered combinations of retinal-v1 odor and retinal channels. All residents
receive the same cue history. The bridge gathers the 3,623 KC rates generated by
the complete 25,563,197-edge recurrent graph; it does not replace them with cue
masks.

The final assay ran 20 counterbalanced trials with the native SIMD kernel in
35.58 seconds alongside the existing live Metal services. The receipt is
`runs/mushroom-fullgraph-assay-v4/assay.json`, SHA-256
`0b37adb129593145657134dfe1b45af547482439c4a0ff6c39d87d0c78a6cccf`.
The report and four exact native snapshots are mirrored under
`/tank/chreatures/runs/mushroom-plasticity/fullgraph-bridge-v1/`.
The frozen cue responses showed that the actual KC population distinguishes the
two engineered inputs while remaining broad: cosine similarity `0.950446`, mean
absolute rate difference `0.00527171`, and RMS difference `0.00701686`.

The plastic history produced maximum absolute recurrent correction
`0.000411199`. Relative to the resident with identical sensory history and no
modulation, the paired cue-A resident's left/right MBON rates changed by
`[-0.000141953, -0.000110504]`; the paired cue-B resident changed by
`[-0.000143651, -0.000198509]`. Cross-history differences were smaller and
side-dependent: `[-0.000002509, 0.000040565]` for cue A and
`[-0.000004804, -0.000061371]` for cue B. This broad generalization is consistent
with the dense, overlapping actual KC activity and the engineered eligibility
rule. It does not establish odor-level biological specificity.

The separately frozen full-graph circuit produced zero correction and zero
between-resident MBON differences. Saving the complete native circuit plus
private plasticity and pending one-step correction, restoring, and replaying the
same request produced bit-exact selected rates, 384 readouts, corrections, and
private state in both plastic and frozen modes.
