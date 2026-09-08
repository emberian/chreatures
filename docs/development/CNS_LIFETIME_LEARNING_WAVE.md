# CNS V5: private learning on measured connections

September 8 afternoon implementation contract. Root owns integration. The public
V4 release and already-running V4 training jobs keep their frozen source and
artifacts. This wave creates new lives and a new artifact/state format. It does
not migrate a living V4 brain in place or keep two production CNS implementations.

The body, retina, BODY807, context12, MOTOR92, full graph and seven neuron-state
fields stay as specified in FLY_ECOLOGY_WAVE.md. The addition is private efficacy
and eligibility on 4,184 existing KC→MBON11 edges. Their altered currents enter
the actual recurrence before its motor and cognitive readouts. No sensory-to-
action bypass, external reward number, teacher identity or prescribed behavior
enters this mechanism.

## Measured substrate and engineered interpretation

Reuse the previously audited bilateral γ1pedc selector: 3,623 connected KCs,
4,184 directed edges and 41,460 measured synapses. MBON11 target rows are 655
and 1306 (body IDs 10704 and 11402). The corresponding PPL101 references have
body IDs 11900 and 11327. Resolve their exact canonical rows from the supplied
neurons artifact. Preserve cross-side KC connections. Do not use direct DAN→KC
edge intersection to invent a volume-transmission mask.

The compartment choice and cue-before-DAN rule follow the anatomical and
physiological motivation recorded in research/fly_embodiment/
LIFETIME_PLASTICITY_PROPOSAL.md and docs/MUSHROOM_PLASTICITY.md. The side-paired
PPL101 teaching gate, rate transduction, timescale, strength and bounds below
are engineered. They assign no pleasure, punishment or motor meaning to PPL101.
This first rule implements bounded depression; it does not supply bidirectional
learning, sleep consolidation or unlimited learning capacity.

The older Python mushroom-body implementation remains a historical research
reference, not a production execution path. Its raw-rate, float64, externally
injected, one-tick-delayed correction must not be copied into V5. V5 uses its
current baseline-centered, release-gated, half-rounded graph in both substeps.

## Immutable artifact additions

CHCNS5 is the one current service format. Add these static tensors to the
authoritative contract, with their hashes in the ordinary array identity:

| Name | Type / shape | Meaning |
|---|---|---|
| plasticity.edge_positions | u32[4184] | Exact positions in canonical graph CSR, grouped by target |
| plasticity.target_ptr | u32[3] | [0,2048,4184] |
| plasticity.target_rows | u32[2] | [655,1306] |
| plasticity.dan_rows | u32[2] | Exact corresponding PPL101 rows, target order |
| plasticity.rule | f32[2,3] | eligibility_tau_s, depression_per_s, maximum_depression |

Seed each rule row with [0.8,0.25,0.8]. These are inherited fixed parameters for
this wave, not fitted physiological constants or optimizer parameters. Validate
finite tau in [0.05,5], depression in [0,1], maximum depression in [0,0.95].
Rate zero is a declared learning-disabled research control, not a second engine.

The compiler validates every selected canonical edge position, source KC and
target MBON identity against the existing selector/candidate artifact. Check
that each source uses the fast transmitter channel. Derived source rows and
baseline coefficients come from graph.col[position] and graph.weight_bits[position].
Do not re-normalize the selected edges, reuse the old bridge's float32 weights,
or add another copy of the baseline current. Copy/decode exactly the canonical
IEEE-half bits, just as for the full graph. Derived GPU buffers are permissible;
they must be created from these validated arrays.

An offline one-way artifact upgrade reads a pinned V4 service and writes a NEW
V5 service with zero initial private plastic state, explicitly recording the
source hash and added selector/rule. It does not rewrite snapshots, invent an
experienced history, or provide a V4 production loader. Freeze the small V4
reader inside that migration tool rather than importing the new V5 reader.

## Private state and chronology

Retain r,a,s,q,mDA,mOA,mHT [N,B]. Add two float32 arrays [4184,B]: efficacy
deviation d, initially 0, and eligibility e, initially 0. GPU vec4 packing may
reserve four lanes, with inactive/reset lanes handled exactly as neuron state.
All learning state is resident-private. No graph-sized per-resident weight copy
is needed (the new state is about 32.7 KiB per resident).

At EACH of the two Jacobi halfsteps, after the ordinary immutable recurrent
sum and before updating rates, add at each selected MBON target j:

```
correction_j = sum_selected_edges_to_j(
    decoded_baseline_weight * d_edge * (r_source-r0_source) * q_source)
fast_j += correction_j
```

Use the same current halfstep r and pre-finalization q as the baseline term.
Freeze d through both halfsteps. Zero d gives exact zero correction, preserving
the V4 neutral fixed point and initial baseline computation. Accumulate each
target's selected edges in canonical order; no nondeterministic edge atomics.
Metal and WGSL accumulate each target sequentially. Torch keeps that exact edge
selection/order but uses its backend's deterministic tree reduction for the two
target sums; cross-backend agreement is numerical, not bitwise. The joined
receipt must report the measured discrepancy rather than claim identical
floating-point accumulation.

After both halfsteps and the existing final adaptation/support/release update,
advance plasticity ONCE at dt=0.01, using final neural rates and release:

```
h_i = min(r0_i, 1-r0_i)
cue_e = clamp(max(0,(r_source-r0_source)/h_source)*q_source, 0, 1)
gate_j = clamp(max(0,(r_PPL101_j-r0_PPL101_j)/h_PPL101_j)*q_PPL101_j, 0, 1)
decayed_e = e * exp(-dt / eligibility_tau_j)
d_next = clamp(d - depression_per_s_j*dt*decayed_e*gate_j,
               -maximum_depression_j, 0)
e_next = max(decayed_e, cue_e)
```

Use OLD decayed eligibility for learning, then enroll the current cue. This is
a discrete cue-before-gate mechanism. Never substitute a synthetic event flag,
the aggregate target mDA field, raw absolute KC rate or an analyst's outcome.
Frozen/absent lanes advance neither state nor learning. A resident reset clears
both arrays as well as the seven neuronal fields. Torch exposes the same state
chronology, with contiguous float32 tensors and gradients through recurrent
state where training requires them; no Python per-edge loop in training.

## Backend, persistence and observatory integration

Torch/ROCm, Rust/Metal and WebGPU implement the same V5 contract. Native/Rust
owns state validation and serialization; WGSL/Metal kernels own GPU updates.
For the browser, use a two-target reduction pass beside recurrent_sum, followed
by an edge-parallel learning pass after finalize_tick. Keep graph traversal
unchanged and avoid a larger recurrent binding group when a small dedicated
group suffices. Metal may use the equivalent separate reduction or target-local
addition, preserving canonical accumulation order and per-halfstep timing.

CHWG5ST, CNSSTATE5 and CHLIVE5 identify new snapshots. Store both plastic arrays,
rule/selector identity, active/reset membership, time, seven neuronal fields,
private cognitive state and the full interacting world. Validate lengths,
finite values and per-target bounds BEFORE mutating any loaded state. A failed
GPU mutation poisons/pauses the life. Do not retry unknown advancement.

The observer may show selected-edge efficacy, eligibility and aggregate change
at the two MBON targets. It must distinguish inherited coefficients from private
changes. It does not feed annotations or archive history back into the creature.
Existing seven-field neuron inspection remains available. No new welfare label
or claim of learned behavior follows just from nonzero efficacy changes.

## One joined evaluation after implementation

Freeze a V5 model exported from an identified V4 parent, then run one substantial
cross-backend sensory chronology and a physical screen/odor/tone world. Compare
all neuronal and plastic state, private-lane independence, and exact same-runtime
checkpoint continuation. Zero learning and zero initial d must preserve the
baseline recurrence; a nonzero private d must alter only its measured-edge
current contributions before propagation. Record actual endogenous PPL/KC
activity and whether plasticity changed; no forced success if those pathways
are not sufficiently recruited by the current interfaces.

A separate, clearly marked research intervention can establish the equation's
cue-before-gate behavior on exact selected CNS cells. Such an assay must not be
mistaken for sensory learning or installed as a resident input. Current physical
training and the unified Rust/MuJoCo host continue alongside this build batch.
No feeding or walking milestone is a prerequisite for building the learning
mechanism; neither is this mechanism a claim that motor competence is solved.
