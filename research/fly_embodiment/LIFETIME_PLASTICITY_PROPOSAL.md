# First MaleCNS lifetime-plasticity substrate

## Recommendation

Use the bilateral `MBON11` KC-input edge set as the first private lifetime-
plasticity mask. It contains 4,184 measured directed KC→MBON edges, 41,460
measured synapses, 3,623 connected KCs, and the two exact target rows 655 and
1306. The corresponding instances are annotated `MBON11(y1pedc>a/B)_L/R`.
MaleCNS also contains the bilateral dopamine-labelled `PPL101(y1ped)_L/R`
neurons and their measured connectivity around these targets. This is the
smallest well-audited substrate in the repository with both connectome support
and direct physiological prior art for timing-dependent KC→MBON plasticity.

The recommendation does not assign reward, punishment, valence, receptor sign,
or behavioral output to either side. Root should initially permit only bounded
private efficacy deltas on the selected existing edges. Measured synapse counts
and immutable baseline graph weights remain separate. Eligibility state,
dopamine traces, update bounds, decay, and the mapping from actual DAN activity
to an efficacy change are engineered dynamics requiring a later cross-backend
contract and experimental validation.

## Full measured candidate landscape

The canonical MaleCNS graph contains 4,064 annotated Kenyon cells, 97 MBON rows,
and 338 cells that are both `class == DAN` and effective-transmitter labelled
dopamine. Across every MBON target there are 61,210 measured KC→MBON directed
edges, 463,640 synapses, and 4,063 connected KCs. All 97 MBON rows receive at
least one direct edge from an annotated dopaminergic DAN: 3,159 directed edges
and 39,710 synapses in total. There are also 145,658 KC→dopaminergic-DAN edges
with 281,857 synapses.

Direct graph convergence is useful evidence that DANs, KCs, and MBONs interact.
It does not locate volume transmission or establish which KC→MBON contacts a
DAN modifies. The flat canonical edge artifact has neuron endpoints and counts,
without per-synapse neuropil coordinates. For that reason, the artifact catalogs
all 61,210 KC→MBON candidates and per-MBON direct dopamine convergence but does
not promote all of them into the first plastic mask.

Parenthetical instance strings such as `y1pedc`, `y5`, or `B'2a` are retained as
`compartment_annotation_hint`. They support inspection and later matching to an
explicit published compartment table. They are not treated as measured synapse
locations or automatically matched across DANs and MBONs.

## Machine-readable artifact

The candidate archive is stored outside runtime packs at:

`~/paperbin/chreatures/research/lifetime-plasticity-v1/male-cns-plasticity-candidates-v1.npz`

SHA-256: `aaa8f1c8c6476e4b528e7b3d11b1b3b00af36da8d84b109bd623479cd17f7081`.

It records canonical CSR edge positions, source/target rows and measured counts
for all KC→MBON candidates, direct dopaminergic-DAN→MBON edges, and KC→DAN
edges. `candidate.recommended_gamma1pedc` is the engineered Boolean proposal
over the measured candidate list. MBON and DAN tables retain exact row, body ID,
type, instance, and side annotations. Loading uses `allow_pickle=False`.

This is CPU extraction and architectural research only. It changes no graph,
resident state, controller, checkpoint, model pack, or active world.

## Literature implications

- Aso et al. 2014 anatomically define 15 MB compartments through overlapping
  KC axons, MBON dendrites, and DAN terminals. This supports compartment-scoped
  eligibility but does not map the MaleCNS flat edge list to synapse locations.
- Hige et al. 2015 report odor-specific, timing-dependent suppression at the
  γ1pedc KC→MBON pathway after PPL1-γ1pedc activation. This supports choosing
  MBON11/PPL101 as the first experiment and retaining causal event order.
- Takemura et al. 2017 show canonical KC→MBON, KC→DAN, and DAN→KC/MBON motifs,
  while noting that direct DAN synapses touch only a subset of KC→MBON sites.
  This prevents using direct-edge intersection as the plasticity mask.
- Li et al. 2020 extend connectomic analysis across all adult MB compartments
  and describe cell-type-specific DAN rules and cotransmitter complications.
  This argues against one universal dopamine sign or immediate all-MBON rollout.
- Bilz et al. 2023 find heterogeneous bouton expression of dopamine-dependent
  KC release plasticity. This argues for edge-private efficacy state and against
  forcing a single compartment scalar onto every edge indefinitely.
- The authors' hemibrain example repository demonstrates connectivity extraction
  through neuPrint/natverse tools. It supports the endpoint-based audit method;
  no code or inferred hemibrain identity was imported into this MaleCNS mask.

See `lifetime-plasticity-source-ledger.json` for exact source links, scope, and
the implementation consequence assigned to each source.
