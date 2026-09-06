# Population search and evidence sources

The population wave uses two bounded archives. A multi-member MAP-Elites archive
retains organism candidates by versioned physical descriptors and whole-life
quality. A separate POET-like archive retains physical environments and their
ancestry. Admission consumes a complete physical evaluation; a process or world
failure is a terminal result, not a missing row. Native GAM fits remain
descriptive evidence and become inheritable only through a later genome birth.

## Mechanisms adopted

- [MAP-Elites](https://arxiv.org/abs/1504.04909) supplies the descriptor-cell and
  local competition mechanism. The author-maintained QDax implementation is
  pinned at commit `46b8feecae26d79414ce7a1ce731e5cfbeb9837d`; the operative code is
  `qdax/core/map_elites.py` (`MAPElites.ask`, `tell`, and `update`) and
  `qdax/core/containers/mapelites_repertoire.py` (`MapElitesRepertoire.add`).
  Chreatures keeps a bounded depth greater than one per cell because a single
  apparent elite is fragile under physical-evaluation noise.
- [Deep Grid](https://arxiv.org/abs/2006.14253) motivates retaining multiple
  independently evaluated candidates per niche. Its robustness result assumes
  invariant noise; Chreatures therefore records world seeds and does not treat
  the method as a cure for a changing environment distribution.
- [AURORA](https://arxiv.org/abs/2106.05648) supplies learned descriptors from
  sensory histories. The author implementation is pinned at
  `167d743454fa16663e933feb5824748fabd69af9`; the corresponding QDax paths are
  `qdax/core/aurora.py` and `qdax/core/containers/unstructured_repertoire.py` at
  the QDax pin above. A changed encoder starts a new immutable descriptor epoch.
  Raw evaluation trajectories remain content-addressed so a new epoch can be
  computed without rewriting old cell assignments.
- [Enhanced POET](https://proceedings.mlr.press/v119/wang20l.html) supplies
  environment mutation, a minimal criterion, novelty, and direct transfer before
  tuning. The author code is pinned at
  `8669a17e6958f80cd547b2de61c51d4518c833d9`; relevant methods are
  `poet_distributed/poet_algo.py::transfer`, `pass_mc`, `get_child_list`, and
  `adjust_envs_niches`, plus
  `poet_distributed/novelty.py::compute_novelty_vs_archive`. PATA-EC coordinates
  depend on the policy panel, so Chreatures freezes and hashes that panel within
  each descriptor epoch.
- [PAIRED](https://proceedings.neurips.cc/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html)
  identifies protagonist-antagonist regret as an environment-generation signal.
  Its coupled three-player optimization is not part of the first population
  campaign; the evidence schema can retain a future regret recipe without
  claiming that it ran.
- [ACCEL](https://proceedings.mlr.press/v162/parker-holder22a.html) combines
  high-regret replay with environment mutation. It is useful when training one
  generalist, while the current campaign preserves distinct genome lineages, so
  it is not the archive owner in this wave.

The native evidence carrier is
[Universal Weave](https://github.com/transkatgirl/universal-weave), pinned at
commit `7a5a0dabb94885e44ad8a6c4355c015d7f38020f`. The adapter uses the upstream
`IndependentWeave` insertion, validation, topological ordering, serde
serialization, reload, and equality paths. Stable node IDs are the first 128 bits
of SHA-256 over the complete source ID; the complete source ID remains in every
node and collisions are rejected.

## Population evidence v1

`chreatures.population_evidence` validates batches before the Rust adapter repeats
the structural checks and builds the native Weave. `fields.parent_roles` is an
object keyed by every parent source ID. Its keys must exactly equal `parent_ids`.
This makes these edges distinct:

- `genome_parent` joins zero, one, or two immutable genome artifacts. Genome
  records reject checkpoint, RNG, optimizer, memory, history, rates, and other
  private-state fields. An optional `inherited_law_fit` edge can name a completed
  prior GAM fit; this records an explicit later genome birth rather than changing
  the evaluated resident in place. Candidate fields retain the current graph,
  ports, base controller, developmental base, population-adapter bank, organism
  interface, and adapter shape identities from native `population-core`.
- `environment_parent` joins immutable topology/resource/profile artifacts.
- `physical_parent_birth` records embodied reproduction, while
  `experimental_initialization` births have no physical-parent edge.
- `life_continuation` is a single non-branching chain from birth through zero or
  more checkpoints to one terminal evaluation. A terminal evaluation ID can
  appear only once as `evaluation_completed` or `evaluation_failed`.
- A pre-allocation `evaluation_failed` has `allocation_status: not_allocated`
  and a `planned_campaign` edge in place of `life_continuation`. It retains the
  planned life identity without inventing a birth.

Descriptor epochs form an immediate predecessor chain. Each
`environment_probe_panel` belongs to one epoch and carries the ordered, unique
policy-artifact hashes. Environments and evaluations identify that frozen panel.
Archive decisions point to terminal evaluations, including explicit rejection of
failed candidates. Every terminal evaluation has exactly one archive decision.
Transfer trials point to separate source and target terminal
evaluations and assert that the target result was measured before fine-tuning.
Completed and failed GAM fits are both retained; only a completed fit may carry a
`gam_law` blob.

Large genomes, environment specifications, checkpoints, evaluation results,
search snapshots, trajectories, fitted laws, and arrays remain external. Their
ledger entries use `urn:sha256:<digest>` blob references with sizes and media
types where known. Weave publication is read-only evidence processing and cannot
advance, merge, or restore a life.

A terminal evaluation carries both `evaluation_result` and `evaluation_trace`
blob roles. The trace is the canonical per-life receipt hashed by the evaluator:
it joins the planned life fields to the verified trajectory snapshot and metrics
for a completion, or to the partial trajectory, checkpoint, and traceback
receipts for a failure. A failure before any committed tick remains terminal with
`committed_ticks: 0`, no birth, and no `descriptor`, `cell`, or `quality`. A
completed evaluation must have at least one committed tick.

Campaign workers emit append-only batch files with format
`chreatures-population-evidence-batch-v1`. A build applies each `batch_id` once,
validates the combined graph, invokes the native adapter, verifies exact
serialize/reload equality, and only then replaces the compact ledger and output
receipts:

```sh
python scripts/build_population_weave.py \
  --ledger runs/population/evidence.json \
  --campaign-id population-wave-1 \
  --description "Population wave 1 physical evidence" \
  --batch runs/population/batches/000001.json

python scripts/build_population_weave.py \
  --ledger runs/population/evidence.json \
  --batch runs/population/batches/000002.json \
  --population-state runs/population/search-state.json
```

`--population-state` reconciles every native terminal evaluation and archive
decision in that durable search snapshot against the combined ledger. The build
fails if a successful or failed candidate was omitted, relabeled, attached to a
different genome/environment, or given a different admission result.

The native population search state has no wall clock or life/checkpoint ownership.
`record_population_campaign.py` joins it to sealed evaluator outputs and their
checkpoint directories. Pass the campaign manifest as `--probe-panel`; the
recorder extracts and authenticates the exact panel whose content hash is frozen
in the native search config:

```sh
python scripts/record_population_campaign.py \
  --search-state runs/population/wave-1/search.json \
  --probe-panel runs/population/wave-1/campaign.json \
  --evaluation-run runs/population/evaluations/batch-000000 \
  --ledger runs/population/wave-1/evidence.json \
  --campaign-id population-wave-1 \
  --description "Population wave 1 physical evidence"
```

The recorder verifies the native state, evaluator identity and terminal content
hashes, the immutable failure copy, every checkpoint component, the latest
pointer, environment identity, and each per-life trace preimage. A step-zero
coupled checkpoint is the allocation proof that creates a `birth`; an identity
plan alone never does. Each content
batch is applied at most once, and rerunning the same inputs reports `unchanged`.
The native adapter then serializes and reloads the full graph before the ledger is
replaced. Missing life provenance is never inferred from archive state.
