# Population campaign coordinator

`scripts/population_campaign.py` turns the native population search state into
authenticated, hardware-sized evaluator assignments. Python performs artifact
validation, scheduling, and physical metric reduction. Genome variation,
parent selection, transfer eligibility, archive cells, and retention remain in
`population-core`.

## Inputs and initialization

`init` accepts an encoded `EmbodiedTrainingProfile.to_value()` at profile
version 7 and a file-executed population-v6 controller NPZ. It strictly requires
format `chreatures-native-developmental-resident-population-v6`, metadata
version 6, and execution `developmental-resident-native-population-v6`. It
verifies the profile, every registered native environment record, the controller
file and internal artifact identities, graph and port identities, organism
interface, adapter-bank identity/count/rank, and biosphere birth source. The
controller version does not change the body-facing organism-interface v4
contract: 4,459 observations, 12 physiology fields, and 12 actions. The exact
encoded profile is copied into the campaign directory. The controller remains
an external content-addressed file because its bytes can be large.

Profile v7 separates semantic identity from transport location.
`profile.sha256` authenticates the physical configuration and every source
asset SHA-256, while `to_value()["locators"]` carries the absolute paths used by
that host. Rebinding locators through `EmbodiedTrainingProfile.from_value(...,
locators=...)` requires the exact same logical keys and verifies every relocated
file before use. Campaign manifests retain the encoded profile file hash and
source path separately from the semantic profile hash.

The compact relocation receipt at
`data/training/profile-v7-relocation.receipt.json` records all 37 source hashes,
the two distinct locator-manifest identities, and the exact regenerated
environment outputs. Reproduce it without retaining the copied source tree:

```bash
.venv/bin/python scripts/probe_profile_relocation.py \
  --expect data/training/profile-v7-relocation.receipt.json
```

```bash
.venv/bin/python scripts/population_campaign.py init \
  --profile /tank/chreatures/campaign-inputs/profile-v7.json \
  --controller /tank/chreatures/campaign-inputs/developmental-resident-reciprocal-v6.npz \
  --population-response-artifact /tank/chreatures/campaign-inputs/population_response_bank.json \
  --seed 20260917 \
  --output /tank/chreatures/runs/population/wave-001
```

Initialization is staged in a sibling directory and published only after all
environments are registered. The campaign stores the native state, copied
profile, immutable configuration identities, and coordinator progress.

## Planning

```bash
.venv/bin/python scripts/population_campaign.py plan \
  --output /tank/chreatures/runs/population/wave-001 \
  --worlds-per-batch 4 \
  --candidate-waves 2 \
  --selection evolve
```

One candidate wave asks for exactly one resident population for every pinned
environment. The resulting worlds are chunked according to `--worlds-per-batch`
(1–64), so larger hosts can advance more independent worlds in parallel. This
controls physical processes, not residents per world; choose it against actual
CPU and shared-memory capacity. The full native ask remains limited to 4,096
candidate/environment assignments. Each assignment contains only the native environment
SHA/selector/seed and full authenticated candidate genomes. The environment SHA
is also its `world_id`; assignment-file identity makes repeated physical lives
distinct.

Before native `ask`, the coordinator durably records a transaction containing
the prior pending set and requested count. A restart with the same arguments
reconstructs the plan from newly pending native candidates. It refuses to ask
again or plan while an earlier batch is outstanding. One transaction is bounded
by the native 4096-candidate ask limit.

Evaluator launch is intentionally separate. Use the copied campaign profile,
the original controller whose file SHA is pinned in `campaign.json`, and one
generated `plans/plan-NNNN/batch-NNNN.json` assignment.

Pass the same `--population-response-artifact /path/to/population_response_bank.json`
to campaign `init` and to `evaluate_population.py` to use the shared three-law GAM
bank. Both boundaries pin its exact bytes and feature contract. Omission is a
separate bank-free condition. Ingestion authenticates the evaluator's sibling
`identity.json` and rejects a different bank, controller or profile; response
models cannot silently change midway through a search campaign.

## Ingestion

```bash
.venv/bin/python scripts/population_campaign.py ingest \
  --output /tank/chreatures/runs/population/wave-001 \
  --result /tank/chreatures/evaluations/batch-0000/result.json
```

`ingest` accepts terminal evaluator success or failure documents. It validates
the document content hash, assignment file hash, and exact assigned
candidate/world/seed population before calling native `tell` once per life.
The evaluation identity contains the physical life ID, evaluation seed,
committed ticks, and authenticated trajectory or failure-trace SHA. Repeating
the same source or resuming after a partial tell is idempotent.

The descriptor recipe uses physical trajectory summaries only:

- mean thrust;
- visited spatial cells divided by the declared 256-cell scale;
- elevation range divided by that environment's height;
- signal activity per valid physical tick;
- allocated conserved mass per model second.

The initial quality is `0.5 * mean_energy + 0.3 * energy_delta - 0.2 *
mean_effort`. It is a versioned bounded search score. It is not ecological
fitness, feeding competence, or evidence that any behavior is useful. Failures,
including startup failures at zero committed ticks, remain terminal native
records with no descriptor, cell, or quality.

Planning first asks the native archive for complete, balanced transfer waves:
retained genomes that have not yet been evaluated in any registered environment
are assigned once to every environment. Native mutation/recombination fills the
remaining resident slots. This makes cross-environment direct evaluation precede
any later fine-tuning eligibility without moving selection into Python.

Current plans require an explicit selection mode. `--selection evolve` performs
those unseen transfers before native mutation. Once complete physical histories
populate the paired archive, `--selection challenge` invokes the native history
scheduler and emits complete environment waves with a recorded transfer or
repeat phase. It does not mutate selected candidates, and infrastructure failures
never count as environment difficulty.

`environment-frontier --output CAMPAIGN` reports authenticated environment
ancestry with physical coverage and difficulty evidence. The optional
`register-challenge-scores --scores ARTIFACT` command installs bounded analyst
pair scores after authenticating every candidate/environment key. Such scores
only order eligible challenges; physical metrics still determine archive quality
and retention.

An infrastructure-failed pair remains blocked from automatic transfer and
challenge scheduling. After the underlying execution fault is diagnosed, the
coordinator can explicitly clear that one pair with
`authorize-infrastructure-retry --candidate SHA --environment SHA`; the retained
failure record remains in its history.

## Separate Torch training lineage

`build_population_training_plan.py` constructs a matched Torch
training cohort without asking or modifying the native archive. It authenticates
the semantic identity of the trainer-generated profile-v7 envelope, the v5
bootstrap and safe identity receipt, and a
prior matched-cohort plan whose physical and neural loci are copied exactly.
Controller/profile bindings are rebased, parents remain empty, and adapter rows
are balanced across the cohort dimensions carried by those authenticated inputs.
Profile host locators do not enter candidate or plan identity. The result
describes an engineered matched research cohort rather than offspring or
continued individuals.

```bash
.venv/bin/python scripts/build_population_training_plan.py \
  --profile /path/to/encoded-profile.json \
  --bootstrap /path/to/rich-worker-population-v5-cold.pt \
  --bootstrap-identity /path/to/rich-worker-population-v5-cold.identity.json \
  --port-bundle /path/to/retinal-v2-maps.npz \
  --source-plan /path/to/prior-matched-candidates32.json \
  --source-freeze SOURCE_REVISION \
  --output /path/to/torch-population-v5-candidates32.json \
  --receipt /path/to/torch-population-v5-candidates32.receipt.json
```

Both output paths must be absent. Publication is atomic. The compact receipt
pins the plan, source plan, recipe, controller, profile, all candidate hashes,
and adapter-row counts. Private state and optimizer state begin fresh.
