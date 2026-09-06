# Population campaign coordinator

`scripts/population_campaign.py` turns the native population search state into
authenticated, hardware-sized evaluator assignments. Python performs artifact
validation, scheduling, and physical metric reduction. Genome variation,
parent selection, transfer eligibility, archive cells, and retention remain in
`population-core`.

## Inputs and initialization

`init` accepts an encoded `EmbodiedTrainingProfile.to_value()` at profile
version 6 and a file-executed population-v5 controller NPZ. It strictly requires
format `chreatures-native-developmental-resident-population-v5`, metadata
version 5, and execution `developmental-resident-native-population-v5`. It
verifies the profile, every registered native environment record, the controller
file and internal artifact identities, graph and port identities, organism
interface, adapter-bank identity/count/rank, and biosphere birth source. The
controller version does not change the body-facing organism-interface v4
contract: 4,459 observations, 12 physiology fields, and 12 actions. The exact
encoded profile is copied into the campaign directory. The controller remains
an external content-addressed file because its bytes can be large.

```bash
.venv/bin/python scripts/population_campaign.py init \
  --profile /tank/chreatures/campaign-inputs/profile-v6.json \
  --controller /tank/chreatures/campaign-inputs/developmental-resident-population-v5.npz \
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
  --candidate-waves 2
```

One candidate wave asks for exactly one resident population for every pinned
environment. The resulting worlds are chunked to at most four physical worlds
per evaluator invocation. Each assignment contains only the native environment
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
