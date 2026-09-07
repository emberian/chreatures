# Fresh embodied recording evidence

The current preparation command accepts a fresh canonical population birth,
its actual current v8 world checkpoint, and a public v4 recording/private body
binding. It does not accept migration receipts or source-life checkpoints.
Archived migration evidence remains in Git history and its original ledger.

Required local inputs are the intact preparation directory (`receipt.json`,
`profile.json`, `founding-bank.json`, `founder-assignments.json`, `search.json`),
the canonical birth export including `neural/`, the recording JSON, its private
binding, and a coherent checkpoint at or after the final recorded tick. Keep
these immutable while preparing evidence. The checkpoint must have no execution
migrations and must contain the birth manifest and physical body order that the
recording's private binding names. A tick-zero checkpoint is not required:
founder provenance comes from the authenticated canonical export and actual
presence comes from the observed checkpoint.

Run `scripts/prepare_living_recording_evidence.py` with `--preparation`,
`--birth-export`, `--recording`, `--binding`, `--checkpoint`, `--ledger`,
`--output-batch`, and `--output-link`. The ledger path may be absent. The command
returns the derived `campaign_id`, `description`, and birth batch path. Its
campaign identity includes the preparation receipt, canonical birth receipt and
actual world identity, so it cannot attach new mechanisms to an old campaign.

Apply that batch using `scripts/build_population_weave.py --ledger ... --batch
... --campaign-id ... --description ...`, using the returned identity and
description verbatim. This existing adapter validates the typed records and
builds/reloads the native Universal Weave before publishing the ledger. Then run
`scripts/link_living_recording.py --ledger ... --recording ... --public-transport
... --link ... --output ...`, and apply the resulting recording batch with
`build_population_weave.py` again. The evidence-link format remains v2 because
its body-to-life-checkpoint mapping did not change; the actual recording must be
v4. Its top-level event chain and each frame's half-open `event_range` are
validated by the current population-evidence adapter.

Preparation authenticates the native search state, bank and assignment file
receipts, every canonical birth output and phenotype file, the checkpoint state,
and the recording's private binding and engine/controller/graph identities.
It initializes the existing population-run, descriptor-epoch and probe-panel
record types from the actual fresh search configuration, then links the native
genome/environment ancestry, founder births and observed life checkpoints.
Search pending assignments remain generation provenance; they do not create
life evaluations. Offspring need an actual captured hatching-parent event or an
already authenticated birth in the same ledger. Repeated preparation is
idempotent, and later checkpoints extend each existing life without branching.

Law-fit associations are optional explicit references to completed fits already
in this campaign's ledger. The preparation command fabricates no events,
evaluations, competence, or associations to old campaign results. All private
bindings, checkpoints and provenance blobs stay outside public site assets.

Implementation has passed its Python startup/compile check. The actual current
v4 recording-to-native-Weave integration will be run when the new recording and
matching coherent checkpoint are available.
