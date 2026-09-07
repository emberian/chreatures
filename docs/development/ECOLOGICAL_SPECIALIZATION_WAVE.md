# Ecological specialization and connected material regions

Approved after the reciprocal ecology wave at `8ea41b3`. Build the coupled
mechanisms in a substantial batch, then run new research worlds. Existing lives,
including the paused tick-9170 continuation, retain their frozen engines.

## Coupled objective

Make material redistribution, physiological specialization, constructed space,
and acquired motor sequences change one another's consequences. There are no
producer, predator, builder or social-role labels in a controller. Genotypes
specify capacities and response laws; experience supplies private acquired state.
The full MaleCNS scaffold and body-local sensory boundary remain explicit.

## Material regions

All chemical inventory remains in MetabolicWeb. Named regional stores are
ordinary compartment rows, never duplicate inventories. Finite physical packets
crossing declared habitat exit faces transfer their complete contents into the
assigned regional row and retire through the existing atomic material/topology
transaction. A failed partial transfer must not leave a falling active remnant.
The exit is recorded as a physical-to-regional transfer, including the cause,
face and actual quantities. Do not widen field sanity bounds or clamp poses.
Non-material resident escape remains a separate physical outcome.

A native batched network proposes finite transfers between regional rows using
concentration, declared conductance, capacities and physical route accessibility.
The same pre-state supplies competing outflows. Regional outlets return finite
contents to actual dormant packet slots at physically checked positions; blocked
outlets retain their contents. No source silently refills. Generated regions have
spatially distinct outlets and material pathways; accumulated resources remain
available after save/restore. Construction can alter path accessibility through
actual physical geometry. Configurations, flow schedules and private network
state are checkpointed. Region labels/coordinates do not enter resident inputs.

## Physiological specialization

Native, budget-constrained enzyme allocation can respond to local substrates
and internal energy under inherited response parameters. Adjustments have
explicit time constants and ATP costs, bounded expression and a shared finite
enzyme budget. The reaction program and material inventory remain authoritative.
This is synthetic physiological regulation, not a claim of measured fly gene
regulation. Offspring receive baseline rules, not an adult's acclimated state.

## Acquired motor sequences

Extend the current native resident with a bounded private library of actually
executed action suffixes and their sensory contexts/consequences. The existing
recurrent predictor evaluates usable suffix candidates alongside current local
control proposals. Deliver the first action and reconsider after sensation.
Imagined rollouts never enter experienced memory. Preserve previous-action
accounting, replacement generations, RNG, private learning and full restoration.
The current native artifact advances to v7; no old execution fallback is added.

## GAM and environmental search

Fit a genotype-by-environment transfer atlas from completed physical campaigns,
with explicit independent units, support and held-out groups. Native GAM fits
should identify where resources, morphology and metabolic capacities interact,
and guide bounded experiment selection. Analyst geometry stays outside the
organism. A fitted association does not establish an intervention effect.
Retain physical outcome vectors and distinct viable niches, not just one winner.

## Ownership and integration

Root owns Biosphere/runtime integration, cross-lane contracts, source revisions,
joined campaigns, publication and diary. Region lane owns native regional matter,
its thin host adapter and material exit transactions. Metabolism lane owns native
regulation and metabolic binding. Environment/growth lane owns spatial region
construction and growth/route coupling. Cognitive lane owns acquired suffixes and
resident v7. GAM lane owns the executed transfer atlas. Compute owns fresh AMD
training and multi-world execution after integrated source is pinned. Native
world-kernels lib.rs registration is coordinated by root. Viewer work follows
actual recorded state. All work remains headless, with bulk data on hbox /tank.

## Native v7 motor-suffix implementation

The v7 resident keeps 32 private suffix slots per resident. Each slot stores a
contiguous 2–8 tick sequence of actual executed 12-channel actions, the private
64-value achieved-goal context at its start, and measured per-tick
`movement_response`, `energy_cost`, and `fatigue_recovery`. Only suffixes of
length 4–8 are recalled for control. Slot replacement is reservoir sampled;
generation counters prevent an action selected from an overwritten slot from
receiving attribution. Support is an execution count, not confidence.

Each decision evaluates eight alternatives in one contiguous native predictor
batch: four current local proposals repeated through H8 and up to four recalled
experienced suffixes. A recalled suffix is scored only over its stored horizon;
native padding is compute storage and is not described as experience. Forecast
validity and goal error use that same horizon. The bounded empirical contribution
is at most 0.10 and is computed from
`tanh(movement_response - energy_cost + fatigue_recovery)`. Context recall uses
negative context RMS plus 0.25 times this empirical utility. Unavailable recalled
slots are masked before selection. The first action is proposed and the host
still supplies the actual executed action and physical consequence receipt.

The native snapshot is current-only format
`chreatures-developmental-resident-native-population-v7`. It contains slot
contents, capture history, replacement RNG, generations, execution counts, and
pending attribution. Cold construction and hatching clear all of that private
state. The immutable export declares these rules under `private_motor_suffix`;
the Python boundary validates every result shape and restores only the v7
snapshot format.

A CPU export from the actual v5 representation and recurrent predictor produced
`/tank/chreatures/scratch/cognitive-v7-integration/artifacts/developmental-resident-population-v7.npz`
(file SHA-256 `6f6a07c9042786704e43ef7fe10fca39871e8a19d25c5a34695e3e398eecf3c5`,
internal artifact identity
`ec690f724c2f920140eee145365b87d07ed392bbe4cae97225db0f79519277c3`).
A ten-boundary B1 integration using an authenticated population candidate filled
eight private slots; its next decision exposed four local and four recalled
candidates with recalled lengths 4, 5, 6, and 7. Snapshot restoration reproduced
all 63 native result fields exactly on the next decision. This verifies the
learning/proposal/scoring/restore path, not useful behavior or calibrated
forecast confidence.
