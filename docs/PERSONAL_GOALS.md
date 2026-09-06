# Private goal-outcome associations

`native/cognitive-core/src/personal_goals.rs` is an optional next-generation
mechanism for changing achieved-goal retrieval after a resident experiences the
physical consequences of pursuing a remembered history. It is separate from
the inherited goal manager, the learned motor worker, and the per-transition
physical-consequence corrector.

The mechanism does not compute reward or inspect the world. After each
committed physical transition, its host passes the existing
`FiniteEnergyObjective.transition` reward. That objective already accounts for
the change in reserve, gut load, fatigue and actual effort. Observed nutrition
is logged by the objective but is not paid again. No object label, world
coordinate, caregiver identity, imagined outcome or extra ingestion reward
enters this module. The pinned default objective identity is
`01ae937a153a056c8cc5fa5be4d55cdfb38dbfcede4dbceb16ec33e19c5f4d00`.

## Association rule

Each resident has four private `f64` weights for each of 128 achieved-memory
slots. At selection time the feature vector is

```text
x = [1, 2*energy-1, 2*gut-1, 2*fatigue-1]
```

The selected slot accumulates the ten observed transition rewards belonging to
the ten committed physical ticks beginning at `selected_at_tick`. On the tenth
outcome,

```text
target = clamp(sum(reward) / 0.01, -1, 1)
error = target - dot(weights[resident, slot], x)
weights += 0.05 * error * x / (1 + dot(x, x))
weights = project_L2(weights, radius=4)
```

The `0.01` return scale, clipping, learning rate, feature choice and weight
radius are engineered defaults. The scale was not fitted as a calibration of
future return. Before a later manager selection, a matching slot receives

```text
logit_bias = 0.35 * tanh(dot(weights[resident, slot], current_x))
```

The bias is bounded to `[-0.35, 0.35]`. It is an associative preference term,
not a confidence, value, pleasure or suffering estimate. Base manager logits
remain inherited. A research configuration may disable updates; it still
tracks completed ten-tick receipts and skipped credit.

## Slot identity and causal credit

A slot model is keyed by both the memory window's `recorded_tick` and a positive
slot `generation`. The achieved-memory owner must increment and expose that
generation whenever reservoir insertion replaces a slot. `replace_slots`
clears all four weights for a changed identity. If replacement happens while a
copied selected goal remains active, the ten outcomes are still recorded but
the final receipt is marked `slot_was_replaced=true`, `attributed=false`, and
no update occurs. This prevents experience from being assigned to the new
memory now occupying the same integer slot.

The public Rust API uses bounded, prevalidated batches:

- `replace_slots(&[GoalSlotReplacement])` registers reservoir changes.
- `selection_biases(resident, recorded_ticks, generations, physiology)` returns
  128 identity-checked biases.
- `begin_goals(&[GoalStart])` starts one pending episode per named resident.
- `observe_rewards(&[GoalReward])` consumes the next exact physical tick and
  returns a receipt only on tick ten.
- `cancel_pending(resident)` handles resets or discontinuities without erasing
  learned slot weights.

The module has no random sampler and consumes no RNG state. The manager remains
the sole owner of stochastic goal selection.

## Private state and replay

The snapshot contains configuration and its SHA-256 identity, slot identities,
all weights, pending features/return/tick count, completion counters and the
last receipt. Dynamic `f64` values are serialized by their IEEE-754 bit
patterns, so restore is exact. Restore rejects a different objective,
learning-toggle configuration, cohort size, slot identity contract, malformed
pending horizon or out-of-bound weights. This state belongs in the resident's
private checkpoint and should not appear in ordinary public observations.

## Bounded mechanism check

The module test used two ten-transition reward histories generated directly by
the current Python `FiniteEnergyObjective.transition` with `dt=0.05` and
reported nutrition zero. Both began at physiology `[0.35,0.10,0.20]`. The
restorative history changed it by `[+0.002,+0.001,-0.001]` per tick at effort
`0.20`; the costly history changed it by `[-0.002,0,+0.002]` at effort `0.80`.
Their returns summed to `0.34104198589921` and `-0.25193826854228973`. The
native accumulator reproduced both sums within
`1e-15`, clipped them to normalized targets `+1` and `-1`, and shifted matched
retrieval biases to approximately `+0.0118321` and `-0.0118321` from the same
starting physiology. Replacing the positive slot during a later pending episode
cleared its learned weights and produced a skipped-attribution receipt. A full
snapshot/restore then reproduced the complete native state exactly.

This check establishes the declared arithmetic, causal slot binding and replay.
It does not establish that the association improves behavior; that requires the
planned matched next-generation closed-loop comparison.
