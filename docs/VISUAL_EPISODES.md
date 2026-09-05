# Delayed native visual episodes

`VisualEpisodeMemory` is a private, bounded NumPy memory for slow native body
vision. It never receives world coordinates, entity identities, object labels,
or an archive. Each observation remains a historical capture even when it is
the newest native feature available.

## Runtime interval contract

At an inherited motor macro boundary `T`, render the first body view. Retain
the exact five transitions `T -> T+1` through `T+4 -> T+5`, then render the
second view at `T+5`. Submit both frames as one native embedding cohort with a
fixed delivery tick `T+25`. If inference has not completed at `T+25`, the
runtime reports `awaiting` and holds model time; it does not change the delivery
tick.

Pass the two completed observations to:

```python
receipt = memory.bind_interval(start_capture, end_capture, steps)
```

Each capture has exactly these fields:

```python
{
    "feature": [...],              # 960 float32-compatible native values
    "feature_sha256": "...",       # SHA-256 of little-endian float32 bytes
    "frame_sha256": "...",         # source PNG hash
    "response_sha256": "...",      # canonical completed cohort response hash
    "capture_tick": T,
    "delivery_tick": T + 25,
    "model_time": 12.5,
    "model_revision": "...",
    "pooling_version": "smolvlm2-native-tiles-mean-v1",
}
```

The pair shares its response hash, delivery tick, model revision, and pooling
contract. Delivery must be later than capture. Each of the five step dictionaries
has exactly:

```python
{
    "from_tick": T + i,
    "to_tick": T + i + 1,
    "dt": 0.05,
    "action": {                     # actual emitted command; no reflex `eat`
        "thrust": ..., "yaw": ..., "gaze_pitch": ..., "grip": ...,
        "signal_low": ..., "signal_mid": ..., "signal_high": ...,
        "posture": ...,
    },
    "outcome": {                   # raw PhysicsWorld.advance result
        "nutrition": ..., "contact": ..., "distance": ..., "effort": ...,
    },
    "physiology_before": {
        "energy": ..., "gut": ..., "fatigue": ..., "speed": ...,
        "angular_velocity": ..., "support": ...,
    },
    "physiology_after": { ...same six fields... },
}
```

The memory retains all five actions. It marks an interval comparable to one
continuous candidate only when the per-channel action span is within the
configured stability tolerance. A sequence of changing commands is preserved
but cannot masquerade as one mean action.

Physics `effort` is a rate. The record therefore keeps the raw per-tick values
and derives separate aggregates:

- `outcome_sum`: sum of raw values, useful for nutrition and distance;
- `outcome_integral`: `sum(outcome * dt)`, used for effort;
- `outcome_maximum`: maximum raw value, used for contact;
- `contextual_physical_outcome`: nutrition from `outcome_sum`, effort from
  `outcome_integral`, and contact from `outcome_maximum`.

This matches `LivingMotorOrgan` and `ContextualMotorRefiner`; raw effort sum is
never passed to bodily utility.

## Recall and controller evidence

```python
result = memory.recall(
    delayed_capture,
    candidate_action_dicts,
    current_tick=current_tick,
)
```

Recall compares the raw 960-value query against past interval starts using
cosine similarity and native RMS distance. It reports per-channel absolute
action errors, mean/max action error, recorded action span, capture ages, and
explicit rejection reasons. Predictions are weighted empirical `.25`-second
physiology deltas and physical outcomes. A single neighbor has zero empirical
dispersion but is not described as low uncertainty.

For the existing optional motor seam:

```python
candidate_evidence = memory.contextual_candidate_evidence(
    delayed_capture,
    current_tick=current_tick,
    current_physiology=current_physiology,
    utility_config=refiner.config,
)
```

The callback accepts the immutable `MOTOR_ACTIONS8` tuples supplied by
`ContextualMotorRefiner`. It computes the refiner's same energy/gut/fatigue
drive reduction plus nutrition and integrated effort utility. Its score is
bounded by `.08` before the refiner's own `.12` external cap, and is reduced by
query age, evidence age, visual support, absolute action support, and sample
count. Unsupported candidates receive zero correction with diagnostics. The
callback cannot be created until at least one completed native interval exists.

`snapshot()` contains the private raw vectors, exact steps, encoder identity,
bounded records, and a canonical state hash. `restore()` revalidates capture
hashes and reconstructs every derived aggregate rather than trusting saved
summaries.

The current snapshot and record schema is v2 with outcome contract
`physics-outcome-v2-effort-rate-integrated`. Restore accepts the initial v1
snapshot, validates its canonical hash, raw feature hashes, interval identity,
and five retained steps, then recomputes every summary under v2. A subsequently
saved snapshot records the source snapshot hash and migration method. This is a
checkpoint-boundary migration; a running process is not mutated in place.
