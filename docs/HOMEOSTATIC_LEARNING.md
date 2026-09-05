# Versioned finite-energy learning interface

`chreatures.homeostasis` is an opt-in reward interface for a future fresh or
continued training stage.  It does not change the completed predictive PPO,
the inherited motor artifacts, existing residents, or live worlds.  Its
purpose is to make physical reward units and causal terms inspectable before
another training run.

## What the completed objective selected

The training world advances at 20 Hz (`dt=0.05 s`), holds one policy action for
five ticks (`0.25 s`), trains in 20-second episodes, and evaluates for 40
seconds.  Its normalized physiology follows these exact rules per physical
tick:

- A contacting resident bites at most `0.34 * eat * dt` gut units.  The supplied
  eating reflex is `clip((1-gut)*(1.1-energy), 0, 1)`; eating was not learned.
- Digestion moves at most `0.032 * dt` from gut to energy with efficiency
  `0.84`.  One full gut therefore takes at least 31.25 seconds to clear, longer
  than a training episode.
- Energy loses `dt * (0.0007 + 0.0042 * effort)`.  These are fractions per
  second, not joules.  Basal depletion takes about 1,429 seconds and maximal
  basal-plus-effort depletion about 204 seconds.
- Effort is a clipped dimensionless proxy:
  `0.45|thrust| + 0.18|yaw| + 0.22|posture| + 0.15*grip + vertical-motion cost`.
  It is not measured joint work and omits gaze, signaling, idle servo work, and
  stabilizer work.
- Fatigue changes by `dt * (0.096*effort - 0.026)`.  It recovers below effort
  `0.270833`, reaches full from zero in 14.29 seconds at effort one, and takes
  38.46 seconds to recover from full at rest.

The old physical reward per tick was:

```text
12 * ((0.85 - energy_before)^2 - (0.85 - energy_after)^2)
+ 3 * nutrition * max(0, 1 - energy_after)
- 0.0002 * dt * effort
```

This objective pays immediately for ingestion and later for energy restored by
digestion.  Its explicit maximal effort charge is only `0.00001` per tick.
The symmetric energy potential also rewards energy loss whenever energy is
above `0.85`.  Episodes are short relative to digestion, recovery, and energy
depletion, so resets truncate much of the physiological consequence.  PPO
adds prediction-learning progress separately, at most `0.004` per macro step.

The final held-out deterministic evaluation shows the resulting tradeoff:

| Policy | Nutrition | Distance | Mean effort | Contacts | Final energy | Physical reward |
|---|---:|---:|---:|---:|---:|---:|
| fixed initial | 0.371626 | 15.3194 | 0.002557 | 2,523 | 0.908322 | -0.390715 |
| learned 20k | 2.858586 | 31.5761 | 0.255938 | 7,277 | 0.900425 | -0.639589 |
| learned, neural features silenced | 2.173134 | 25.6784 | 0.163814 | 8,306 | 0.894865 | -0.318102 |

Relative to initialization, the learned actor ingested 7.69 times as much,
traveled 2.06 times as far, and used 100.1 times the effort.  Silencing reduced
learned ingestion by 24.0%, supporting sensor-dependent ingestion in this
specific evaluation, but the learned bodily return was worse.

The training trace gives a more precise diagnosis than “more locomotion.”  In
successive 4,000-step bins, mean effort rose from `0.3280` to `0.3802` and mean
contacts from `120.53` to `139.72`, while distance per macro remained near
`1.18`.  From 8,000 to 20,000 steps, held-out ingestion rose 280.4% while
distance fell 5.7%.  Training selected a high-output/contact tendency, while
later ingestion improvement did not come from greater displacement alone.
Aggregate summaries do not retain ingestion timing or per-resident energy
transitions, so the final drive and ingestion reward terms cannot be exactly
reconstructed.  The learned evaluation's explicit effort penalty is about
`-0.02457`, too small by itself to explain its total reward.

## Finite-energy objective v1

`FiniteEnergyObjective` treats `energy + 0.84*gut` as an assimilable reserve.
Its state potential, measured in normalized body-energy fractions, is the
negative sum of:

- a smooth one-sided reserve shortfall around `0.85` with temperature `0.08`;
- `0.08 * fatigue²`;
- `0.08 * max(gut - 0.55, 0)²` for gut overload.

The smooth shortfall is monotonic in reserve, so burning energy never improves
it above the target.  Gut enters reserve immediately and digestion conserves
`energy + 0.84*gut` away from clipping; actual ingestion is therefore credited
once through state change.  `nutrition` is retained as a causal audit term but
does not receive a second bonus.  A separate cost charges 25% of the world's
known action-dependent energy rate, `0.25 * 0.0042 * effort * dt`, so high
effort remains costly even while reserve is sufficient.  Potential change and
cost are multiplied by 12 reward units per energy fraction to retain an
interpretable scale.
This deliberately makes action effort visible twice below sufficiency: once
through its realized energy loss and once through the 25% regularizer.  The
declared extra weight is a behavioral efficiency preference, not a second
physical-energy claim, and should be ablated before it becomes a default.

```python
from chreatures.homeostasis import FiniteEnergyConfig, FiniteEnergyObjective

config = FiniteEnergyConfig()
objective = FiniteEnergyObjective(config)
reward, terms = objective.transition(
    {"energy": energy0, "gut": gut0, "fatigue": fatigue0},
    {"energy": energy1, "gut": gut1, "fatigue": fatigue1},
    nutrition=actual_nutrition,
    effort=world_effort,
    dt=0.05,
)
manifest_value = config.to_value()  # versioned, canonical SHA-256
```

Inputs may be scalar, batched dictionaries, or arrays ending in
`[energy, gut, fatigue]`.  Unknown physiology fields are rejected.  The API
returns a resident reward array and named arrays for before/after reserve,
shortfall, fatigue and gut costs, potential delta, effort cost, observed
nutrition, a diagnostic hunger gate, and final reward.  Prediction progress
must remain a separately logged training term so physical reward stays
auditable.  No object kind, position, identity, or goal enters this interface.

This is a proposed objective, not a demonstrated improvement.  Before adopting
it, a new stage should log every component per resident, preserve continuing
physiology or use episodes longer than digestion and fatigue recovery, and
compare ingestion, stored reserve, fatigue, effort, distance, and return on
matched held-out worlds.  Mechanical-work accounting would require additional
world instrumentation; this version honestly uses the existing effort proxy.
