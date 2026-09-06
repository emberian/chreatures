# Joined sensorimotor development

`scripts/develop_sensorimotor_worker.py` is a research runner for online 20 Hz
development in shared chemical 3-D worlds. It combines the canonical full
MaleCNS circuit, 351 permitted senses, six local physiology values, a private
continuous worker GRU, executed-action efference copy, and the existing finite
energy objective. It does not modify production runtime code or existing
residents.

The frozen v2 autoencoder supplies 64-dimensional keys for four-frame windows
that each resident has already experienced. Each resident keeps its own bounded
128-entry reservoir, achievement timestamps, RNG, recurrent state, and current goal. Goals
are sticky for ten physics ticks. No other resident's history, future frame,
position, object label, or identity is available to the policy.

At a goal boundary, `SlowGoalManager` encodes the worker hidden state after the
current observation, all 384 canonical neural readouts, and six physiology
values. Its learned 64-dimensional query scores only that resident's valid
achieved keys. The initial final query layer is zero, so selection begins
uniformly; a small trainable gain and the query weights then learn through the
manager PPO objective. The selected goal applies to the following action and
ten-tick return. The preceding transition is always credited against the goal
that was fixed before that action.

The worker horizon input is the normalized remaining time in the current
ten-tick attempt, `log1p(max(1, 10 - goal_age_ticks)) / log(41)`. It is distinct
from the achieved memory's age, which remains checkpoint provenance. Action and
manager choices use each resident's saved NumPy RNG; worker inverse-CDF sampling
is vectorized as one batched GPU operation from a `[B,12]` uniform array.

The 20 Hz worker samples its existing categorical signed axes and joint hurdle
axes. PPO replays complete time-major rollout chunks from their exact initial
private hidden states and updates the inherited shared worker and critic only at
rollout boundaries. The slow manager has a separate ten-tick return and value
estimate but shares the declared optimizer boundary. The physical reward comes
from `FiniteEnergyObjective`. An engineered auxiliary term, default coefficient
`0.01`, rewards reduction in Euclidean distance between the frozen current
history code and frozen sticky-goal code. This is logged separately and is not
claimed to preserve the optimal policy under discounting.

Periodic coherent checkpoints include world snapshots, complete canonical
neural arrays, shared model and optimizer state, private GRU and action state,
private goal reservoirs and RNGs, incomplete manager returns, and an explicitly
empty pending 20 Hz rollout. They pin the graph, ports, chemical profile,
bootstrap worker and normalizer identity, action order, and 0.05-second interval.
The first run is an integrated developmental experiment; offline or training
reward changes alone do not establish physical competence.

## First joined run

The frozen `bec8a49` runner completed one uninterrupted hbox run with four
three-resident chemical worlds. It executed 20,480 world ticks, 160 shared PPO
updates, and 245,760 resident transitions in 670.717 seconds (366.41 resident
transitions/second). The final artifact is
`/tank/chreatures/runs/development/sensorimotor-worker-online-v1b-seed20260911/development.pt`,
SHA-256 `0f20e21df906c83873415034dee060bd0a57fc12ccdd8ad259647c71d6dd9304`.

The shared-memory world transport consumed 242.787 seconds in advance calls and
45.127 seconds in observations, 42.93% of total wall time. The remaining time
includes full-MaleCNS steps, worker and manager inference, PPO, and coherent
checkpoints. On the last batch, physical reward averaged `-0.0007458`, the
separately logged engineered goal-progress term averaged `+0.0019252`, worker
entropy was `18.0265`, and manager entropy was `4.8396`. These are training
measurements. They do not show that a resident learned useful goal selection or
improved physical behavior.
