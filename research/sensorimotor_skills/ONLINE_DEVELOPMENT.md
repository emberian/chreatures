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
