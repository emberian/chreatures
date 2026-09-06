# Learning what to retain

The existing motor carries a private 32-value recurrent context, but its three
context transforms are fixed random buffers. The policy can learn to read that
context; it cannot learn the write or retention mechanism. The current
`gated-v1` prototype makes those transforms trainable and learns an elementwise
write gate. It is a working-memory baseline, not an implemented hippocampus.
No existing resident has received it, and behavioral improvement is unproven.

## Research that informs the choice

The investigation used authenticated Scry SQL over its documented
`openalex.works` relation, followed by primary papers and author repositories.
The retained query receipts are in `research/working_memory/searches.json`.
Kagi's documented v1 endpoint returned 404 in this environment, and the v0
endpoint rejected the available key with 401. Scry queries succeeded. No keys
or account details are retained in the repository.

[Tallec and Ollivier, *Can recurrent neural networks warp time?*](https://arxiv.org/pdf/1804.11188)
derive a gated leaky recurrence and relate gate values to characteristic
timescales. Their equations 8–10 directly describe the family used here. Our
log-spaced initial retention scales, 0.5–32 model seconds, are an engineering
choice; they do not reproduce the paper's complete chrono-initialization
experiments or establish biological memory timescales.

[Ni, Eysenbach and Salakhutdinov, *Recurrent Model-Free RL Can Be a Strong
Baseline for Many POMDPs*](https://arxiv.org/html/2110.05038)
show that recurrent architecture, algorithm and context length materially
affect results. Their [author implementation](https://github.com/twni2016/pomdp-baselines)
provides stronger comparisons than a fixed random reservoir. Their results
favor separate recurrent actor/critic encoders and off-policy learning in
several settings. Our shared PPO trunk does not reproduce that implementation;
critic interference and sample efficiency remain explicit risks, not settled
by installing a gate.

[POPGym](https://arxiv.org/html/2303.01859) compares recurrent, attention and
state-space approaches under PPO. Its findings warn against choosing memory
from supervised sequence benchmarks alone, or testing memory solely through
navigation. A policy with more parameters can improve without remembering.
The [POPGym Arcade paper, revision 8](https://arxiv.org/abs/2503.01450v8)
adds a particularly relevant warning: value functions can assign credit to
irrelevant history, and unusual observations can contaminate later decisions.
That motivates interventions on experienced history and irrelevant
distractors, not just longer retention curves.

## Implemented mechanism

Let `q` be the existing projected neural readout, `a` the preceding action,
and `h` private working state. The candidate is
`z = tanh(Wq q + Wa a + Wh h)`. A learned sigmoid gate reads the same inputs;
the update is `h_next = h + gate * (z - h)`. The original anatomical circuit
continues to evolve separately. The gate does not read entity identities,
coordinates, experiment labels or the observer's archive.

Training must reconstruct resident sequences with gradients through memory.
Flattening recorded, detached contexts into unrelated training samples would
leave the write mechanism without the intended future credit. The prototype
therefore trains contiguous chunks, initializes each from its recorded private
state, masks padding and resets each resident at its own real episode boundary.
Future observations enter the context only before the following decision.
The default training chunk spans 32 decisions (8 seconds at the current
cadence). Truncation still limits learnable credit, even when the gate can
retain state longer. A retained boundary state also becomes stale as parameters
change during PPO epochs; this is another reason not to infer competence from
an architecture or gradient check.

The inherited artifact contains shared weights and normalization only. A
fresh child receives zero working state. Introducing gates from a reservoir
parent is a declared developmental architecture change, with a fresh optimizer,
not an exact continuation. Adult snapshots retain private context, pending
action, predictor state and RNG. Deployment is being moved into the native
motor kernel; Torch remains the training implementation.

## What this does not solve

The chemical acquisition audit found 23 mouth-contact ticks among 36,727
ordinary contact ticks. Eating was automatically requested throughout the
run, and the policy holds a command for five physics steps. Better working
memory does not supply precise head alignment or a learned feeding decision.
The complementary [sensorimotor skills investigation](LEARNED_SENSORIMOTOR_SKILLS.md)
therefore targets a learned controller responding on every physical tick.

Before promotion, compare learned gates with frozen gates and a matched
recurrent baseline under the same experienced sequences. Test both a useful
delayed cue and an irrelevant distraction. Keep world and resident histories
held out, report physiological cost, and intervene on memory after the cue
has disappeared. Numeric parity and checkpoint continuation prove execution
properties; only those behavioral comparisons can establish useful memory.
