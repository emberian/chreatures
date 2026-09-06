# Recurrent predictive state organ

The existing predictive PPO context is a fixed random reservoir, and its
auxiliary predictor estimates one next-step projected feature delta. It cannot
learn recurrent state dynamics or roll an executed action sequence forward.
`chreatures.predictive_state` supplies that missing capability as a compact,
separate organ. Policy/planning integration remains explicit future work owned by
the runtime integration lane.

## Inputs and sequence contract

The organ receives only information available to the resident:

- normalized anonymous neural features;
- six local physiology values: energy, gut, fatigue, bounded local speed,
  bounded local angular velocity, and neural support;
- the eight executed motor actions in the established action order;
- reset and valid masks plus sequence order.

`PredictiveSequence` is time-major. `action[t]` is the action executed after
`observation[t]` and before `observation[t+1]`. `reset[t]` clears state before
observing row `t`; a rollout import derives it from `done[t-1]`. Shapes are
`features[T,B,F]`, `physiology[T,B,6]`, `actions[T,B,8]`, and boolean
`reset/valid[T,B]`. Validation rejects nonfinite values and streams that do not
reset initially. Names, account identity, scenario identity, rewards, object
kinds, positions, PPO context, and unobserved world state are excluded.

The initial smoke source is the read-only archived real rollout
`rollout-step-0012835.npz`: 54 decisions by 48 residents, 384 neural features,
six physiology values, and eight actions at 0.25 seconds per decision. It spans
only 13.5 seconds per resident and is evidence for startup and sequence mechanics,
not whole-life predictive learning. Longer continuous datasets require a bounded
recorder in the active runner lane; no synthetic rows are substituted.

## Trainable loop

An observation encoder and `observe_cell` GRU infer posterior latent state from
the current observation, previous executed action, and previous private state.
A distinct `transition_cell` GRU advances that state from actions alone. During
training, the transition rolls actual future action sequences for horizons 1, 2,
4, and 8 and predicts the corresponding future neural/physiology observation.
Targets beyond resets or invalid rows are masked. A latent consistency term
aligns imagined state with the detached future posterior.

The decoder produces a mean and bounded diagonal conditional residual scale.
Gaussian negative log likelihood trains both. This scale combines aleatoric
variation and model misfit; it is not calibrated epistemic or OOD uncertainty.
Heldout NLL is reported separately at each trained horizon. Planning inference
also emits `exp(-h/max_trained_horizon)` horizon support so consumers can
attenuate unsupported long rollouts. An ensemble or another explicit epistemic
method is still needed for OOD confidence.

`model.imagine(state, actions)` is inference only. Its dreamed observations and
states must never be inserted into the experience dataset as if sensed.

## Persistence and native export

Trainer checkpoints contain model, optimizer, update counter, NumPy RNG, Torch
RNG, architecture format, and config. Atomic replacement prevents a partial file
from becoming a checkpoint. Restore rejects other formats or versions.

Immutable export is an NPZ of contiguous row-major float32 arrays. Metadata pins
every tensor name, shape, dtype, and SHA-256, action/physiology ordering, feature
layout, dimensions, and PyTorch GRU gate order (`reset, update, new`). The caller
owns each resident's mutable latent state; the exported weights are shareable.
This layout is intended for a direct Rust inference implementation without an
inner language model.

## Executed real-rollout smoke test

Three CPU updates used the first 43 chronological rows and held out the final 11.
Aggregate heldout NLL decreased from `0.08930` to `0.07114`. The post-training
per-horizon values were H1 `0.06338`, H2 `0.06692`, H4 `0.08396`, and H8
`0.07979`, covering 1,392 valid targets. Gradient norm before clipping was
`0.1008`. Checkpoint restore reproduced all heldout metrics exactly; the training
script also restores two branches, performs one identical update, and requires
exact model and metric replay.

Run the same bounded training path with:

```bash
python scripts/train_predictive_state.py ROLLOUT.npz \
  --output /tank/chreatures/predictive-state/run --epochs 25 --device cpu
```

The executed artifacts are in
`/tank/chreatures/predictive-state-src-v1/artifacts/smoke-neutral-scale`. They are
smoke artifacts, not a production organ trained on sufficient lived history.
