# State-conditioned population variance

> **Historical research record.** The Python implementation and operational
> commands described here were retired from the current tree after the native
> rich developmental controller replaced them. Reproduce this experiment from
> Git commit `0caa7ef`; the findings and design rationale below remain part of
> the research record.

The predictive PPO policy now has two explicit variance architectures. The
default `PredictivePPOConfig(std_profile="global-v1")` retains the established
learned eight-element `log_std` vector and its exact action distribution.
`std_profile="state-conditioned-v2"` adds a population-shared head driven by the
policy's actual trunk hidden state. That state already combines normalized neural
features, the six local physiology channels, and private recurrent context.

For action `a` in hidden state `h`, v2 uses

```text
offset(h, a) = 2 tanh(linear(h, a) / 2)
log_std(h, a) = clamp(global_log_std(a) + offset(h, a), -3.5, 0.3)
```

This is learned population policy output. There is no hand-authored fatigue,
effort, object, or world-state gate. The mean policy, value function, predictor,
context dynamics, action squashing, PPO objective, and local observations are
unchanged.

## Compatibility and upgrade

The v2 linear weight and bias are exactly zero initialized. Its construction
saves and restores PyTorch's RNG state, so it neither changes legacy module
weights nor consumes the next action sample. `distribution(mean, hidden)` is
used consistently during action collection and PPO updates.

Checkpoint and genome architecture identities are explicit. Global checkpoints
and genomes retain version/format 1; state-conditioned artifacts use version 2,
`state-conditioned-v2`, and include the head arrays. Restore rejects an identity
that disagrees with its serialized config.

Upgrade an existing global checkpoint explicitly:

```python
from chreatures.learning import PredictivePPOTrainer

trainer, extra = PredictivePPOTrainer.upgrade_state_conditioned(
    "checkpoint.pt", device="cuda"
)
```

The upgrade retains every shared v1 parameter, Adam step and moment, private
resident context, normalization moments, prediction-error traces, counters,
NumPy RNG, PyTorch RNG, device RNG, and caller metadata. It appends zero Adam
state for the zero head. The source checkpoint is not overwritten; a later
snapshot from the returned trainer is an explicit v2 branch.

## Executed probe

`scripts/probe_state_conditioned_std.py` ran with CPU Torch in an isolated hbox
source tree. It first trained a global learner so that Adam had nonzero state,
saved it, restored it, and upgraded a separate branch. At the upgrade boundary:

- paired latent samples, squashed actions, log probabilities, values, and
  predictions had maximum absolute delta zero;
- legacy optimizer moments and private state had maximum absolute delta zero;
- the new head and its Adam state were exactly zero.

One actual PPO update changed the head weight by `0.0012013`, changed effective
standard deviation by `0.0010689`, and produced a `0.0005486` standard-deviation
difference between two hidden states. A v2 checkpoint followed by stochastic
action, restore, and replay had maximum absolute delta zero for all returned
policy values. The compact report is retained at
`/tank/chreatures/std-head-src-v2/artifacts/probe.json`.

This mechanism gives lifelong population learning direct control over variance
as a function of sensed neural/physiological state. It does not itself assert
that the current inherited variances are energetically optimal; sustained
training and evaluation must learn and test that behavior.
