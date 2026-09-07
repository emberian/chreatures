# Learned sequence control: implementation contract

Status: **SUPERSEDED INPUT CONTRACT, 2026-09-07; not implemented or trained**.
Latest user steering prohibits raw sensory/physiology bypasses around MaleCNS.
The 668/128 feature tables and their concrete head dimensions below are withdrawn
and must not be connected to a trainer or resident. Root owns the replacement
CNS-derived adapter architecture. The generic conditional Bernoulli/categorical
likelihood, authenticated execution copy, and versioned research rollout boundary
remain design material for that replacement. This document preserves the exact
superseded proposal for review; it is not evidence of physical competence.

## Purpose and scope

Native v8 samples among four local and four recalled candidates every tick.
With near-equal scores, interruption rate is approximately 0.5 × 7/8 = 0.4375
per tick and H8 completion probability after initiation is 8^-7. Thus roughly
1,310 interruptions in 2,995 ticks and no completions are expected. An active
cursor reserved among candidates does not create temporal abstraction.

First wave freezes inherited sensory encoders, worker/proposal weights, goal
manager weights, predictor and consequence-law weights. Existing private
memories and adaptation continue evolving under their native rules. Train only
the shared selector, termination hazard and value heads below. All recurring
inference, candidate construction, sampling, memory and execution remain Rust.
Torch performs batched optimization on recorded native decision inputs.

No continuation reward, switching cost, completion bonus or minimum duration.
The initial hazard is an explicit, trainable timescale prior: beta = 1/8.
Policy inputs contain sensation, proprioception, private acquired history and
predictions from them. No positions, object labels, identities, archive contents,
resident IDs, slot IDs, generations or global tick enter the heads.

## Native input tensors (float32, finite, row-major)

`state[B,668]`, `proposal[B,8,128]`, `active[B,128]`,
`proposal_mask[B,8]` bool and `active_mask[B]` bool are the exact first-wave
interface. Feature coordinates are zero-based half-open intervals.

| State coordinates | Contents |
| --- | --- |
| 0:128 | Current worker recurrent state plus inherited recurrent adapter |
| 128:512 | Current 384 neural readouts, exactly the existing native worker inputs |
| 512:524 | Current 12 physiology channels in existing native order/scaling |
| 524:536 | Previous acknowledged 12-action vector in `ACTION_NAMES` order |
| 536:600 | Current 64-dimensional achieved-history key; zero until four frames exist |
| 600:664 | Current selected goal key; zero when goal invalid |
| 664 | Goal valid, 0 or 1 |
| 665 | `log1p(goal_remaining_ticks)`; zero when invalid |
| 666 | `recent_code_count / 4`, bounded [0,1] |
| 667 | Active execution available, exactly `active_mask` |

Each proposal or active feature row has the same 128 coordinates:

| Coordinates | Contents |
| --- | --- |
| 0:96 | Remaining actions flattened time-major `[8,12]`, normalized motor units; zero padding after actual duration |
| 96 | Actual remaining execution duration / 8 |
| 97 | Already acknowledged phase / 8 |
| 98 | Original sequence length / 8 |
| 99 | Acquired suffix flag, 0 or 1 |
| 100 | `log1p(complete_execution_support)`; zero for primitive |
| 101 | Existing contextual recall score clipped [-8,1]; zero for primitive |
| 102 | Existing empirical suffix utility in [-1,1]; zero for primitive |
| 103 | GAM out-of-domain flag |
| 104 | Predictor valid at forecast horizon, 0 or 1 |
| 105 | Goal forecast comparison valid, 0 or 1 |
| 106:118 | Ensemble-mean predicted absolute physiology at forecast horizon; zero if predictor invalid |
| 118 | Goal forecast progress / frozen `forecast_goal_rms`; zero if comparison invalid |
| 119 | Goal forecast disagreement / same RMS; zero if comparison invalid |
| 120 | Existing centered GAM/personal consequence score tilt |
| 121 | Existing population response tilt; zero when absent |
| 122 | Existing goal forecast tilt; zero if invalid |
| 123 | Existing combined native score before unavailable-candidate sentinel |
| 124:127 | Personal expected consequences: movement response, energy cost, fatigue recovery; zero if GAM out-of-domain |
| 127 | Forecast horizon / 8 |

Preserve existing channel units; no additional learned/running normalizer in
this version. State and proposal features are recorded exactly as supplied to
the native heads. No policy input depends on a future outcome. Absent proposal
rows and absent active row are all zero; their masks determine participation.
Unavailable rows must not affect centering, pooling or any policy probability.
Center existing legacy GAM/goal tilts over available replacement proposals;
evaluate the active row against those same centers. Invalid comparisons do not
enter the relevant center, and an empty valid set has center zero.

Indices 0..3 are freshly sampled local primitives. Their execution duration and
original length are 1, phase is 0, and only their first action is populated in
0:96. Preserve the existing predictor's H8 constant-action counterfactual for
these primitives, hence coordinate 127 is 1. This forecast is not a commitment.

Indices 4..7 are the best available acquired sequences **starting at phase zero**
under existing contextual recall ranking. Their duration equals original length
(4..8), and their forecast horizon equals that duration. This bank never reserves
the active remainder. If its source also ranks as a new proposal, that proposal
explicitly means restarting at zero after termination.

`active` is the separate remaining suffix at its acknowledged phase; its forecast
horizon equals remaining duration (1..7 after initiation). Native prediction
therefore evaluates up to **nine** rows per resident: eight replacements and one
active remainder. Forecast-only tail padding repeats the final action as current
predictor code requires; policy action features above remain zero-padded. H1..3
goal forecasts preserve v8's actual-history/predicted-history splice.

## Shared head architecture and parameter layout

All layers are float32 affine transforms `y = x @ weight.T + bias`; named weight
arrays are stored in Torch `[out,in]` orientation. Rust may transpose internally.
There is no dropout, batch normalization or hidden recurrence in these heads.

| Prefix | Weight shape | Output |
| --- | --- | --- |
| `state_encoder` | [128,668] | tanh, called `s` |
| `candidate_encoder` | [64,128] | tanh, shared across proposals and active, called `c` |
| `selector_hidden` | [64,192] | tanh of concatenated `[s,c_k]` |
| `selector_out` | [1,64] | Raw candidate logit |
| `hazard_hidden` | [64,256] | tanh of `[s,c_active,c_pool]` |
| `hazard_out` | [1,64] | Termination logit |
| `value_hidden` | [64,256] | tanh of `[s,c_active,c_pool]` |
| `value_out` | [1,64] | Scalar pre-decision value |

Each layer also has bias `[out]`. `c_pool` is the arithmetic mean of available
proposal embeddings; `c_active` is **zero after encoding** if active absent.
Four local proposals guarantee a nonempty pool. Use unconstrained learned
selector logits; legacy scores are inputs, not extra added logits. Initialize
hidden layers with seeded Xavier-uniform weights and zero bias. Initialize all
three output weights and biases to zero except `hazard_out.bias=-ln(7)`.
Archive initialized tensors; seed alone is not an artifact identity.

The critic uses the exact same nonprivileged features. Sharing encoders means
their gradients are updated by all three losses; inherited worker/encoder
weights remain frozen. The declared initial selector is uniform over available
proposals. The initial termination prior yields (7/8)^7 ≈ 39% H8 survival after
initiation, absent other cancellation. It is trainable, not a reward term.

## Boundary, masks and likelihood

1. Acknowledge the previous action and its sensory/body consequence. Advance
   active phase only for exact contiguous accepted execution. Finish a sequence
   only when every phase has an authentic receipt. Process resets/cancellations
   before making the next decision. Update normal sensory/private memory state.
2. Construct all features and proposals once, before sampling the control heads.
   Frozen local-proposal sampling still runs every tick, including continuation;
   its private RNG state persists. Give control sampling a separate private RNG.
3. If `active_mask`, sample termination `z ~ Bernoulli(sigmoid(hazard_logit))`.
   Set `hazard_mask=true`. If z=0, deliver the active next action, with no selector
   draw. If z=1, clear that execution attempt and sample the replacement selector.
4. If active absent, set `hazard_mask=false`, z=0 sentinel, and sample selector.
   Set `selector_mask=true` exactly when replacement selection actually occurs.
   Unavailable proposal logits are masked out of the categorical normalization.
5. Primitive selection delivers one action without starting active execution.
   Acquired selection starts a fresh private execution copy and delivers phase 0.
   No phase increments until receipt. Continuing sets selected candidate to -1.

For one physical decision, old/new PPO log probability is:

```
logp = hazard_mask * log Bernoulli(z; beta)
     + selector_mask * log Categorical(k; masked_selector_logits)
```

Implement masked terms by branching, never multiplying invalid/infinite logp by
zero. Inactive z and k are sentinels, not sampled actions. Bernoulli logp uses
stable log-sigmoid; categorical logp uses masked log-sum-exp. Store both component
old log probabilities and their sum. Never substitute primitive `action_nll`.
The proposal-generator likelihood cancels from ratios only because its weights
and feature-generating inherited transforms are frozen in this wave. The sampled
proposal set is an observed pre-decision auxiliary input for head optimization.

Entropy regularization, if used, has separate declared coefficients and masks:
hazard entropy only on active decisions, selector entropy only when selection
occurs. Default hazard entropy coefficient is zero so an undisclosed entropy
pressure does not erase the timescale prior. No fixed termination regularizer.

All acknowledged transitions train the critic and whichever head decisions
occurred, including partial attempts. Use per-physical-tick discounted returns
and GAE; document reward-component weights, discount and lambda in the training
manifest. Natural episode termination zeroes bootstrap; time-limit truncation
bootstraps the last coherent state. A distributed tick failure pauses and does
not fabricate a transition or retry an unknown mutation. Override receipts are
recorded with actual action and cancellation reason: first wave excludes those
rows from actor loss but may fit critic on acknowledged reward/next state.

Rollout-tail bootstrap must obtain next-boundary inputs/value without issuing an
extra unacknowledged action. Provide a read-only preview on a fork of private
state/RNG, or an explicit persisted prepare/decide stage. The real next boundary
must consume sensing, memory updates and proposal RNG exactly once. Never fake
an action receipt, silently discard a pending decision or advance physics merely
to obtain the bootstrap value. Root/native integration selects the API mechanism;
this no-extra-mutation requirement is fixed.

Track cancellation reasons separately: voluntary termination, host override,
tick gap, reset, invalid source. Active execution actions are copied privately
at initiation so reservoir replacement cannot erase their remaining actions.
Slot/generation are attribution metadata only. If the source is replaced while
the copy executes, completion still counts as execution but cannot update the
replacement slot; record attribution unavailable. Birth/reset clears execution.

## Rollout record and parameter-version transaction

Native output adds exact `state`, `proposal`, `active` and masks above; raw head
logits and value; z/k and decision masks; both component behavior logp and total;
proposed motor action; active source slot/generation and phase as **metadata**;
and immutable control parameter version/hash. Host records acknowledged action,
before/after physiology, reward components, next input tensors, terminal versus
truncation flags, cancellation/censor reason and actor-valid mask. Record private
snapshot identities, resident/episode/tick mapping and inherited artifact hashes
as metadata, never model inputs. Pending decisions cannot be training samples
until acknowledged. Collection must archive rollout tensors, not infer proposal
choices retrospectively from final motor actions.

Torch replays recorded head tensors through these exact layers, recomputes joint
logp with masks, and applies clipped PPO plus scalar value loss. First-wave
inference and Torch training own the same names, dimensions and formulas above.
No old native trajectory is relabeled as being produced by the new policy.

Only an explicitly created **research-training cohort** supports parameter swap.
Ordinary resident/live-world instances reject it. After a rollout, every resident
must have no pending unacknowledged action and no uncertain distributed tick.
Pause the cohort; checkpoint its full private/native/world/neural state and RNGs
with the currently loaded parameter identity. Archive the old parameter tensor
artifact before optimizing. Torch creates a new immutable artifact containing
the head tensors, contract version, parent hash, optimizer/update provenance and
inherited dependency hashes. Validate all dimensions, finite values and hashes
before any mutation. Stage all training participants, then commit the new version
at that same coherent boundary; any failed staging leaves old version active.
An uncertain commit remains paused until explicit participant reconciliation.

Write a swap receipt linking old checkpoint, old/new parameter hashes and next
rollout index. Take a new checkpoint identity after commit. Preserve private
sensory memory, execution copies/phases and all RNG state across this authorized
research swap; resume only after every participant reports the same committed
version. Each rollout has exactly one head parameter version. Restoration loads
the archived exact version from its checkpoint, never latest weights. Frozen
running lives remain on their loaded architecture and parameters; no silent
replacement or compatibility reinterpretation is permitted.

## Evidence and source limits

Inspected native sites: `developmental.rs` candidate dimensions/softmax/refinement,
`motor_suffix.rs` recall/select/receipt/capture; `runtime3d.py` sample configuration.
Current `develop_rich_sensorimotor.py` PPO samples the Torch worker directly and
does not optimize the deployed native selector. Current native collector stores
actual actions/outcomes but lacks control-choice likelihoods; extend collection
with this contract before claiming native-controller PPO.

[Bacon, Harb and Precup, The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
motivates separately learning temporally extended behavior and its termination
from task consequences. This contract adapts that separation to finite acquired
motor sequences and PPO; it does not claim their results transfer to Chreatures.
The [author implementation](https://github.com/jeanharb/option_critic/blob/master/neural_net.py)
contains distinct termination/action heads and an optional `termination_reg`.
That optional regularizer is deliberately absent here. These primary sources
were read directly after Scry schema succeeded but its pinned paper lookup
returned HTTP 429 host memory pressure; discovery is not implementation evidence.

No training, live-state mutation or physical validation was performed in drafting
this contract. Integration must establish native/Torch head likelihood agreement
and then assess the joined physical rollout: returns, learned duration, authentic
completion and sensory-conditioned interruption, not completion count alone.
