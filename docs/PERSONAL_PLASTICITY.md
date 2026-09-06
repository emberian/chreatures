# Private lifetime motor plasticity

`chreatures.personal_plasticity` is an opt-in online adapter around the frozen
inherited motor policy. It changes a resident's continuous motor tendency from
that resident's own action and later physical consequence. It does not mutate
the shared `MotorArtifact`, write a genome, choose a goal, or receive an object
kind, object identity, position, language description, or personality label.
Existing resident snapshots restore without it; a new lineage must opt in.

The version-1 actor is a zero-initialized `8 x 65` private linear map. Its input
is the inherited organ's frozen 64-component projection plus a bias. Its output
passes through `tanh` and is capped at `0.32` in inherited pre-tanh Gaussian
mean units. The initial offset is therefore exactly zero and each action's
eventual correction remains bounded. This is 520 learned actor scalars, small
enough to checkpoint per resident without introducing another neural model.

The adapter samples the inherited diagonal Gaussian around the adapted mean.
After the selected action has run for one five-tick macro, it receives the five
actual physical tick records. Each record contains only before/after energy,
gut and fatigue, measured nutrition, measured effort, and physical `dt`. Reward
comes from the explicitly versioned `FiniteEnergyObjective` v1. The objective's
one-sided reserve shortfall never rewards burning energy above an arbitrary
target, and ingestion is credited once through the measured gut change. No
intermediate physiological state is fabricated for macro accounting.

The actor update is the baseline-proposal Gaussian mean score multiplied by a
bounded one-step TD error. A private linear value baseline and exponentially
decayed eligibility traces assign delayed consequences over a short action history.
Actor and critic gradients, weights, advantages, and mean corrections all have
declared limits. This remains ordinary online actor-critic learning; the value
estimate is not a calibrated forecast and a finite lifetime provides no
performance guarantee.

## Optional state-conditioned exploration

The original `fixed-inherited-v1` variance remains the default. Its diagonal
Gaussian scale is supplied by the immutable inherited policy and frozen for
each decision. A global-v1 artifact returns one global vector; a
state-conditioned-v2 artifact derives it from the same decision hidden state.
Neither changes from private lifetime learning, so even an actor that learns a
zero mean retains a nonzero activity floor. In the current physical effort
equation that floor can exceed the fatigue recovery threshold. Mean adaptation
alone therefore cannot express physical rest.

`PersonalPlasticityConfig(variance_adaptation="state-log-std-v2")` enables a
separate private log-standard-deviation head. Its exact feature profile is
`projected64-physiology6-bias-v2`: the frozen 64-component inherited projection,
the six normalized body-local values produced by
`MotorOrgan.physiology_vector` (energy, gut, fatigue, bounded speed, bounded
angular velocity, and support), and a bias. No position, object identity, named
goal, or simulator-wide value enters this head. The mean actor and critic keep
their original 64-plus-bias inputs.
The profile freezes that normalization locally: speed is `tanh(speed/2)`,
angular velocity is `tanh(angular_velocity/4)`, the four bounded body values
remain in `[0,1]`, sensory projection and physiology groups are divided by
`sqrt(64)` and `sqrt(6)` respectively, and the bias remains one.

The `8 x 71` variance head begins at exact zero. It adds at most 1.5 log units
in either direction and clamps the effective log standard deviation to
`[-5.0, 0.3]`. Its score-function derivative is
`standard_noise**2 - 1`, multiplied by the bounded head Jacobian and the same
proposal advantage used by the mean actor. Gradients, traces, and weights use
the existing bounds. This rule learns from physical consequences; it contains
no fatigue threshold, rest command, or authored state gate.

When variance-v2 is active, contextual alternatives are still formed as
`zj = z0 + inherited_sigma*c*epsilon_j`. Their perturbation scale is the
frozen decision-time inherited standard deviation, while only baseline `z0`
uses the private effective standard deviation. Consequently the downstream candidate
generator has no additional private-variance path after conditioning on `z0`,
and the baseline proposal-density score remains valid. Decision provenance
records inherited and effective log standard deviations, the exact sampled
noise, decision-time physiology vector, feature-profile version, and the fixed
alternative-scale contract.

## Integration boundary

At a motor macro boundary, a caller can use the adapter's private sampler:

```python
from chreatures.personal_plasticity import PersonalMotorPlasticity

plasticity = PersonalMotorPlasticity(enabled=True, learning=True)
feature64 = motor.projected(motor.normalize(raw_neural_features))
inherited_mean, _, hidden = motor.forward(
    motor.normalize(raw_neural_features),
    motor.physiology_vector(local_physiology),
)
inherited_log_std = motor.distribution_log_std(hidden)
proposal = plasticity.propose(
    feature64,
    inherited_mean,
    inherited_log_std,
    local_physiology=motor.physiology_vector(local_physiology),
)
physical_vector = plasticity.commit(proposal)
# Commit physical_vector through MotorOrgan and retain five actual tick records.
result = plasticity.observe(next_feature64, transitions=actual_tick_records)
```

`adapt_mean(feature64, inherited_mean)` is a smaller callback seam for a caller
that owns sampling. When the adapter is disabled it returns an unmodified copy
without performing floating-point arithmetic. A runtime seeking bit-exact old
behavior should branch directly to `MotorOrgan.tick` when the option is off;
the adapter then consumes neither its private RNG nor any motor state.

`propose(..., inherited_noise=noise)` accepts a standard-normal draw from an
existing sampler. With the adapter disabled, the resulting latent action is
bit-identical to
`inherited_mean + exp(clip(log_std, -3.5, 0.3)) * noise`. Without that
argument, the resident's private adapter RNG owns the draw.

`LivingMotorOrgan(..., plasticity=True)` performs this join at its existing
five-tick boundary. It draws the ancestral standard-normal sample from the
resident's `MotorOrgan.rng`, passes that exact noise and resulting pre-tanh
baseline through personal adaptation, and gives the same latent baseline to
the contextual refiner. The refiner's other candidates still come from its own
private RNG. A false `plasticity` option follows the original refiner and motor
RNG path, and snapshots written before this option restore with it absent.
Variance adaptation remains a second explicit choice:

```python
config = PersonalPlasticityConfig(
    variance_adaptation="state-log-std-v2",
)
living = LivingMotorOrgan(
    artifact,
    plasticity=True,
    plasticity_config=config,
)
```

No current resident or default new resident is upgraded by this option.

An existing `LivingMotorOrgan` can schedule explicit developmental maturation
with `queue_state_log_std_v2()`. If an old mean-only proposal is in flight, the
queue remains inert until all five physical outcomes have been recorded and
that proposal's actor/critic update has completed. The quiescent boundary then
calls `enable_state_log_std_v2()` before drawing the next proposal. The first
variance-v2 sample uses an exact-zero head and therefore matches the inherited
sample and RNG stream; only its later measured outcome can update variance.

The queue and applied record are stored only after this method is called, so
untouched snapshots keep their old schema. The record contains canonical
source and target config hashes, queue/apply macro counters, the old pending
decision sequence, zero-head checks, and the explicit credit boundary. It is
visible as `view()["variance_maturation"]`. Shared motor artifacts, old mean
weights, learned critic/history, and body mechanics remain unchanged.

Opting into plasticity also selects contextual utility
`finite-energy-v1`. The actor and contextual refiner receive the same
`FiniteEnergyConfig` object and persist its canonical SHA-256 identity. A
caller-supplied legacy contextual config is rejected for a new plastic organism
because the two learning organs would otherwise value the same transition
differently. Restoring an old nonplastic snapshot still selects its exact
legacy contextual semantics.

The living wrapper retains before/after physiology and the measured outcome
for each of the five intervening physics ticks. At the next boundary it records
the relational transition, gives the actual tick sequence to personal
plasticity, advances inherited recurrent context, and samples the next macro.
Its snapshot stores both the partially accumulated physical record and the
plasticity organ's matching pending proposal, so a mid-macro restore does not
lose or duplicate credit.

Personal plasticity sums the finite-energy objective over the five actual
0.05-second records. Contextual and external evidence score the matching
before/after states over one 0.25-second interval, using the accumulated effort
integral. The potential telescopes and the effort integral is linear, so these
three paths produce the same reward up to float32 accumulation rounding.

A focused articulated MuJoCo check held one physical action for five steps and
fed the same measured body history through all three routes. Personal
plasticity scored `-0.002316420228`; contextual macro accounting scored
`-0.002316420199`; and visual episodic evidence, recalling that exact interval,
scored `-0.002316420199`. The actor/context difference was `2.91e-11`, and the
visual/context difference was zero. The accumulated effort was `0.07339955`
over 0.25 seconds. A separate above-target check showed the retained legacy
quadratic scoring a 1% energy loss as `+0.0228`, while finite-energy-v1 scored
it as `-0.0280522`.

The default credit rule is explicitly named `latent-proposal-v2`. Let the
adapted actor sample baseline latent `z0 ~ N(mu_theta, sigma)`. The refiner then
constructs each alternative as `zj = z0 + sigma*c*epsilon_j`, where its private
`epsilon_j` is independent of the actor parameters, and scores all transformed
candidates with a selector frozen at that decision. Conditional on `z0`, the
candidate generator, selector, physics and reward have no other path to
`theta`. The likelihood-ratio identity therefore gives

```text
gradient_theta E[R] = E[R * gradient_theta log p_theta(z0)]
```

even when the selector executes `zj` rather than `z0`. This is credit for the
internal baseline proposal distribution of the composite controller. It does
not assign a Gaussian density to the executed action.

The contract is narrow and machine-readable. `ContextualMotorRefiner` labels
its generated-around-baseline pipeline as
`latent-proposal-downstream-selector-v2`. External candidate evidence must
declare `proposal_credit_contract="candidate-and-frozen-state-only-v1"`; an
undeclared callback remains ineligible because Python closures could otherwise
read private actor parameters. The current delayed visual-memory callback meets
this condition by closing over a frozen capture, memory result, physiology and
utility configuration, then receiving only immutable candidate vectors. It
must emit that declaration when the runtime enables proposal credit.

Caller-supplied candidate tables, a directly replaced physical action, an
undeclared external callback, and any other unknown selection pipeline skip
the actor update and clear its eligibility chain. The critic can still fit the
experienced return. `executed-match-v1` remains available in
`PersonalPlasticityConfig.credit_assignment` as the older conservative rule;
it credits only a directly executed unchanged proposal. Deterministic proposals
never receive score-function credit under either mode.

Every completed call to `observe` requires a nonempty contiguous sequence of
real records with this schema:

```python
{
    "before": {"energy": 0.55, "gut": 0.04, "fatigue": 0.03},
    "after":  {"energy": 0.5499, "gut": 0.04, "fatigue": 0.0301},
    "nutrition": 0.0,
    "effort": 0.31,
    "dt": 0.05,
}
```

`set_learning(False)` freezes actor, critic, and eligibility state while still
allowing proposals and auditable reward observations. `set_enabled(False)`
also returns zero offsets. `discard_pending()` is only for an aborted physical
transition; it deliberately assigns no outcome.

## Private continuation

`snapshot_value()` contains the configuration, exact `FiniteEnergyConfig`
with its own checksum, actor and critic weights, both eligibility traces,
private RNG, counters, recent diagnostics, and any pending sampled decision.
The whole value has a canonical SHA-256 checksum. `restore_value()` validates
dimensions, finite values, bounds, decision provenance, and checksum. The
inherited artifact is absent because it remains shared and immutable.

The compact `view()` reports enabled/frozen state, the versioned credit rule,
decision, proposal-credit and skip counts, reward/TD diagnostics, recent
correction, parameter norms, and objective version. It exposes no private
feature vectors or action history.

Mean-only snapshots retain their previous config, decision, array, and checksum
schema; missing variance fields restore as `fixed-inherited-v1`. Variance-v2
snapshots additionally store the 71-input head, eligibility trace, update
counter, last offset, feature-profile identity, and decision-time body vector.
`enable_state_log_std_v2()` is the only checkpoint conversion seam. It requires
no pending action, retains existing mean/critic state, and creates the new head
and trace at exact zero. Restore never calls it implicitly.

A mid-macro queue check snapshotted and restored an actual articulated run,
completed the old five-tick action, and applied maturation at macro decision
one. The old update reported `variance_updated=False`; the next pending
decision carried `projected64-physiology6-bias-v2`; both new arrays had zero
nonzero elements; and its action exactly matched an unmigrated control with the
same motor and refiner RNG states. The post-application snapshot also restored
exactly.

## Variance checks

A three-million-sample common-random-number check differentiated a composite
five-candidate selector through the variance bias at zero. Alternatives used
fixed inherited perturbation scale. The proposal score estimate was
`-0.4167643 ± 0.0007858`; central finite difference was `-0.4165719`, a
`0.245` standard-error difference. This checks proposal-density credit through
candidate selection without treating the executed alternative as Gaussian.

An articulated MuJoCo comparison then drove two identical bodies for 400 real
0.05-second steps with the same standard-normal draws and zero means. The
inherited distribution produced mean measured effort `0.32940`, above the
physics fatigue-recovery threshold `0.27083`, and fatigue rose by `0.11245`.
Putting the bounded variance head near its learned contraction limit reduced
all eight log standard deviations by `1.44604`; measured effort fell to
`0.09055`, and fatigue recovered from `0.20` to `0.0`. This establishes that
the new action distribution can represent rest in the actual articulated
mechanics. It does not claim that a particular lifetime will learn the maximum
contraction or that lower exploration is always beneficial.

## Measured physical development check

The implementation was exercised on local MuJoCo 3.12.0
`ArticulatedSensoriumWorld` histories. One articulated resident began at the
same pose and physiology on flat terrain. An otherwise identical static berry
was placed 0.235 m in front for one lifetime and 0.27 m behind for the paired
lifetime. Each of 800 decisions sampled the full eight-channel Gaussian from
the nursery-8000 standard deviations, drove the actual hinges for five 0.05 s
steps, ran contact ingestion, energy, gut and fatigue dynamics, and supplied
all five measured records to `observe`.

For this causal isolation check, both learners received the same fixed
64-component projection of an actual empty-habitat body observation. That
removes scene discrimination from the question and tests whether different
experienced action/outcome histories change the same motor query. With the
default learning configuration and seed 440, the final thrust mean correction
at that common query was `+0.01027` after the front-resource history and
`-0.02688` after the rear-resource history. Action/nutrition correlation in the
recorded histories was `+0.5242` and `-0.6440`, respectively. A learning-frozen
control processed 800 front-resource macros and retained an exactly zero actor
and zero correction. The correction cap was not approached.

A JSON round trip after the 800-macro front history was continued for eight
more physical macros on both the original and restored copies. Proposed
actions, physical outcomes, actor/critic weights, eligibility traces, RNG and
the public view remained exact. A separate check passed a reranked action and
confirmed one critic update, zero actor updates, and one recorded actor skip
because that direct intervention did not declare a proposal-independent
generated-candidate pipeline.

The joined `LivingMotorOrgan` path was also checked directly. With plasticity
false, 23 ticks were compared against the pre-join implementation using
identical neural vectors and physical outcomes; every action, both motor RNG
states, both refiner RNG states, and the complete snapshot were exact. An
opted-in articulated MuJoCo run advanced 21 real physics ticks across four
complete macros and restored midway through a macro. A declared frozen
candidate callback selected alternative candidate 1. All four macros updated
the baseline-proposal actor under `latent-proposal-v2`; actor norm reached
`0.04760` and the largest current mean correction was `0.00715`. The original
and restored copies produced identical actions, physical outcomes, world
snapshots, and private motor snapshots. Repeating the external selection
without its independence declaration produced a critic update and an actor
skip.

## Composite-gradient numerical check

A common-random-number Monte Carlo check used two million scalar samples with
`sigma=0.63`, four alternatives at `c=0.42`, inherited penalties computed only
from the independent perturbations, a nonlinear action-dependent selector, and
a nonlinear reward of its selected action. Alternatives were selected in
65.41% of samples. At `theta=0.12`, the centered baseline-proposal score
estimate was `0.738048` with Monte Carlo standard error `0.000838`. The common
random-number central finite difference was `0.738588` at step `0.005` and was
stable at `0.739006` and `0.737525` for steps `0.01` and `0.0025`. The main
estimate differed by 0.64 standard errors, supporting the composite credit
identity rather than executed-action Gaussian credit.

Argmax selection and sparse physiological reward can still make this estimator
high variance. The private value baseline reduces variance, while advantage,
gradient and weight clipping deliberately trade unbiasedness for bounded
lifetime changes. The identity would fail if a selector or perturbation read
current private actor parameters through a path other than `z0`, or if an
external action replacement were credited without a known generative path.

These are controlled development histories, not evidence of spontaneous
individuality or improved foraging. The fixed feature intentionally omits the
harder problem of learning under a drifting or aliased neural projection. The
observed offsets are small, reward contact is sparse, and nutrition over this
short check is too noisy to establish a performance gain. A runtime evaluation
should compare matched residents over continuing physiology, report offset and
actor-skip distributions, and retain a disabled bit-exact control.
