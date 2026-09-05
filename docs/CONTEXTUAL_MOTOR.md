# Contextual motor refinement

`chreatures.contextual_motor.ContextualMotorRefiner` is an optional private
organ that lets experienced relational memory make a small change to an
inherited continuous motor decision. It does not create goals or search the
world. At a motor macro boundary it starts with an ancestral draw from the
inherited pre-tanh Gaussian and adds a few nearby alternatives, asks the resident's
`RelationalContextMemory` what followed similar actions in similar sensory
contexts, and may select a different candidate.

The organ has no API for simulator positions, entity identifiers, object
classes, or authored rewards. Its input is an anonymous sensory feature vector,
the resident's own energy, gut and fatigue levels, and outcomes measured after
the selected action was physically executed. A useful runtime feature can be a
private visual/context embedding. Raw physical senses also work provided world
positions and identity-bearing fields are left out.

## Decision boundary

The action order is the `motor_inheritance.ACTIONS` order:

```
thrust, yaw, gaze_pitch, grip,
signal_low, signal_mid, signal_high, posture
```

At an open inherited-organ macro boundary, use the joined method so the chosen
candidate enters the motor predictor and hold state as well as the world:

```python
decision = refiner.refine_and_commit(
    motor,
    context_memory,
    context_feature,
    raw_motor_senses,
    local_physiology,
    dt=0.05,
)
physical_action = decision["action"]
```

For a stochastic inherited organ, candidate zero is its actual full-standard-
deviation ancestral Gaussian draw. A deterministic organ uses `mean` instead.
The remaining default candidates are at most eight lower-scale alternatives
around that baseline, with five total by default. Their inherited score is a
penalty relative to candidate zero, which has score zero. With memory disabled
or no covered experience, selection therefore returns candidate zero exactly
and preserves the inherited policy's exploration distribution. As in
`MotorOrgan.tick`, negative grip and signal values project to zero before
memory queries and execution. Candidate generation and relational queries
happen only at the macro boundary. After advancing physics with the
returned first tick, ticks two through five use
`motor.continue_macro_action(dt)`. Once the fifth tick and its observations are
complete, normalize the next raw motor senses and call
`motor.open_macro_boundary(next_normalized)`. This performs the inherited
predictor/context update and opens the boundary for the next refinement. The
selected physical vector is deliberately the action recorded by the inherited
predictor on this external-selector path.
The caller-supplied candidate hook exists for replay and controlled evaluation;
runtime candidates should still originate from the inherited policy.

Every decision records the complete physical and pre-tanh candidate vectors,
the inherited mean and log standard deviation, inherited scores, predicted
outcomes, support, uncertainty heuristic, contextual corrections, selected and
inherited-selected indices, policy artifact SHA-256, memory revision, and
decision count. Provenance also distinguishes the stochastic ancestral,
deterministic mean, and caller-supplied baselines. This preserves whether memory
changed the choice and which inherited artifact supplied the candidate
distribution. For caller-supplied replay candidates, the recorded pre-tanh
value is explicitly labeled as an inverse-tanh proxy.

## Experienced outcome contract

After the held action finishes, call `record` with the exact feature and
physiology before execution, the action actually executed, the next feature
and physiology, and the aggregate physical outcome. The relational outcome
vector has this fixed order:

```
energy_delta, gut_delta, fatigue_delta, nutrition, effort, contact
```

`nutrition`, `effort`, and `contact` must come from the physical world's
outcome. The organ learns reduction in a smooth energy/gut/fatigue drive plus a
small direct nutrition term and effort cost. It also learns a regularized
linear value over resulting anonymous features after eight transitions. It
does not assign value to named things or locations.

`record` requires the memory's current observation to equal the supplied
pre-action feature. This catches stale or reordered asynchronous results. A
trajectory begins with `memory.begin(feature)`; subsequent calls naturally use
the next feature left by the preceding record. Episode resets call
`memory.begin(feature)` again.

The contextual correction is bounded to 0.32 score units by default. Under the
new `absolute-v2` coverage rule, predicted drive improvement is multiplied by
the historical repetition and uncertainty gates plus absolute coverage:

* a support gate that rises from minimum support 0.5 to full support 5;
* an uncertainty gate that falls to zero at heuristic uncertainty 0.58.
* an action-distance kernel gate, rising between similarity 0.20 and 0.80;
* an unnormalized action-match-mass gate, rising between 0.12 and 0.80;
* an observation-similarity gate, rising between 0.25 and 0.75.

The three absolute gates combine by their minimum before multiplying the first
two. A far query can therefore have a large historical effective count yet
receive exactly zero correction. Snapshots written before this diagnostic
retain `effective-v1` behavior explicitly on restore; their action scores do
not silently change.

An action with no matching experience receives zero correction. A one-shot
edge can have a small influence; repeated consistent experience increases it.
The inherited score remains part of every combined score, so context cannot
promote an arbitrary action outside the inherited candidate set.

Relational-memory uncertainty is a coverage heuristic. In the existing real
trajectory evaluation its uncertainty/error correlation was only 0.068. It is
not calibrated confidence or a probability. The gate uses it conservatively
alongside actual edge support; decision provenance states this limitation.

An optional `candidate_evidence` callback can add another private memory's
candidate-level evidence without importing that memory implementation. It is
called once per macro boundary with an immutable tuple of physical candidate
vectors and returns a source label, one correction per candidate, and aligned
JSON diagnostics. Applied corrections are separately clipped to 0.12 score
units and audited with requested values. `None` does not call external code or
change actions, scores, or random streams. Disabling refinement makes both
relational and external applied corrections zero.

## Controls and private state

`refiner.freeze(True)` disables updates in both the relational graph and the
feature-value model while leaving learned action refinement available.
`refiner.freeze(False)` resumes learning. Setting `refiner.enabled = False`
sets all contextual corrections to zero without discarding state.
`refiner.clear(memory)` explicitly removes that resident's relational graph and
feature values. Clearing does not affect the inherited policy artifact.

`refiner.snapshot(memory)` serializes the refiner, its RNG, counters, last
decision, and the complete bounded relational memory. `restore` reconstructs
both objects and reproduces its local candidate samples exactly. Runtime
ancestral baselines use `MotorOrgan.rng`, so exact continuation of the joined
decision also requires the motor organ's private snapshot. This is private
per-resident state. The inherited motor artifact is referenced in decision
provenance by SHA-256 and remains under the motor organ's own snapshot contract.

An empty-memory comparison used two `MotorOrgan` instances with the same seed
and trained artifact. Direct `tick` and `refine_and_commit` returned the exact
same physical action, selected candidate zero, and left the two motor RNG states
identical. A disabled-refiner check with an explicitly supplied ancestral
baseline likewise returned that baseline exactly.

Live orchard checkpoints at ticks 1,187, 1,526, and 2,011 showed that the frozen
representation was not collapsed: no neural value clipped during normalization,
projection saturation was zero for Fern and Mica and about 5--6% for Pip, and
within-resident checkpoint changes reached 0.44--0.55 RMS. The original 0.95
context-allocation threshold merged those moderate changes. A
`projection-v2` LivingMotor profile with observation bandwidth 0.28,
new-context threshold 0.34, and action bandwidth 0.20 is available explicitly
for research organisms. It is not the default. Three sparse checkpoints formed
two rather than one states for Fern and Mica under that profile, which is too
little evidence to establish a general improvement. Pip's sharper physical
interaction already formed three path contexts under the legacy profile.

At tick 2,011 the live gates were active, but candidate correction differences
were usually below 0.001 while inherited alternative penalties ranged from
0.024 to 0.251. Sparse nutrition was diluted across hundreds of transitions.
Zero changed choices in that run therefore does not show a dead projection or
inactive gate, and it also does not show useful contextual control yet.

## Physical alternate-history probe

The integration probe used `ArticulatedSensoriumWorld` with `body-v1` vision, a
twelve-joint body, ground contact, and one ordinary static food sphere. The
365-value feature contained retina, odor, touch, sound, illumination, body
motion, tarsal contact, and joint state. It excluded antenna world positions,
resident physiology, simulator coordinates, and IDs. Each candidate was held
for five 0.05 s physics steps, and each action/history pair was repeated six
times from an identical MuJoCo checkpoint.

With the food 0.22 m ahead, the first forward episode ingested 0.085 units and
the reverse episode 0.017; memory selected forward. With the food mirrored
0.22 m behind, the corresponding episodes ingested 0.034 and 0.085; that
resident's separate memory selected reverse. The inherited candidate scores
were equal in both cases. The established edges had support 6, contextual
corrections remained below the 0.32 cap, and the physical measurements rather
than a food label produced the different choices.

As a stricter history control, both learned memories were then queried with the
exact same ahead-scene feature, physiology, candidates, and inherited scores.
The ahead-fed history still chose forward, with contextual corrections 0.254
versus 0.136. The rear-fed history chose reverse, with corrections 0.148 versus
0.225. Thus the choice difference survived when the current input was held
fixed; it came from private experienced transition outcomes.

The same probe verified that a frozen contradictory physical transition did
not change the memory revision or learned arrays, clearing returned selection
to the inherited tie-break candidate, and a JSON round-trip of the combined
private snapshot reproduced both state and the next sampled decision exactly.
This demonstrates a narrow learned affordance under familiar coverage. It does
not establish calibrated uncertainty, long-horizon planning, or general object
understanding.

The absolute-v2 gate was then evaluated with six training episodes per history
using different MuJoCo seeds, headings, and lateral starts, followed by an
independent seventh episode. In the held-out front-food episode, forward and
reverse ingested 0.085 and 0.017 and the refiner chose forward. In the held-out
rear-food episode they ingested 0.034 and 0.085 and it chose reverse, changing
the inherited tie-break to the physically better candidate. Exact-action match
mass was 0.969 and observation similarity was 0.979/0.985, so the new absolute
gate passed covered queries. When both memories received the identical current
input they still chose in opposite directions, preserving the earlier causal
history control. This remains a small controlled physical affordance test.
