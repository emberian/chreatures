# Action-conditioned relational context memory

> **Historical research record.** The Python implementation and operational
> commands described here were retired from the current tree after the native
> rich developmental controller replaced them. Reproduce this experiment from
> Git commit `0caa7ef`; the findings and design rationale below remain part of
> the research record.

`chreatures.context_memory.RelationalContextMemory` is an optional personal organ for learning reusable transition structure from experience. Its complete input is:

- an anonymous dense sensory feature vector;
- the continuous action actually executed;
- the resulting anonymous sensory vector and outcome vector.

It has no world reference, simulator coordinates, object or resident identifiers, scene labels, target query, reward policy, or planning objective. It does not choose behavior. A runtime can use its predicted consequences and uncertainty as inputs to a separate learned policy later.

## Structure

The memory maintains a bounded set of latent contexts and action-conditioned directed transitions. A context has an online sensory-emission prototype and visit statistics. A transition has source and destination contexts plus online means and variances for action, sensory change, and outcome. Inference combines current sensory emission similarity with the posterior over contexts established by the preceding path.

When a familiar-looking emission arrives through a novel established route, the organ may allocate a second latent clone. The clones initially look alike but can acquire different outgoing transitions. This lets path history disambiguate perceptually aliased views without assigning a place ID. Prediction weights matching outgoing transitions by context posterior, action similarity, experience count, and a smaller sensory fallback that permits reuse before the exact clone has sampled an action.

Predicted sensation blends local transition delta with the learned destination prototype. The former preserves continuous short-timescale change; the latter represents the reusable relational map. Predicted outcomes and uncertainty use observed edge statistics. Uncertainty includes limited support, action novelty, successor variance, context entropy, and sensory novelty. It is an empirical confidence signal, not a probability calibrated for safety decisions.

Prediction diagnostics are versioned as `absolute-match-v2`. The historical
`support` field remains the effective count of the normalized contributing
edges. It is retained for compatibility, but it is not an absolute coverage
measure: even tiny raw kernel weights normalize to one before their edge counts
are averaged. New fields therefore report unnormalized `action_match_mass`,
the best edge match, nearest action distance and kernel similarity, and nearest
observation distance and similarity. Consumers should use these absolute
diagnostics before allowing a consequence estimate to affect action.

`ContextMemoryConfig` fixes context and transition capacities. When full, low-visit stale contexts or edges are replaced and incident edges are removed. One experience creates a queryable transition immediately. Snapshots include every prototype, moment, edge, posterior, counter, and current anonymous observation and validate dimensions, finiteness, bounds, and graph references on restore.

## Relationship to existing organs

`VisualMemory` owns raw visual episodes, learned/replaced projection versions, and reprojection of personal records. `RelationalContextMemory` does not learn or version a sensory encoder. It accepts a stable vector produced by that layer or another sensor encoder and adds a discrete relational posterior and transition topology. `AdaptiveOrgan` and `PredictivePPO` retain their continuous recurrent context and policy learning; this module is not wired into either runtime yet.

The design takes two limited ideas from the literature. The Tolman-Eichenbaum Machine separates reusable relational structure from sensory grounding, while clone-structured cognitive graphs represent the same observation with different hidden states when sequence context differs. See Whittington et al., [*The Tolman-Eichenbaum Machine*](https://doi.org/10.1016/j.cell.2020.10.024), and George et al., [*Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps*](https://doi.org/10.1038/s41467-021-22559-5). This module is a small engineering prototype inspired by those principles. It is neither a TEM/CSCG implementation nor a claim about recovered fly memory physiology.

## Real-world trajectory evaluation

`scripts/develop_context_memory.py` generates open-loop trajectories in the actual `ArticulatedSensoriumWorld` with `body-v1` vision. Every transition advances the MuJoCo world for four 0.05 s substeps. The evaluated vector contains the 5×16 physical retina, bilateral odor, sound, touch, shade, illumination, local linear and angular velocity, six tarsal loads, and twelve joint positions and velocities. It excludes root pose, coordinates, headings, entity metadata, and IDs. A frozen 64-dimensional random projection provides the anonymous feature boundary; it is fit only for scale statistics on training trajectories.

The comparison baseline is a bounded five-neighbor store keyed by current sensation, current action, and the previous two action/outcome pairs. Train and test trajectories use disjoint world seeds and physical starting states. The alias subset is the held-out top quartile by local successor disagreement divided by current observation/action distance: these are views that match training inputs but have divergent nearby successors.

The default deterministic run uses eight training trajectories, three held-out trajectories, 96 transitions each, and equal 640-transition capacity. On the current local MuJoCo 3.12.0 run:

| Held-out metric | Relational contexts | Two-step history kNN |
| --- | ---: | ---: |
| Overall next-feature MSE | 0.2045 | 0.2531 |
| Aliased-subset next-feature MSE, 72 transitions | 0.1860 | 0.2193 |
| Outcome MSE | 0.0177 | 0.0296 |

The relational model used its configured 160 contexts and 223 transition prototypes after 768 training transitions. Uncertainty/error correlation was positive but weak (`0.068`), so the uncertainty formula still needs calibration on more varied experience before runtime use. These figures are one seeded research run rather than evidence of general navigation or biological cognition.

The rapid-binding check selected the largest-change physical training transition. Before binding, the empty memory's next-feature MSE was `1.0820`; immediately after one experience it was below `5e-16` for the same anonymous observation and action. This check demonstrates immediate storage and retrieval, while the disjoint held-out trajectory metrics above measure generalization.

Run it with:

```bash
.venv/bin/python scripts/develop_context_memory.py
```

The script also performs a one-transition rapid-binding measurement and exact snapshot round-trip. It prints one JSON report and does not write training state or alter a live resident.

## Minimal use

```python
from chreatures.context_memory import ContextMemoryConfig, RelationalContextMemory

memory = RelationalContextMemory(ContextMemoryConfig(
    feature_dim=64,
    action_dim=3,
    outcome_dim=4,
))

memory.begin(first_sensory_vector, learn=True)
prediction = memory.predict(executed_action)
memory.step(executed_action, next_sensory_vector, outcome_vector, learn=True)
```

For held-out inference, call `reset()`, then `begin(..., learn=False)` and `step(..., learn=False)`. The observed next sensation advances the posterior without changing stored contexts or transitions.
