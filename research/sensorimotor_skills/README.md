# Sensorimotor play dataset

> **Historical research record.** The Python implementation and operational
> commands described here were retired from the current tree after the native
> rich developmental controller replaced them. Reproduce this experiment from
> Git commit `0caa7ef`; the findings and design rationale below remain part of
> the research record.

`trajectory-schema-v1.json` is the immutable first collector contract. Each
episode packet stores observation sequences one element longer than its action
sequence so achieved future sensory windows can be sampled without reconstructing
or advancing a world.

Partition keys such as `episode-000/world-002/resident-01` are bookkeeping only.
They never enter model columns. The manifest binds the graph, ports, motor organ,
physical habitat, chemical birth configuration, encounter conditions, training
profile, native extension, source code, exploration configuration, seeds, and
every packet and birth-checkpoint checksum.

The collector uses no food position, bearing, distance, entity identity, or
object label. Exploration is temporally correlated motor babble around the
inherited stochastic action. Its stop intervals are general motor coverage, not
a response to sensed or analyst-known food.

`oral_command` records the additional executed eating cause. Collector v1 uses
the supplied body-state law `clip((1-gut)*(1.1-energy), 0, 1)`; eating is not a
learned ninth motor axis in this dataset.

The inherited organ updates its private context with its own five-tick proposed
action. Per-tick exploration is applied afterward, so the organ context does not
encode the exact delivered exploratory action history. The manifest discloses
this bootstrap limitation and downstream training must use `executed_actions`.

The collector defaults to eight worlds. A training collection requires at least
six worlds so the data boundary can reserve two whole validation worlds and two
whole heldout worlds, and it requires at least 44 transitions per episode so all
five future-goal buckets through offset 40 are available. These checks run before
the recurrence graph is loaded or allocated. Smaller mechanics checks must opt in
with `--smoke`; their manifests set `training_readiness.ready` to false and give
the reason. A smoke record is never a worker fitting, selection, or evaluation
dataset, even when its requested dimensions happen to exceed the minima.

The first offline worker at source commit `545ca7a` fits its goal autoencoder on
training worlds and freezes it before recurrent worker optimization. Worker loss
is a per-axis action negative log likelihood weighted equally between all valid
steps and steps where the quantized action changed. Future offsets are sampled by
choosing uniformly among five buckets (`1..2`, `3..5`, `6..10`, `11..20`, and
`21..40`) and then uniformly within the chosen bucket. A forward-consistency loss
remains a proposed extension; it was not part of the recorded v1 result.
