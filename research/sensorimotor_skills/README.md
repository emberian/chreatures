# Sensorimotor play dataset

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
