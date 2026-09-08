# Fly embodiment atlas

This directory joins the actual MaleCNS motor and sensory annotations to the
pinned NeuroMechFly body identities without inventing muscle directions.

Build the compact atlas from the canonical derived neuron table and body
schema:

```bash
.venv/bin/python research/fly_embodiment/export_atlas.py \
  --neurons /tank/chreatures/data/malecns/derived/neurons.npz \
  --body-schema native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json \
  --physical-fixture native/browser-world/fixtures/fly-ecology/physics.json \
  --output research/fly_embodiment/fly-body-neural-atlas-v1.npz
```

`atlas.motor_mask[92,815]` is a structural mask for learned CNS-derived motor
weights. Position outputs use signed centered decoding; the adhesion, pump and
salivary outputs are bounded to `[0,1]`. It is not a muscle recruitment matrix. The optional 156-cohort layer
retains published descriptive leg, wing and haltere identities for audits and
auxiliary losses. Sensory masks retain the canonical nonvisual afferent rows;
they do not introduce a world-state bypass around the CNS.

`atlas.body_mask[11798,807]` connects the frozen BODY807 physical channel bank
to the nonvisual afferent rows. `body807-channel-schema.json` gives every channel
name, unit, raw range, evidence grade and mapping basis.
`motor92-channel-schema.json` similarly owns the canonical neural MOTOR92
semantic order. The physical fixture retains separate hashes for its compiled
M90 numeric actuator map and sensor-anchor/optic calibration payload.

See `source_ledger.json` for source revisions, hashes, claims and the concrete
effect of each source on the mapping.
