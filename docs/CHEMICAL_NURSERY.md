# Chemical resource encounters

`chemical-encounters-v4` is an opt-in training profile for learning from
physical encounters with finite chemical packets. It composes the existing
`EmbodiedTrainingWorld`, MuJoCo body, `Biosphere`, shared metabolic web,
diffusion field, acoustics, and full neural circuit. It does not add a second
simulator or expose packet coordinates, identities, distances, or bearings to
the controller.

The versioned conditions are
[`data/training/chemical-resource-encounters-v1.json`](../data/training/chemical-resource-encounters-v1.json).
They are environmental conditions rather than species preferences:

- three residents start with adequate body ATP and reserve and empty guts;
- four finite physical material packets retain their ordinary chemical stocks;
- stage 0 places packets 0.24--0.36 m away within 0.7 rad of body heading;
- stage 1 expands to 0.28--0.55 m and either side of the body;
- stage 2 expands to 0.32--0.90 m over the full circle;
- held-out seeds use 0.30--0.95 m over the full circle.

Placement is deterministic for the world seed, curriculum stage, and profile
hash. A downward ray through the actual MuJoCo scene selects a static supporting
surface near the resident's starting height. Each proposed position is then
applied to the complete habitat and rejected when the actual initial MuJoCo
contacts include the packet. The accepted positions, supporting entities, and
retry count are embedded in the physical world specification and therefore in
checkpoints. Terrain and manipulable objects remain present.

The controller receives only the existing 351 body-local sensor channels. The
full MaleCNS recurrence produces 384 readouts, and the policy additionally
receives six local physiology/activity values. The policy emits eight motor
coordinates. Eating remains a local physiological request derived from current
energy and gut readouts; material moves only after a positive eating request
and an actual contact within the mouth radius.

Every macro log records the executed action mean, absolute mean, and standard
deviation for all eight motor coordinates. Training and evaluation receipts
separately accumulate:

- positive eating-request steps;
- ordinary physical contact steps while eating;
- unique mouth-radius material contacts;
- material mass ingested into the gut;
- nutrition absorbed from the gut into the body.

This separation prevents an ordinary collision, a mouth encounter, ingestion,
and later absorption from being treated as the same event. Episode-end records
also preserve private chemical compartments and conserved chemistry accounting.

A canonical AMD run from the inherited state-conditioned learner uses:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
TRITON_CACHE_DIR=/tank/chreatures/cache/triton \
/tank/chreatures/envs/rocm-dev/bin/python scripts/learn_affordances.py \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --output /tank/chreatures/runs/learning/chemical-encounters-canonical-v1 \
  --worlds 16 --workers 16 --steps 20000 --episode-steps 5000 \
  --macro-steps 5 --rollout-decisions 64 --checkpoint-every 5000 \
  --eval-worlds 4 --eval-steps 5000 --seed 20260906 \
  --std-profile state-conditioned-v2 --reward-objective finite-energy-v1 \
  --training-profile chemical-encounters-v4 \
  --chemical-habitat data/habitats/chemical-reef.json \
  --chemical-biosphere data/biosphere/chemical-reef-v1.json \
  --chemical-conditions data/training/chemical-resource-encounters-v1.json \
  --warm-start-learner /tank/chreatures/runs/learning/embodied-homeostasis-v2-state-std-16x3-v1/checkpoints/learner-step-0009885.pt \
  --physical-backend fast --device cuda --brain-backend tiled
```

This is a new private cohort. The shared model, optimizer, feature-normalizer,
and state-conditioned variance parameters are transferred, while resident
context, neural state, bodies, chemical state, and worlds are reset. Results
must report this as transfer learning rather than exact continuation.

The preceding short frozen-layout comparison has a sanitized public
[`data/training/chemical-transfer-paired-v1.receipt.json`](../data/training/chemical-transfer-paired-v1.receipt.json)
receipt. Its source bulk receipt has SHA-256
`6c8969977a6bf320a614ab8af77b391581f67783523ac0a1c3231bd811fe0a5c`;
the public receipt has SHA-256
`9be32376902e9ae8700f95b653305b055128fd8bea08c8495239ca0cace53115`.
Neither canonical MaleCNS nor the derived G2 graph acquired a material packet in
that assay. Its inherited deterministic evaluations therefore establish only a
zero-acquisition transfer baseline, not autonomous chemical regulation or a
causal anatomical advantage.
