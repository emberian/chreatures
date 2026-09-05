# 3D life observatory

The observatory is a read-only evidence archive over the saved adult 3D world
and the eight-world developmental run. It verifies source hashes before reading
them, freezes compact time series, evaluates nonlinear dynamics on complete
held-out worlds, and links sources, summaries, models, and comparisons in a
native Universal Weave DAG. It never writes to a habitat or imports archive
records into a resident's personal memory.

Build the checked-in evidence with the isolated native GAM environment:

```bash
integrations/.venv/bin/python scripts/build_observatory.py
```

Inputs default to `runs/hollow-garden.json` and
`runs/development/initial-8x3-20260905`. Outputs default to
`integrations/artifacts/observatory3d/`. `manifest.json` hashes every published
artifact as a set; `observatory.json` contains verified source receipts and
physical, ecological, cognitive, and outcome summaries.

`adult_world_timeseries.csv` records the checkpoint's bounded resident history.
`development_world_timeseries.csv` aggregates each step within each independent
physical world while retaining nutrition, contact, effort, activity, prediction
error, learning progress, and next-step state/outcomes. It excludes neural
features and autobiographical memory contents.

## Whole-world evaluation

The native GAM analysis holds out worlds 03 and 07 in full. No time-adjacent row
from either world appears in training. One model predicts next-step mean energy
from current physiology and physical outcomes; another predicts next-step
locomotor distance from bodily state, activity, and effort; a third predicts
next-step forward-model error from current error, activity, learning progress,
phase, and developmental time. Each is compared with persistence and
training-mean baselines on the held-out worlds. Results, warnings, failures,
native build identity, persisted model hashes, and exact reload parity are
retained. These are descriptive predictions, not causal or behavioral-skill
claims. Persisted native model payloads are stored with deterministic gzip
compression (`*.gam.gz`); decompress one to a `.gam` file before passing it to
`gamfit.load`.

## Evidence graph and API

The Weave request uses content-addressed `urn:sha256:…` artifact references.
Universal Weave validates the independent DAG, serializes it, reloads the native
type, and checks equality. The final comparison has multiple parents: adult 3D
state, developmental population evidence, and successful held-out models.
`dag-view.json` provides frontend-ready nodes, edges, levels, lanes, and artifact
references.

The module exports `create_observatory_router()` and a default `router`. A server
can include it without giving the observatory any mutation route:

```python
from chreatures.observatory import router as observatory_router

app.include_router(observatory_router)
```

The router serves `GET /api/observatory`, `GET /api/observatory/graph`, and
receipt-checked `GET /api/observatory/artifacts/{artifact_name}`. Artifact names
must be present in the immutable manifest; arbitrary filesystem paths are never
served.
