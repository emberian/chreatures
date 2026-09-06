# Remote full-graph brain service

`scripts/serve_brain.py` keeps the complete curated MaleCNS sparse graph and
private state for each resident on one AMD GPU. It binds to `127.0.0.1` by
default and is intended for an SSH tunnel from the local 3D runtime. It does
not expose the service on the LAN or internet.

The backend uses three sparse CSR matrices and never constructs a dense
neuron-by-neuron matrix:

- the signed, incoming-row-normalized full recurrent graph;
- a sparse annotation-derived afferent channel map;
- 48 disjoint, row-normalized, annotation-derived efferent readouts.

Every resident has private rates, adaptation, support, and simulated time.
The recurrence uses two substeps, `tau=0.16`, gain `0.92`, the rectified tanh
target, slow adaptation, and bounded support. The interface maps are engineered
from source annotations; their names and memberships do not claim fitted
physiology.

## Start and connect

On the project compute node, generated files, caches, logs, and snapshots belong
under `/tank/chreatures`. The project-owned replacement environment is used for
new launches; the original shared environment disappeared during development.
For the compact 16-input/48-readout interface:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=/tank/chreatures/service-src \
PYTORCH_KERNEL_CACHE_PATH=/tank/chreatures/cache/pytorch \
/tank/chreatures/envs/rocm-dev/bin/python \
  /tank/chreatures/service-src/scripts/serve_brain.py \
  --graph /tank/chreatures/data/malecns/derived \
  --device cuda --capacity 16 --bind 127.0.0.1 --port 8765 \
  --snapshot-dir /tank/chreatures/runs/server/malecns-v1/snapshots \
  --pid-file /tank/chreatures/runs/server/malecns-v1/brain.pid
```

From the local machine, tunnel a different local port if desired:

```sh
ssh -N -L 18765:127.0.0.1:8765 hbox
curl http://127.0.0.1:18765/v1/health
```

The server refuses non-loopback binds. Request bodies are capped at 1 MiB.
Full neural state never travels in an HTTP request: snapshot and restore use a
safe checkpoint ID for compressed files inside the configured server snapshot
directory and return a SHA-256 receipt.

### Explicit derived circuit service

The same AMD `RemoteBrain` service can run a compiled `CircuitBlueprint` graph.
Derived loading is opt-in and requires the expected graph, manifest, and port
bundle hashes. The canonical default does not infer or accept a derived graph.
For the checked generation-two research circuit:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=/tank/chreatures/service-src \
/tank/chreatures/envs/rocm-dev/bin/python \
  /tank/chreatures/service-src/scripts/serve_brain.py \
  --graph-kind derived \
  --graph /tank/chreatures/data/circuit-blueprints/lineage-g2-seed40 \
  --expected-graph-sha256 d4d927715b988e00249730dba3cb496210fc442b9fad5288cb2f0d8438331163 \
  --expected-graph-manifest-sha256 42a781ac806733b8819c87fffa393f3759dcfb79170732869fa98d9da21302a2 \
  --port-bundle /tank/chreatures/data/circuit-blueprints/lineage-g2-seed40/ports.npz \
  --expected-port-bundle-sha256 01b0e6038de2ef41fc631d8d7f055d507ffad111430ae2b09d11ada327ed7d0e \
  --device cuda --capacity 3 --microbatch-size 3 \
  --bind 127.0.0.1 --port 8770 \
  --snapshot-dir /tank/chreatures/runs/server/derived-g2/snapshots \
  --pid-file /tank/chreatures/runs/server/derived-g2/brain.pid
```

Startup verifies the graph's base and derived artifacts before allocating
state. Metadata reports the selected graph kind, exact graph and manifest
hashes, direct parent, measured root, blueprint, full port spec, and bundle
hashes. The derived identity is also embedded in the existing version-three
snapshot `ports` metadata, so restore rejects another derivation, source
manifest, or port artifact even when dimensions happen to match.

The 351-channel physical encoder is allowed across graph-specific port specs
only when its complete declarative preprocessing document has the same hash.
This covers channel order, shapes, scaling, contact limits, and optional feature
ports. The graph routing and readout identities remain distinct and continue to
be pinned by the full port spec and bundle hashes.

An isolated run used this command on hbox, then connected through an SSH
tunnel with the ordinary `NeuralClient`. One direct resident produced 384
readouts; its checksum-pinned snapshot replayed with maximum absolute delta
zero. A fresh three-resident articulated `Habitat3D` then ran two physical
steps, saved its whole checkpoint, and reproduced the next physical and neural
step exactly after restore. The private service and tunnel were stopped after
the check. The evidence is recorded in
`data/circuit-blueprints/derived-g2-service-v1.receipt.json`. This verifies
integration and identity enforcement, not behavioral benefit from the
synthetic circuit.

## Ordered API

`GET /v1/health` reports liveness, residents, and the next mutation sequence.
`GET /v1/metadata` reports the graph source/hash/counts, device/runtime,
dynamics, afferent channels, 48 feature names, residents, PID, and port.

Every POST is a mutation and must carry the exact integer `seq` reported as
`next_seq`. Failed requests do not consume a sequence number. This rejects
missing, duplicated, and reordered physics-to-neural updates.

Create resident state:

```json
{"seq":0,"resident_ids":["mica","fern","pip"]}
```

Send one ordered neural timestep to `POST /v1/step`:

```json
{
  "seq": 1,
  "dt": 0.05,
  "residents": [
    {"id":"mica","senses":{"<afferent-channel>":0.8}},
    {"id":"fern","senses":{"<afferent-channel>":0.2}},
    {"id":"pip","senses":{}}
  ]
}
```

Channel names come from `/v1/metadata`. Each value must be finite and in
`[0,1]`; omitted channels are zero. Each resident result contains:

```json
{
  "id": "mica",
  "time": 0.05,
  "features": [0.0046, 0.0047, 0.0045],
  "readouts": {"annotation-derived feature name": 0.0},
  "activity": 0.0,
  "activity_peak": 0.0,
  "support": 1.0
}
```

The example abbreviates the 48-element feature array. The response also
includes `feature_names` in vector order and the graph hash.
The local `AdaptiveOrgan(feature_dim=48)` consumes `features`; raw world
coordinates, object IDs, and target identities are absent.

Create and restore server-side checkpoints:

```json
{"seq":2,"name":"garden-000002"}
```

to `POST /v1/snapshot`, then:

```json
{"seq":3,"name":"garden-000002","sha256":"<receipt hash>"}
```

to `POST /v1/restore`. A checkpoint stores active resident IDs, all three full
neural state arrays, resident times, graph hash, mapping names, and dynamics.
Restore rejects a different graph, map, dynamics configuration, checksum,
shape, non-finite state, or resident set larger than server capacity.

Other ordered mutations are `POST /v1/residents/remove` and
`POST /v1/shutdown`. SIGTERM also exits cleanly. The PID file is removed only
by the process that owns it.

## Custom annotation maps

`--mapping-json` accepts `inputs`, optional `input_gains`, `readouts`, and
optional `readout_gains`, which are passed to `MaleCNSGraph.build_input_map`
and `build_readout_map`. The service validates sparse `N x C` and `48 x N`
shapes at startup. The default maps come from the pinned artifact manifest and
are preferable for reproducible runs.

Selector overrides are disabled for derived mode. A derived service must use
the port bundle emitted by its blueprint compilation and pin that file on the
command line.

## Measured full-graph run

The production service is live on hbox at `127.0.0.1:8765`, PID `3377017`.
Its run directory is `/tank/chreatures/runs/server/malecns-v1`, with log
`server.log`, measurement `measurement.json`, PID file `brain.pid`, and
checkpoints under `snapshots/`. The service was intentionally left running for
the local 3D controller. Its next mutation sequence after measurement is 133,
with residents `mica`, `fern`, and `pip` loaded.

| Measurement | Result |
| --- | ---: |
| Graph | 165,122 neurons; 25,563,197 directed edges; 124,025,046 synapses |
| Dataset SHA-256 | `48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625` |
| Startup to ready PID | 5.908 s |
| Stream | 128 requests × 3 residents at `dt=0.05` |
| Median / p95 request latency | 9.82 ms / 10.60 ms |
| Mean / maximum request latency | 13.34 ms / 190.34 ms |
| Throughput | 74.95 requests/s; 224.85 resident-steps/s |
| PyTorch allocated GPU memory | 238,679,040 bytes |
| Driver-reported VRAM after stream | 577,626,112 of 12,868,124,672 bytes |
| Checkpoint | 4,489,421 bytes; SHA-256 `63b4de5724a1442b49ea392a584652625a29c5a611f60b7b8db7d342928da37e` |
| Restore replay maximum delta | 0.0 |

The maximum latency was the first warmup request; p95 remains well inside a
50 ms, 20 Hz physics timestep. Three different bilateral odor/color/tone and
contact streams produced distinct resident feature vectors after 6.4 simulated
seconds. Pairwise maximum feature differences were `1.11e-4` (Mica/Fern),
`1.05e-4` (Mica/Pip), and `2.61e-5` (Fern/Pip).

The snapshot was restored by ID plus checksum and the same subsequent sensory
step was replayed. All 48 features, activity, support, and resident time matched
exactly. The complete evidence is mirrored locally under
`runs/server/malecns-v1/`.

The deployed source hashes match the local files:

- `remote_brain.py`: `8289254aa548f7dd89368cea23af1f9b2496ffeebebfc755b32ea8470297ec2e`
- `serve_brain.py`: `b766448739ce4b421362c80a8b57535e92321873e9ce340f26f6a334937008eb`
- `malecns.py`: `5105c4ba8444baeca0114e49adaf4f0e8d9fc838e46d27dfb875cb8b68cb8b2f`
