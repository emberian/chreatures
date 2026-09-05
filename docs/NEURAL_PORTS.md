# Versioned neural ports

`retinal-v1` is a separate rich sensory and readout interface for the full
MaleCNS graph. It leaves the live 16-input/48-readout service unchanged. A new
service can load this bundle as an explicit `RemoteBrain` override when the 3D
runtime is ready to consume the larger schema.

The interface has 351 physical input channels and 384 readouts. Both maps are
SciPy CSR matrices. The graph remains the same 165,122-neuron,
25,563,197-edge MaleCNS organism; the port neither removes nor creates a graph
edge and never forms a dense neuron-by-neuron matrix.

## Physical vector

The first 320 channels preserve the complete `retina3d` tensor in row-major
`[elevation=5, azimuth=16, red/green/blue/proximity=4]` order. Names such as
`retina/e02/a07/green` make every coordinate explicit. The rest are:

| Observation | Channels | Encoding |
| --- | ---: | --- |
| bilateral odor, three identities | 6 | clipped concentration divided by 4 |
| local linear velocity XYZ | 6 | positive/negative opponent pairs, scale 4 m/s |
| local angular velocity XYZ | 6 | positive/negative opponent pairs, scale 8 rad/s |
| local contact normals | 7 | maximum positive/negative XYZ projection across up to eight contacts, plus contact count/8 |
| bilateral touch | 2 | clipped raw touch values |
| three tones | 3 | clipped amplitude divided by 2 |
| shade | 1 | clipped raw value |

Opponent channels keep zero motion at zero drive and preserve direction
without a hidden signed-to-unsigned projection. Contact normals are a fixed
directional envelope rather than the complete variable-length contact list;
the contact count retains whether several surfaces contributed.

```python
from chreatures.neural_ports import sensory_channel_dict

channels = sensory_channel_dict(world.sense("mica"))
```

Malformed shapes, non-finite values, excessive contact count, or out-of-range
retina/normal data are rejected. `sensory_channel_dict` returns the exact names
accepted by a service built from the same port bundle.

## Retinal routing and its limits

The builder reads the official, checksum-pinned MaleCNS annotation Feather and
matches 23,720 traced optic-lobe neurons to the loaded graph by exact body ID.
Their `assignedOlHex1` and `assignedOlHex2` fields provide measured optic-lobe
column coordinates. Those coordinates are linearly binned into the 16 by 5
synthetic camera raster. Because the optic-lobe lattice has nonrectangular
corners, 72 of 320 raster/component ports use the nearest measured hex
coordinate from the same declared type cohort. Each fallback is marked in the
serialized per-port metadata.

This supports a real spatial organization, but it does not make the whole map
measured retinotopy. The choice of which hex axis corresponds to camera
azimuth/elevation, rectangular binning, left/right camera alignment, RGB and
proximity cell-type cohorts, and gains are engineered. The renderer's RGB
channels are not fly photoreceptor spectra, and proximity has no claim to be a
native sensory quantity. The code never describes body-ID or hash ordering as
retinal anatomy.

Other channel families use exact MaleCNS annotations:

- Odor uses olfactory afferent class and anatomical side. The world's three
  fictional odor identities are an engineered partition of exact cell types;
  neurons without a resolved side join both sides instead of receiving fake
  laterality.
- Sound uses the auditory subclass. The three fictional tone bands are an
  engineered exact-type partition.
- Linear/angular motion uses mechanosensory proprioceptive afferents. Local
  XYZ/sign is mapped to leg entry nerve, side, and type cohorts as an explicit
  engineered interface.
- Contact direction and touch use tactile/mechanosensory afferents grouped by
  entry nerve and side. Contact count reaches the full 4,302-cell population.
- Shade uses the 91 annotated hygro/thermosensory afferents.

Every serialized input port records neuron count and exact superclass, class,
type, side, and soma-neuromere counts. This makes the engineered decisions
inspectable instead of burying them in an opaque projection.

## Readout atlas

The 384-row readout is an observation atlas across four domains:

| Domain | Rows | Covered neurons |
| --- | ---: | ---: |
| visual | 160 | 103,270 |
| mushroom body | 96 | 5,645 |
| navigation | 80 | 6,659 |
| efferent | 48 | 925 |

Visual includes optic-lobe intrinsic/sensory and visual projection/centrifugal
superclasses. Mushroom body includes ALPN, ALLN, ALIN, ALON, Kenyon cell, MBON,
and DAN classes. Navigation includes the CX class and ascending, descending,
sensory-ascending, and sensory-descending superclasses. Efferent includes motor
and efferent superclasses.

Within each domain, neurons are partitioned by exact type (falling back through
class, subclass, and superclass), side, and soma-neuromere or exit-nerve
region. The largest `quota - 1` exact signatures receive individual rows and
the remaining signatures form that domain's `other` row. Every row is nonempty
and computes a population mean. Per-row metadata records its signature and
source category counts. These are annotation-stratified observations, not a
trained behavioral decoder.

## Build, serialize, and use

On the bulk host:

```python
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import build_neural_port

graph = MaleCNSGraph.load("/tank/chreatures/data/malecns/derived")
ports = build_neural_port(
    graph,
    "data/ports/retinal-v1.json",
    annotation_path=(
        "/tank/chreatures/data/malecns/source/"
        "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ),
)
receipt = ports.save("/tank/chreatures/data/ports/retinal-v1-maps.npz")
```

The completed bundle is 413,005 bytes with SHA-256
`56fdf4657358628843412c5d72a11c4464eea75127616ad0626bb7bb3f0865b2`.
Its sparse input map is `[165122, 351]` with 62,716 nonzeros. Its sparse
readout map is `[384, 165122]` with 116,499 nonzeros. Building needs PyArrow
to read the pinned retinal coordinates; loading the serialized bundle does
not:

```python
from chreatures.neural_ports import NeuralPortBundle
from chreatures.remote_brain import RemoteBrain

ports = NeuralPortBundle.load(
    "/tank/chreatures/data/ports/retinal-v1-maps.npz", graph
)
brain = RemoteBrain(
    graph,
    capacity=16,
    device="cuda",
    **ports.remote_brain_kwargs(),
)
resident_senses = ports.channel_dict(world.sense("mica"))
```

The bundle pins the graph hash and refuses to load against another graph. It
contains input and readout names, CSR pointers/indices/values, the semantic
port spec, and per-port provenance. Serialization uses only numeric/string
NumPy arrays with `allow_pickle=False` on load.

`serve_brain.py` loads a serialized interface directly and checks all three
identities before allocating resident state: the graph hash embedded in the
bundle, the bundle's semantic spec hash, and the bundle file checksum and size
recorded by the spec. For example:

```shell
python scripts/serve_brain.py \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --port-spec /tank/chreatures/data/ports/retinal-v1.json \
  --device cuda --capacity 16 --port 8767 \
  --snapshot-dir /tank/chreatures/runs/server/malecns-retinal-v1/snapshots \
  --pid-file /tank/chreatures/runs/server/malecns-retinal-v1/brain.pid
```

`GET /v1/metadata` reports `brain.ports.mode = "versioned_bundle"`, the bundle and
spec hashes, and the 351 ordered input names and 384 ordered readout names in
the existing `brain.inputs` and `brain.readouts` fields. Omitting
`--port-bundle` keeps the original 16-input/48-readout defaults. A caller may
still use `--mapping-json` for explicit selector maps; the two mapping options
are mutually exclusive.

Snapshot requests may include `resident_ids` to create a named cohort
snapshot. Its receipt records `scope = "cohort"` and the exact ordered resident
IDs. Restoring that receipt changes only those residents, creates missing cohort
members when capacity permits, and leaves every other resident and its clock
untouched. Requests without `resident_ids` retain the original full-service
snapshot behavior. Snapshot metadata also pins the port interface, so state
cannot be restored into a graph service with different port mappings.

The isolated rich service on hbox runs at `127.0.0.1:8767`; the compact 16/48
service remains at `127.0.0.1:8765`. A bounded GPU replay on the rich service
produced 384 features for each resident, reproduced a two-resident cohort with
maximum absolute feature delta 0, and preserved an independent resident whose
clock advanced from 0.10 to 0.15 seconds across the cohort restore. The check
residents were removed afterward, leaving all 16 slots available.

## Optional learned features

Pretrained dense features can be added only through declared feature ports.
Add ordered entries to `physical_inputs.feature_ports` in a new versioned spec,
provide one explicit MaleCNS selector per named feature to
`build_neural_port(feature_ports=...)`, and pass a `[0,1]` value for every name
to `encode(..., feature_values=...)`. Missing, extra, non-finite, or out-of-range
features are rejected. There is no automatic resize, random projection, or
unnamed feature vector hidden behind the physical interface.
