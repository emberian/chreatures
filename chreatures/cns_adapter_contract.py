"""Authoritative immutable embodiment-driven MaleCNS V4 service contract."""
from __future__ import annotations
import hashlib, json, os, struct
from pathlib import Path
import numpy as np

FORMAT = "chreatures-cns-service-v4"
CONTROLLER_FORMAT = "chreatures-cns-context-controller-v1"
MAGIC = b"CHCNS4\0\0"
DIMENSIONS = dict(
    neurons=165122,
    edges=25563197,
    sites=1771,
    receptors=4107,
    receptor_types=10,
    site_edges=4669,
    body_targets=11798,
    neuron_types=11752,
    body_inputs=807,
    context_inputs=12,
    context_targets=1314,
    motor_outputs=92,
    motor_targets=815,
    latent=512,
    readout_rank=64,
    modulator_families=3,
)
ARRAY_SPECS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight_bits", "<u2", (25563197,)),
    ("graph.channel", "<u4", (165122,)),
    ("atlas.receptor_rows", "<u4", (4107,)),
    ("atlas.receptor_type", "<u4", (4107,)),
    ("atlas.receptor_ptr", "<u4", (4108,)),
    ("atlas.site_indices", "<u4", (4669,)),
    ("atlas.site_weight", "<f4", (4669,)),
    ("atlas.body_rows", "<u4", (11798,)),
    ("atlas.body_mask", "<f4", (11798, 807)),
    ("atlas.context_rows", "<u4", (1314,)),
    ("atlas.motor_rows", "<u4", (815,)),
    ("atlas.motor_mask", "<f4", (92, 815)),
    ("atlas.neuron_type", "<u4", (165122,)),
    ("optic.spectral_logits", "<f4", (10, 3)),
    ("optic.gain_raw", "<f4", (10,)),
    ("optic.bias", "<f4", (10,)),
    ("body.mean", "<f4", (807,)),
    ("body.scale", "<f4", (807,)),
    ("body.weight", "<f4", (11798, 807)),
    ("body.bias", "<f4", (11798,)),
    ("context.weight", "<f4", (1314, 12)),
    ("context.bias", "<f4", (1314,)),
    ("dynamics.baseline_raw", "<f4", (11752,)),
    ("dynamics.recurrent_gain_raw", "<f4", (11752,)),
    ("dynamics.tau_raw", "<f4", (11752,)),
    ("dynamics.adaptation_gain_raw", "<f4", (11752,)),
    ("dynamics.adaptation_tau_raw", "<f4", (11752,)),
    ("dynamics.release_tau_raw", "<f4", (11752,)),
    ("dynamics.release_use_raw", "<f4", (11752,)),
    ("dynamics.mod_gain_raw", "<f4", (11752, 3)),
    ("dynamics.mod_adaptation_raw", "<f4", (11752, 3)),
    ("dynamics.modulation_tau_raw", "<f4", (3,)),
    ("afferent.neutral_drive", "<f4", (165122,)),
    ("readout.projection.weight", "<f4", (64, 165122)),
    ("readout.output.weight", "<f4", (512, 64)),
    ("readout.output.bias", "<f4", (512,)),
    ("motor.reference_rate", "<f4", (815,)),
    ("motor.rate_scale", "<f4", (815,)),
    ("motor.weight", "<f4", (92, 815)),
    ("motor.intercept", "<f4", (92,)),
)
STATIC_NAMES = {n for n, _, _ in ARRAY_SPECS[:15]}
PARAMETER_ORDER = tuple(
    n for n, _, _ in ARRAY_SPECS[15:] if n != "afferent.neutral_drive"
)
PARAMETER_COUNT = sum(
    int(np.prod(s)) for n, _, s in ARRAY_SPECS if n in PARAMETER_ORDER
)
SENSORY_DIM = 6120
LATENT_DIM = 512
MOTOR_DIM = 92
CONTEXT_DIM = 12
GRAPH_QUANTIZATION = {
    "storage": "ieee-754-binary16-bits-little-endian",
    "rounding": "round-to-nearest-ties-to-even",
    "compute": "decode-once-to-float32",
}


def canonical(v):
    return json.dumps(
        v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _hash(v):
    if (
        not isinstance(v, str)
        or len(v) != 64
        or any(c not in "0123456789abcdef" for c in v)
    ):
        raise ValueError("expected lowercase SHA-256")


def adapter_identity_payload(m):
    return {
        k: m[k]
        for k in (
            "format",
            "graph_sha256",
            "atlas_sha256",
            "anatomy_sha256",
            "morphology_sha256",
            "sensory_schema_sha256",
            "actuator_schema_sha256",
            "motor_calibration_sha256",
            "graph_source_weight_sha256",
            "graph_quantization",
            "readout_mask_sha256",
            "dimensions",
            "parameter_order",
            "array_sha256",
        )
    }


def service_identity(m, file_hash):
    _hash(file_hash)
    if m.get("format") != FORMAT or m.get("dimensions") != DIMENSIONS:
        raise ValueError("CNS service metadata differs")
    if m.get("graph_quantization") != GRAPH_QUANTIZATION:
        raise ValueError("CNS graph quantization differs")
    for k in (
        "graph_sha256",
        "atlas_sha256",
        "anatomy_sha256",
        "morphology_sha256",
        "sensory_schema_sha256",
        "actuator_schema_sha256",
        "motor_calibration_sha256",
        "graph_source_weight_sha256",
        "readout_mask_sha256",
        "adapter_sha256",
    ):
        _hash(m[k])
    return dict(
        format=FORMAT,
        graph_sha256=m["graph_sha256"],
        atlas_sha256=m["atlas_sha256"],
        anatomy_sha256=m["anatomy_sha256"],
        morphology_sha256=m["morphology_sha256"],
        sensory_schema_sha256=m["sensory_schema_sha256"],
        actuator_schema_sha256=m["actuator_schema_sha256"],
        motor_calibration_sha256=m["motor_calibration_sha256"],
        graph_source_weight_sha256=m["graph_source_weight_sha256"],
        graph_quantization=m["graph_quantization"],
        readout_mask_sha256=m["readout_mask_sha256"],
        adapter_sha256=m["adapter_sha256"],
        service_artifact_sha256=file_hash,
        sensory_dim=SENSORY_DIM,
        latent_dim=LATENT_DIM,
        motor_dim=MOTOR_DIM,
        context_dim=CONTEXT_DIM,
    )


def neutral_afferent_drive(a):
    out = np.zeros(165122, dtype="<f4")
    supported = np.diff(a["atlas.receptor_ptr"]) > 0
    out[a["atlas.receptor_rows"]] = (
        1 / (1 + np.exp(-a["optic.bias"][a["atlas.receptor_type"]])) * supported
    )
    out[a["atlas.body_rows"]] = 1 / (1 + np.exp(-a["body.bias"]))
    return out


def validate_arrays(a):
    if set(a) != {x[0] for x in ARRAY_SPECS}:
        raise ValueError("CNS service arrays differ from V4 contract")
    for n, d, s in ARRAY_SPECS:
        x = a[n]
        if not isinstance(x, np.ndarray) or x.shape != s or x.dtype != np.dtype(d):
            raise ValueError(f"CNS tensor {n} must be {d}{s}")
        if x.dtype.kind == "f" and not np.isfinite(x).all():
            raise ValueError(f"CNS tensor {n} is nonfinite")
    if (
        a["graph.crow"][0] != 0
        or a["graph.crow"][-1] != 25563197
        or np.any(np.diff(a["graph.crow"].astype(np.int64)) < 0)
    ):
        raise ValueError("invalid graph CSR")
    if np.any(a["graph.col"] >= 165122) or np.any(a["graph.channel"] > 4):
        raise ValueError("invalid graph index/channel")
    decoded_weight = a["graph.weight_bits"].view("<f2").astype("<f4")
    if not np.isfinite(decoded_weight).all():
        raise ValueError("quantized graph weights must decode to finite float32")
    source_channel = a["graph.channel"][a["graph.col"]]
    if np.any(decoded_weight[source_channel == 0] != 0):
        raise ValueError("unknown-transmitter graph edges must remain zero")
    ptr = a["atlas.receptor_ptr"]
    if ptr[0] != 0 or ptr[-1] != 4669 or np.any(np.diff(ptr.astype(np.int64)) < 0):
        raise ValueError("invalid receptor CSR")
    if np.any(a["atlas.receptor_type"] >= 10) or np.any(
        a["atlas.site_indices"] >= 1771
    ):
        raise ValueError("invalid receptor type/site index")
    if np.any(a["atlas.site_weight"] <= 0):
        raise ValueError("receptor site weights must be positive")
    for start, stop in zip(ptr[:-1], ptr[1:], strict=True):
        if stop > start and not np.isclose(
            a["atlas.site_weight"][start:stop].sum(), 1, atol=2e-6
        ):
            raise ValueError("receptor mixture must sum to one")
    for n, b in (
        ("atlas.receptor_rows", 165122),
        ("atlas.body_rows", 165122),
        ("atlas.context_rows", 165122),
        ("atlas.motor_rows", 165122),
        ("atlas.neuron_type", 11752),
    ):
        if np.any(a[n] >= b):
            raise ValueError(f"out-of-range {n}")
    for n in (
        "atlas.receptor_rows",
        "atlas.body_rows",
        "atlas.context_rows",
        "atlas.motor_rows",
    ):
        rows = a[n]
        if np.any(rows[1:] <= rows[:-1]):
            raise ValueError(f"{n} must be sorted and unique")
    injected = np.concatenate(
        (a["atlas.receptor_rows"], a["atlas.body_rows"], a["atlas.context_rows"])
    )
    if np.unique(injected).size != injected.size:
        raise ValueError("injected CNS row sets overlap")
    if np.intersect1d(a["atlas.motor_rows"], injected).size:
        raise ValueError("motor rows overlap injected rows")
    if np.any((a["atlas.body_mask"] != 0) & (a["atlas.body_mask"] != 1)) or np.any(
        (a["atlas.motor_mask"] != 0) & (a["atlas.motor_mask"] != 1)
    ):
        raise ValueError("structural masks must be binary")
    if np.any(a["body.scale"] <= 0):
        raise ValueError("body scale must be positive")
    if np.any(a["motor.rate_scale"] <= 0):
        raise ValueError("motor rate scale must be positive")
    if np.any((a["motor.reference_rate"] < 0) | (a["motor.reference_rate"] > 1)):
        raise ValueError("motor reference rates must be in [0,1]")
    mask = np.ones(165122, np.uint8)
    mask[
        np.concatenate(
            (a["atlas.receptor_rows"], a["atlas.body_rows"], a["atlas.context_rows"])
        )
    ] = 0
    if not np.allclose(
        a["afferent.neutral_drive"], neutral_afferent_drive(a), rtol=0, atol=2e-7
    ):
        raise ValueError("neutral drive differs")
    return mask


def metadata_for(
    arrays,
    *,
    graph_sha256,
    atlas_sha256,
    anatomy_sha256,
    morphology_sha256,
    sensory_schema_sha256,
    actuator_schema_sha256,
    motor_calibration_sha256,
    graph_source_weight_sha256,
    training_status,
    provenance=None,
):
    for v in (
        graph_sha256,
        atlas_sha256,
        anatomy_sha256,
        morphology_sha256,
        sensory_schema_sha256,
        actuator_schema_sha256,
        motor_calibration_sha256,
        graph_source_weight_sha256,
    ):
        _hash(v)
    if training_status not in {"initialized-untrained", "trained"}:
        raise ValueError("invalid training status")
    mask = validate_arrays(arrays)
    m = dict(
        format=FORMAT,
        graph_sha256=graph_sha256,
        atlas_sha256=atlas_sha256,
        anatomy_sha256=anatomy_sha256,
        morphology_sha256=morphology_sha256,
        sensory_schema_sha256=sensory_schema_sha256,
        actuator_schema_sha256=actuator_schema_sha256,
        motor_calibration_sha256=motor_calibration_sha256,
        graph_source_weight_sha256=graph_source_weight_sha256,
        graph_quantization=GRAPH_QUANTIZATION,
        readout_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        dimensions=DIMENSIONS,
        parameter_order=list(PARAMETER_ORDER),
        training_status=training_status,
        provenance=provenance or {},
        array_sha256={
            n: hashlib.sha256(np.ascontiguousarray(arrays[n]).tobytes()).hexdigest()
            for n, _, _ in ARRAY_SPECS
        },
    )
    m["adapter_sha256"] = hashlib.sha256(
        canonical(adapter_identity_payload(m))
    ).hexdigest()
    return m


def write_service_artifact(path, arrays, **kwargs):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    meta = metadata_for(arrays, **kwargs)
    payload = canonical(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("xb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)
            for n, d, _ in ARRAY_SPECS:
                f.write(np.ascontiguousarray(arrays[n], dtype=d).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": h.hexdigest(),
        "metadata": meta,
    }


def load_service_artifact(path):
    """Memory-map and authenticate one current CHCNS3 artifact."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("requires a CHCNS4 service artifact")
        metadata_bytes = stream.read(4)
        if len(metadata_bytes) != 4:
            raise ValueError("truncated CHCNS4 metadata length")
        metadata_length = struct.unpack("<I", metadata_bytes)[0]
        metadata = json.loads(stream.read(metadata_length))
    if metadata.get("format") != FORMAT or metadata.get("dimensions") != DIMENSIONS:
        raise ValueError("CHCNS4 metadata contract differs")
    if metadata.get("graph_quantization") != GRAPH_QUANTIZATION:
        raise ValueError("CHCNS4 graph quantization differs")
    for name in (
        "graph_sha256",
        "atlas_sha256",
        "anatomy_sha256",
        "morphology_sha256",
        "sensory_schema_sha256",
        "actuator_schema_sha256",
        "motor_calibration_sha256",
        "graph_source_weight_sha256",
        "readout_mask_sha256",
        "adapter_sha256",
    ):
        _hash(metadata.get(name))
    offset = 12 + metadata_length
    arrays = {}
    for name, dtype, shape in ARRAY_SPECS:
        value = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        expected = metadata.get("array_sha256", {}).get(name)
        if expected is None or hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"CHCNS4 tensor receipt differs: {name}")
        arrays[name] = value
        offset += value.nbytes
    if path.stat().st_size != offset:
        raise ValueError("CHCNS4 artifact has trailing or missing bytes")
    validate_arrays(arrays)
    return arrays, metadata
