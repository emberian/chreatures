"""Immutable CNS service tensors and binary boundary; no simulation numerics.

The large readout belongs to the neural service. Resident artifacts bind this
artifact's identity rather than copying its weights or receiving raw senses.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from collections.abc import Mapping

import numpy as np

FORMAT = "chreatures-cns-service-v1"
CONTROLLER_FORMAT = "chreatures-cns-only-controller-v1"
MAGIC = b"CHCNS1\0\0"
DIMENSIONS = dict(neurons=165122, edges=25563197, sites=1771, receptors=4107,
                  receptor_types=10, site_edges=4669, body_targets=11233,
                  neuron_types=11752, body_inputs=43, body_hidden=128,
                  inputs=5356, latent=512)
ARRAY_SPECS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight", "<f4", (25563197,)),
    ("atlas.receptor_rows", "<u4", (4107,)),
    ("atlas.receptor_type", "<u4", (4107,)),
    ("atlas.receptor_ptr", "<u4", (4108,)),
    ("atlas.site_indices", "<u4", (4669,)),
    ("atlas.site_weight", "<f4", (4669,)),
    ("atlas.body_rows", "<u4", (11233,)),
    ("atlas.neuron_type", "<u4", (165122,)),
    ("optic.spectral_logits", "<f4", (10, 3)),
    ("optic.gain_raw", "<f4", (10,)),
    ("optic.bias", "<f4", (10,)),
    ("body.mean", "<f4", (43,)),
    ("body.scale", "<f4", (43,)),
    ("body.input.weight", "<f4", (128, 43)),
    ("body.input.bias", "<f4", (128,)),
    ("body.output.weight", "<f4", (11233, 128)),
    ("body.output.bias", "<f4", (11233,)),
    ("dynamics.bias_raw", "<f4", (11752,)),
    ("dynamics.tau_raw", "<f4", (11752,)),
    ("dynamics.source_raw", "<f4", (11752,)),
    ("dynamics.target_raw", "<f4", (11752,)),
    ("dynamics.excitability_raw", "<f4", (11752,)),
    ("readout.weight", "<f4", (512, 165122)),
    ("readout.bias", "<f4", (512,)),
)
PARAMETER_ORDER = tuple(name for name, _, _ in ARRAY_SPECS[10:])
PARAMETER_COUNT = sum(int(np.prod(shape)) for _, _, shape in ARRAY_SPECS[10:])
SENSORY_DIM = DIMENSIONS["inputs"]
LATENT_DIM = DIMENSIONS["latent"]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def adapter_identity_payload(metadata):
    """Portable hash projection contains only strings, integer shapes and hashes.

    Floating provenance is bound by the file hash, avoiding differing Rust/Python
    JSON spellings of exponents in the internal mathematical artifact identity.
    """
    names = ("format", "graph_sha256", "atlas_sha256", "readout_mask_sha256",
             "dimensions", "parameter_order", "array_sha256")
    return {name: metadata[name] for name in names}


def _hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")


def service_identity(metadata, service_artifact_sha256):
    """Exact identity envelope shared by neural host and resident exporter."""
    _hash(service_artifact_sha256)
    if metadata.get("format") != FORMAT or metadata.get("dimensions") != DIMENSIONS:
        raise ValueError("CNS service metadata format/dimensions differ")
    names = ("graph_sha256", "atlas_sha256", "readout_mask_sha256", "adapter_sha256")
    for name in names:
        _hash(metadata[name])
    return dict(format=FORMAT, **{name: metadata[name] for name in names},
                service_artifact_sha256=service_artifact_sha256,
                sensory_dim=SENSORY_DIM, latent_dim=LATENT_DIM)


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate exact topology/packing and return the enforced readout mask."""
    if set(arrays) != {x[0] for x in ARRAY_SPECS}:
        raise ValueError("CNS service arrays differ from the one current contract")
    for name, dtype, shape in ARRAY_SPECS:
        a = arrays[name]
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != np.dtype(dtype):
            raise ValueError(f"CNS tensor {name} must be {dtype}{shape}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"CNS tensor {name} is nonfinite")
    for prefix, end in (("graph.crow", 25563197), ("atlas.receptor_ptr", 4669)):
        ptr = arrays[prefix]
        if ptr[0] != 0 or ptr[-1] != end or np.any(ptr[1:] < ptr[:-1]):
            raise ValueError(f"invalid CSR pointer: {prefix}")
    for name, bound in (("graph.col", 165122), ("atlas.receptor_rows", 165122),
                        ("atlas.body_rows", 165122), ("atlas.receptor_type", 10),
                        ("atlas.site_indices", 1771), ("atlas.neuron_type", 11752)):
        if np.any(arrays[name] >= bound):
            raise ValueError(f"out-of-range CNS index: {name}")
    receptors, body = arrays["atlas.receptor_rows"], arrays["atlas.body_rows"]
    if np.any(receptors[1:] <= receptors[:-1]) or np.any(body[1:] <= body[:-1]):
        raise ValueError("afferent rows must be unique and in ascending graph order")
    if np.intersect1d(receptors, body).size:
        raise ValueError("optic and body afferents overlap")
    ptr, weight = arrays["atlas.receptor_ptr"], arrays["atlas.site_weight"]
    if np.any(weight <= 0):
        raise ValueError("receptor site weights must be positive")
    for start, stop in zip(ptr[:-1], ptr[1:], strict=True):
        if stop > start and not np.isclose(weight[start:stop].sum(), 1, atol=2e-6):
            raise ValueError("supported receptor site mixtures must sum to one")
    if np.any(arrays["body.scale"] <= 0):
        raise ValueError("body normalizer scale must be positive")
    mask = np.ones(165122, dtype=np.uint8)
    mask[receptors] = 0
    mask[body] = 0
    return mask


def metadata_for(arrays, *, graph_sha256, atlas_sha256, training_status, provenance=None):
    _hash(graph_sha256)
    _hash(atlas_sha256)
    if training_status not in {"initialized-untrained", "trained"}:
        raise ValueError("training status must explicitly distinguish initialization")
    mask = validate_arrays(arrays)
    meta = dict(format=FORMAT, graph_sha256=graph_sha256, atlas_sha256=atlas_sha256,
                readout_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                dimensions=DIMENSIONS, parameter_order=list(PARAMETER_ORDER),
                training_status=training_status, provenance=provenance or {},
                array_sha256={name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
                              for name, _, _ in ARRAY_SPECS})
    meta["adapter_sha256"] = hashlib.sha256(canonical(adapter_identity_payload(meta))).hexdigest()
    return meta


def write_service_artifact(path, arrays, **metadata_arguments):
    """Write a new frozen artifact; existing artifacts are never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    meta = metadata_for(arrays, **metadata_arguments)
    payload = canonical(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(MAGIC)
            stream.write(struct.pack("<I", len(payload)))
            stream.write(payload)
            for name, dtype, _ in ARRAY_SPECS:
                stream.write(np.ascontiguousarray(arrays[name], dtype=dtype).tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication refuses a destination created concurrently.
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()
    with path.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                file_sha256=file_hash, metadata=meta)
