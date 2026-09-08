#!/usr/bin/env python3
"""Derive compact antennal collision proxies from pinned NeuroMechFly meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"truncated STL: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + triangles * 50:
        raise ValueError(f"expected binary STL records: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.frombuffer(data, dtype=dtype, count=triangles, offset=84)
    # The pinned MJCF applies mesh scale=1000, converting source mesh lengths
    # to its millimetre model coordinates.
    return records["vertices"].reshape(-1, 3).astype(np.float64) * 1000.0


def ellipsoid_proxy(vertices: np.ndarray) -> tuple[list[float], list[float], float]:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    center = 0.5 * (low + high)
    half = np.maximum(0.5 * (high - low), 1e-6)
    normalized_radius = np.sqrt(np.sum(((vertices - center) / half) ** 2, axis=1))
    inflation = float(np.quantile(normalized_radius, 0.95))
    size = half * inflation
    coverage = float(np.mean(np.sum(((vertices - center) / size) ** 2, axis=1) <= 1.0 + 1e-12))
    return center.tolist(), size.tolist(), coverage


def capsule_proxy(vertices: np.ndarray) -> tuple[list[float], float, float]:
    center = vertices.mean(axis=0)
    covariance = np.cov(vertices - center, rowvar=False)
    _, axes = np.linalg.eigh(covariance)
    axis = axes[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projection = (vertices - center) @ axis
    radial = (vertices - center) - np.outer(projection, axis)
    radius = float(np.quantile(np.linalg.norm(radial, axis=1), 0.95))
    start = center + projection.min() * axis
    end = center + projection.max() * axis
    nearest_projection = np.clip(projection, projection.min(), projection.max())
    nearest = center + np.outer(nearest_projection, axis)
    coverage = float(np.mean(np.linalg.norm(vertices - nearest, axis=1) <= radius + 1e-12))
    return [*start.tolist(), *end.tolist()], radius, coverage


def mirror_y(values: list[float]) -> list[float]:
    mirrored = values.copy()
    for index in range(1, len(mirrored), 3):
        mirrored[index] = -mirrored[index]
    return mirrored


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    default_source = here / "assets" / "neuromechfly-2.1.0-ca65a510-ypr"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=here / "assets" / "antenna-contact-proxies-v1.json")
    args = parser.parse_args()
    source = args.source.resolve()
    schema = json.loads((source / "schema.json").read_text())
    proxies = []
    for segment in ("pedicel", "funiculus", "arista"):
        mesh = source / "model" / f"l_{segment}.stl"
        vertices = binary_stl_vertices(mesh)
        if segment == "arista":
            fromto, radius, coverage = capsule_proxy(vertices)
            left = {"shape": "capsule", "fromto_mm": fromto, "size_mm": [radius]}
        else:
            position, size, coverage = ellipsoid_proxy(vertices)
            left = {"shape": "ellipsoid", "position_mm": position, "size_mm": size}
        for side in ("l", "r"):
            proxy = dict(left)
            if side == "r":
                if "position_mm" in proxy:
                    proxy["position_mm"] = mirror_y(proxy["position_mm"])
                if "fromto_mm" in proxy:
                    proxy["fromto_mm"] = mirror_y(proxy["fromto_mm"])
            proxy.update(
                id=f"{side}_{segment}-contact-proxy",
                segment=f"{side}_{segment}",
                source_mesh=f"l_{segment}.stl",
                source_mesh_sha256=sha256(mesh),
                source_mesh_scale_to_model=[1000.0, -1000.0 if side == "r" else 1000.0, 1000.0],
                fit_vertex_coverage=coverage,
                fit_quantile=0.95,
                evidence_grade="ENGINEERED_PROXY_FROM_AUTHOR_MESH",
            )
            proxies.append(proxy)
    payload = {
        "format": "chreatures-antenna-contact-proxies-v1",
        "source": {
            **schema["source"],
            "model_xml_sha256": schema["files"]["model/model.xml"],
        },
        "units": {"length": "model millimetre"},
        "fit": {
            "pedicel_funiculus": "axis-aligned ellipsoid inflated to contain 95% of triangle-record vertices",
            "arista": "principal-axis capsule with radius equal to the 95th percentile distance from its axis",
            "limit": "Triangle-record sampling weights tessellation density. Proxies are synthetic contact approximations, not measured cuticle or receptor geometry.",
        },
        "physics_contract": {
            "mass_model_units": 0.0,
            "inertia_effect": "none; author segment mesh retains its original mass and inertia",
            "same_resident_contact": "excluded by resident collision bits",
            "environment_and_conspecific_contact": "enabled",
            "contact_parameters": {
                "margin_mm": 0.001,
                "solref_time_constant_s": 0.002,
                "solimp": [0.98, 0.99, 1e-5, 0.5, 3.0],
                "friction": [1.0, 1.0, 0.02],
                "evidence_grade": "ENGINEERED_COMPLIANT_CONTACT",
                "limit": "The 2 ms contact time constant is a stable numerical compliance and is not a measured antennal cuticle or Johnston-organ calibration.",
            },
            "controller_visibility": "only existing BODY807 joint position, velocity, generalized load and segment-contact channels",
        },
        "proxies": proxies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha256(args.output), "proxies": len(proxies)}))


if __name__ == "__main__":
    main()
