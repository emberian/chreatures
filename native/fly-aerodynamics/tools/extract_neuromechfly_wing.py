#!/usr/bin/env python3
"""Reduce the pinned NeuroMechFly left-wing STL to deterministic blade strips.

This is a build-time provenance tool. The production load kernel has no Python
dependency. It uses the convex projection because the STL is a closed, thick
visual/collision mesh; summing triangle areas would count both wing surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REVISION = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
EXPECTED_STL_SHA256 = "142f77f15008fd75e1490b546bb859cf98c056121f1abd65acea2052c4e7764c"
EXPECTED_MODEL_SHA256 = "67aeb93868ac0cbf27c6f2645148a04f3cc77f0b0105bd9fa4ea15fc8755aa06"
EXPECTED_BODY_SCHEMA_SHA256 = "d8c3ff3d22b7f68ec8fb752ba210689ce6531820df57edc96c3bd305766a2d5a"
STRIP_COUNT = 24


def read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_STL_SHA256:
        raise ValueError("left-wing STL does not match the pinned author asset")
    if len(raw) < 84:
        raise ValueError("truncated STL")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError("only the pinned binary STL representation is supported")
    vertices = np.empty((count * 3, 3), dtype=np.float64)
    for index in range(count):
        record = 84 + index * 50
        vertices[index * 3 : index * 3 + 3] = np.asarray(
            struct.unpack_from("<9f", raw, record + 12), dtype=np.float64
        ).reshape(3, 3)
    # The author meshes are meters on disk and scaled by 1000 in the MJCF.
    return np.unique(vertices, axis=0) * 1000.0


def cross2(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - origin[0]) * (b[1] - origin[1]) -
                 (a[1] - origin[1]) * (b[0] - origin[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(ordered) < 3:
        raise ValueError("projected mesh has no planform")
    lower: list[np.ndarray] = []
    for point_tuple in ordered:
        point = np.asarray(point_tuple)
        while len(lower) >= 2 and cross2(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point_tuple in reversed(ordered):
        point = np.asarray(point_tuple)
        while len(upper) >= 2 and cross2(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def clip_span(poly: np.ndarray, boundary: float, keep_greater: bool) -> np.ndarray:
    if len(poly) == 0:
        return poly
    result: list[np.ndarray] = []
    previous = poly[-1]
    previous_inside = previous[0] >= boundary if keep_greater else previous[0] <= boundary
    for current in poly:
        current_inside = current[0] >= boundary if keep_greater else current[0] <= boundary
        if current_inside != previous_inside:
            fraction = (boundary - previous[0]) / (current[0] - previous[0])
            result.append(previous + fraction * (current - previous))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(result)


def polygon_area_centroid(poly: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * shifted[:, 1] - shifted[:, 0] * poly[:, 1]
    signed_area = 0.5 * float(cross.sum())
    if signed_area <= 0.0:
        raise ValueError("strip polygon is degenerate or clockwise")
    centroid = ((poly + shifted) * cross[:, None]).sum(axis=0) / (6.0 * signed_area)
    return signed_area, centroid


def vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(attribute: str) -> list[float]:
    return [float(value) for value in attribute.split()]


def body_audit(model_path: Path, schema_path: Path) -> dict:
    if file_sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("model.xml does not match the pinned imported body")
    if file_sha256(schema_path) != EXPECTED_BODY_SCHEMA_SHA256:
        raise ValueError("body schema does not match the pinned imported body")
    root = ET.parse(model_path).getroot()
    schema = json.loads(schema_path.read_text())
    result = {}
    for side in ("l", "r"):
        body_name = f"fly/{side}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"missing {body_name}")
        geom = body.find(f"geom[@name='{body_name}']")
        if geom is None:
            raise ValueError(f"missing {body_name} geom")
        short_name = f"{side}_wing"
        segment = next(item for item in schema["body_segments"] if item["id"] == short_name)
        joints = [item for item in schema["joint_dofs"] if item["child"] == short_name]
        actuators = [item for item in schema["actuators"] if item["target"] in {j["id"] for j in joints}]
        result[short_name] = {
            "body_pos_parent_frame_mm": values(body.attrib["pos"]),
            "body_quaternion_wxyz": values(body.attrib["quat"]),
            "mass_model_units": float(geom.attrib["mass"]),
            "base_compiled_body_id": segment["compiled_body_id"],
            "joints_yaw_pitch_roll": joints,
            "position_servos_yaw_pitch_roll": actuators,
        }
    return {
        "body_schema_sha256": EXPECTED_BODY_SCHEMA_SHA256,
        "model_xml_sha256": EXPECTED_MODEL_SHA256,
        "total_mass_model_units": schema["fixture"]["total_mass_model_units"],
        "mass_unit_status": schema["fixture"]["mass_unit_status"],
        "model_units": {"length": "millimeter", "time": "second", "angle": "radian"},
        "wing_bodies": result,
    }


def build_payload(stl_path: Path, model_path: Path, schema_path: Path) -> dict:
    vertices_mm = read_binary_stl(stl_path)
    centered = vertices_mm - vertices_mm.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(vertices_mm))
    normal = eigenvectors[:, 0]
    if normal[2] < 0.0:
        normal = -normal
    span = eigenvectors[:, 2]
    projected_from_hinge = vertices_mm - np.outer(vertices_mm @ normal, normal)
    farthest = projected_from_hinge[np.argmax(np.linalg.norm(projected_from_hinge, axis=1))]
    if np.dot(span, farthest) < 0.0:
        span = -span
    chord = np.cross(span, normal)
    chord /= np.linalg.norm(chord)
    span /= np.linalg.norm(span)
    normal = np.cross(chord, span)
    normal /= np.linalg.norm(normal)

    coordinates = np.column_stack((vertices_mm @ span, vertices_mm @ chord))
    hull = convex_hull(coordinates)
    full_area_mm2, _ = polygon_area_centroid(hull)
    span_min = float(hull[:, 0].min())
    span_max = float(hull[:, 0].max())
    edges = np.linspace(span_min, span_max, STRIP_COUNT + 1)
    normal_offset_mm = float(vertices_mm @ normal @ np.ones(len(vertices_mm)) / len(vertices_mm))

    strips = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        clipped = clip_span(hull, float(lo), True)
        clipped = clip_span(clipped, float(hi), False)
        area_mm2, center_sc = polygon_area_centroid(clipped)
        chord_extent_mm = float(clipped[:, 1].max() - clipped[:, 1].min())
        center_mm = span * center_sc[0] + chord * center_sc[1] + normal * normal_offset_mm
        strips.append({
            "index": index,
            "span_interval_mm": [float(lo), float(hi)],
            "center_local_mm": vector(center_mm),
            "area_mm2": area_mm2,
            "chord_mm": chord_extent_mm,
        })

    payload = {
        "format": "chreatures.neuromechfly-wing-planform.v1",
        "source": {
            "project": "FlyGym / NeuroMechFly",
            "version": "2.1.0",
            "revision": REVISION,
            "asset": "model/l_wing.stl",
            "asset_sha256": EXPECTED_STL_SHA256,
            "license": "Apache-2.0",
            "mesh_length_unit": "meter",
            "mjcf_mesh_scale": 1000.0,
        },
        "derivation": {
            "method": "PCA_PLANE_CONVEX_HULL_EQUAL_SPAN_STRIPS",
            "strip_count": STRIP_COUNT,
            "vertex_count_unique": int(len(vertices_mm)),
            "triangle_count": int((stl_path.stat().st_size - 84) // 50),
            "pca_eigenvalues_mm2": vector(eigenvalues),
            "caveat": "Convex projection fills concavities and ignores thickness/corrugation.",
        },
        "left": {
            "semantic_id": "l_wing",
            "handed_frame": "chord_cross_span_equals_normal",
            "chord_axis_local": vector(chord),
            "span_axis_local": vector(span),
            "normal_axis_local": vector(normal),
            "span_interval_mm": [span_min, span_max],
            "span_extent_mm": span_max - span_min,
            "planform_area_mm2": full_area_mm2,
            "mean_chord_mm": full_area_mm2 / (span_max - span_min),
            "elements": strips,
        },
        "right": {
            "semantic_id": "r_wing",
            "derivation": "author Y reflection; normal uses axial-vector reflection",
            "handed_frame": "chord_cross_span_equals_normal",
        },
        "imported_body_audit": body_audit(model_path, schema_path),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["derivation_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def rust_vec(values: list[float], scale: float = 1.0) -> str:
    return "Vec3::new(" + ", ".join(f"{value * scale:.17e}" for value in values) + ")"


def write_rust(payload: dict, path: Path) -> None:
    left = payload["left"]
    reflection = lambda values: [values[0], -values[1], values[2]]
    right_normal = [-value for value in reflection(left["normal_axis_local"])]
    lines = [
        "// @generated by tools/extract_neuromechfly_wing.py; do not hand edit.",
        "use crate::{BladeElement, Vec3, WingGeometry, WingSide};",
        "",
        f'pub const SOURCE_STL_SHA256: &str = "{EXPECTED_STL_SHA256}";',
        f'pub const DERIVATION_SHA256: &str = "{payload["derivation_sha256"]}";',
        f"pub const ELEMENT_COUNT: usize = {STRIP_COUNT};",
        "",
        "pub static LEFT_ELEMENTS: [BladeElement; ELEMENT_COUNT] = [",
    ]
    for element in left["elements"]:
        lines.append(
            "    BladeElement { center_local_m: "
            + rust_vec(element["center_local_mm"], 1e-3)
            + f", area_m2: {element['area_mm2'] * 1e-6:.17e}, chord_m: {element['chord_mm'] * 1e-3:.17e} }},"
        )
    lines.extend(["] ;".replace(" ", ""), "", "pub static RIGHT_ELEMENTS: [BladeElement; ELEMENT_COUNT] = ["])
    for element in left["elements"]:
        center = reflection(element["center_local_mm"])
        lines.append(
            "    BladeElement { center_local_m: "
            + rust_vec(center, 1e-3)
            + f", area_m2: {element['area_mm2'] * 1e-6:.17e}, chord_m: {element['chord_mm'] * 1e-3:.17e} }},"
        )
    lines.extend([
        "];", "",
        "pub static LEFT_WING: WingGeometry = WingGeometry {",
        '    semantic_id: "l_wing",',
        "    side: WingSide::Left,",
        "    source_asset_sha256: SOURCE_STL_SHA256,",
        f"    chord_local: {rust_vec(left['chord_axis_local'])},",
        f"    span_local: {rust_vec(left['span_axis_local'])},",
        f"    normal_local: {rust_vec(left['normal_axis_local'])},",
        f"    planform_area_m2: {left['planform_area_mm2'] * 1e-6:.17e},",
        "    elements: &LEFT_ELEMENTS,", "};", "",
        "pub static RIGHT_WING: WingGeometry = WingGeometry {",
        '    semantic_id: "r_wing",',
        "    side: WingSide::Right,",
        "    source_asset_sha256: SOURCE_STL_SHA256,",
        f"    chord_local: {rust_vec(reflection(left['chord_axis_local']))},",
        f"    span_local: {rust_vec(reflection(left['span_axis_local']))},",
        f"    normal_local: {rust_vec(right_normal)},",
        f"    planform_area_m2: {left['planform_area_mm2'] * 1e-6:.17e},",
        "    elements: &RIGHT_ELEMENTS,", "};", "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--body-schema", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.stl, args.model_xml, args.body_schema)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.rust.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    write_rust(payload, args.rust)
    print(json.dumps({
        "derivation_sha256": payload["derivation_sha256"],
        "planform_area_mm2": payload["left"]["planform_area_mm2"],
        "span_extent_mm": payload["left"]["span_extent_mm"],
        "elements": len(payload["left"]["elements"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
