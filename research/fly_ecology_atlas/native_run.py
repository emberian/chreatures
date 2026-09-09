#!/usr/bin/env python3
"""Execute one independent committed-growth atlas world through the native host."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import select
import subprocess
import time
from pathlib import Path

import numpy as np


FORMAT = "chreatures-native-fly-ecology-run-v3"
GROWTH_FIELDS = (
    "construction_proposed",
    "birth_proposed",
    "construction_clearance_approved",
    "birth_clearance_approved",
    "construction_clearance_blocked",
    "birth_clearance_blocked",
    "construction_capacity_rejected",
    "birth_capacity_rejected",
    "construction_resource_rejected",
    "birth_resource_rejected",
    "committed_constructions",
    "committed_births",
    "aborted_transactions",
    "construction_material_allocated",
    "birth_material_endowed",
    "last_committed",
)


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_exclusive(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def rpc(process: subprocess.Popen, request: dict) -> dict:
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 180)[0]:
        raise TimeoutError("native host response timeout")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("native host closed its response stream")
    response = json.loads(line)
    if not response.get("ok") or response.get("id") != request["id"]:
        raise RuntimeError(response)
    return response


def vec(value: object, name: str, count: int | None = None) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.ndim != 1 or (count is not None and len(result) != count):
        raise ValueError(f"{name} shape differs")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is nonfinite")
    return result


def development_colonies(ecology: dict) -> list[dict]:
    colonies = [
        item
        for item in ecology["organisms"]
        if item.get("anchored_region") is not None
        and item.get("genotype", {}).get("development") is not None
    ]
    if not colonies or len({item["id"] for item in colonies}) != len(colonies):
        raise ValueError("development colony identities differ")
    return colonies


def committed_structures(ecology: dict, entity_ids: set[str]) -> list[dict]:
    owners = {item["id"] for item in development_colonies(ecology)}
    structures = []
    for item in ecology["structures"]:
        if item.get("owner_id") not in owners:
            continue
        binding = item.get("physics_binding")
        if not isinstance(binding, str) or binding not in entity_ids:
            raise ValueError("committed structure lacks current physical entity")
        material = vec(item["initial_material"], "structure initial material")
        current = vec(item["material"]["quantity"], "structure material", len(material))
        if (material < 0).any() or (current < 0).any():
            raise ValueError("committed structure material outside bounds")
        structures.append(
            {
                "id": item["id"],
                "owner_id": item["owner_id"],
                "physics_binding": binding,
                "active": bool(item["active"]),
                "position_m": vec(item["position_m"], "structure position", 3).tolist(),
                "orientation_xyzw": vec(
                    item["orientation_xyzw"], "structure orientation", 4
                ).tolist(),
                "nominal_length_m": float(item["nominal_length_m"]),
                "nominal_radius_m": float(item["nominal_radius_m"]),
                "initial_material": material.tolist(),
                "material": current.tolist(),
            }
        )
    structures.sort(key=lambda item: item["id"])
    return structures


def route_metrics(ecology: dict, fixture: dict) -> tuple[np.ndarray, dict]:
    host = ecology["native_host"]
    routes = fixture["ecology"]["routes"]
    openness = vec(host["route_open_fraction"], "route openness", len(routes))
    if ((openness < 0) | (openness > 1)).any():
        raise ValueError("route openness outside bounds")
    area_per_length = np.asarray(
        [route["cross_section_m2"] / route["length_m"] for route in routes],
        np.float64,
    )
    base = np.asarray([route["base_open_fraction"] for route in routes], np.float64)
    if (
        not np.isfinite(area_per_length).all()
        or not np.isfinite(base).all()
        or (area_per_length <= 0).any()
        or ((base < 0) | (base > 1)).any()
    ):
        raise ValueError("route geometry outside bounds")
    effective = openness * base
    weight = area_per_length / area_per_length.sum()
    return openness, {
        "mean_aperture": float(openness.mean()),
        "minimum_aperture": float(openness.min()),
        "blocked_fraction": float(np.mean(openness < 1.0)),
        "area_weighted_permeability": float(np.dot(weight, effective)),
    }


def summary(sample: dict, fixture: dict, baseline_structures: set[str] | None) -> dict:
    ecology = sample["ecology"]
    host = ecology["native_host"]
    growth = host.get("growth")
    if not isinstance(growth, dict) or set(GROWTH_FIELDS) - set(growth):
        raise ValueError("committed growth observer schema differs")
    entity_ids = sample.get("entityIds")
    if not isinstance(entity_ids, list) or len(entity_ids) != len(set(entity_ids)):
        raise ValueError("physical entity identities differ")
    structures = committed_structures(ecology, set(entity_ids))
    initial_ids = baseline_structures or set()
    new_structures = [item for item in structures if item["id"] not in initial_ids]
    committed_count = int(growth["committed_constructions"])
    if baseline_structures is not None and committed_count < len(new_structures):
        raise ValueError("committed construction counter is below physical structure set")
    material_allocated = vec(
        growth["construction_material_allocated"], "allocated construction material"
    )
    if (material_allocated < 0).any():
        raise ValueError("allocated material outside bounds")
    structure_material = (
        np.asarray([item["initial_material"] for item in new_structures], np.float64).sum(0)
        if new_structures
        else np.zeros_like(material_allocated)
    )
    if baseline_structures is not None and (
        len(material_allocated) != len(structure_material)
        or (material_allocated + 1e-12 < structure_material).any()
    ):
        raise ValueError("committed material counter is below retained physical structures")
    by_binding = {item["physics_binding"]: item for item in structures}
    for commit in growth["last_committed"]:
        if commit["kind"] == "constructed_geometry":
            structure = by_binding.get(commit["physics_binding"])
            if structure is None or structure["owner_id"] != commit["owner_id"]:
                raise ValueError("last committed construction lacks postcommit structure")
            quantity = vec(
                commit["material_quantity"], "last committed material", len(material_allocated)
            )
            if (quantity < 0).any():
                raise ValueError("last committed material outside bounds")
    colonies = development_colonies(ecology)
    resource = 0.0
    for colony in colonies:
        quantity = vec(colony["material"]["quantity"], "colony material")
        atp = float(colony["atp"])
        if not np.isfinite(atp) or atp < 0 or (quantity < 0).any():
            raise ValueError("colony resource outside bounds")
        resource += atp + float(quantity.sum())
    openness, route = route_metrics(ecology, fixture)
    residual = float(ecology["accounting"]["maximum_absolute_residual"])
    if not np.isfinite(residual):
        raise ValueError("ecology accounting residual is nonfinite")
    return {
        "time_s": float(ecology["time_s"]),
        "route": {**route, "open_fraction": openness.tolist()},
        "route_plan_sha256": host["route_plan_sha256"],
        "topology_revision": int(host["topology_revision"]),
        "colony_resource": resource,
        "captured_photon_energy": float(ecology["accounting"]["captured_photon_energy"]),
        "growth": growth,
        "committed_structures": structures,
        "new_committed_structures": new_structures,
        "new_committed_structure_count": len(new_structures),
        "active_new_structure_count": sum(item["active"] for item in new_structures),
        "new_structure_material": structure_material.tolist(),
        "maximum_elemental_residual": residual,
    }


def derived_fixture(base_path: Path, fixture: dict, unit: dict, work_path: Path) -> str:
    source_dir = base_path.parent.resolve()
    fixture["scene_xml"] = str((source_dir / fixture["scene_xml"]).resolve())
    for asset in fixture["mesh_assets"]:
        asset["path"] = str((source_dir / asset["path"]).resolve())
    fixture["ecology"]["seed"] = unit["ecology_seed"]
    for organism in fixture["ecology"]["organisms"]:
        development = organism.get("genotype", {}).get("development")
        if organism.get("anchored_region") is not None and development is not None:
            development.update(unit["genotype"]["genes"])
    text = json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n"
    with work_path.open("x") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    plan_sha = sha(args.plan)
    unit = next(item for item in plan["units"] if item["unit_id"] == args.unit)
    layout = unit["layout"]
    fixture_path = Path(layout["fixture"])
    if (
        sha(plan["native_binary"]) != plan["native_binary_sha256"]
        or sha(__file__) != plan["runner_sha256"]
        or sha(fixture_path) != layout["fixture_sha256"]
    ):
        raise ValueError("sealed native campaign input changed")
    fixture = json.loads(fixture_path.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The native host recompiles its in-memory MJCF beside the fixture whenever
    # committed growth changes topology.  The authored MJCF deliberately uses
    # mesh basenames, so the ephemeral derived fixture must retain the source
    # fixture's asset directory as its scene_dir.  Putting it beside the result
    # makes initial loading work through the absolute scene path but causes the
    # first topology recompile to lose the mesh VFS root.
    work_path = fixture_path.parent / f".atlas-{unit['unit_id']}.world.tmp.json"
    stderr_path = args.output.with_suffix(".stderr.log")
    if work_path.exists() or stderr_path.exists():
        raise FileExistsError("unit work output already exists")
    derived_sha = derived_fixture(fixture_path, fixture, unit, work_path)
    started = time.monotonic()
    process = None
    ready = None
    trace = []
    completed = False
    error = None
    request_id = 0
    stderr_file = stderr_path.open("x")
    try:
        process = subprocess.Popen(
            [
                plan["native_binary"],
                "--scene",
                str(work_path),
                "--seed",
                str(unit["physics_seed"]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )
        if not select.select([process.stdout], [], [], 180)[0]:
            raise TimeoutError("native host ready timeout")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("native host failed before ready")
        ready = json.loads(line)
        if not ready.get("ok") or ready.get("event") != "ready":
            raise RuntimeError(ready)
        if ready.get("engine") != plan["engine"]:
            raise ValueError("native host engine differs")
        residents = int(ready["residents"])
        if residents <= 0:
            raise ValueError("native host resident count differs")
        motor = base64.b64encode(bytes(residents * 92 * 4)).decode()
        request_id += 1
        initial_sample = rpc(process, {"id": request_id, "command": "sample"})["sample"]
        preliminary = summary(initial_sample, fixture, None)
        baseline = {item["id"] for item in preliminary["committed_structures"]}
        trace.append({"tick": 0, **summary(initial_sample, fixture, baseline)})
        for tick in range(1, plan["ticks"] + 1):
            request_id += 1
            rpc(
                process,
                {"id": request_id, "command": "advance", "motor92_base64": motor},
            )
            if tick % plan["sample_every_ticks"] == 0 or tick == plan["ticks"]:
                request_id += 1
                sample = rpc(process, {"id": request_id, "command": "sample"})["sample"]
                trace.append({"tick": tick, **summary(sample, fixture, baseline)})
        request_id += 1
        rpc(process, {"id": request_id, "command": "close"})
        if process.wait(timeout=15) != 0:
            raise RuntimeError("native host exited unsuccessfully")
        completed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        stderr_file.close()
        work_path.unlink(missing_ok=True)
    stderr_sha = sha(stderr_path)
    if stderr_path.stat().st_size == 0:
        stderr_path.unlink()
        stderr_sha = None
    report = {
        "format": FORMAT,
        "completed": completed,
        "unit": unit,
        "seconds": plan["seconds"],
        "ticks": plan["ticks"],
        "sample_every_ticks": plan["sample_every_ticks"],
        "trace": trace,
        "initial": trace[0] if trace else None,
        "final": trace[-1] if completed else None,
        "error": error,
        "provenance": {
            "plan_sha256": plan_sha,
            "runner_sha256": plan["runner_sha256"],
            "native_binary_sha256": plan["native_binary_sha256"],
            "base_fixture_sha256": layout["fixture_sha256"],
            "derived_fixture_sha256": derived_sha,
            "ready": ready,
            "stderr_sha256": stderr_sha,
        },
        "wall_seconds": time.monotonic() - started,
        "claim_limit": plan["claim_limit"],
    }
    write_exclusive(args.output, report)
    print(json.dumps({"output": str(args.output), "completed": completed, "error": error}))
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
