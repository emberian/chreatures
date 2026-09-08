#!/usr/bin/env python3
"""Execute one sealed atlas cell through the current native MuJoCo host."""
from __future__ import annotations
import argparse, base64, hashlib, json, select, subprocess, time
from pathlib import Path
import numpy as np

FORMAT = "chreatures-native-fly-ecology-run-v2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rpc(process, request):
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 120)[0]:
        raise TimeoutError("native host response timeout")
    response = json.loads(process.stdout.readline())
    if not response.get("ok") or response.get("id") != request["id"]:
        raise RuntimeError(response)
    return response


def summary(ecology):
    host = ecology["native_host"]
    open_fraction = np.asarray(host["route_open_fraction"], np.float64)
    colonies = [
        item
        for item in ecology["organisms"]
        if item.get("anchored_region") and item.get("genotype", {}).get("development")
    ]
    resource = sum(
        float(item["atp"]) + sum(item["material"]["quantity"]) for item in colonies
    )
    structures = [
        item
        for item in ecology["structures"]
        if item.get("owner") in {c["id"] for c in colonies}
    ]
    return dict(
        time_s=float(ecology["time_s"]),
        mean_route_aperture=float(open_fraction.mean()),
        minimum_route_aperture=float(open_fraction.min()),
        blocked_route_fraction=float(np.mean(open_fraction < 0.999)),
        colony_resource=float(resource),
        branch_count=len(structures),
        active_branch_count=sum(bool(item["active"]) for item in structures),
        topology_revision=int(host["topology_revision"]),
        growth=host["growth"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--genotype", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    plan_sha = sha(a.plan)
    genotype = next(x for x in plan["genotypes"] if x["genotype_id"] == a.genotype)
    layout = next(x for x in plan["layouts"] if x["layout_id"] == a.layout)
    if (
        sha(plan["native_binary"]) != plan["native_binary_sha256"]
        or sha(__file__) != plan["runner_sha256"]
        or sha(layout["fixture"]) != layout["fixture_sha256"]
    ):
        raise ValueError("sealed native campaign input changed")
    fixture = json.loads(Path(layout["fixture"]).read_text())
    for organism in fixture["ecology"]["organisms"]:
        if organism.get("anchored_region") and organism.get("genotype", {}).get(
            "development"
        ):
            organism["genotype"]["development"].update(genotype["genes"])
    derived = Path(layout["fixture"]).parent / f".{genotype['genotype_id']}.world.json"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    started = time.monotonic()
    process = subprocess.Popen(
        [
            plan["native_binary"],
            "--scene",
            str(derived),
            "--seed",
            str(layout["world_seed"]),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = json.loads(process.stdout.readline())
        trace = []
        motor = base64.b64encode(bytes(2 * 92 * 4)).decode()
        for tick in range(plan["ticks"]):
            rpc(
                process,
                {"id": tick * 2 + 1, "command": "advance", "motor92_base64": motor},
            )
            if tick == 0 or (tick + 1) % 200 == 0 or tick + 1 == plan["ticks"]:
                sample = rpc(process, {"id": tick * 2 + 2, "command": "sample"})[
                    "sample"
                ]
                trace.append({"tick": tick + 1, **summary(sample["ecology"])})
        rpc(process, {"id": plan["ticks"] * 2 + 3, "command": "close"})
        process.wait(timeout=10)
        report = dict(
            format=FORMAT,
            completed=True,
            genotype=genotype,
            layout=layout,
            seconds=plan["seconds"],
            ticks=plan["ticks"],
            trace=trace,
            initial=trace[0],
            final=trace[-1],
            provenance={
                "plan_sha256": plan_sha,
                "runner_sha256": plan["runner_sha256"],
                "native_binary_sha256": plan["native_binary_sha256"],
                "ready": ready,
            },
            wall_seconds=time.monotonic() - started,
            claim_limit=plan["claim_limit"],
        )
        a.output.write_text(
            json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
        )
    finally:
        derived.unlink(missing_ok=True)
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    main()
