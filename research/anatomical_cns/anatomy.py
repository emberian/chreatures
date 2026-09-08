#!/usr/bin/env python3
"""Export annotation-derived MaleCNS V3 interface masks and slow edges.

This is an offline artifact builder.  It reads the canonical memory-mapped CSR
without changing it or contacting a running neural service.  Rows in every
interface CSR are named channels and columns are canonical local neuron indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

GRAPH_SHA256 = "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
SCHEMA = "chreatures-anatomical-cns-v3"
SIDES = ("L", "R")
LEG_REGIONS = (("front", "T1", "ProLN"), ("middle", "T2", "MesoLN"), ("hind", "T3", "MetaLN"))
# Exact first-resident joint order in site/live/fixtures/garden.xml.  Each entry
# expands hip then knee.  Keep body channels, fatigue, feet, and motor pairs in
# this shared order.
FIXTURE_LEGS = (
    ("lf", "front", "L", "T1", "ProLN"),
    ("lm", "middle", "L", "T2", "MesoLN"),
    ("lh", "hind", "L", "T3", "MetaLN"),
    ("rf", "front", "R", "T1", "ProLN"),
    ("rm", "middle", "R", "T2", "MesoLN"),
    ("rh", "hind", "R", "T3", "MetaLN"),
)
MODULATORS = ("dopamine", "octopamine", "serotonin")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _afferent(superclasses: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (("_sensory" in value) or value.startswith("sensory_") for value in superclasses),
        dtype=bool, count=len(superclasses),
    )


def sensory_masks(z: np.lib.npyio.NpzFile) -> tuple[list[str], list[np.ndarray], list[dict]]:
    # Frozen V3 body scope deliberately retains the V2 body atlas.  In
    # particular, sensory_ascending is outside it even when its subtype is known.
    aff = np.isin(z["superclasses"], ["cb_sensory", "vnc_sensory"])
    side, cls, sub, nerve = (z[k] for k in ("sides", "classes", "subclasses", "entry_nerves"))
    names: list[str] = []; masks: list[np.ndarray] = []; notes: list[dict] = []
    def add(name: str, mask: np.ndarray, rule: str, status: str = "mapped") -> None:
        names.append(name); masks.append(mask); notes.append({"name": name, "count": int(mask.sum()), "rule": rule, "status": status})

    olf = aff & (cls == "olfactory")
    for s in SIDES:
        cohort = olf & ((side == s) | (side == "unknown") | (side == ""))
        for identity in range(3):
            add(f"odor/{s}/{identity}", cohort, "olfactory; anatomical side; learned chemical identity tuning")

    proprio = aff & (cls == "mechanosensory_proprioceptive")
    linear = aff & np.isin(sub, ["haltere", "wind_gravity"])
    angular = aff & ((sub == "haltere") | (sub == "wind_gravity"))
    for family, cohort in (("linear", linear), ("angular", angular)):
        for axis in "xyz":
            for sign_name in ("positive", "negative"):
                add(f"{family}/{axis}/{sign_name}", cohort, f"{family} mechanosensory cohort; learned axis/direction tuning")

    tactile = aff & np.isin(cls, ["mechanosensory_tactile", "mechanosensory"])
    for axis in "xyz":
        for sign_name in ("positive", "negative"):
            add(f"contact/normal_{axis}/{sign_name}", tactile, "tactile/mechanosensory; learned direction tuning")
    add("contact/count", tactile, "all tactile/mechanosensory afferents")
    for s in SIDES:
        add(f"contact/touch/{s}", tactile & (side == s), "tactile/mechanosensory; anatomical side")

    auditory = aff & (sub == "auditory")
    for band in range(16):
        add(f"auditory/band_{band:02d}", auditory, "all typed auditory afferents; trainable per-cell band tuning (annotation has no measured frequency assignment)")
    shade = aff & np.isin(cls, ["hygrosensory", "thermosensory"])
    add("shade", shade, "hygro/thermosensory proxy; engineered and non-photometric")

    visceral = aff & np.isin(sub, ["abdomen", "pharyngeal sensillum"])
    for i in range(12):
        add(f"intero/{i:02d}", visceral, "abdominal/pharyngeal afferent proxy; learned signal tuning", "proxy")

    leg_cohorts: dict[str, np.ndarray] = {}
    for code, _leg, s, _neuromere, leg_nerve in FIXTURE_LEGS:
        leg_cohorts[code] = proprio & (nerve == leg_nerve) & (side == s)
    # Contract ranges [56:68], [68:80], [80:92] are family-major.
    for family in ("joint_angle", "joint_velocity", "joint_load"):
        for code, leg, s, _neuromere, leg_nerve in FIXTURE_LEGS:
            for joint in ("proximal", "distal"):
                add(f"{family}/{code}/{joint}", leg_cohorts[code], f"proprioceptive; fixture {code}; {leg} entry nerve {leg_nerve}; anatomical side {s}; learned joint tuning")
    for code, leg, s, _neuromere, leg_nerve in FIXTURE_LEGS:
        add(f"foot_contact/{code}", tactile & (side == s) & (nerve == leg_nerve), f"tactile/mechanosensory; fixture {code}; {leg} entry nerve {leg_nerve}; side {s}")
    for code, leg, s, _neuromere, leg_nerve in FIXTURE_LEGS:
        for joint in ("proximal", "distal"):
            add(f"fatigue/{code}/{joint}", leg_cohorts[code], f"proprioceptive proxy for local muscle state; fixture {code}; {leg} entry nerve {leg_nerve}; side {s}", "proxy")
    assert len(names) == 110
    return names, masks, notes


def motor_masks(z: np.lib.npyio.NpzFile) -> tuple[list[str], list[np.ndarray], list[dict]]:
    motor=np.isin(z["superclasses"],["vnc_motor","cb_motor"])
    side, soma, manc, exit_nerve = (z[k] for k in ("sides","soma_neuromeres","manc_types","exit_nerves"))
    names=[]; masks=[]; notes=[]
    def add(name,mask,rule,status="mapped"):
        names.append(name); masks.append(mask); notes.append({"name":name,"count":int(mask.sum()),"rule":rule,"status":status})
    patterns=(
        ("proximal/positive", ("extensor", "promotor", "anterior rotator", "levator")),
        ("proximal/negative", ("flexor", "remotor", "reductor", "posterior rotator")),
        ("distal/positive", ("ti extensor", "ta levator")),
        ("distal/negative", ("ti flexor", "ta depressor")),
    )
    lower=np.char.lower(manc.astype(str))
    for code, leg, s, neuromere, nerve in FIXTURE_LEGS:
        base=motor & (side==s) & ((soma==neuromere)|(exit_nerve==nerve))
        for joint in ("proximal", "distal"):
            for label, pats in patterns:
                if not label.startswith(joint):
                    continue
                match=np.zeros(len(motor),bool)
                for pat in pats: match |= np.char.find(lower,pat)>=0
                add(f"joint/{code}/{label}",base&match,f"engineered synthetic {code} {joint} direction mapping; motor side {s}; {neuromere}/{nerve}; MANC type contains {pats}","engineered_assumption")
    wing=(exit_nerve=="ADMN")
    head=np.isin(exit_nerve,["PhN","MxLbN","AN","ON"])
    abdomen=np.char.startswith(exit_nerve.astype(str),"AbN")|(exit_nerve=="AbNT")
    tarsal=(np.char.find(lower,"ta ")>=0)
    ancillary=(
        ("gaze_pitch",head), ("posture",np.isin(soma,["T1","T2","T3"])),
        ("grip",tarsal),
        ("signal_low",wing), ("signal_mid",wing), ("signal_high",wing),
        ("eat",np.isin(exit_nerve,["PhN","MxLbN"])),
        ("release",np.isin(exit_nerve,["PhN","MxLbN"])),
        ("secrete",abdomen), ("allocate",motor),
    )
    for name,selector in ancillary:
        selector=motor&selector
        add(f"ancillary/{name}",selector,"engineered body action over exact motor exit nerve/type", "mapped" if selector.any() else "missing")
    assert len(names)==34
    return names,masks,notes


def graph_dynamics(graph: Path, z: np.lib.npyio.NpzFile, output: dict[str,np.ndarray], summary: dict) -> None:
    indptr=np.load(graph/"indptr.npy",mmap_mode="r",allow_pickle=False)
    indices=np.load(graph/"indices.npy",mmap_mode="r",allow_pickle=False)
    counts=np.load(graph/"counts.npy",mmap_mode="r",allow_pickle=False)
    nts=z["effective_nt"]
    channel=np.zeros(len(nts),np.uint32)
    fast_sign={"acetylcholine":1.0,"gaba":-1.0,"glutamate":-1.0,"histamine":-1.0}
    for nt in fast_sign: channel[nts==nt]=1
    for code,nt in enumerate(MODULATORS,2): channel[nts==nt]=code
    weight=np.zeros(len(indices),np.float32)
    edge_counts={name:0 for name in ("fast",)+MODULATORS}
    synapse_counts={name:0 for name in ("fast",)+MODULATORS}
    for target in range(len(nts)):
        lo,hi=int(indptr[target]),int(indptr[target+1])
        if lo==hi: continue
        src=np.asarray(indices[lo:hi]); raw=np.asarray(counts[lo:hi],dtype=np.float32)
        source_nt=nts[src]
        all_incoming=float(raw.sum())
        fast=channel[src]==1
        if fast.any():
            signs=np.fromiter((fast_sign.get(v,0.0) for v in source_nt[fast]),np.float32,count=int(fast.sum()))
            weight[lo:hi][fast]=raw[fast]*signs/max(all_incoming,1.0)
            edge_counts["fast"]+=int(fast.sum()); synapse_counts["fast"]+=int(raw[fast].sum())
        for code,nt in enumerate(MODULATORS,2):
            keep=channel[src]==code
            family_total=float(raw[keep].sum())
            if family_total:
                weight[lo:hi][keep]=raw[keep]/family_total
                edge_counts[nt]+=int(keep.sum()); synapse_counts[nt]+=int(raw[keep].sum())
    output["graph.channel"]=channel
    output["graph.weight"]=weight
    summary.update({name:{"sources":int((channel==(1 if name=='fast' else MODULATORS.index(name)+2)).sum()),"edges":edge_counts[name],"synapses":synapse_counts[name],"normalization":"all incoming target counts" if name=="fast" else "positive within-family incoming target counts"} for name in edge_counts})


def build_anatomy(graph: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Build the frozen V3 structural arrays and a human-readable report."""
    manifest=json.loads((graph/"manifest.json").read_text())
    if manifest.get("dataset_hash") != GRAPH_SHA256: raise ValueError("canonical graph identity differs")
    with np.load(graph/"neurons.npz",allow_pickle=False) as z:
        arrays: dict[str,np.ndarray]={"schema":np.asarray(SCHEMA),"graph_sha256":np.asarray(GRAPH_SHA256)}
        report={"schema":SCHEMA,"graph_sha256":GRAPH_SHA256,"interfaces":{},"graph_channels":{}}
        body_rows=np.flatnonzero(np.isin(z["superclasses"],["cb_sensory","vnc_sensory"])).astype(np.uint32)
        body_names,body_full,body_notes=sensory_masks(z)
        body_mask=np.stack([m[body_rows] for m in body_full],axis=1).astype(np.float32)
        context_rows=np.flatnonzero(z["superclasses"]=="descending_neuron").astype(np.uint32)
        motor_names,motor_full,motor_notes=motor_masks(z)
        motor_rows=np.flatnonzero(np.isin(z["superclasses"],["vnc_motor","cb_motor"])).astype(np.uint32)
        motor_mask=np.stack([m[motor_rows] for m in motor_full]).astype(np.float32)
        expected=((body_rows,(11233,),"body rows"),(body_mask,(11233,110),"body mask"),(context_rows,(1314,),"context rows"),(motor_rows,(815,),"motor rows"),(motor_mask,(34,815),"motor mask"))
        for value,shape,label in expected:
            if value.shape != shape: raise ValueError(f"{label} differs: {value.shape} != {shape}")
        arrays.update({"atlas.body_rows":body_rows,"atlas.body_mask":body_mask,"atlas.context_rows":context_rows,"atlas.motor_rows":motor_rows,"atlas.motor_mask":motor_mask,"atlas.body_names":np.asarray(body_names),"atlas.motor_names":np.asarray(motor_names)})
        report["interfaces"]={
            "body":{"shape":list(body_mask.shape),"nnz":int(np.count_nonzero(body_mask)),"channels":body_notes,"omissions":{"sensory_ascending_chordotonal":int(((z['superclasses']=='sensory_ascending')&(z['subclasses']=='chordotonal organ')).sum()),"annotation_missing_auditory":int(((z['superclasses']=='')&(z['subclasses']=='auditory')).sum())}},
            "context":{"shape":list(context_rows.shape),"selection":"all annotated descending_neuron rows; 12 learned context columns have unrestricted support on these rows and no assigned biological roles"},
            "motor":{"shape":list(motor_mask.shape),"nnz":int(np.count_nonzero(motor_mask)),"rows":len(motor_rows),"channels":motor_notes},
        }
        graph_dynamics(graph,z,arrays,report["graph_channels"])
    return arrays,report


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("graph",type=Path); p.add_argument("output",type=Path); args=p.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("refusing to overwrite an anatomy artifact or manifest; choose a new versioned output name")
    arrays,report=build_anatomy(args.graph)
    arrays["metadata_json"]=np.asarray(json.dumps(report,sort_keys=True,separators=(",",":")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    receipt={"artifact":args.output.name,"bytes":args.output.stat().st_size,"sha256":_digest(args.output),**report}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"artifact":str(args.output),"bytes":receipt["bytes"],"sha256":receipt["sha256"],"graph_channels":report["graph_channels"],"interfaces":{k:{q:v[q] for q in v if q in ('shape','nnz','rows','omissions')} for k,v in report["interfaces"].items()}},indent=2))


if __name__ == "__main__": main()
