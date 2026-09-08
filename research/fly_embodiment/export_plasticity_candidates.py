#!/usr/bin/env python3
"""Extract measured MaleCNS KC→MBON candidate edges and DA convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

GRAPH_SHA256 = "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
NEURONS_SHA256 = "0e6706229b93cbdcab48905504bd3b8f6075534488c65ad2c8f844668036ea9a"
FORMAT = "chreatures-male-cns-lifetime-plasticity-candidates-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compartment_hint(instance: str) -> str:
    """Retain the exact parenthetical annotation as an unverified hint."""
    match = re.search(r"\(([^)]*)\)", instance)
    return match.group(1) if match else ""


def incoming_edges(
    targets: np.ndarray, source_mask: np.ndarray, indptr: np.ndarray,
    indices: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions=[]; pre=[]; post=[]; synapses=[]
    for target in targets:
        lo,hi=int(indptr[target]),int(indptr[target+1])
        sources=np.asarray(indices[lo:hi]); keep=source_mask[sources]
        if keep.any():
            offsets=np.flatnonzero(keep)+lo
            positions.append(offsets.astype(np.uint32))
            pre.append(sources[keep].astype(np.uint32))
            post.append(np.full(int(keep.sum()),target,np.uint32))
            synapses.append(np.asarray(counts[lo:hi])[keep].astype(np.uint32))
    empty=np.empty(0,np.uint32)
    return tuple(np.concatenate(x) if x else empty.copy() for x in (positions,pre,post,synapses))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("graph",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite versioned candidate artifact")
    manifest_path=args.graph/"manifest.json"; neurons_path=args.graph/"neurons.npz"
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("dataset_hash") != GRAPH_SHA256: raise ValueError("graph identity differs")
    if sha256(neurons_path) != NEURONS_SHA256: raise ValueError("neuron metadata differs")
    indptr=np.load(args.graph/"indptr.npy",mmap_mode="r",allow_pickle=False)
    indices=np.load(args.graph/"indices.npy",mmap_mode="r",allow_pickle=False)
    counts=np.load(args.graph/"counts.npy",mmap_mode="r",allow_pickle=False)
    with np.load(neurons_path,allow_pickle=False) as z:
        kc_mask=z["classes"]=="Kenyon_Cell"
        mbon_rows=np.flatnonzero(z["classes"]=="MBON").astype(np.uint32)
        dan_annotation=z["classes"]=="DAN"
        dopamine=(z["effective_nt"]=="dopamine")
        dan_dopamine=dan_annotation&dopamine
        dan_rows=np.flatnonzero(dan_dopamine).astype(np.uint32)
        kc_edges=incoming_edges(mbon_rows,kc_mask,indptr,indices,counts)
        direct_da=incoming_edges(mbon_rows,dan_dopamine,indptr,indices,counts)
        all_da=incoming_edges(mbon_rows,dopamine,indptr,indices,counts)
        kc_to_dan=incoming_edges(dan_rows,kc_mask,indptr,indices,counts)
        edge_pos,pre,post,syn=kc_edges
        target_lookup=np.full(len(z["body_ids"]),np.iinfo(np.uint16).max,np.uint16)
        target_lookup[mbon_rows]=np.arange(len(mbon_rows),dtype=np.uint16)
        target_index=target_lookup[post]
        recommended=(z["types"][post]=="MBON11")
        da_target_edges=np.bincount(target_lookup[direct_da[2]],minlength=len(mbon_rows)).astype(np.uint32)
        da_target_syn=np.bincount(target_lookup[direct_da[2]],weights=direct_da[3],minlength=len(mbon_rows)).astype(np.uint64)
        all_da_target_edges=np.bincount(target_lookup[all_da[2]],minlength=len(mbon_rows)).astype(np.uint32)
        all_da_target_syn=np.bincount(target_lookup[all_da[2]],weights=all_da[3],minlength=len(mbon_rows)).astype(np.uint64)
        arrays={
            "schema":np.asarray(FORMAT),"graph_sha256":np.asarray(GRAPH_SHA256),
            "candidate.edge_positions":edge_pos,"candidate.pre_rows":pre,"candidate.post_rows":post,
            "candidate.synapse_counts":syn,"candidate.target_index":target_index,
            "candidate.recommended_gamma1pedc":recommended,
            "mbon.rows":mbon_rows,"mbon.body_ids":z["body_ids"][mbon_rows],"mbon.types":z["types"][mbon_rows],
            "mbon.instances":z["instances"][mbon_rows],"mbon.sides":z["sides"][mbon_rows],
            "mbon.compartment_annotation_hint":np.asarray([compartment_hint(x) for x in z["instances"][mbon_rows]]),
            "mbon.direct_annotated_dan_edges":da_target_edges,"mbon.direct_annotated_dan_synapses":da_target_syn,
            "mbon.direct_all_dopamine_edges":all_da_target_edges,"mbon.direct_all_dopamine_synapses":all_da_target_syn,
            "dan.rows":dan_rows,"dan.body_ids":z["body_ids"][dan_rows],"dan.types":z["types"][dan_rows],
            "dan.instances":z["instances"][dan_rows],"dan.sides":z["sides"][dan_rows],
            "dan.compartment_annotation_hint":np.asarray([compartment_hint(x) for x in z["instances"][dan_rows]]),
            "dan_to_mbon.edge_positions":direct_da[0],"dan_to_mbon.pre_rows":direct_da[1],
            "dan_to_mbon.post_rows":direct_da[2],"dan_to_mbon.synapse_counts":direct_da[3],
            "kc_to_dan.edge_positions":kc_to_dan[0],"kc_to_dan.pre_rows":kc_to_dan[1],
            "kc_to_dan.post_rows":kc_to_dan[2],"kc_to_dan.synapse_counts":kc_to_dan[3],
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**arrays)
        selected=recommended
        selected_pre=np.unique(pre[selected])
        target_rows=np.unique(post[selected])
        report={
            "format":f"{FORMAT}-manifest","artifact":args.output.name,"artifact_sha256":sha256(args.output),
            "artifact_bytes":args.output.stat().st_size,"graph_sha256":GRAPH_SHA256,
            "sources":{"graph_manifest_sha256":sha256(manifest_path),"neurons_sha256":NEURONS_SHA256},
            "measured":{
                "kenyon_cells":int(kc_mask.sum()),"mbon_rows":len(mbon_rows),"dopaminergic_DAN_rows":len(dan_rows),
                "all_kc_to_mbon_edges":len(edge_pos),"all_kc_to_mbon_synapses":int(syn.sum()),
                "connected_kcs":len(np.unique(pre)),"direct_DAN_to_MBON_edges":len(direct_da[0]),
                "direct_DAN_to_MBON_synapses":int(direct_da[3].sum()),"KC_to_DAN_edges":len(kc_to_dan[0]),
                "KC_to_DAN_synapses":int(kc_to_dan[3].sum()),
            },
            "recommended_first_subset":{
                "selector":"exact type MBON11 among measured KC→MBON edges",
                "edges":int(selected.sum()),"synapses":int(syn[selected].sum()),"connected_kcs":len(selected_pre),
                "target_rows":target_rows.astype(int).tolist(),"target_body_ids":z["body_ids"][target_rows].astype(int).tolist(),
                "reason":"retains the previously audited bilateral gamma1pedc target and PPL101 references; smallest substrate with direct physiology support",
            },
            "interpretation":{
                "measured":"edge positions, endpoints, synapse counts, neuron annotations, and direct graph convergence",
                "inferred":"parenthetical instance text is retained only as compartment_annotation_hint; no spatial synapse coordinates exist in this flat artifact",
                "engineered":"recommended mask and any future eligibility, learning rate, bounds, dopamine-to-update transform, receptor response, or valence semantics",
                "warning":"direct DAN→MBON/KC edges neither delimit volume transmission nor prove which KC→MBON synapses a DAN modifies",
            },
        }
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
