#!/usr/bin/env python3
"""Prepare and benchmark the experimental fixed-B3 Metal MaleCNS kernel."""
from __future__ import annotations
import argparse, json, subprocess, sys, time, statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from chreatures.malecns import MaleCNSGraph
from chreatures.metal_circuit import MetalCircuit
from chreatures.neural_ports import NeuralPortBundle

def complete_check(graph_dir:Path,artifact:Path,port_bundle:Path,iterations:int,kernel:str)->dict:
    graph=MaleCNSGraph.load(graph_dir,mmap=True); ports=NeuralPortBundle.load(port_bundle,graph); recurrent=graph.matrix(); rng=np.random.default_rng(7302); rates=np.zeros((graph.n,3),np.float32); adaptation=np.zeros_like(rates); support=np.ones_like(rates); ids=["metal-a","metal-b","metal-c"]; readout_delta=physiology_delta=0.0
    streams=[]
    for _ in range(4):
        channels=rng.random((351,3),dtype=np.float32);channels[rng.random(channels.shape)<.8]=0;streams.append(channels)
    with MetalCircuit(artifact,port_bundle,kernel=kernel) as circuit:
        circuit.add_residents(ids)
        for channels in streams:
            request=[{"id":rid,"senses":dict(zip(ports.input_names,channels[:,j].astype(float),strict=True))} for j,rid in enumerate(ids)]; actual=circuit.step(request,.05);drive=ports.input_map@channels
            for _ in range(2):
                target=np.maximum(np.tanh(.005+drive+.92*(recurrent@rates)-.1*adaptation),0);rates+=(.05/2/.16)*(target*support-rates)
            adaptation+=.05/5*(rates-adaptation);support=np.clip(support+.05*(.024*(1-support)-.003*rates),.65,1);expected=ports.readout_map@rates;got=np.asarray([x["features"] for x in actual],np.float32).T;readout_delta=max(readout_delta,float(np.max(np.abs(got-expected))));expected_phys=np.asarray([rates.mean(0),rates.max(0),support.mean(0)]);got_phys=np.asarray([[x["activity_mean"],x["activity_peak"],x["support_mean"]] for x in actual]).T;physiology_delta=max(physiology_delta,float(np.max(np.abs(got_phys-expected_phys))))
        fixed=[{"id":rid,"senses":dict(zip(ports.input_names,streams[-1][:,j].astype(float),strict=True))} for j,rid in enumerate(ids)];samples=[];gpu=[]
        for i in range(iterations+3):
            started=time.perf_counter();result=circuit.step(fixed,.05);elapsed=(time.perf_counter()-started)*1000
            if i>=3:samples.append(elapsed);gpu.append(result[0]["gpu_ms"])
        snapshot=circuit.snapshot(ROOT/"runs/metal-benchmark","replay");first=circuit.step(fixed,.05);circuit.restore(ROOT/"runs/metal-benchmark","replay",snapshot["sha256"]);second=circuit.step(fixed,.05);replay=float(np.max(np.abs(np.asarray(first[0]["features"])-np.asarray(second[0]["features"]))))
    return {"kernel":kernel,"readout_max_abs_delta":readout_delta,"physiology_max_abs_delta":physiology_delta,"snapshot_replay_max_abs_delta":replay,"request_median_ms":statistics.median(samples),"request_minimum_ms":min(samples),"gpu_median_ms":statistics.median(gpu),"iterations":iterations}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--graph",type=Path,default=ROOT/"data/metal-brain",help="canonical graph, needed only for --complete reference validation"); p.add_argument("--artifact",type=Path,default=ROOT/"data/metal-brain/metal-csr-v2.bin"); p.add_argument("--port-bundle",type=Path,default=ROOT/"data/ports/retinal-v1-maps.npz");p.add_argument("--iterations",type=int,default=20);p.add_argument("--complete",action="store_true");p.add_argument("--kernel",choices=("row","simd"),default="row"); a=p.parse_args()
    if not a.artifact.is_file(): p.error("artifact is missing; run scripts/prepare_metal_brain.py first")
    receipt={"bytes":a.artifact.stat().st_size}
    binary=ROOT/"native/metal-brain/target/release/metal-brain"
    subprocess.run(["cargo","build","--release"],cwd=binary.parent.parent.parent,check=True)
    result=json.loads(subprocess.check_output([binary,a.artifact,str(a.iterations)],text=True))
    output={"artifact":receipt,"recurrent_benchmark":result}
    if a.complete: output["complete_backend"]=complete_check(a.graph,a.artifact,a.port_bundle,a.iterations,a.kernel)
    print(json.dumps(output,indent=2,sort_keys=True))
if __name__=="__main__":main()
