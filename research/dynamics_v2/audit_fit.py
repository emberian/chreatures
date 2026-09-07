"""Joined canonical Torch V2 trained replay, real-graph ablation and gradient audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from research.dynamics_v2.model import FullGraph,load_service
from research.dynamics_v2.train import DynamicsProbe,sequence,evaluate
from research.sensorimotor_skills.cns_adapter import SensoryPredictionHeads,write_parameter_artifact
from scripts.train_cns_adapter import ProceduralOpticSequences,optic_coordinates,compute_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--service',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(20260907)
    a,meta=load_service(args.service)
    graph=FullGraph(a,identity={'graph_dataset_sha256':meta['graph_sha256'],'atlas_file_sha256':meta['atlas_sha256']})
    model=DynamicsProbe(graph,'cuda',.9);fit=np.load(args.fit)
    with torch.no_grad():
        for target,name in [(model.gain_raw,'dynamics.recurrent_gain_raw'),(model.tau_raw,'dynamics.tau_raw'),
                            (model.projection.weight,'readout.projection.weight'),(model.output.weight,'readout.output.weight'),
                            (model.output.bias,'readout.output.bias')]:
            target.copy_(torch.tensor(fit[name],device='cuda'))
        named=dict(model.named_parameters())
        for name in fit.files:
            if name.startswith('probe.'):named[name.removeprefix('probe.')].copy_(torch.tensor(fit[name],device='cuda'))
        model.projection.weight[:,~model.cns.readout_mask.bool()]=0
    atlas=np.load(args.atlas)['site_side_hex']
    drives,targets=[torch.tensor(x,device='cuda') for x in sequence(graph,atlas,np.random.default_rng(777),40,4,True)]
    intact=evaluate(model,drives,targets)
    # Canonical parameter artifact contains all afferents and five type arrays.
    parameter_receipt=write_parameter_artifact(args.output/'canonical-trained-parameters.npz',model.cns,
        training_status='trained',provenance={'source_fit_sha256':hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        'scope':'controlled optical/body temporal probe; not embodied controller','source_adapter_sha256':meta['adapter_sha256']})
    # One actual full-graph backward through current production loss/generator.
    for parameter in model.cns.parameters():parameter.requires_grad_(True)
    heads=SensoryPredictionHeads(device=torch.device('cuda'))
    generator=ProceduralOpticSequences(optic_coordinates(args.atlas,torch.device('cuda')),ticks=8,batch_size=2)
    loss,metrics,_=compute_loss(model.cns,heads,generator.make(20260921),4)
    loss.backward()
    gradient={name:float(parameter.grad.norm()) if parameter.grad is not None else None
              for name,parameter in model.cns.named_parameters()}
    if not all(value is not None and np.isfinite(value) for value in gradient.values()):
        raise RuntimeError('canonical fullgraph gradient missing/nonfinite')
    # Local model only: all actual chemical edges zeroed, no live service touched.
    with torch.no_grad():model.w.values().zero_()
    ablated=evaluate(model,drives,targets)
    report={'format':'chreatures-canonical-v2-joined-training-audit-v1','source_fit_sha256':hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        'intact':intact,'zero_edges':ablated,'canonical_loss':float(loss.detach()),
        'canonical_metrics':{k:float(v) for k,v in metrics.items()},'canonical_gradient_norms':gradient,
        'parameter_artifact':parameter_receipt,'torch':torch.__version__,'hip':torch.version.hip,
        'scope':'fullgraph replay, masked edge-dependence, joint optic/body/type/readout gradient; not behavior'}
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
