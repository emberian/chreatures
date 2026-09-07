"""One-way v1 anatomy/afferent seed plus V2 learned parameters -> current service."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chreatures.cns_adapter_contract import ARRAY_SPECS,write_service_artifact
from research.dynamics_v2.model import FullGraph,load_service


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed-service',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True);p.add_argument('--training-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old,meta=load_service(args.seed_service);graph=FullGraph(old);fitted=np.load(args.parameters)
    arrays={name:np.ascontiguousarray(fitted[name] if name in fitted else old[name],dtype=dtype)
            for name,dtype,shape in ARRAY_SPECS}
    if not np.allclose(arrays['afferent.neutral_drive'],graph.neutral[:,0],rtol=0,atol=2e-7):
        raise ValueError('neutral current does not match immutable learned afferents')
    receipt=write_service_artifact(args.output,arrays,graph_sha256=meta['graph_sha256'],atlas_sha256=meta['atlas_sha256'],
        training_status='trained',provenance={'model_family':'operating-point-full-malecns-v2',
        'anatomical_afferent_seed_adapter_sha256':meta['adapter_sha256'],
        'parameters_file_sha256':hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
        'training_receipt':json.loads(args.training_receipt.read_text()),
        'trained_scope':'per-type recurrence gain/tau and factorized masked Z512 readout; supervised probe only',
        'controller_status':'not trained','neutral_reference':'optic RGB .5 and standardized body 0',
        'source_import':'explicit one-way anatomy and afferent initialization; no private state migration'})
    args.output.with_suffix('.bin.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'path':receipt['path'],'bytes':receipt['bytes'],'sha256':receipt['file_sha256'],'adapter_sha256':receipt['metadata']['adapter_sha256']}))

if __name__=='__main__':main()
