"""Export canonical V2 parameters and actual full-graph Torch execution fixture."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from research.dynamics_v2.model import FullGraph,load_service,N
from research.dynamics_v2.train import DynamicsProbe,logit


def main():
    p=argparse.ArgumentParser();p.add_argument('--service',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    arrays,meta=load_service(args.service);graph=FullGraph(arrays);model=DynamicsProbe(graph,args.device,.9)
    fit=np.load(args.fit)
    with torch.no_grad():
        for target,name in [(model.gain_raw,'dynamics.recurrent_gain_raw'),(model.tau_raw,'dynamics.tau_raw'),
                            (model.projection.weight,'readout.projection.weight'),(model.output.weight,'readout.output.weight'),
                            (model.output.bias,'readout.output.bias')]:
            target.copy_(torch.tensor(fit[name],device=args.device))
    parameters={name:np.asarray(fit[name]) for name in fit.files if name.startswith(('dynamics.','readout.'))}
    parameters.update({'dynamics.baseline_raw':np.full(11752,logit((.2-.05)/.4),np.float32),
                       'dynamics.adaptation_gain_raw':np.full(11752,logit(.15/.5),np.float32),
                       'dynamics.adaptation_tau_raw':np.full(11752,logit((1.5-.25)/4.75),np.float32),
                       'afferent.neutral_drive':graph.neutral[:,0].copy()})
    # Directly injected columns are exactly zero, even in the stored projection.
    parameters['readout.projection.weight'][:,~graph.mask]=0
    np.savez_compressed(args.output/'canonical-v2-parameters.npz',**parameters)
    sensory=np.zeros((8,2,5356),np.float32)
    sensory[:,:,:5313]=.5
    sensory[1,0,:5313]=0;sensory[1,1,:5313]=1
    for t in range(2,8):
        stripe=(np.sin(np.arange(1771)*.11-t*.5)>.0).astype(np.float32)
        sensory[t,0,:5313]=np.repeat(stripe,3);sensory[t,1,:5313]=np.repeat(1-stripe,3)
        sensory[t,:,5313]=[np.sin(t),np.cos(t)]
    x=torch.zeros((N,2),device=args.device);adapt=x.clone();support=x+1
    rates=[];adaptations=[];supports=[];latents=[]
    with torch.no_grad():
        for step in sensory:
            drive=graph.drive(step[:,:5313].reshape(2,1771,3),step[:,5313:])
            x,adapt,support=model.tick(torch.tensor(drive,device=args.device),x,adapt,support)
            z=torch.tanh(model.output(model.projection((x*model.mask).T)))
            rates.append((x+.2).cpu().numpy());adaptations.append(adapt.cpu().numpy())
            supports.append(support.cpu().numpy());latents.append(z.cpu().numpy())
    np.savez_compressed(args.output/'torch-v2-fixture.npz',sensory=sensory,
        rates=np.stack(rates),adaptation=np.stack(adaptations),support=np.stack(supports),latent=np.stack(latents))
    report={'format':'chreatures-dynamics-v2-export-fixture-v1','source_adapter_sha256':meta['adapter_sha256'],
            'dt':.05,'ticks':8,'batch':2,'state_order':'tick,neuron,batch','latent_order':'tick,batch,512',
            'sensory_order':'tick,batch,5356','initial_state':'rate=.2,adaptation=0,support=1',
            'training_status':'temporal-probe-trained; controller-not-trained',
            'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.glob('*.npz')}}
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
