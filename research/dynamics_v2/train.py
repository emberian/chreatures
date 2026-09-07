"""ROCm research fit of full-graph V2 type gains/time constants.

This does not export a competent controller: heads are supervised probes and
raw target features never enter recurrence or the policy readout.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from research.dynamics_v2.model import FullGraph,load_service,N
from research.sensorimotor_skills.cns_adapter import (
    CNSState, CNSStaticArrays, TrainableCNSAdapter,
)


def csr(value,device):
    return torch.sparse_csr_tensor(torch.tensor(value.indptr,dtype=torch.int32,device=device),
        torch.tensor(value.indices,dtype=torch.int32,device=device),torch.tensor(value.data,device=device),
        size=value.shape,device=device)


def logit(x):return float(np.log(x/(1-x)))


class DynamicsProbe(nn.Module):
    """Training-only targets around the one canonical current Torch CNS model."""
    def __init__(self,graph,device,gain=.9):
        super().__init__()
        identity={**graph.identity,'source':'authenticated sealed anatomy/afferent research initialization'}
        static=CNSStaticArrays.from_service_arrays(graph.arrays,identity)
        self.cns=TrainableCNSAdapter(static,device=torch.device(device))
        # Explicit one-way afferent initialization; old dynamics/readout never load.
        with torch.no_grad():
            for target,name in [(self.cns.optic_spectral_logits,'optic.spectral_logits'),
                                (self.cns.optic_gain_raw,'optic.gain_raw'),(self.cns.optic_bias,'optic.bias'),
                                (self.cns.body_input.weight,'body.input.weight'),(self.cns.body_input.bias,'body.input.bias'),
                                (self.cns.body_output.weight,'body.output.weight'),(self.cns.body_output.bias,'body.output.bias'),
                                (self.cns.body_mean,'body.mean'),(self.cns.body_scale,'body.scale')]:
                target.copy_(torch.tensor(np.asarray(graph.arrays[name]),device=device))
            self.cns.dynamics_recurrent_gain_raw.fill_(logit((gain-.5)/1.5))
        for name,parameter in self.cns.named_parameters():
            parameter.requires_grad_(name in {'dynamics_recurrent_gain_raw','dynamics_tau_raw'} or name.startswith('readout_'))
        self.register_buffer('frozen_neutral',torch.tensor(graph.neutral,device=device))
        self.current=nn.Linear(512,3,device=device)
        self.temporal=nn.GRUCell(512,128,device=device)
        self.future=nn.Linear(128,3,device=device)
        nn.init.normal_(self.current.weight,std=.0001);nn.init.zeros_(self.current.bias)
        nn.init.normal_(self.future.weight,std=.0001);nn.init.zeros_(self.future.bias)
        self.gain_prior=self.gain_raw.detach().clone()
        self.tau_prior=self.tau_raw.detach().clone()

    @property
    def gain_raw(self):return self.cns.dynamics_recurrent_gain_raw
    @property
    def tau_raw(self):return self.cns.dynamics_tau_raw
    @property
    def projection(self):return self.cns.readout_projection
    @property
    def output(self):return self.cns.readout_output
    @property
    def w(self):return self.cns.recurrent
    @property
    def mask(self):return self.cns.readout_mask[:,None]

    def tick(self,drive,x,a,s):
        dynamics=self.cns.effective_dynamics();baseline=dynamics[0]
        state=CNSState(x+baseline,a,s)
        state=self.cns.step_from_drive(drive,state,dynamics=dynamics,
            neutral_drive=self.frozen_neutral.expand_as(drive))
        return state.rates-baseline,state.adaptation,state.support

    def forward(self,drives):
        state=self.cns.initial_state(drives.shape[-1])
        baseline=self.cns.effective_dynamics()[0]
        x=state.rates-baseline;a=state.adaptation;s=state.support
        predictions=[];hidden=drives.new_zeros((drives.shape[-1],128))
        for t in range(len(drives)):
            x,a,s=checkpoint(self.tick,drives[t],x,a,s,use_reentrant=False)
            latent=self.cns.readout_from_state(CNSState(x+baseline,a,s))
            hidden=self.temporal(latent,hidden)
            predictions.append(torch.cat((self.current(latent),self.future(hidden)),dim=-1))
        return torch.stack(predictions)


def sequence(graph,atlas,rng,ticks,batch,heldout=False):
    xy=atlas[:,1:].astype(np.float32);xy=(xy-xy.mean(0))/np.maximum(xy.std(0),1)
    # Nonoverlapping frequency bands across training and heldout sources.
    def frequency():
        if heldout:return rng.uniform(.75,.9,batch)
        low=rng.uniform(.35,.75,batch);high=rng.uniform(.9,1.6,batch)
        return np.where(rng.random(batch)<.5,low,high)
    freq=frequency()
    direction=rng.choice([-1,1],batch);phase=rng.uniform(0,2*np.pi,batch)
    bodyfreq=frequency();bodyphase=rng.uniform(0,2*np.pi,batch)
    t=np.arange(ticks+4,dtype=np.float32)*.05
    angle=t[:,None]*freq[None,:]*direction[None,:]*2*np.pi+phase[None,:]
    body=np.zeros((ticks+4,batch,43),np.float32)
    body[:,:,0]=1.5*np.sin(t[:,None]*bodyfreq[None,:]*2*np.pi+bodyphase[None,:])
    target=np.stack([np.sin(angle),np.cos(angle),body[:,:,0]],axis=-1).astype(np.float32)
    current=target[:ticks];delta=target[4:]-target[:-4]
    targets=np.concatenate([current,delta],axis=-1)
    drives=[]
    for i in range(ticks):
        optic=np.repeat((.5+.35*np.sin(xy[None,:,0]*1.7+angle[i,:,None]))[:,:,None],3,axis=-1).astype(np.float32)
        drives.append(graph.drive(optic,body[i]))
    return np.stack(drives),targets


def evaluate(model,drives,targets):
    with torch.no_grad():
        predictions=model(drives)[10:]; truth=targets[10:]
        mse=((predictions-truth)**2).mean((0,1));base=(truth**2).mean((0,1))
    return {'mse':mse.cpu().tolist(),'zero_or_persistence_mse':base.cpu().tolist(),
            'skill':(1-mse/base.clamp_min(1e-9)).cpu().tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--service',type=Path,required=True);p.add_argument('--atlas',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=64)
    p.add_argument('--ticks',type=int,default=40);p.add_argument('--batch',type=int,default=4);p.add_argument('--gain',type=float,default=.9)
    p.add_argument('--device',default='cuda');p.add_argument('--initialize',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(20260907)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('ROCm unavailable')
    a,meta=load_service(args.service);graph=FullGraph(a,identity={'graph_dataset_sha256':meta['graph_sha256'],'atlas_file_sha256':meta['atlas_sha256']});atlas=np.load(args.atlas)['site_side_hex']
    model=DynamicsProbe(graph,args.device,args.gain)
    if args.initialize:
        init=np.load(args.initialize)
        with torch.no_grad():
            for target,name in [(model.gain_raw,'dynamics.recurrent_gain_raw'),(model.tau_raw,'dynamics.tau_raw'),
                                (model.projection.weight,'readout.projection.weight'),(model.output.weight,'readout.output.weight'),
                                (model.output.bias,'readout.output.bias')]:
                target.copy_(torch.tensor(init[name],device=args.device))
    optimizer=torch.optim.Adam([{'params':[model.gain_raw,model.tau_raw],'lr':.006},
                                {'params':list(model.projection.parameters())+list(model.output.parameters())+list(model.current.parameters())+list(model.temporal.parameters())+list(model.future.parameters()),'lr':.001}])
    def tensor(a):return torch.tensor(a,device=args.device)
    eval_drives,eval_targets=map(tensor,sequence(graph,atlas,np.random.default_rng(777),args.ticks,args.batch,True))
    before=evaluate(model,eval_drives,eval_targets)
    rng=np.random.default_rng(20260907);began=time.monotonic();history=[]
    # Equal target weighting makes delta improvement explicit; no variance/activity reward.
    for update in range(args.updates):
        drives,targets=map(tensor,sequence(graph,atlas,rng,args.ticks,args.batch))
        optimizer.zero_grad(set_to_none=True);prediction=model(drives)[10:];truth=targets[10:]
        mse=(prediction-truth).square().mean((0,1))
        prior=((model.gain_raw-model.gain_prior)**2).mean()+((model.tau_raw-model.tau_prior)**2).mean()
        scale=mse.new_tensor([.5,.5,1.125])
        loss=(mse[:3]/scale).mean()+2*(mse[3:]/scale).mean()+.01*prior
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if update%8==0:
            rec={'update':update,'loss':float(loss.detach()),'seconds':time.monotonic()-began}
            history.append(rec);print(json.dumps(rec),flush=True)
    after=evaluate(model,eval_drives,eval_targets)
    arrays={'dynamics.recurrent_gain_raw':model.gain_raw.detach().cpu().numpy(),
            'dynamics.tau_raw':model.tau_raw.detach().cpu().numpy(),
            'readout.projection.weight':model.projection.weight.detach().cpu().numpy(),
            'readout.output.weight':model.output.weight.detach().cpu().numpy(),
            'readout.output.bias':model.output.bias.detach().cpu().numpy(),
            **{'probe.'+name:parameter.detach().cpu().numpy() for name,parameter in model.named_parameters() if name.startswith(('current.','temporal.','future.'))}}
    np.savez_compressed(args.output/'fitted-parameters.npz',**arrays)
    report={'format':'chreatures-dynamics-v2-temporal-research-fit-v1','source_adapter_sha256':meta['adapter_sha256'],
            'neurons':N,'edges':graph.w.nnz,'device':str(torch.cuda.get_device_name() if args.device=='cuda' else args.device),
            'torch':torch.__version__,'hip':torch.version.hip,'updates':args.updates,'initialization':str(args.initialize) if args.initialize else 'new','frequency_split':'training .35-.75 or .9-1.6 Hz; heldout .75-.9 Hz; independent optical/body phases','supervised_head':'current linear Z512; future GRU128 on Z512 only','seconds':time.monotonic()-began,
            'target_order':['current_visual_sin','current_visual_cos','current_body0','future_visual_sin_delta','future_visual_cos_delta','future_body0_delta'],
            'before':before,'after':after,'history':history,'status':'research-fit-not-controller',
            'fixed_parameters':{'baseline':.2,'adaptation_gain':.15,'adaptation_tau':1.5,'support_recovery':.024,'support_cost':.003},
            'limitations':['procedural optic/body excitation, not embodied competence','body adapter and optical adapter frozen','Z512 plus 6 dimensional supervised head, not a trained controller']}
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
