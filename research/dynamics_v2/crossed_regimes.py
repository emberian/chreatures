"""Crossed actual-fullgraph physiological regimes for native GAM response fitting."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from research.dynamics_v2.model import FullGraph,load_service,N
from research.dynamics_v2.train import csr


def fit_probe(x,target,ntrain=8,horizon=4):
    y=target[16+horizon:]-target[16:-horizon];x=x[16:-horizon]
    train=x[:,:ntrain].reshape(-1,x.shape[-1]);test=x[:,ntrain:].reshape(-1,x.shape[-1])
    yt=y[:,:ntrain].reshape(-1,3);yv=y[:,ntrain:].reshape(-1,3)
    mu=train.mean(0);sd=np.maximum(train.std(0),1e-6)
    train=(train-mu)/sd;test=(test-mu)/sd
    train=np.column_stack((train,np.ones(len(train))));test=np.column_stack((test,np.ones(len(test))))
    coef=np.linalg.solve(train.T@train+30*np.eye(train.shape[1]),train.T@yt)
    mse=np.mean((test@coef-yv)**2,axis=0);base=np.mean(yv**2,axis=0)
    return float(mse.mean()),float(base.mean()),float(np.mean(1-mse/np.maximum(base,1e-12)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--service',type=Path,required=True);p.add_argument('--atlas',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ticks',type=int,default=128);p.add_argument('--baseline-only',action='store_true');p.add_argument('--setting',type=float,nargs=3,metavar=('GAIN','TAU','ADAPT'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    a,meta=load_service(args.service);g=FullGraph(a);w=csr(g.w,'cuda')
    typ=torch.tensor(np.asarray(a['atlas.neuron_type'],dtype=np.int64),device='cuda')
    rng=np.random.default_rng(20260908);sites=np.load(args.atlas)['site_side_hex'][:,1:].astype(np.float32)
    sites=(sites-sites.mean(0))/np.maximum(sites.std(0),1)
    freq=np.r_[np.repeat([.4,.8,1.2,1.6],2),[.6,1.,1.4,.9]]
    direction=np.r_[np.tile([-1,1],4),[-1,1,-1,1]]
    phase=rng.uniform(0,2*np.pi,12);bodyphase=rng.uniform(0,2*np.pi,12)
    bodyfreq=freq[rng.permutation(12)]
    t=np.arange(args.ticks)*.05;angle=t[:,None]*freq[None,:]*direction[None,:]*2*np.pi+phase
    targets=np.stack([np.sin(angle),np.cos(angle),1.5*np.sin(t[:,None]*bodyfreq*2*np.pi+bodyphase)],axis=-1).astype(np.float32)
    drives=[]
    for index in range(args.ticks):
        optic=np.repeat((.5+.35*np.sin(sites[None,:,0]*1.7+angle[index,:,None]))[:,:,None],3,axis=-1).astype(np.float32)
        body=np.zeros((12,43),np.float32);body[:,0]=targets[index,:,2]
        drives.append(g.drive(optic,body)-g.neutral)
    drives=torch.tensor(np.stack(drives),device='cuda')
    sensory=np.zeros(N,np.float32);sensory[g.receptors]=1;sensory[g.body]=1
    reach=np.abs(g.w)@sensory;reach[~g.mask]=0
    rows=np.unique(np.r_[np.argsort(reach)[-192:],rng.choice(np.flatnonzero(g.mask),192,replace=False)])
    rows=torch.tensor(rows,device='cuda');mask=torch.tensor(g.mask,device='cuda')
    report={'format':'chreatures-v2-crossed-fullgraph-regimes-v1','source_adapter_sha256':meta['adapter_sha256'],
            'neurons':N,'edges':g.w.nnz,'ticks':args.ticks,'batch':12,'train_streams':8,'heldout_streams':4,
            'baseline':.2,'adaptation_tau':1.5,'metric':'heldout 0.2-second delta decoder skill against persistence',
            'recovery_memory_definition':'nonafferent RMS 1s after neutral / last driven RMS; descriptive, not task utility','records':[]}
    began=time.monotonic()
    if args.baseline_only:
        with torch.no_grad():
            def arr(name):return torch.tensor(np.asarray(a[name]),device='cuda')
            bias=(.5*torch.tanh(arr('dynamics.bias_raw')))[typ,None]
            tau=(.025+.475*torch.sigmoid(arr('dynamics.tau_raw')))[typ,None]
            source=(.5+torch.sigmoid(arr('dynamics.source_raw')))[typ,None]
            target=(.5+torch.sigmoid(arr('dynamics.target_raw')))[typ,None]
            exc=(.5+torch.sigmoid(arr('dynamics.excitability_raw')))[typ,None]
            neutral=torch.tensor(g.neutral,device='cuda')
            rate=torch.zeros((N,12),device='cuda');adapt=rate.clone();support=rate+1;features=[]
            alpha=(.025/tau).clamp_max(1)
            for centered in drives:
                drive=centered+neutral
                for _ in range(2):
                    q=torch.relu(torch.tanh(bias+exc*(drive+.92*target*torch.sparse.mm(w,source*rate))-.1*adapt))
                    rate=rate+alpha*(support*q-rate)
                adapt=adapt+.01*(rate-adapt)
                support=(support+.05*(.024*(1-support)-.003*rate)).clamp(.65,1)
                features.append(rate[rows].T.cpu().numpy())
            mse,persist,skill=fit_probe(np.stack(features),targets)
        report['v1_baseline']={'temporal_probe_mse':mse,'persistence_mse':persist,'skill':skill,'seconds':time.monotonic()-began}
        report['completed']=True;(args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['v1_baseline']),flush=True)
        return
    with torch.no_grad():
        for gain,tau,k in ([args.setting] if args.setting else itertools.product([.7,1.05,1.4],[.04,.08,.16],[.05,.15,.35])):
            x=torch.zeros((N,12),device='cuda');adapt=x.clone();support=x+1
            alpha=float(-np.expm1(-.025/tau));features=[];sat=[]
            def tick(drive,x,adapt,support):
                for _ in range(2):
                    u=drive+gain*torch.sparse.mm(w,x)-k*adapt
                    x=x+alpha*(support*.2*torch.tanh(u/.2)-x)
                adapt=adapt+float(-np.expm1(-.05/1.5))*(x-adapt)
                support=(support+.05*(.024*(1-support)-.003*x.abs()/.2)).clamp(.65,1)
                return x,adapt,support
            for drive in drives:
                x,adapt,support=tick(drive,x,adapt,support)
                features.append(x[rows].T.cpu().numpy());sat.append(float((x[mask].abs()>.19).float().mean()))
            before=float(x[mask].square().mean().sqrt())
            for _ in range(20):x,adapt,support=tick(torch.zeros_like(x),x,adapt,support)
            recovery=float(x[mask].square().mean().sqrt())/max(before,1e-12)
            mse,persist,skill=fit_probe(np.stack(features),targets)
            rec={'gain':gain,'tau':tau,'adaptation_gain':k,'temporal_probe_mse':mse,'persistence_mse':persist,
                 'skill':skill,'recovery_memory':recovery,'saturation':float(np.mean(sat)),'elapsed_seconds':time.monotonic()-began}
            report['records'].append(rec);(args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(rec),flush=True)
    report['completed']=True;(args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
