"""One joined real-graph baseline/regime audit and held-out temporal decoding fit."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from research.dynamics_v2.model import FullGraph,Regime,load_service,N


def audit_recordings(graph, clip, blank):
    c=np.load(clip/'neural_rates.npy',mmap_mode='r')
    b=np.load(blank/'neural_rates.npy',mmap_mode='r')
    out={}
    for name,mask in [('afferent',~graph.mask),('nonafferent',graph.mask)]:
        cs=np.asarray(c[:,mask]);bs=np.asarray(b[:,mask])
        out[name]={'clip_blank_rms':float(np.sqrt(np.mean((cs-bs)**2))),
                   'blank_from_initial_rms':float(np.sqrt(np.mean((bs-bs[:1])**2))),
                   'clip_temporal_rms':float(np.sqrt(np.mean(np.diff(cs,axis=0)**2))),
                   'blank_temporal_rms':float(np.sqrt(np.mean(np.diff(bs,axis=0)**2))),
                   'blank_zero_fraction':float(np.mean(bs<1e-6))}
    return out


def stimuli(atlas,ticks):
    """Four complete independent clips: two train, two held out; no shuffled ticks."""
    sites=np.load(atlas)['site_side_hex'].astype(np.float32)
    xy=sites[:,1:];xy=(xy-xy.mean(0))/np.maximum(xy.std(0),1)
    phase=np.array([.4,2.2,1.3,3.1],np.float32)
    frequency=np.array([.55,1.35,.8,1.05],np.float32)
    direction=np.array([1,-1,-1,1],np.float32)
    t=np.arange(ticks,dtype=np.float32)*.05
    angle=phase[None,:]+t[:,None]*frequency[None,:]*direction[None,:]*2*np.pi
    # Hidden target describes visual phase, future change and one body afferent.
    luminance=.5+.35*np.sin(xy[None,None,:,0]*1.7+angle[:,:,None])
    optic=np.repeat(luminance[:,:,:,None],3,axis=-1).astype(np.float32)
    body=np.zeros((ticks,4,43),np.float32)
    body[:,:,0]=1.5*np.sin(t[:,None]*np.array([1.1,.65,.8,1.3])[None,:]+phase)
    targets=np.stack([np.sin(angle),np.cos(angle),body[:,:,0]],axis=-1)
    return optic,body,targets


def fit_probe(features,targets):
    """Ridge fit readout on two lives; held-out lives never influence scaling/fit."""
    # Predict actual 0.2 s changes, evaluated against persistence (zero delta).
    horizon=4
    x=features[20:-horizon]; y=targets[20+horizon:]-targets[20:-horizon]
    train=x[:,:2].reshape(-1,x.shape[-1]); test=x[:,2:].reshape(-1,x.shape[-1])
    yt=y[:,:2].reshape(-1,3); yv=y[:,2:].reshape(-1,3)
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    train=(train-mean)/scale;test=(test-mean)/scale
    train=np.concatenate([train,np.ones((len(train),1))],axis=1)
    test=np.concatenate([test,np.ones((len(test),1))],axis=1)
    coefficients=np.linalg.solve(train.T@train+10*np.eye(train.shape[1]),train.T@yt)
    prediction=test@coefficients
    mse=np.mean((prediction-yv)**2,axis=0); persistence=np.mean(yv**2,axis=0)
    return {'heldout_future_delta_mse':mse.tolist(),'persistence_mse':persistence.tolist(),
            'skill_vs_persistence':(1-mse/np.maximum(persistence,1e-12)).tolist()},coefficients,mean,scale


def main():
    p=argparse.ArgumentParser();p.add_argument('--service',type=Path,required=True);p.add_argument('--atlas',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ticks',type=int,default=160)
    p.add_argument('--clip',type=Path);p.add_argument('--blank',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    a,meta=load_service(args.service);g=FullGraph(a);optic,body,targets=stimuli(args.atlas,args.ticks)
    rng=np.random.default_rng(20260907)
    # Half broad actual CNS rows, half strongly reached direct postsynaptic rows.
    sensory=np.zeros(N,np.float32);sensory[g.receptors]=1;sensory[g.body]=1
    reach=np.abs(g.w)@sensory;reach[~g.mask]=0
    top=np.argsort(reach)[-192:]
    rows=np.unique(np.r_[top,rng.choice(np.flatnonzero(g.mask),192,replace=False)])
    assert g.mask[rows].all()
    report={'format':'chreatures-dynamics-v2-regime-audit-v1','source_adapter_sha256':meta['adapter_sha256'],
            'neurons':N,'edges':g.w.nnz,'seed':20260907,'fit':'two whole train clips, two disjoint frequency/phase heldout clips',
            'probe_neurons':len(rows),'target_order':['visual_sin_delta','visual_cos_delta','body_channel0_delta'],
            'sealed_body_neutral_mean':float(g.neutral[g.body].mean()),'regimes':[]}
    if args.clip and args.blank:report['recording_audit']=audit_recordings(g,args.clip,args.blank)
    # Drives are boundary targets, never readout inputs; every fit feature is masked CNS.
    drives=[g.drive(optic[t],body[t]) for t in range(args.ticks)]
    for label,regime in [('v1',None),('v2-g09',Regime(gain=.9)),('v2-g13',Regime(gain=1.3)),('v2-g18',Regime(gain=1.8))]:
        began=time.monotonic();state=(np.zeros((N,4),np.float32),np.zeros((N,4),np.float32),np.ones((N,4),np.float32))
        features=[];norms=[];sat=[]
        for drive in drives:
            state=g.step_v1(drive,state) if regime is None else g.step_v2(drive,state,regime)
            features.append(state[0][rows].T.copy());norms.append(float(np.sqrt(np.mean(state[0][g.mask]**2))))
            sat.append(float(np.mean(np.abs(state[0][g.mask])>.19)))
        feat=np.stack(features);fit,coef,mean,scale=fit_probe(feat,targets)
        rec={'name':label,'parameters':asdict(regime) if regime else 'sealed_v1','seconds':time.monotonic()-began,
             'rms_mean':float(np.mean(norms)),'saturation_fraction':float(np.mean(sat)),**fit}
        report['regimes'].append(rec)
        np.savez_compressed(args.output/(label+'-probe.npz'),rows=rows,weight=coef,mean=mean,scale=scale,features=feat,targets=targets)
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(rec),flush=True)
    # Exact neutral is an invariant for V2, unlike the v1 zero birth transient.
    z=np.zeros((N,1),np.float32)
    neutral=g.step_v2(g.neutral,(z,z.copy(),np.ones_like(z)),Regime())
    report['neutral_max_deviation']=float(np.abs(neutral[0]).max())
    report['completed']=True
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
