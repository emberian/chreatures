"""Actual-world sensory information probe and bounded existing-CNS fine tuning.

All probe inputs are CNS Z512. Raw afferents are CNS inputs and offline targets;
geometry, teacher labels, reward and recorded commands are never model inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
import torch
from torch import nn
from chreatures.cns_adapter_contract import ARRAY_SPECS,MAGIC,PARAMETER_ORDER
from research.sensorimotor_skills.cns_adapter import (
    CNSStaticArrays,TrainableCNSAdapter,load_export_arrays,write_parameter_artifact,
)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def service(path,expected):
    if sha(path)!=expected:raise ValueError('service SHA mismatch')
    with path.open('rb') as f:
        if f.read(8)!=MAGIC:raise ValueError('requires current CNS service')
        n=struct.unpack('<I',f.read(4))[0];meta=json.loads(f.read(n))
    offset=12+n;arrays={}
    for name,dtype,shape in ARRAY_SPECS:
        a=np.memmap(path,dtype=dtype,shape=shape,offset=offset,mode='r');offset+=a.nbytes
        if hashlib.sha256(a).hexdigest()!=meta['array_sha256'][name]:raise ValueError(name+' checksum mismatch')
        arrays[name]=a
    if offset!=path.stat().st_size:raise ValueError('service length mismatch')
    static=CNSStaticArrays.from_service_arrays(arrays,{'graph_dataset_sha256':meta['graph_sha256'],'atlas_file_sha256':meta['atlas_sha256']})
    model=TrainableCNSAdapter(static,device=torch.device('cuda'))
    load_export_arrays(model,{name:arrays[name] for name in PARAMETER_ORDER})
    return model,meta


def load_corpus(root,expected_manifest):
    manifest_path=root/'combined-receipt.json'
    if sha(manifest_path)!=expected_manifest:raise ValueError('combined receipt mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True or len(manifest['episodes'])!=8:raise ValueError('incomplete corpus')
    senses=[];latents=[];lineage=[]
    for i,rec in enumerate(manifest['episodes']):
        if rec['episode_index']!=i or rec['split']!=('train' if i<6 else 'heldout-worlds'):raise ValueError('world split differs')
        side=root/rec['sidecar_directory'];mp=side/'manifest.json'
        if sha(mp)!=rec['sidecar_manifest_sha256']:raise ValueError('sidecar manifest mismatch')
        m=json.loads(mp.read_text())
        if m['cns_service_artifact_sha256']!=manifest['cns_service_artifact_sha256']:raise ValueError('CNS lineage mismatch')
        if m['source_episode_sha256']!=rec['sha256']:raise ValueError('episode binding mismatch')
        values={}
        for name,r in m['buffers'].items():
            p=side/name
            if p.stat().st_size!=r['byteLength'] or sha(p)!=r['sha256']:raise ValueError('sidecar buffer mismatch: '+name)
            values[name]=np.memmap(p,dtype=r['dtype'],shape=tuple(r['shape']),mode='r')
        sensory=values['afferent-input.f32']
        if sensory.shape!=(513,3,5356) or not np.isfinite(sensory).all():raise ValueError('afferent layout differs')
        if not np.array_equal(values['tick-index.u32'],np.arange(513)):raise ValueError('nonchronological ticks')
        if not np.allclose(np.diff(values['input-time-seconds.f64']),.05,rtol=0,atol=1e-7):raise ValueError('dt differs')
        if not np.all(values['active-mask.u8']==7):raise ValueError('this bounded probe requires all three residents active')
        ep=root/rec['file']
        if sha(ep)!=rec['sha256']:raise ValueError('episode file mismatch')
        with np.load(ep,allow_pickle=False) as f:
            z=f['cns_latent'].copy();reset=f['reset'].copy()
        if z.shape!=(513,3,512) or not np.isfinite(z).all():raise ValueError('latent shape differs')
        if reset[1:].any():raise ValueError('unexpected within-world reset')
        senses.append(np.asarray(sensory));latents.append(z)
        lineage.append({k:rec[k] for k in ['episode_index','split','world_seed','variation_seed','layout_identity','sha256','sidecar_manifest_sha256']})
    # World-major source becomes time,world*resident,channel; no split mixing.
    return np.concatenate(senses,axis=1),np.concatenate(latents,axis=1),manifest,lineage


def make_targets(sensory,atlas):
    sites=np.load(atlas)['site_side_hex'];side=sites[:,0];xy=sites[:,1:]
    masks=[];names=[]
    for s in np.unique(side):
        local=side==s;mid=np.median(xy[local],axis=0)
        for x in range(2):
            for y in range(2):
                mask=local&((xy[:,0]>=mid[0])==bool(x))&((xy[:,1]>=mid[1])==bool(y))
                if not mask.any():raise ValueError('empty optic target sector')
                masks.append(mask);names.extend([f'optic_side{int(s)}_sector{x}{y}_{c}' for c in 'rgb'])
    optic=sensory[:,:,:5313].reshape(513,24,1771,3)
    features=np.concatenate([optic[:,:,m].mean(2) for m in masks],axis=-1)
    names.extend([f'body_{i:02d}' for i in range(43)])
    return np.concatenate([features,sensory[:,:,5313:]],axis=-1).astype(np.float32),len(names)-43,names


class Probe(nn.Module):
    def __init__(self,outputs,delta_scale):
        super().__init__();self.register_buffer('delta_scale',delta_scale)
        self.gru=nn.GRU(512,64);self.current=nn.Linear(512,outputs);self.future=nn.Linear(64,outputs)
        nn.init.zeros_(self.current.weight);nn.init.zeros_(self.current.bias)
        nn.init.zeros_(self.future.weight);nn.init.zeros_(self.future.bias)
    def forward(self,z,hidden=None):
        h,hidden=self.gru(z,hidden)
        return self.current(z),self.future(h)*self.delta_scale,hidden


def metric(prediction,truth,baseline,active):
    mse=(prediction-truth).square().mean((0,1));base=(baseline-truth).square().mean((0,1))
    selected=active&(base>1e-8)
    if not selected.any():return {'channels':0,'mse':None,'baseline_mse':None,'skill':None}
    return {'channels':int(selected.sum()),'mse':float(mse[selected].mean()),'baseline_mse':float(base[selected].mean()),
            'skill':float(1-mse[selected].sum()/base[selected].sum()),
            'per_channel_skill':[float(1-mse[i]/base[i]) if bool(selected[i]) else None for i in range(len(mse))]}


def evaluate(probe,z,target,active,optic_dim):
    with torch.no_grad():current,delta,_=probe(z)
    # Ignore the first 16 physical ticks for predictor initialization.
    current=current[16:-4];delta=delta[16:-4];now=target[16:-4];future=target[20:]
    result={}
    for name,sl in [('optic',slice(0,optic_dim)),('body',slice(optic_dim,None))]:
        result[name+'_current']=metric(current[:,:,sl],now[:,:,sl],torch.zeros_like(now[:,:,sl]),active[sl])
        result[name+'_future_delta']=metric(delta[:,:,sl],future[:,:,sl]-now[:,:,sl],torch.zeros_like(now[:,:,sl]),active[sl])
    return result


def probe_loss(current,delta,target,optic_dim,active,delta_scale,burn=8):
    # Unit scale is fixed from training worlds, equal optical/body family weight.
    now=target[:-4];future=target[4:]-now
    loss=current.new_zeros(())
    for sl in [slice(0,optic_dim),slice(optic_dim,None)]:
        use=active[sl]
        if use.any():
            loss+=((current[:-4,:,sl][burn:][:,:,use]-now[:,:,sl][burn:][:,:,use])**2).mean()
            loss+=2*(((delta[:-4,:,sl][burn:][:,:,use]-future[:,:,sl][burn:][:,:,use])/delta_scale[sl][use])**2).mean()
    return loss


def run_cns(model,sensory,batch_indices,*,grad=False,state=None):
    packet=sensory[:,batch_indices]
    optic=packet[:,:,:5313].reshape(len(packet),len(batch_indices),1771,3);body=packet[:,:,5313:]
    reset=torch.zeros(packet.shape[:2],dtype=torch.bool,device=packet.device)
    return model.forward_sequence(optic,body,reset,state,checkpoint_ticks=4)


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--manifest-sha',required=True)
    p.add_argument('--service',type=Path,required=True);p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-updates',type=int,default=160);p.add_argument('--fine-tune-epochs',type=int,default=2);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);began=time.monotonic();torch.manual_seed(20260909);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7)
    sensory_np,z_np,manifest,lineage=load_corpus(args.corpus,args.manifest_sha)
    targets,optic_dim,names=make_targets(sensory_np,args.atlas)
    mean=targets[:,:18].mean((0,1));std=targets[:,:18].std((0,1));active=std>1e-5
    scale=np.maximum(std,1e-3);target=torch.tensor((targets-mean)/scale,device='cuda')
    active=torch.tensor(active,device='cuda');z=torch.tensor(z_np,device='cuda');sensory=torch.tensor(sensory_np,device='cuda')
    delta_scale=(target[4:,:18]-target[:-4,:18]).std((0,1)).clamp_min(.02)
    probe=Probe(target.shape[-1],delta_scale).cuda();optimizer=torch.optim.Adam(probe.parameters(),lr=.001)
    rng=np.random.default_rng(20260909)
    report={'format':'chreatures-physical-cns-probe-v1','source_manifest_sha256':args.manifest_sha,
            'service_sha256':manifest['cns_service_artifact_sha256'],'lineage':lineage,'target_names':names,
            'target_scale':'training worlds only, current std floor .001; normalized delta std floor .02; channels train std<=1e-5 excluded',
            'probe_inputs':'CNS Z512 only, no commands/rewards/geometry/teacher labels','horizon_seconds':.2,
            'training_worlds':list(range(6)),'heldout_worlds':[6,7],
            'scope':'offline physical sensory calibration; never updates public CNS or resident binding'}
    def publish(stage):
        report['stage']=stage;report['elapsed_seconds']=time.monotonic()-began
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'stage':stage,'seconds':report['elapsed_seconds']}),flush=True)
    for update in range(args.probe_updates):
        start=int(rng.integers(0,513-68));idx=rng.choice(18,12,replace=False)
        optimizer.zero_grad(set_to_none=True);cur,delt,_=probe(z[start:start+68,idx])
        loss=probe_loss(cur,delt,target[start:start+68,idx],optic_dim,active,probe.delta_scale);loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(),1);optimizer.step()
    report['frozen_probe_train']=evaluate(probe,z[:,:18],target[:,:18],active,optic_dim)
    report['frozen_probe_heldout']=evaluate(probe,z[:,18:],target[:,18:],active,optic_dim)
    torch.save({'probe':probe.state_dict(),'mean':mean,'scale':scale,'active':active.cpu(),'target_names':names},args.output/'frozen-probe.pt')
    publish('frozen-probe-complete')
    model,meta=service(args.service,manifest['cns_service_artifact_sha256']);model.eval()
    # Verify the recorded frozen latents against actual afferent replay before fit.
    with torch.no_grad():replay,_=run_cns(model,sensory[:16],list(range(3)))
    parity=float((replay-z[:16,:3]).abs().max());report['recorded_replay_max_abs']=parity
    if parity>5e-4:publish('stopped-replay-mismatch');raise RuntimeError('recorded CNS alignment differs')
    bias_latent=torch.tanh(model.readout_output.bias).detach()
    constant=bias_latent[None,None].expand(513,6,-1)
    report['frozen_probe_zero_edges']=evaluate(probe,constant,target[:,18:],active,optic_dim)
    publish('frozen-cns-replay-verified')
    if args.fine_tune_epochs:
        optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':5e-5},{'params':probe.parameters(),'lr':3e-4}])
        model.train();history=[]
        # Chronological TBPTT: two complete worlds/6 lives per group, no world seams.
        for epoch in range(args.fine_tune_epochs):
            for group in range(3):
                idx=list(range(group*6,(group+1)*6));state=None;hidden=None
                for start in range(0,480,32):
                    optimizer.zero_grad(set_to_none=True)
                    latent,next_state=run_cns(model,sensory[start:start+36],idx,state=state,grad=True)
                    current,delta,next_hidden=probe(latent,hidden)
                    loss=probe_loss(current,delta,target[start:start+36,idx],optic_dim,active,probe.delta_scale,burn=0)
                    loss.backward();nn.utils.clip_grad_norm_([*model.parameters(),*probe.parameters()],1);optimizer.step()
                    # Four lookahead inputs are not consumed into the next chunk state.
                    # Recompute the 32 tick boundary with updated weights, no gradient.
                    with torch.no_grad():
                        latent32,state=run_cns(model,sensory[start:start+32],idx,state=state)
                        _,_,hidden=probe(latent32,hidden)
                    state=state.detach();hidden=hidden.detach()
                history.append({'epoch':epoch,'group':group,'loss':float(loss.detach()),'seconds':time.monotonic()-began})
                report['fine_tune_history']=history;publish('fine-tuning')
        model.eval()
        with torch.no_grad():
            fitted=[]
            for group in range(4):
                out,_=run_cns(model,sensory,list(range(group*6,(group+1)*6)));fitted.append(out)
            fitted=torch.cat(fitted,dim=1)
        report['candidate_train']=evaluate(probe,fitted[:,:18],target[:,:18],active,optic_dim)
        report['candidate_heldout']=evaluate(probe,fitted[:,18:],target[:,18:],active,optic_dim)
        constant=torch.tanh(model.readout_output.bias)[None,None].expand(513,6,-1)
        report['candidate_zero_edges']=evaluate(probe,constant,target[:,18:],active,optic_dim)
        report['candidate_parameters']=write_parameter_artifact(args.output/'research-candidate-parameters.npz',model,training_status='trained',
            provenance={'source_service_sha256':manifest['cns_service_artifact_sha256'],'physical_corpus_sha256':args.manifest_sha,
                        'training_worlds':list(range(6)),'heldout_worlds':[6,7],'scope':'offline sensory probe candidate only; never deployed'})
        np.save(args.output/'candidate-latents.npy',fitted.cpu().numpy())
        torch.save({'probe':probe.state_dict(),'mean':mean,'scale':scale,'active':active.cpu(),'target_names':names},args.output/'candidate-probe.pt')
    # One actual zero-edge fullgraph probe verifies the constant-bias ablation identity.
    with torch.no_grad():
        model.recurrent.values().zero_();zero,_=run_cns(model,sensory[:8],list(range(18,24)))
        report['zero_edge_constant_max_abs']=float((zero-torch.tanh(model.readout_output.bias)[None,None]).abs().max())
    report['completed']=True;publish('complete')

if __name__=='__main__':main()
