#!/usr/bin/env python3
"""Summarize complete rich-development transition telemetry without loading Torch."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np

FORMAT="chreatures-rich-development-behavior-analysis-v1"
ACTIONS=("thrust","yaw","gaze_pitch","grip","signal_low","signal_mid","signal_high","posture")
OUTCOMES=("nutrition","contact","distance","effort","mechanical_work","ingested_mass","mouth_material_contacts","homeostatic_reward")
PHYSIOLOGY=("energy","gut","fatigue","speed_tanh","angular_velocity_tanh","circuit_physiology")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def q(v:np.ndarray)->dict[str,float]:
 a=np.asarray(v,dtype=np.float64)
 return {"mean":float(a.mean()),"std":float(a.std()),"q01":float(np.quantile(a,.01)),"q10":float(np.quantile(a,.1)),"q50":float(np.quantile(a,.5)),"q90":float(np.quantile(a,.9)),"q99":float(np.quantile(a,.99)),"min":float(a.min()),"max":float(a.max())}
def atomic(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp')
 t.write_text(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');os.replace(t,p)
def main()->int:
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 files=sorted(a.run.glob('telemetry-*.npz')); assert files
 parts={k:[] for k in ('episode','tick','world_slot','resident_slot','physiology','executed_action','oral','next_physiology','outcomes','goal_progress','goal_distance_t','goal_distance_t1','goal_switch','goal_attempt_age')}
 for p in files:
  with np.load(p,allow_pickle=False) as z:
   for k in parts:parts[k].append(z[k].reshape((-1,)+z[k].shape[2:]) if z[k].ndim>=3 else z[k].reshape(-1))
 d={k:np.concatenate(v) for k,v in parts.items()};n=len(d['episode']); assert n==491520
 action=d['executed_action']; oral=d['oral']; outcome=d['outcomes']; before=d['physiology'];after=d['next_physiology']
 action_stats={name:q(action[:,i])|{"fraction_abs_ge_0_95":float(np.mean(np.abs(action[:,i])>=.95))} for i,name in enumerate(ACTIONS)}
 outcome_stats={name:q(outcome[:,i]) for i,name in enumerate(OUTCOMES)}
 groups=[]
 for ep in sorted(set(d['episode'].tolist())):
  for w in sorted(set(d['world_slot'].tolist())):
   for r in sorted(set(d['resident_slot'].tolist())):
    m=(d['episode']==ep)&(d['world_slot']==w)&(d['resident_slot']==r)
    if not m.any():continue
    ix=np.flatnonzero(m); oo=outcome[m]; aa=action[m]
    groups.append({"episode":int(ep),"world":int(w),"resident":int(r),"steps":int(m.sum()),"energy_start":float(before[ix[0],0]),"energy_end":float(after[ix[-1],0]),"gut_end":float(after[ix[-1],1]),"fatigue_end":float(after[ix[-1],2]),"reward_sum":float(oo[:,7].sum()),"effort_sum":float(oo[:,3].sum()),"distance_sum":float(oo[:,2].sum()),"nutrition_sum":float(oo[:,0].sum()),"ingested_mass_sum":float(oo[:,5].sum()),"contact_fraction":float(np.mean(oo[:,1]>0)),"mouth_contact_steps":int(np.sum(oo[:,6]>0)),"oral_mean":float(oral[m].mean()),"action_abs_mean":float(np.abs(aa).mean()),"action_saturation_fraction":float(np.mean(np.abs(aa)>=.95)),"goal_progress_mean":float(d['goal_progress'][m].mean()),"goal_switches":int(d['goal_switch'][m].sum())})
 identity=json.load((a.run/'identity.json').open()); result_json=json.load((a.run/'result.json').open())
 reward=outcome[:,7]; effort=outcome[:,3]
 episode_summary=[]
 for ep in sorted(set(d['episode'].tolist())):
  rows=[row for row in groups if row['episode']==ep]
  episode_summary.append({
   "episode":int(ep),"resident_trajectories":len(rows),
   "mean_energy_delta":float(np.mean([row['energy_end']-row['energy_start'] for row in rows])),
   "mean_reward_sum":float(np.mean([row['reward_sum'] for row in rows])),
   "mean_effort_sum":float(np.mean([row['effort_sum'] for row in rows])),
   "mean_distance_sum":float(np.mean([row['distance_sum'] for row in rows])),
   "ingested_mass_sum":float(np.sum([row['ingested_mass_sum'] for row in rows])),
   "mouth_contact_steps":int(np.sum([row['mouth_contact_steps'] for row in rows])),
   "mean_action_saturation_fraction":float(np.mean([row['action_saturation_fraction'] for row in rows])),
  })
 goal_coefficient=float(identity['arguments']['goal_progress_coefficient'])
 result={"format":FORMAT,"scope":{"transitions":n,"packets":len(files),"episodes":len(set(d['episode'].tolist())),"worlds":len(set(d['world_slot'].tolist())),"residents_per_world":len(set(d['resident_slot'].tolist())),"dt_seconds":identity['dt_seconds']},"source":{"run":str(a.run),"identity_sha256":sha(a.run/'identity.json'),"result_sha256":sha(a.run/'result.json'),"final_checkpoint_sha256":sha(a.run/'checkpoint-update-000160.pt'),"artifact_sha256":result_json['artifact_sha256'],"profile_sha256":identity['profile']['sha256'],"bootstrap_sha256":identity['bootstrap_sha256']},"action_order":list(ACTIONS),"physiology_order":list(PHYSIOLOGY),"outcome_order":list(OUTCOMES),"actions":action_stats,"oral":q(oral)|{"fraction_ge_0_95":float(np.mean(oral>=.95)),"fraction_le_0_05":float(np.mean(oral<=.05))},"outcomes":outcome_stats,"physiology":{"before":{name:q(before[:,i]) for i,name in enumerate(PHYSIOLOGY)},"after":{name:q(after[:,i]) for i,name in enumerate(PHYSIOLOGY)},"energy_delta_total":float(np.sum(after[:,0]-before[:,0]))},"goals":{"progress":q(d['goal_progress']),"distance_before":q(d['goal_distance_t']),"distance_after":q(d['goal_distance_t1']),"switch_fraction":float(d['goal_switch'].mean()),"attempt_age":q(d['goal_attempt_age']),"coefficient":goal_coefficient,"mean_shaping_reward":float(d['goal_progress'].mean()*goal_coefficient),"mean_shaping_to_abs_physical_reward_ratio":float(abs(d['goal_progress'].mean()*goal_coefficient)/abs(reward.mean()))},"relationships":{"effort_reward_correlation":float(np.corrcoef(effort,reward)[0,1]),"effort_energy_delta_correlation":float(np.corrcoef(effort,after[:,0]-before[:,0])[0,1]),"oral_gut_correlation":float(np.corrcoef(oral,before[:,1])[0,1])},"episode_summary":episode_summary,"per_episode_world_resident":groups,"interpretation_limits":["Executed action saturation is measured after policy sampling and clipping; telemetry does not identify pre-clipping actor logits.","Audit world/resident indices and outcomes were not controller inputs.","This is one inherited lineage and seed; distributions do not establish behavioral improvement against a matched control."]}
 atomic(a.output,result);print(json.dumps({"output":str(a.output),"sha256":sha(a.output),"transitions":n,"groups":len(groups)}));return 0
if __name__=='__main__':raise SystemExit(main())
