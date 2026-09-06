#!/usr/bin/env python3
"""Describe observed candidate/environment GAM support without imputing crossed pairs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 d=np.load(a.data); split=json.loads(a.split.read_text()); target=['energy_state_delta','fatigue_state_delta','effort']; cols=[0,1,2]; rows=[]
 pairs=sorted(set(zip(d['candidate_unit'].astype(str),d['environment_unit'].astype(str))))
 held=set(split['heldout_candidates']); val=set(split['validation_candidates'])
 for candidate,environment in pairs:
  mask=(d['candidate_unit'].astype(str)==candidate)&(d['environment_unit'].astype(str)==environment)
  rows.append({'candidate_sha256':candidate,'environment_sha256':environment,'split':'heldout' if candidate in held else 'validation' if candidate in val else 'training','transitions':int(mask.sum()),'observed_mean':{name:float(d['targets'][mask,col].mean()) for name,col in zip(target,cols)}})
 out={'format':'chreatures-population-candidate-environment-support-v1','data_sha256':sha(a.data),'unit_of_analysis':'observed candidate-environment assignment with resident transitions nested within it','observed_pairs':rows,'candidate_count':len(set(x[0] for x in pairs)),'environment_count':len(set(x[1] for x in pairs)),'observed_pair_count':len(pairs),'possible_crossed_pair_count':len(set(x[0] for x in pairs))*len(set(x[1] for x in pairs)),'proposal_guidance':'collect each candidate or lineage in repeated environments before emitting authenticated challenge scores','challenge_score_artifact':None,'limitation':'each candidate was observed in only one environment and no pair has independent repeats; genotype-by-environment transfer ranking is unidentified'}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(sha(a.output))
if __name__=='__main__':main()
