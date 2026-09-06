#!/usr/bin/env python3
"""Measure runtime domain coverage and candidate-varying action support."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--schema',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    d=np.load(a.data); x=np.asarray(d['features'],float); s=json.loads(a.schema.read_text()); b=json.loads(a.bank.read_text()); fs=b['fitted']['features']
    cand=d['candidate_unit']; env=d['environment_unit']; lin=d['lineage_unit']
    test=np.isin(cand,s['split'].get('heldout_candidates',[]))|np.isin(env,s['split'].get('heldout_environments',[]))|np.isin(lin,s['split'].get('heldout_lineages',[]))
    validation=(~test)&np.isin(cand,s['split'].get('validation_candidates',[]))
    if not validation.any(): validation=(~test)&(d['world_unit'].astype(int)%int(s['split'].get('validation_world_mod',5))==0)
    train=(~test)&(~validation)
    scopes={'all':np.ones(len(x),bool),'training_pool':train,'validation':validation,'final_holdout':test}
    law_masks={}; sensitivity={}
    for law in b['fitted']['laws']:
        ok=np.isfinite(x).all(1); action_component=np.zeros(len(x)); action_spans=[]
        for term in law['terms']:
            i=int(term['feature']); f=fs[i]; ok &= (x[:,i]>=f['minimum'])&(x[:,i]<=f['maximum'])
            if i>=16:
                z=(x[:,i]-f['mean'])/f['scale']; vals=np.interp(z,term['knots'],term['values']);action_component+=vals
                action_spans.append({'feature':f['name'],'latent_span_over_fitted_domain':float(max(term['values'])-min(term['values']))})
        law_masks[law['name']]=ok
        sensitivity[law['name']]={'action_terms':action_spans,'executed_action_component_std_latent':float(action_component.std()),'candidate_varying':bool(action_spans and action_component.std()>1e-12),'shared_state_history_terms_are_common_on_latent_scale_only':True,'nonlinear_response_transform_can_modulate_physical_action_differences':True}
    union=np.logical_and.reduce(list(law_masks.values()))
    stat=lambda mask,scope:{'rows':int(scope.sum()),'in_domain_rows':int((mask&scope).sum()),'fraction':float(mask[scope].mean())}
    out={'format':'chreatures-population-gam-support-report-v1','bank_sha256':sha(a.bank),'data_sha256':sha(a.data),'schema_sha256':sha(a.schema),'runtime_contract':'all inputs finite; range rejection only on each law smooth features; bank requires every scored law in domain','coverage':{k:stat(union,v) for k,v in scopes.items()},'per_law_all':{k:stat(v,scopes['all']) for k,v in law_masks.items()},'action_sensitivity':sensitivity,'interpretation':'descriptive support under executed actions; action terms permit candidate-varying scores but are not causal effects'}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'sha256':sha(a.output),'coverage':out['coverage'],'action_sensitivity':sensitivity},indent=2))
if __name__=='__main__':main()
