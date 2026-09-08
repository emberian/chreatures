#!/usr/bin/env python3
"""Native GAM atlas of actual MuJoCo developmental outcomes; offline only."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.dynamics_v2.gam_fit import _require_native_gamfit,_capture_native_stderr
from research.fly_ecology_atlas.prepare import FEATURES,RANGES,FORMAT,sha,write
TARGETS=('height_mm','canopy_span_mm','route_permeability','branch_count','blocked_rate')
FORMULAS={'joint':'response ~ duchon('+','.join(FEATURES)+',centers=12)',
          'additive':'response ~ '+'+'.join(f's({k},k=3)' for k in FEATURES)}
def scores(pred,truth):
    error=np.asarray(pred,np.float64)-truth
    return dict(rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error))),max_abs=float(np.max(np.abs(error))))
def metrics(record,plan):
    if not record['completed']:return None
    final=record['final'];ecology=final['ecology'];structures=ecology['structures']
    # Required stable host observer contract; never substitute requested openness.
    growth=final['growth'];routes=final['routeMeasurements']
    openness=np.asarray(routes['open'],np.float64)
    fixture=json.loads(Path(record['layout']['fixture']).read_text())
    area=np.array([r['cross_section_m2']/r['length_m'] for r in fixture['ecology']['routes']])
    base_open=np.array([r['base_open_fraction'] for r in fixture['ecology']['routes']])
    diffusivity=np.array([p['diffusivity_m2_s'] for p in fixture['ecology']['pools']])
    if openness.shape!=area.shape or not np.isfinite(openness).all():raise ValueError('Measured route openness missing/invalid')
    geometry=final['geometry'];heights=[];bounds=[];by_owner={}
    binding_owner={s['physics_binding']:s['owner_id'] for s in structures}
    for g in geometry:
        p=np.asarray(g['position'],np.float64);rotation=np.asarray(g['rotation'],np.float64).reshape(3,3);size=g['size']
        extent=np.abs(rotation[:,2])*size[1]+size[0]
        heights.append(p[2]+extent[2]);bound=(p[:2]-extent[:2],p[:2]+extent[:2]);bounds.append(bound)
        owner=next((owner for binding,owner in binding_owner.items() if g['name']==f'ecology:{binding}:geom'),None)
        if owner is None:raise ValueError('Compiled branch lacks a committed native owner')
        by_owner.setdefault(owner,[]).append(bound)
    habitat_span=0. if not bounds else float(np.linalg.norm(np.max([b[1] for b in bounds],axis=0)-np.min([b[0] for b in bounds],axis=0)))
    colony_ids=[o['id'] for o in fixture['ecology']['organisms'] if o['anchored_region'] and o['genotype']['development']]
    local_spans=[float(np.linalg.norm(np.max([b[1] for b in by_owner[owner]],axis=0)-np.min([b[0] for b in by_owner[owner]],axis=0))) if owner in by_owner else 0. for owner in colony_ids]
    span=float(np.mean(local_spans))
    material=np.sum([s['initial_material'] for s in structures],axis=0,dtype=np.float64) if structures else np.zeros(len(fixture['ecology']['pools']))
    programs={o['id']:o['genotype']['development'] for o in fixture['ecology']['organisms']}
    atp=sum(programs[s['owner_id']]['atp_cost'] for s in structures)
    proposed=growth['proposed'];rejected=growth['blocked']
    return dict(height_mm=float(max(heights,default=0)),canopy_span_mm=span,habitat_canopy_span_mm=habitat_span,route_permeability=float(np.dot(area,openness*base_open)/area.sum()),branch_count=float(len(structures)),
                blocked_rate=float(rejected/proposed) if proposed else 0.,proposed=int(proposed),blocked=int(rejected),clearance_accepted=int(growth['clearanceAccepted']),
                route_diffusive_conductance_m3_s=(area[:,None]*openness[:,None]*base_open[:,None]*diffusivity[None,:]).tolist(),
                effective_route_advection_m3_s=(np.asarray(routes['flowsM3S'])*openness*base_open).tolist(),
                captured_photon_energy=float(ecology['accounting']['captured_photon_energy']),
                lateral_count=sum(o['development_state']['lateral_cursor'] for o in ecology['organisms']),
                active_branch_count=sum(s['active'] for s in structures),construction_material_by_pool=material.tolist(),construction_atp=float(atp),
                maintenance_atp=float(ecology['accounting']['maintenance_atp_spent']),regulation_atp=float(ecology['accounting']['regulation_atp_spent']),
                maximum_elemental_residual=float(ecology['accounting']['maximum_absolute_residual']))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--analysis-recipe',type=Path);a=p.parse_args()
    plan=json.loads(a.plan.read_text());plan_hash=sha(a.plan);records=[];failures=[]
    if a.analysis_recipe:
        recipe=json.loads(a.analysis_recipe.read_text())
        if recipe['plan_sha256']!=plan_hash or recipe['formulas']!=FORMULAS or recipe['analysis_source_sha256']!=sha(__file__):raise ValueError('Analysis recipe binding differs')
    for setting in plan['settings']:
        for layout in plan['layouts'][:3]:
            path=a.results/(setting['setting_id']+'--'+layout['layout_id']+'.json')
            if not path.exists():raise ValueError('Missing actual outcome: '+str(path))
            r=json.loads(path.read_text())
            if r['provenance']['plan_sha256']!=plan_hash or r['setting']!=setting or r['layout']!=layout:raise ValueError('Outcome identity differs')
            expected_runner=plan['runner_sha256']
            if r['provenance']['runner_sha256']!=expected_runner or r['provenance']['runtime_sha256']!=plan['runtime_sha256'] or r['provenance']['core_wasm_sha256']!=plan['core_wasm_sha256']:raise ValueError('Run source or native build differs')
            m=metrics(r,plan)
            item=dict(setting=setting,layout=layout['layout_id'],source_sha256=sha(path),metrics=m,error=r.get('error'))
            if m is None:failures.append(item)
            else:records.append(item)
    a.output.mkdir(parents=True,exist_ok=False);gam=_require_native_gamfit();warnings=[]
    train=[r for r in records if r['setting']['split']=='train'];test=[r for r in records if r['setting']['split']=='heldout']
    diagnostics={};models={}
    def rows(records,target):return [{**r['setting']['genes'],'response':r['metrics'][target]} for r in records]
    def fit(records,target,formula):
        data=rows(records,target);gam.validate_formula(data,formula,family='gaussian')
        model,msgs=_capture_native_stderr(lambda:gam.fit(data,formula,family='gaussian'));warnings.extend(msgs);return model
    for target in TARGETS:
        y=np.array([r['metrics'][target] for r in train],np.float64)
        if len(train)<24 or len({r['setting']['setting_id'] for r in train})<12:
            diagnostics[target]={'status':'insufficient-completed-settings','training_rows':len(train)};continue
        if not np.isfinite(y).all():raise ValueError('Nonfinite physical target')
        if np.ptp(y)<=1e-12:
            diagnostics[target]={'status':'constant-target-not-fitted','value':float(y[0]),'sd_float64':float(y.std())};continue
        entry={'training_rows':len(train),'heldout_rows':len(test),'sd_float64':float(y.std()),'models':{}}
        for label,formula in FORMULAS.items():
            try:
                model=fit(train,target,formula);path=a.output/(target+'-'+label+'.gam');model.save(path);restored=gam.load(path)
                original=np.asarray(model.predict(rows(train,target))).reshape(-1);reloaded=np.asarray(restored.predict(rows(train,target))).reshape(-1)
                if not np.allclose(original,reloaded,atol=1e-10,rtol=0):raise ValueError('Native serialized GAM reload differs')
                models[(target,label)]=restored;setting_pred=[];setting_truth=[];setting_mean=[]
                for sid in sorted({r['setting']['setting_id'] for r in train}):
                    fold_train=[r for r in train if r['setting']['setting_id']!=sid];fold_test=[r for r in train if r['setting']['setting_id']==sid]
                    fold=fit(fold_train,target,formula);setting_pred.extend(np.asarray(fold.predict(rows(fold_test,target))).reshape(-1).tolist());setting_truth.extend(r['metrics'][target] for r in fold_test);setting_mean.extend([np.mean([r['metrics'][target] for r in fold_train])]*len(fold_test))
                world_pred=[];world_truth=[];world_mean=[]
                for lid in sorted({r['layout'] for r in train}):
                    fold_train=[r for r in train if r['layout']!=lid];fold_test=[r for r in train if r['layout']==lid]
                    fold=fit(fold_train,target,formula);world_pred.extend(np.asarray(fold.predict(rows(fold_test,target))).reshape(-1).tolist());world_truth.extend(r['metrics'][target] for r in fold_test);world_mean.extend([np.mean([r['metrics'][target] for r in fold_train])]*len(fold_test))
                heldout=np.array([r['metrics'][target] for r in test])
                entry['models'][label]=dict(status='native-fit-serialized-reloaded',artifact_sha256=sha(path),formula=formula,reload_max_abs=float(np.max(np.abs(original-reloaded))),
                    train=scores(original,y),heldout=scores(np.asarray(restored.predict(rows(test,target))).reshape(-1),heldout) if test else None,
                    heldout_mean=scores(np.full(len(test),y.mean()),heldout) if test else None,
                    leave_setting_out=scores(setting_pred,np.array(setting_truth)),leave_setting_out_mean=scores(setting_mean,np.array(setting_truth)),
                    leave_world_out=scores(world_pred,np.array(world_truth)),leave_world_out_mean=scores(world_mean,np.array(world_truth)))
            except Exception as exc:
                entry['models'][label]={'status':'native-fit-or-validation-rejected','error':str(exc)}
                models.pop((target,label),None)
        diagnostics[target]=entry
    # Select by training leave-setting-out error only; heldout is never used to
    # tune the model or choose settings. Confirmation remains actual physics.
    rng=np.random.default_rng(plan['seed']+41);points=[{k:float(lo+x*(hi-lo)) for k,(lo,hi),x in zip(FEATURES,RANGES,row)} for row in rng.uniform(.1,.9,(256,4))]
    predictions={};selected_models={}
    for target in TARGETS:
        if diagnostics[target].get('status')=='constant-target-not-fitted':
            predictions[target]=np.full(len(points),diagnostics[target]['value']);selected_models[target]='observed-constant-mean-no-GAM'
        candidates=[(entry['leave_setting_out']['rmse'],label) for label,entry in diagnostics[target].get('models',{}).items() if entry['status']=='native-fit-serialized-reloaded']
        if candidates:
            _,label=min(candidates);selected_models[target]=label;predictions[target]=np.asarray(models[(target,label)].predict(points)).reshape(-1)
    proposals=[]
    if models and all(k in predictions for k in TARGETS):
        eligible=np.flatnonzero(predictions['branch_count']>=1)
        if len(eligible):
            normalized={k:(v-v.mean())/max(v.std(),1e-12) for k,v in predictions.items()}
            objectives={'high-canopy':normalized['height_mm']+.3*normalized['route_permeability'],
                        'wide-canopy':normalized['canopy_span_mm']-.2*normalized['route_permeability'],
                        'permeable-branches':normalized['route_permeability']+.2*normalized['branch_count']}
            used=[]
            for label,objective in objectives.items():
                order=eligible[np.argsort(objective[eligible])[::-1]]
                index=next((int(i) for i in order if all(np.linalg.norm(np.array([(points[i][k]-points[j][k])/(hi-lo) for k,(lo,hi) in zip(FEATURES,RANGES)]))>.25 for j in used)),None)
                if index is None:continue
                used.append(index);proposal=dict(setting_id='confirmation-'+label,split='confirmation',genes=points[index],intent=label,predicted={k:float(v[index]) for k,v in predictions.items()},selection='diversity objective on training-only native GAM; no physical success inferred')
                write(a.output/(proposal['setting_id']+'.json'),proposal);proposals.append(proposal)
    report=dict(format=FORMAT,status='actual-native-gam-fit-confirmation-pending' if proposals else 'actual-fit-no-viable-diversity-proposal',plan_sha256=plan_hash,
        source_sha256=sha(__file__),analysis_recipe_sha256=sha(a.analysis_recipe) if a.analysis_recipe else None,formulas=FORMULAS,planned_source_sha256=plan['source_sha256'],native_build=gam.build_info(),native_version=gam.__version__,diagnostics=diagnostics,records=records,failed_runs=failures,
        selected_models=selected_models,confirmation_proposals=proposals,native_messages=warnings,
        claim_limit='Actual MuJoCo colony-development sensitivity with neutral diagnostic flies; no CNS computation or motor-learning competence claim. Unsuccessful physical runs and rejected fits retained.')
    write(a.output/'report.json',report)
    write(a.output/'receipt.json',dict(format=FORMAT,status=report['status'],plan_sha256=plan_hash,raw_report_sha256=sha(a.output/'report.json'),analysis_recipe_sha256=report['analysis_recipe_sha256'],analysis_source_sha256=sha(__file__),native_version=gam.__version__,native_engine=gam.build_info()['engine_crate'],diagnostics=diagnostics,completed_runs=len(records),failed_runs=[dict(setting=r['setting']['setting_id'],layout=r['layout'],error=r['error']) for r in failures],confirmation_proposals=proposals,claim_limit=report['claim_limit']))
    print(json.dumps({'report':str(a.output/'report.json'),'sha256':sha(a.output/'report.json'),'proposals':len(proposals)}))
if __name__=='__main__':main()
