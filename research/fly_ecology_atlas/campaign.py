#!/usr/bin/env python3
"""Execute the frozen whole-world campaign with a wall-clock launch budget."""
import argparse,concurrent.futures,json,subprocess,time
from pathlib import Path
from research.fly_ecology_atlas.prepare import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--budget-minutes',type=float,default=55);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 if not 1<=a.workers<=8:raise ValueError('Use one to eight independent native worlds')
 plan=json.loads(a.plan.read_text());a.results.mkdir(parents=True,exist_ok=True);started=time.monotonic();jobs=[(s,l)for s in plan['settings']for l in plan['layouts'][:3]]
 if a.pilot:jobs=jobs[:1]
 def run(job):
  setting,layout=job;name=setting['setting_id']+'--'+layout['layout_id'];path=a.results/(name+'.json')
  if path.exists():
   r=json.loads(path.read_text())
   if r['provenance']['plan_sha256']!=sha(a.plan):raise ValueError('Existing run belongs to another plan')
   return {'name':name,'retained':True,'completed':r['completed'],'wall_seconds':r['wall_seconds']}
  command=['node',plan['runner'],'--plan',str(a.plan),'--setting',setting['setting_id'],'--layout',layout['layout_id'],'--output',str(path)]
  with (a.results/(name+'.stdout.log')).open('x') as out,(a.results/(name+'.stderr.log')).open('x') as err:
   result=subprocess.run(command,stdout=out,stderr=err)
  r=json.loads(path.read_text()) if path.exists() else {}
  return {'name':name,'completed':bool(r.get('completed')),'exit_code':result.returncode,'wall_seconds':r.get('wall_seconds'),'receipt_present':path.exists()}
 # Independent worlds may run concurrently. Once a native world starts it is
 # allowed to finish its coherent bounded run; the budget stops NEW launches.
 done=[];pending=iter(jobs);exhausted=False
 with concurrent.futures.ThreadPoolExecutor(max_workers=1 if a.pilot else a.workers) as pool:
  active={}
  while active or not exhausted:
   while len(active)<(1 if a.pilot else a.workers) and not exhausted:
    if time.monotonic()-started>a.budget_minutes*60:exhausted=True;break
    job=next(pending,None)
    if job is None:exhausted=True;break
    active[pool.submit(run,job)]=job
   if not active:break
   finished,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in finished:
    active.pop(future);record=future.result();done.append(record);print(json.dumps(record),flush=True)
 report={'format':'chreatures-fly-ecology-campaign-receipt-v1','plan_sha256':sha(a.plan),'pilot':a.pilot,'workers':1 if a.pilot else a.workers,'wall_seconds':time.monotonic()-started,'runs':done,'requested_runs':len(jobs),'launch_budget_minutes':a.budget_minutes}
 name='pilot-receipt.json' if a.pilot else 'campaign-receipt.json';write(a.results/name,report)
if __name__=='__main__':main()
