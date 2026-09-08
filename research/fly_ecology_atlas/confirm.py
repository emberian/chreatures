#!/usr/bin/env python3
"""Bind fresh physical outcomes to native-GAM diversity proposals."""
import argparse,json
from pathlib import Path
from research.fly_ecology_atlas.prepare import sha,write,FORMAT
from research.fly_ecology_atlas.fit import metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 plan=json.loads(a.plan.read_text());fit=json.loads((a.fit/'report.json').read_text());records=[]
 for proposal in fit['confirmation_proposals']:
  path=a.results/(proposal['setting_id']+'--confirmation-layout.json');r=json.loads(path.read_text())
  if r['setting']!=proposal or r['layout']['layout_id']!='confirmation-layout' or r['provenance']['plan_sha256']!=sha(a.plan):raise ValueError('Physical confirmation binding differs')
  observed=metrics(r,plan);records.append(dict(intent=proposal['intent'],proposal_sha256=sha(a.fit/(proposal['setting_id']+'.json')),run_sha256=sha(path),completed=r['completed'],predicted=proposal['predicted'],observed=observed,
   absolute_error={k:abs(v-observed[k])for k,v in proposal['predicted'].items()} if observed else None,error=r.get('error')))
 report=dict(format=FORMAT,status='physical-confirmation-executed',plan_sha256=sha(a.plan),gam_report_sha256=sha(a.fit/'report.json'),records=records,
  claim_limit='These are actual new-layout physical outcomes, including failure. Execution does not imply predictive accuracy or successful diversity; no CNS/controller substitution or promoted genotype.')
 write(a.output,report);print(json.dumps({'report':str(a.output),'sha256':sha(a.output)}))
if __name__=='__main__':main()
