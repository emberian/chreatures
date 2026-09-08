#!/usr/bin/env python3
"""Freeze a gene-space design and actual MJCF layout variants; no simulation."""
from __future__ import annotations
import argparse, copy, hashlib, json, shutil, subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

FEATURES = ('branch_angle_rad','lateral_probability','phototropism','contact_avoidance')
RANGES = ((.25,1.15),(.08,.75),(.1,1.6),(.2,2.))
FORMAT = 'chreatures-fly-ecology-development-atlas-v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def design(seed):
    rng=np.random.default_rng(seed);best=None;score=-1
    for _ in range(128):
        x=np.column_stack([(rng.permutation(24)+rng.random(24))/24 for _ in FEATURES])
        d=((x[:,None]-x[None,:])**2).sum(-1);np.fill_diagonal(d,np.inf)
        minimum=d.min()
        if minimum>score:best=x;score=minimum
    return [dict(setting_id=f'setting-{i:02d}',split='train' if i<18 else 'heldout',
                 genes={k:float(lo+x[j]*(hi-lo)) for j,(k,(lo,hi)) in enumerate(zip(FEATURES,RANGES))},
                 coordinates={k:float(2*x[j]-1) for j,k in enumerate(FEATURES)}) for i,x in enumerate(best)]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260910);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);base=json.loads(a.fixture.read_text());scene=a.fixture.parent/base['scene_xml']
    if sha(scene)!=base['source_mjcf_sha256']:raise ValueError('Base scene hash differs')
    if base['ecology']['format']!='chreatures-ecology-config-v2':raise ValueError('Current native developmental ecology v2 required')
    # Freeze the host and current Wasm module; binary dependencies remain shared
    # by explicit symlink, and their exact installed file hashes are recorded.
    runtime=a.output/'runtime';runtime.mkdir();shutil.copy2(a.runtime,runtime/'runtime.mjs');shutil.copytree(a.runtime.parent/'pkg',runtime/'pkg')
    (runtime/'node_modules').symlink_to((a.runtime.parent/'node_modules').resolve(),target_is_directory=True)
    shutil.copy2(Path(__file__).with_name('run.mjs'),a.output/'run.mjs')
    fixtures=a.output/'layouts';fixtures.mkdir()
    layouts=[dict(layout_id='layout-0',screen_pos=[18,0,7],brightness=1.,obstacle=None),
             dict(layout_id='layout-1',screen_pos=[-18,5,7],brightness=.55,obstacle=None),
             dict(layout_id='layout-2',screen_pos=[18,-10,5],brightness=.85,obstacle=dict(pos=[-6,-6,1.1],size=[.6,2.,.6])),
             dict(layout_id='confirmation-layout',screen_pos=[16,12,6],brightness=.9,obstacle=dict(pos=[7,7,1.5],size=[.6,1.5,.7]))]
    for i,layout in enumerate(layouts):
        xml=ET.parse(scene);screen=next(n for n in xml.iter('geom') if n.get('name')=='ecology/visual-screen');screen.set('pos',' '.join(map(str,layout['screen_pos'])))
        if layout['obstacle']:
            obstacle=next(n for n in xml.iter('geom') if n.get('name')=='ecology/curved-bark-03')
            for k,v in layout['obstacle'].items():obstacle.set(k,' '.join(map(str,v)))
        folder=fixtures/layout['layout_id'];folder.mkdir();xmlpath=folder/'scene.xml';xml.write(xmlpath,encoding='unicode')
        f=copy.deepcopy(base);f['fixture_id']=base['fixture_id']+'-'+layout['layout_id'];f['source_mjcf_sha256']=f['scene_xml_sha256']=sha(xmlpath);f['scene_xml']='scene.xml';f['ecology']['seed']=a.seed+1000+i
        write(folder/'world.json',f);layout.update(fixture=str((folder/'world.json').resolve()),fixture_sha256=sha(folder/'world.json'),scene_xml_sha256=sha(xmlpath),world_seed=a.seed+1000+i)
    plan=dict(format=FORMAT,status='planned-not-executed',seed=a.seed,control_dt=.01,ticks=800,features=list(FEATURES),ranges=dict(zip(FEATURES,RANGES)),settings=design(a.seed),layouts=layouts,
              runner=str((a.output/'run.mjs').resolve()),runner_sha256=sha(a.output/'run.mjs'),runtime=str((runtime/'runtime.mjs').resolve()),asset_root=str(a.fixture.parent.resolve()),
              neutral_motor='all 92 outputs zero; first84 neutral servo targets,last8 adhesion/pump/saliva off; diagnostic only, no CNS',
              mujoco_sha256={name:sha(a.runtime.parent/'node_modules/@mujoco/mujoco'/name) for name in ['package.json','mujoco.js','mujoco.wasm']},
              node_version=subprocess.check_output(['node','--version'],text=True).strip(),
              base_fixture_sha256=sha(a.fixture),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('run.mjs'),Path(__file__).with_name('fit.py'),a.runtime]},
              runtime_sha256=sha(runtime/'runtime.mjs'),core_wasm_sha256=sha(runtime/'pkg/chreatures_browser_world_bg.wasm'),core_js_sha256=sha(runtime/'pkg/chreatures_browser_world.js'))
    write(a.output/'plan.json',plan);print(json.dumps({'plan':str(a.output/'plan.json'),'sha256':sha(a.output/'plan.json')}))
if __name__=='__main__':main()
