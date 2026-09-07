"""One-way export of the current articulated MJCF and measured optic membership.
No residents are restored/advanced; fixture is a newly authored browser epoch.
Run from repository root: .venv/bin/python native/browser-world/export_fixture.py
"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from chreatures.articulated import ArticulatedWorld
from chreatures.physical_batch import OPTIC_ATLAS_SHA256, OPTIC_SITE_SIDE_HEX, OPTIC_SUPPORTED_SITE_MASK
out=Path(__file__).parent/'fixtures'
spec=json.loads((ROOT/'data/habitats/hollow-garden.json').read_text())
spec['entities'].append({'id':'physical-screen','mobility':'static','material':'cream','physical_material':'rock','position':[2.0,2.15,0.55],'shapes':[{'type':'box','size':[0.012,0.72,0.5]}],'components':[]})
spec['bodies'][0]['heading']=0.0
w=ArticulatedWorld(spec=spec)
xml=w._xml
(out/'garden.xml').write_text(xml)
def name(kind,i): return mujoco.mj_id2name(w.model,kind,i)
bodies=[]
for b in w.bodies:
    a=w._resident_articulation[b.id]
    joints=[w._leg_joints[b.id][l['name']][k] for l in a['legs']['layout'] for k in ('hip','knee')]
    bodies.append({'id':b.id,'root':w._body_mj[b.id], 'head':mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_GEOM,f'resident:{b.id}:geom:head'), 'qpos':[int(w.model.jnt_qposadr[j]) for j in joints], 'dofs':[int(w.model.jnt_dofadr[j]) for j in joints], 'legs':a['legs']['layout'],'controller':a['controller'],'physiology':[b.energy,b.gut,b.fatigue,0,0,1,1,0,0,0,0,0], 'eyes':[[a['trunk']['head_size'][0]+.004,s*a['trunk']['head_size'][1]*.72,0] for s in (1,-1)]})
geoms=[]
for i in range(w.model.ngeom):
    mat=int(w.model.geom_matid[i]); rgba=w.model.mat_rgba[mat] if mat>=0 else w.model.geom_rgba[i]
    geoms.append({'id':i,'name':name(mujoco.mjtObj.mjOBJ_GEOM,i),'type':int(w.model.geom_type[i]),'size':w.model.geom_size[i].tolist(),'rgba':rgba.tolist(),'body':int(w.model.geom_bodyid[i])})
entities=[]
for e in w._entities:
    cs=e.get('components',[]); food=next((c for c in cs if c['type']=='food'),{}); scent=next((c for c in cs if c['type']=='scent'),{})
    entities.append({'id':e['id'],'body':w._entity_mj[e['id']],'free':e['mobility']=='free','food':food.get('amount',0),'nutrition':food.get('nutrition',1),'odor':scent.get('odor',-1),'strength':scent.get('strength',0),'growth':.002 if food else 0,'geoms':[g['id'] for g in geoms if g['body']==w._entity_mj[e['id']]]})
manifest={'format':'chreatures-browser-world-v1','engine':'mujoco-3.12.0-wasm-browser-epoch-1','source_mjcf_sha256':hashlib.sha256(xml.encode()).hexdigest(),'atlas_sha256':OPTIC_ATLAS_SHA256,'anatomical_sites':OPTIC_SITE_SIDE_HEX.tolist(),'supported_sites':OPTIC_SUPPORTED_SITE_MASK.tolist(),'optics':'Engineered affine hex optics; measured atlas membership, not measured viewing directions.','bodies':bodies,'geoms':geoms,'entities':entities,'screen_geom':mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_GEOM,'entity:physical-screen:geom:0'),'dt':.05,'physics_dt':float(w.model.opt.timestep),'world_size':spec['size'],'body_notice':w.articulation_spec['description'],'data_notice':'Derived anatomical site membership from data/ports/optic-anatomy-audit-v1.npz; preserve repository NOTICE.md and original source notices.'}
(out/'garden.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
print(json.dumps({'bodies':len(bodies),'geoms':len(geoms),'sites':len(manifest['anatomical_sites']),'xml_sha256':manifest['source_mjcf_sha256']}))
