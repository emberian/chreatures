#!/usr/bin/env python3
"""Record a fresh coupled CNS/physical-screen research life, entirely headlessly."""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.metal_circuit import MetalCircuit
from chreatures.population import CandidateGenome, content_sha256
from chreatures.resident_birth import FORMAT as BIRTH_FORMAT, controller_identity, validate_manifest
from chreatures.runtime3d import Habitat3D, MODEL_DT
from chreatures.visual_stimulus import FORMAT as STIMULUS_FORMAT
from scripts.capture_recorded_body_frames import MuJoCoBodyFrameCapture
from scripts.serve_metal import Sequenced, handler_type


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_birth(args, out):
    spec = json.loads(args.habitat.read_text())
    bio = json.loads(args.biosphere.read_text())
    old_birth = json.loads(args.physical_founders.read_text())
    identity = controller_identity(args.controller)
    founders = []
    for row in old_birth['residents']:
        candidate = copy.deepcopy(row['candidate'])
        candidate['base_controller_sha256'] = identity['file_sha256']
        candidate['sha256'] = content_sha256(candidate)
        CandidateGenome(candidate)
        founders.append({'candidate': candidate, 'cns_adapter_sha256': identity['cns_adapter_sha256']})
    birth = validate_manifest({'format': BIRTH_FORMAT, 'controller': identity, 'residents': founders})
    if len(founders) != len(spec['bodies']):
        raise ValueError('physical founder count differs')
    if any(row['id'] == 'cns-cinema-screen' for row in spec['entities']):
        raise ValueError('screen identity already exists')
    # A declared assay installation, placed before birth. No runtime controller
    # sees these coordinates or the identity of the screen.
    spec['entities'].append({'id': 'cns-cinema-screen', 'mobility': 'static',
        'material': 'cream', 'physical_material': 'light', 'position': args.screen_position,
        'shapes': [{'type': 'box', 'size': [0.025, args.screen_width/2, args.screen_height/2]}],
        'components': []})
    frames = np.load(args.stimulus, mmap_mode='r', allow_pickle=False)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < args.frames:
        raise ValueError('stimulus requires enough uint8 RGB frames')
    frame_source = args.stimulus.resolve()
    if args.condition == 'blank':
        frame_source = out/'blank.npy'
        np.save(frame_source, np.zeros((1, *frames.shape[1:]), dtype=np.uint8), allow_pickle=False)
    spec['optic_screen'] = {'entity': 'cns-cinema-screen', 'shape_index': 0, 'front': '-x',
                            'texture_shape': list(frames.shape[1:3])}
    spec['optic_stimulus'] = {'format': STIMULUS_FORMAT, 'frames_path': str(frame_source),
        'frames_sha256': sha(frame_source), 'fps': 20, 'start_tick': args.warmup_ticks, 'end': 'hold'}
    save_json(out/'habitat.json', spec)
    save_json(out/'biosphere.json', bio)
    save_json(out/'resident-birth.json', birth)
    return spec, frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service',type=Path,required=True)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--habitat',type=Path,required=True)
    p.add_argument('--biosphere',type=Path,required=True)
    p.add_argument('--physical-founders',type=Path,required=True)
    p.add_argument('--stimulus',type=Path,required=True)
    p.add_argument('--stimulus-video',type=Path,required=True)
    p.add_argument('--stimulus-receipt',type=Path,required=True)
    p.add_argument('--training-receipt',type=Path)
    p.add_argument('--atlas',type=Path,default=ROOT/'data/ports/optic-anatomy-audit-v1.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',choices=['clip','blank'],default='clip')
    p.add_argument('--frames',type=int,default=600)
    p.add_argument('--warmup-ticks',type=int,default=40)
    p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--screen-position',type=float,nargs=3,default=[2.9,2.1,1.5])
    p.add_argument('--screen-width',type=float,default=3.6)
    p.add_argument('--screen-height',type=float,default=2.7)
    p.add_argument('--camera-azimuth',type=float,default=135.0)
    p.add_argument('--camera-elevation',type=float,default=-24.0)
    p.add_argument('--camera-distance',type=float,default=3.0)
    p.add_argument('--no-render',action='store_true')
    args=p.parse_args()
    if not 2 <= args.frames <= 3600 or args.warmup_ticks < 0:
        raise ValueError('bounded recording extent required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec, source_frames=prepare_birth(args,out)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save_json(out/'intent.json',{'format':'chreatures-cns-screen-assay-v1','condition':args.condition,
        'seed':args.seed,'warmup_ticks':args.warmup_ticks,'frames':args.frames,'dt':MODEL_DT,
        'source_revision':revision,'service_sha256':sha(args.service),'controller_sha256':sha(args.controller),
        'stimulus_sha256':sha(args.stimulus),'meaning':'fresh research life; actual CNS and physical response, not learned motor competence'})
    count=len(spec['bodies'])
    rates=np.lib.format.open_memmap(out/'neural_rates.npy',mode='w+',dtype=np.float32,shape=(args.frames,165122))
    commands=np.lib.format.open_memmap(out/'commands.npy',mode='w+',dtype=np.float32,shape=(args.frames,count,12))
    positions=np.lib.format.open_memmap(out/'positions.npy',mode='w+',dtype=np.float64,shape=(args.frames,count,3))
    times=np.lib.format.open_memmap(out/'time_seconds.npy',mode='w+',dtype=np.float64,shape=(args.frames,))
    latents=np.lib.format.open_memmap(out/'cns_latents.npy',mode='w+',dtype=np.float32,shape=(args.frames,count,512))
    capture=None if args.no_render else MuJoCoBodyFrameCapture(out,frame_count=args.frames,width=640,height=480,distance=args.camera_distance,azimuth=args.camera_azimuth,elevation=args.camera_elevation)
    started=time.perf_counter()
    try:
        with MetalCircuit(args.service,capacity=count) as brain:
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_type(Sequenced(brain,out/'neural-state')))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                habitat=Habitat3D(seed=args.seed,brain_url=f'http://127.0.0.1:{server.server_port}',spec=spec,
                    biosphere=out/'biosphere.json',resident_artifact=args.controller,population_birth=out/'resident-birth.json')
                habitat.branch='research-cns-screen-'+args.condition
                for _ in range(args.warmup_ticks):habitat.step()
                habitat.paused=True;habitat.save(out/'world-before-clip.json');habitat.paused=False
                for index in range(args.frames):
                    habitat.step()
                    receipt=habitat.neural.capture_rates(f'frame-{index:05d}',habitat.remote_ids[habitat.world.bodies[0].id])
                    if abs(receipt['time']-habitat.world.time)>1e-7:
                        raise RuntimeError('neural and physical capture times differ')
                    rate_path=Path(receipt['path']);rates[index]=np.fromfile(rate_path,dtype='<f4');rate_path.unlink()
                    commands[index]=habitat.actual_previous
                    positions[index]=[(b.x,b.y,b.z) for b in habitat.world.bodies]
                    latents[index]=[habitat.neural_state[b.id]['features'] for b in habitat.world.bodies]
                    times[index]=habitat.world.time
                    if capture:capture.append(habitat.world)
                    if index%100==0:
                        progress={'frames':index+1,'target':args.frames,'tick':habitat.tick,'seconds':time.perf_counter()-started}
                        save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                habitat.paused=True;habitat.save(out/'world-after-clip.json')
                final_tick=habitat.tick
                final_counts=[habitat.cognition_state[b.id]['memory_count'] for b in habitat.world.bodies]
                view=habitat.view()
                save_json(out/'final-public-view.json',view)
                # One whole-loop continuation check on this designated research
                # instance: restore the complete pre-clip state and repeat its
                # first input, including private controller and stochastic state.
                restored=Habitat3D.load(out/'world-before-clip.json',brain_url=habitat.neural.url,resident_artifact=args.controller)
                restored.paused=False;restored.step()
                repeated=restored.neural.capture_rates('restored-first-frame',restored.remote_ids[restored.world.bodies[0].id])
                repeated_path=Path(repeated['path'])
                restore_checks={
                    'neural_rates_exact':bool(np.array_equal(rates[0],np.fromfile(repeated_path,dtype='<f4'))),
                    'delivered_commands_exact':bool(np.array_equal(commands[0],restored.actual_previous)),
                    'body_positions_exact':bool(np.array_equal(positions[0],[(b.x,b.y,b.z) for b in restored.world.bodies])),
                    'time_exact':float(times[0])==restored.world.time,
                }
                repeated_path.unlink();restored.neural.close()
                if not all(restore_checks.values()):
                    raise RuntimeError(f'whole-loop continuation differs: {restore_checks}')
                model_identity=copy.deepcopy(brain.cns_identity)
                habitat.neural.close()
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)
        for array in (rates,commands,positions,times,latents):array.flush()
        body_receipt=None if capture is None else capture.seal(checkpoint_hashes=[sha(out/'world-before-clip.json'),sha(out/'world-after-clip.json')],source_world_revision=revision)
        with np.load(args.atlas,allow_pickle=False) as atlas:
            soma=np.asarray(atlas['soma_xyz_source_units'],dtype=np.float32)
            valid=np.isfinite(soma).all(axis=1) & (soma >= 0).all(axis=1)
        np.save(out/'soma_positions.npy',soma,allow_pickle=False);np.save(out/'soma_valid.npy',valid,allow_pickle=False)
        summary={'format':'chreatures-cns-screen-response-v1','status':'completed','condition':args.condition,
            'frames':args.frames,'final_tick':final_tick,'seconds':time.perf_counter()-started,'memory_count':final_counts,
            'cns_identity':model_identity,'whole_loop_restore':restore_checks,'rate_capture_sha256':sha(out/'neural_rates.npy'),
            'commands_sha256':sha(out/'commands.npy'),'positions_sha256':sha(out/'positions.npy'),
            'controller_sha256':sha(args.controller),'world_source_revision':revision,
            'frame_semantics':'input is sampled before each 50ms tick; CNS and physical outputs are recorded after that same tick'}
        save_json(out/'response.receipt.json',summary)
        if capture:
            if args.condition=='clip':
                if args.frames==len(source_frames):
                    shutil.copyfile(args.stimulus,out/'stimulus_frames.npy')
                else:np.save(out/'stimulus_frames.npy',source_frames[:args.frames],allow_pickle=False)
            else:np.save(out/'stimulus_frames.npy',np.zeros((args.frames,*source_frames.shape[1:]),dtype=np.uint8),allow_pickle=False)
            streams={}
            for name in ('time_seconds','stimulus_frames','neural_rates','soma_positions','soma_valid','body_frames'):
                path=out/(name+'.npy');a=np.load(path,mmap_mode='r',allow_pickle=False)
                streams[name]={'file':path.name,'sha256':sha(path),'shape':list(a.shape),'dtype':str(a.dtype)}
            axes=[0,1];bounds=[[float(soma[valid,i].min()),float(soma[valid,i].max())] for i in axes]
            max_delta=max(float(np.max(np.abs(rates[i]-rates[0]))) for i in range(args.frames))
            manifest={'format':'chreatures-bad-apple-response-bundle-v1','status':'recorded synchronized source, rate-model, and physical frames',
                'frame_count':args.frames,'playback_fps':20,'streams':streams,
                'neural_projection':{'axes':axes,'bounds_source_units':bounds,'display_mode':'signed delta rate',
                    'delta_reference':'first recorded frame','rate_floor':0,'rate_ceiling':max(max_delta,1e-8),'rate_unit':'normalized model-rate change'},
                'sources':{'stimulus_sha256':sha(args.stimulus_video),'stimulus_receipt_sha256':sha(args.stimulus_receipt),
                    'malecns_graph_sha256':model_identity['graph_sha256'],'rate_capture_sha256':sha(out/'neural_rates.npy'),
                    'body_recording_sha256':sha(out/'body-frames.receipt.json'),'world_source_revision':revision},
                'neural_model':{'status':'trained rate model with cited training receipt' if args.training_receipt else 'initialized, untrained rate model',
                    'artifact_sha256':sha(args.service),'training_receipt_sha256':sha(args.training_receipt) if args.training_receipt else None},
                'controller':{'status':'initialized controller; motor and memory core untrained','artifact_sha256':sha(args.controller),'training_receipt_sha256':None},
                'frame_semantics':summary['frame_semantics']}
            save_json(out/'manifest.json',manifest)
        print(json.dumps(summary),flush=True)
    except Exception as error:
        if capture:capture.abort()
        save_json(out/'failed.json',{'type':type(error).__name__,'message':str(error),'seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':main()
