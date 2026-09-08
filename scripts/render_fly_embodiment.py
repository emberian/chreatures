#!/usr/bin/env python3
"""Offscreen MuJoCo renders of the imported fly or a saved physical world.

This is an observer artifact exporter. It never advances or controls a life.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / 'native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/model'
WORLD = ROOT / 'native/browser-world/fixtures/fly-ecology'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--azimuth', type=float, default=135)
    parser.add_argument('--elevation', type=float, default=-24)
    parser.add_argument('--distance', type=float)
    args = parser.parse_args()
    if mujoco.__version__ != '3.12.0':
        raise RuntimeError('The observer requires the current MuJoCo3.12 pin')
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else None
    if snapshot:
        xml = snapshot['xml']
        assets = {item['path']: (WORLD / item['path']).read_bytes()
                  for item in snapshot['fixture']['mesh_assets']}
        model = mujoco.MjModel.from_xml_string(xml, assets=assets)
        data = mujoco.MjData(model)
        for stored, native in [('geomSize', 'geom_size'), ('geomPos', 'geom_pos'),
                               ('geomRGBA', 'geom_rgba'), ('geomContype', 'geom_contype'),
                               ('geomConaffinity', 'geom_conaffinity'),
                               ('actuatorForceRange', 'actuator_forcerange'),
                               ('actuatorGainParameters', 'actuator_gainprm')]:
            target = getattr(model, native)
            target[:] = np.asarray(snapshot[stored]).reshape(target.shape)
        mujoco.mj_setConst(model, data)
        state = np.asarray(snapshot['physical'], np.float64)
        if state.size != mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError('Snapshot physical layout differs')
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        scope = 'Saved physical world geometry and pose; observer lighting only, no simulation step.'
    else:
        xml = (BODY / 'model.xml').read_text()
        model = mujoco.MjModel.from_xml_path(str(BODY / 'model.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        scope = 'Author imported anatomical reference pose, not a learned behavior.'
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 1000
    model.vis.headlight.ambient[:] = [.45, .46, .44]
    model.vis.headlight.diffuse[:] = [.8, .77, .70]
    model.vis.headlight.specular[:] = [.15, .15, .15]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 1.5] if snapshot else data.body('fly/c_thorax').xpos + [-0.45, 0, -0.25]
    camera.distance = args.distance or (48 if snapshot else 8)
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    options = mujoco.MjvOption()
    options.label = mujoco.mjtLabel.mjLABEL_NONE
    options.frame = mujoco.mjtFrame.mjFRAME_NONE
    with mujoco.Renderer(model, width=1600, height=1000) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        pixels = renderer.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', '1600x1000', '-i', 'pipe:0',
                    '-frames:v', '1', '-quality', '92', '-y', str(args.output)],
                   input=pixels.tobytes(), check=True)
    receipt = {'format': 'chreatures-native-anatomical-observer-v1', 'scope': scope,
               'mujoco': mujoco.__version__, 'width': 1600, 'height': 1000,
               'source_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
               'source_snapshot_sha256': sha(args.snapshot) if args.snapshot else None,
               'model_time': data.time, 'camera': {'distance_mm': camera.distance,
                 'azimuth': camera.azimuth, 'elevation': camera.elevation, 'lookat_mm': camera.lookat.tolist()},
               'output_sha256': sha(args.output), 'renderer_source_sha256': sha(Path(__file__))}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
