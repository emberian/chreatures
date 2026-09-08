"""Stage one current fly physics generation. Does not publish or advance lives."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'site/live'
SOURCE = HERE / 'fixtures/fly-ecology'
fixture = json.loads((SOURCE/'world.json').read_text())
items = {
    'world-runtime.mjs': HERE/'runtime.mjs',
    'fixtures/fly-ecology/world.json': SOURCE/'world.json',
    'fixtures/fly-ecology/scene.xml': SOURCE/'scene.xml',
    'fixtures/fly-ecology/motor92.json': ROOT/'research/fly_embodiment/motor92-channel-schema.json',
    'pkg/chreatures_browser_world.js': HERE/'pkg/chreatures_browser_world.js',
    'pkg/chreatures_browser_world_bg.wasm': HERE/'pkg/chreatures_browser_world_bg.wasm',
    'vendor/mujoco/mujoco.js': HERE/'node_modules/@mujoco/mujoco/mujoco.js',
    'vendor/mujoco/mujoco.wasm': HERE/'node_modules/@mujoco/mujoco/mujoco.wasm',
    'vendor/mujoco/LICENSE': HERE/'licenses/MuJoCo-LICENSE',
    'fixtures/fly-ecology/FlyGym-LICENSE': ROOT/'native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/author-source/FlyGym-LICENSE',
}
for mesh in fixture['mesh_assets']:
    path = mesh['path']
    if Path(path).name != path:
        raise ValueError('Mesh VFS path must be a flat authenticated filename')
    source = SOURCE/path
    if hashlib.sha256(source.read_bytes()).hexdigest() != mesh['sha256']:
        raise ValueError(f'Body mesh hash differs: {path}')
    items[f'fixtures/fly-ecology/{path}'] = source
assets = {}
for target, source in items.items():
    dest = OUT/target
    dest.parent.mkdir(parents=True,exist_ok=True)
    if target == 'world-runtime.mjs':
        dest.write_text('// GENERATED from native/browser-world/runtime.mjs.\n'+source.read_text().replace('import("@mujoco/mujoco")','import("./vendor/mujoco/mujoco.js")'))
    else:
        shutil.copyfile(source,dest)
    raw = dest.read_bytes()
    assets[target] = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
# Superseded generic-body files remain in Git history, never another production path.
for old in ['fixtures/garden.json','fixtures/garden.xml']:
    (OUT/old).unlink(missing_ok=True)
manifest = {
    'format':'chreatures-browser-physics-assets-v4','engine':fixture['engine'],
    'upstream':{'name':'@mujoco/mujoco','version':'3.12.0','source':'https://github.com/google-deepmind/mujoco/tree/3.12.0/wasm','license':'Apache-2.0','license_path':'vendor/mujoco/LICENSE'},
    'body_source':{'name':'NeuroMechFly/FlyGym','revision':'ca65a510c2afe6ac61c51df4f274c8d190c2f95f','license':'Apache-2.0','license_path':'fixtures/fly-ecology/FlyGym-LICENSE'},
    'original_code_license':'AGPL-3.0-or-later',
    'fixture_notice':'Micro-CT-derived female fly body paired explicitly with male CNS; 126 axes, 84 effective position servos, six adhesion and two native oral actuators. Servo and physiological dynamics are engineered assumptions, not identified muscles.',
    'assets':assets,
}
(OUT/'physics-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'staged':str(OUT),'files':len(items),'bytes':sum(v['bytes'] for v in assets.values())}))
