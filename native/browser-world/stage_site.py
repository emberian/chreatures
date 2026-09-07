"""Stage generated browser physics assets; canonical source remains native/browser-world.
No publishing or remote mutation. Re-run after building Rust/Wasm.
"""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'site/live'
items={
 'world-runtime.mjs':HERE/'runtime.mjs',
 'fixtures/garden.json':HERE/'fixtures/garden.json',
 'fixtures/garden.xml':HERE/'fixtures/garden.xml',
 'pkg/chreatures_browser_world.js':HERE/'pkg/chreatures_browser_world.js',
 'pkg/chreatures_browser_world_bg.wasm':HERE/'pkg/chreatures_browser_world_bg.wasm',
 'vendor/mujoco/mujoco.js':HERE/'node_modules/@mujoco/mujoco/mujoco.js',
 'vendor/mujoco/mujoco.wasm':HERE/'node_modules/@mujoco/mujoco/mujoco.wasm',
 'vendor/mujoco/LICENSE':HERE/'licenses/MuJoCo-LICENSE',
}
assets={}
for target,source in items.items():
    dest=OUT/target;dest.parent.mkdir(parents=True,exist_ok=True)
    if target=='world-runtime.mjs':
        dest.write_text('// GENERATED from native/browser-world/runtime.mjs; edit the canonical source.\n'+source.read_text().replace('import("@mujoco/mujoco")','import("./vendor/mujoco/mujoco.js")'))
    else: shutil.copyfile(source,dest)
    raw=dest.read_bytes();assets[target]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
manifest={'format':'chreatures-browser-physics-assets-v1','engine':'mujoco-3.12.0-wasm-browser-epoch-1','upstream':{'name':'@mujoco/mujoco','version':'3.12.0','source':'https://github.com/google-deepmind/mujoco/tree/3.12.0/wasm','license':'Apache-2.0','license_path':'vendor/mujoco/LICENSE'},'original_code_license':'AGPL-3.0-or-later','fixture_notice':'Engineered twelve-hinge body, anatomical optic membership; source and data notices in repository NOTICE.md and fixture metadata.','assets':assets}
(OUT/'physics-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'staged':str(OUT),'files':len(items),'bytes':sum(v['bytes'] for v in assets.values())}))
