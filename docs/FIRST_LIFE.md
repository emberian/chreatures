# First life

The first species is a little bilateral, six-legged animal in a persistent top-down terrarium. Its antennae sample three diffusing scents; a coarse retinal fan sees occluded colored objects and boundaries; contact, sound, motion and internal physiology complete its sensory stream. It can walk, turn, eat on contact, push light objects and signal. Food, scented objects, shelter, landmarks, other residents and a caregiver's placed signals make an environment with consequences.

Actual FlyWire connectivity supplies a recurrent circuit. This is a synthetic species with measured anatomical ancestry: rate dynamics, sensory coding, output bridges, body and learning rules are our construction. Never call it a recovered fly or invent measured connections. The wiring must be able to influence real actions; silencing or rewiring is a research-instance control.

The runtime is Python with NumPy/SciPy, a local HTTP/WebSocket surface and a canvas habitat. The browser draws the authoritative Python world; it never substitutes its own moving sprites. GPU batches use the same equations in Torch on AMD. Bulk datasets and build caches belong on hbox /tank/chreatures. Laptop is the local interactive home; remote workers run developmental populations and native analysis.

## Shared implementation contracts (v0.1)

Coordinates are habitat units in [0,width] × [0,height], +x right and +y down. Heading radians, 0=right, positive clockwise. Model dt=0.05 seconds. There is one RNG owned by World for world events; each Brain has its own saved RNG. All public snapshot fields are JSON compatible.

`chreatures.world.World(seed=7)` owns .width (1200), .height (800), .time, .rng, .bodies (list of Body), .objects (list of Object). Dataclasses support JSON dictionaries. Body fields: id:str, name:str, x,y,heading:float, radius:float=9, energy:float[0,1], gut:float[0,1], fatigue:float[0,1], speed:float, angular_velocity:float, age:float, color:str; extra fields allowed. Initially three bodies. World objects have id,kind,x,y,radius,color and optional odor:int,amount:float,movable:bool. Kinds: food, stone, shelter, toy, flower, beacon. Source scents are 0/1/2. Food can be moved/depleted; objects persist.

`World.sense(body_id)` returns ONLY local observations: `odor` shape (2,3) left/right antennae concentration floats; `vision` shape (16,4) RGB[0,1], proximity[0,1] using ray fan with occlusion; `touch` shape(2,), `sound` shape(3,), `shade`:float, `speed`:float, `angular_velocity`:float. Outputs JSON lists; no target coordinates, account IDs or secret state. World may add own-body energy/gut/fatigue. `World.advance(actions:dict[str,dict], dt:float)` moves/collides/pushes/ingests/digests bodies and returns dict bodyid -> outcomes `{nutrition:float, contact:float, distance:float}`. Actions forward[0,1], turn[-1,1], eat[0,1], signal[0,1]. Metabolism and fatigue are continuous and bounded, never instant reset sleep. Movement requires no global goal logic.

`World.command(command:dict)` supports add(kind,x,y,odor?), move(id,x,y), remove(id), signal(x,y,tone?), with numeric/schema validation. Root server implements pause,speed,save and load; do not put them in World. Commands are serialized by server. `World.snapshot()` and `World.restore(snapshot)` (classmethod, returns World) roundtrip all world/body/object/RNG/delayed state for exact same-runtime continuation. `World.view()` returns public renderable state {width,height,time,bodies,objects,...}. No brain data inside world class.

`Brain(connectome_path, seed, genome?)` owns connectome state, local support/adaptation, recurrent context, learned readouts and RNG. `Brain.step(senses:dict, physiology:dict, dt, reward=0.0)` returns action dict. Each creature shares immutable graph but has private tensors. Brain snapshot/restore captures all evolving state. Neural anatomy schema data/connectome/circuit.npz: ids Unicode strings, pre/post int32, count float32, sign float32 per neuron; labels/side/type optional. manifest.json records source revision, transforms and schema. Actual filename may be communicated by data deputy.

Server runtime owns World + brain per body, atomic versioned checkpoint, queued input, activity history. Run local only by default, bind127.0.0.1. API GET /api/state, POST /api/command. Browser never holds authoritative organism state. Standardized probe experiments and native libraries are separate from live resident advancement.

## Owned work lanes

Root: initial architecture; brain, runtime, server, interactive UI, integration.
World deputy: chreatures/world.py, tests/test_world.py.
Data deputy: acquisition script, data/connectome, docs/CONNECTOME.md, data tests.
Compute deputy: scripts/compute_probe.py, scripts/remote_probe.sh, docs/COMPUTE.md.
Library deputy: integrations/, docs/LIBRARIES.md, tests/test_integrations.py.

Tests should challenge actual causal or persistence behavior. Compare lived histories and neural interventions; do not spend this cycle producing ceremonial schemas without a creature. The interface should invite watching and interacting first, with neural inspection available alongside it.
