/** Thin host boundary over MuJoCo Wasm and Rust/Wasm. No browser/DOM dependency.
 * Policy clients receive only sample(): optic B*5313 and body B*807.
 * observe()/snapshot() belong exclusively to the observer/checkpoint owner.
 */
export const ACTION_NAMES = Object.freeze([
  ...Array.from({ length: 84 }, (_, i) => `signed_servo_${String(i).padStart(2, "0")}`),
  ...["lf", "lm", "lh", "rf", "rm", "rh"].map((leg) => `adhesion_${leg}`),
  "pharyngeal_pump",
  "salivary_drive",
]);
export const ENGINE = "mujoco-3.12.0-neuromechfly-cns-v4";
const SITES = 1771,
  RAY_STRIDE = 3 + SITES * 3,
  CNS_MOTOR = 92,
  PHYSICAL_CONTROL = 90,
  BODY_CHANNELS = 807,
  SEGMENTS = 69,
  JOINTS = 126,
  PHYSICS_DT = 0.0001,
  CONTROL_DT = 0.01,
  SUBSTEPS = 100,
  CONTACT_SAMPLES = 10,
  CONTACT_STRIDE = 20;
const arr = (v) => Array.from(v);
const GROWTH_RAYS = Object.freeze([
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
].map((v) => {
  const n = Math.hypot(...v);
  return Object.freeze(v.map((x) => x / n));
}));
function normalize3(v, fallback = [0, 0, 1]) {
  const n = Math.hypot(...v);
  return n > 1e-12 ? v.map((x) => x / n) : Array.from(fallback);
}
function matVec3(m, v) {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}
function matTransposeVec3(m, v) {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}
function quatFromZ(direction) {
  const [x, y, z] = normalize3(direction);
  if (z < -0.999999) return [1, 0, 0, 0];
  const s = Math.sqrt(2 * (1 + z));
  return [-y / s, x / s, 0, s / 2];
}
function checkNumbers(v, length, name) {
  if (!v || v.length !== length || !Array.from(v).every(Number.isFinite))
    throw new Error(`${name} requires ${length} finite scalars`);
}
async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}
function assetBytes(assets, path) {
  const value = assets instanceof Map ? assets.get(path) : assets?.[path];
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new Error(`Missing binary MuJoCo VFS asset: ${path}`);
}
function compileModel(mj, xml, fixture, assets) {
  const vfs = new mj.MjVFS();
  try {
    for (const asset of fixture.mesh_assets) vfs.addBuffer(asset.path, assetBytes(assets, asset.path));
    return mj.MjModel.from_xml_string(xml, vfs);
  } finally {
    vfs.delete();
  }
}
function xmlAttribute(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;");
}
function capsuleXml({ physics_binding, position_m, orientation_xyzw, radius_m, length_m }) {
  const id = xmlAttribute(physics_binding), p = position_m.map((v) => v * 1000);
  const [x, y, z, w] = orientation_xyzw;
  const size = [radius_m * 1000, length_m * 500];
  return `<body name="ecology:${id}" pos="${p.join(" ")}" quat="${[w, x, y, z].join(" ")}"><geom name="ecology:${id}:geom" type="capsule" size="${size.join(" ")}" rgba="0.42 0.31 0.16 1" friction="0.9 0.02 0.004"/></body>`;
}
export async function createBrowserWorld({
  fixture,
  xml,
  assets,
  seed = 7,
  mujocoFactory,
  coreModule,
  coreWasm,
  mujocoOptions = {},
} = {}) {
  if (!fixture || typeof xml !== "string" || fixture.engine !== ENGINE || !Array.isArray(fixture.mesh_assets))
    throw new Error("Current compiled fly fixture/XML/VFS assets required");
  if (await sha256(new TextEncoder().encode(xml)) !== fixture.source_mjcf_sha256)
    throw new Error("MJCF fixture digest differs");
  for (const asset of fixture.mesh_assets)
    if (!asset?.path || await sha256(assetBytes(assets, asset.path)) !== asset.sha256)
      throw new Error(`MuJoCo VFS asset receipt differs: ${asset?.path}`);
  if (!mujocoFactory) mujocoFactory = (await import("@mujoco/mujoco")).default;
  if (!coreModule)
    coreModule = await import("./pkg/chreatures_browser_world.js");
  await coreModule.default(coreWasm ? { module_or_path: coreWasm } : undefined);
  const mj = await mujocoFactory(mujocoOptions);
  const model = compileModel(mj, xml, fixture, assets);
  return new BrowserWorld(mj, coreModule.WorldCore, fixture, xml, assets, model, seed);
}
class BrowserWorld {
  #mj;
  #Core;
  #model;
  #data;
  #core;
  #fixture;
  #xml;
  #assets;
  #buffers;
  #frame;
  #width = 2;
  #height = 2;
  #paused = false;
  #closed = false;
  #busy = false;
  #visitorCounter = 0;
  #visitorForces = [];
  #physicsSensed = false;
  #baseActuatorForceRange;
  #baseActuatorGainPrm;
  #meshes;
  #researchBodyMap;
  #retinalSiteIndices;
  #lastRetinalTrace = [];
  #clearanceScratch = null;
  #clearanceScratchBuilds = 0;
  #clearanceChecks = 0;
  #lastIllumination = [];
  #growthClearanceMemory = new Map();
  #growthStats = {
    proposed: 0,
    clearanceAccepted: 0,
    blocked: 0,
    offspringRejected: 0,
  };
  #routeMeasurements;
  constructor(mj, Core, fixture, xml, assets, model, seed) {
    if (mj.mj_versionString() !== "3.12.0")
      throw new Error("MuJoCo engine pin differs");
    this.#mj = mj;
    this.#Core = Core;
    this.#fixture = structuredClone(fixture);
    this.#xml = xml;
    this.#assets = assets;
    this.#model = model;
    this.#data = new mj.MjData(this.#model);
    const neutralKey = fixture.neutral_keyframe?.key_id;
    if (neutralKey !== 0 || this.#model.nkey < 1)
      throw new Error("Compiled fly neutral keyframe differs");
    mj.mj_resetDataKeyframe(this.#model, this.#data, neutralKey);
    this.#core = new Core(JSON.stringify(fixture), seed);
    mj.mj_forward(this.#model, this.#data);
    this.#validateCompiledFixture();
    this.#baseActuatorForceRange = Float64Array.from(this.#model.actuator_forcerange);
    this.#baseActuatorGainPrm = Float64Array.from(this.#model.actuator_gainprm);
    this.#meshes = this.#compileMeshObserver();
    this.#researchBodyMap = Object.freeze({
      bodies: structuredClone(this.#fixture.bodies),
      residents: structuredClone(this.#fixture.residents),
    });
    this.#retinalSiteIndices = [[], []];
    for (let site = 0; site < SITES; site++) {
      if (!this.#fixture.supported_sites[site]) continue;
      const eye = this.#fixture.anatomical_sites[site]?.[0] - 1;
      if (eye !== 0 && eye !== 1) throw new Error("Supported retinal site has no anatomical eye");
      this.#retinalSiteIndices[eye].push(site);
    }
    this.#retinalSiteIndices = Object.freeze(this.#retinalSiteIndices.map((indices) => Object.freeze(indices)));
    if (this.#retinalSiteIndices[0].length + this.#retinalSiteIndices[1].length !== 1486)
      throw new Error("Retinal atlas supported-site count differs");
    const retinalRays = Math.max(...this.#retinalSiteIndices.map((indices) => indices.length));
    this.#routeMeasurements = {
      open: Float64Array.from(this.#fixture.ecology.routes, (route) => route.base_open_fraction),
      flows: new Float64Array(this.#fixture.ecology.routes.length),
    };
    this.#frame = new Float32Array(12);
    this.#buffers = {
      velocity: new mj.DoubleBuffer(6),
      hits: new mj.IntBuffer(retinalRays),
      distances: new mj.DoubleBuffer(retinalRays),
      normals: new mj.DoubleBuffer(retinalRays * 3),
      retinalHits0: new mj.IntBuffer(this.#retinalSiteIndices[0].length),
      retinalHits1: new mj.IntBuffer(this.#retinalSiteIndices[1].length),
      retinalDistances0: new mj.DoubleBuffer(this.#retinalSiteIndices[0].length),
      retinalDistances1: new mj.DoubleBuffer(this.#retinalSiteIndices[1].length),
      retinalNormals0: new mj.DoubleBuffer(this.#retinalSiteIndices[0].length * 3),
      retinalNormals1: new mj.DoubleBuffer(this.#retinalSiteIndices[1].length * 3),
      contactForce: new mj.DoubleBuffer(6),
      geomDistance: new mj.DoubleBuffer(6),
      state: new mj.DoubleBuffer(
        mj.mj_stateSize(this.#model, mj.mjtState.mjSTATE_INTEGRATION.value),
      ),
    };
  }
  #validateCompiledFixture() {
    const m = this.#model, f = this.#fixture;
    const counts = f.compiled_counts;
    if (f.format !== "chreatures-fly-ecology-compiled-v1" || m.opt.timestep !== PHYSICS_DT || f.physics_dt !== PHYSICS_DT || f.control_dt !== CONTROL_DT || !Array.isArray(f.bodies) || f.bodies.length < 1 || f.mesh_assets.length !== 39 || !counts ||
      [["nq", m.nq], ["nv", m.nv], ["nu", m.nu], ["nbody", m.nbody], ["njnt", m.njnt], ["ngeom", m.ngeom], ["nmesh", m.nmesh], ["nsite", m.nsite], ["nsensor", m.nsensor], ["nsensordata", m.nsensordata]].some(([name, value]) => counts[name] !== value) ||
      !Array.isArray(f.geoms) || f.geoms.length !== m.ngeom || f.geoms.some((g, i) => g.id !== i || g.body !== m.geom_bodyid[i]))
      throw new Error("Compiled fly timing or resident fixture differs");
    const illumination = f.illumination;
    checkNumbers(illumination?.sky_direction_world, 3, "Physical sky direction");
    if (![f.ray_distance_mm, illumination.sky_intensity, illumination.screen_intensity, illumination.photon_energy_per_second]
      .every((value) => Number.isFinite(value) && value > 0) ||
      illumination.sky_intensity > 1 || illumination.screen_intensity > 1 ||
      f.supported_sites?.length !== SITES || f.anatomical_sites?.length !== SITES)
      throw new Error("Physical illumination or retinal support fixture differs");
    const assignedActuators = [];
    for (const b of f.bodies) {
      for (const [name, length, bound] of [
        ["segments", SEGMENTS, m.nbody], ["qpos", JOINTS, m.nq],
        ["dofs", JOINTS, m.nv], ["actuators", PHYSICAL_CONTROL, m.nu],
      ]) if (!Array.isArray(b[name]) || b[name].length !== length || b[name].some((x) => !Number.isInteger(x) || x < 0 || x >= bound))
        throw new Error(`Compiled resident ${name} differs`);
      if (!Number.isInteger(b.root) || !Number.isInteger(b.head))
        throw new Error("Compiled fly root/head IDs differ");
      assignedActuators.push(...b.actuators);
    }
    if (new Set(assignedActuators).size !== assignedActuators.length)
      throw new Error("Compiled fly actuator IDs overlap residents");
  }
  #compileMeshObserver(m = this.#model) {
    return Object.freeze(Array.from({ length: m.nmesh }, (_, id) => {
      const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id];
      const na = m.mesh_normaladr[id], nn = m.mesh_normalnum[id];
      const fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
      const mesh = {
        id,
        positions: Float32Array.from(m.mesh_vert.subarray(va * 3, (va + vn) * 3)),
        faces: Uint32Array.from(m.mesh_face.subarray(fa * 3, (fa + fn) * 3)),
      };
      // Compiled normals can have their own index stream. Expose them only
      // when they line up with vertices; otherwise the view derives normals.
      if (nn === vn)
        mesh.normals = Float32Array.from(m.mesh_normal.subarray(na * 3, (na + nn) * 3));
      return Object.freeze(mesh);
    }));
  }
  get residents() {
    return this.#fixture.bodies.length;
  }
  get engine() {
    return ENGINE;
  }
  get time() {
    return this.#core.time();
  }
  get paused() {
    return this.#paused;
  }
  #assertOpen() {
    if (this.#closed) throw new Error("World disposed");
    if (this.#busy) throw new Error("World mutation already pending");
  }
  #velocities() {
    const out = new Float64Array(this.#model.nbody * 6);
    for (let body = 0; body < this.#model.nbody; body++) {
      this.#mj.mj_objectVelocity(
        this.#model,
        this.#data,
        this.#mj.mjtObj.mjOBJ_XBODY.value,
        body,
        this.#buffers.velocity,
        0,
      );
      out.set(this.#buffers.velocity.GetView(), body * 6);
    }
    return out;
  }
  #jointLoads() {
    const d = this.#data, out = new Float64Array(this.#model.nv);
    for (let i = 0; i < out.length; i++)
      out[i] = d.qfrc_actuator[i] + d.qfrc_constraint[i] + d.qfrc_applied[i];
    return out;
  }
  #applyActuatorCapacity() {
    const m = this.#model, capacity = this.#core.actuator_capacity();
    checkNumbers(capacity, this.residents * PHYSICAL_CONTROL, "Physical actuator capacity");
    for (let row = 0; row < this.residents; row++) {
      const ids = this.#fixture.bodies[row].actuators;
      for (let k = 0; k < PHYSICAL_CONTROL; k++) {
        const value = capacity[row * PHYSICAL_CONTROL + k];
        if (value < 0 || value > 1) throw new Error("Physical actuator capacity outside [0,1]");
        const id = ids[k];
        if (k < 84) {
          m.actuator_forcerange[id * 2] = this.#baseActuatorForceRange[id * 2] * value;
          m.actuator_forcerange[id * 2 + 1] = this.#baseActuatorForceRange[id * 2 + 1] * value;
        } else {
          const width = m.actuator_gainprm.length / m.nu;
          for (let j = 0; j < width; j++)
            m.actuator_gainprm[id * width + j] = this.#baseActuatorGainPrm[id * width + j] * value;
        }
      }
    }
    return Float64Array.from(capacity);
  }
  #contactRecords() {
    const values = [], contacts = this.#data.contact;
    try {
      for (let i = 0; i < contacts.size(); i++) {
        const c = contacts.get(i);
        try {
          this.#mj.mj_contactForce(this.#model, this.#data, i, this.#buffers.contactForce);
          values.push(c.geom1, c.geom2, ...c.pos, ...c.frame, ...this.#buffers.contactForce.GetView());
        } finally { c?.delete(); }
      }
    } finally { contacts.delete(); }
    if (values.length % CONTACT_STRIDE) throw new Error("MuJoCo contact packet stride differs");
    return values;
  }
  #sampledPhysics(samples) {
    const nbody = this.#model.nbody;
    const positions = new Float64Array(samples.length * nbody * 3);
    const rotations = new Float64Array(samples.length * nbody * 9);
    const velocities = new Float64Array(samples.length * nbody * 6);
    const contacts = [], offsets = new Uint32Array(samples.length + 1);
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i];
      positions.set(sample.positions, i * nbody * 3);
      rotations.set(sample.rotations, i * nbody * 9);
      velocities.set(sample.velocities, i * nbody * 6);
      contacts.push(...sample.contacts);
      offsets[i + 1] = contacts.length / CONTACT_STRIDE;
    }
    return { positions, rotations, velocities, contacts: Float64Array.from(contacts), offsets };
  }
  #capturePhysics() {
    return {
      positions: Float64Array.from(this.#data.xpos),
      rotations: Float64Array.from(this.#data.xmat),
      velocities: this.#velocities(),
      contacts: this.#contactRecords(),
    };
  }
  #irradiance() {
    const out = new Float64Array(this.residents), d = this.#data;
    for (let row = 0; row < this.residents; row++) {
      const body = this.#fixture.bodies[row], head = body.head;
      const rotation = d.xmat.subarray(head * 9, head * 9 + 9);
      const normal = normalize3([rotation[2], rotation[5], rotation[8]]);
      const originM = arr(d.xpos.subarray(head * 3, head * 3 + 3)).map((x) => x * 0.001);
      out[row] = this.#withResidentRayMask(row, (geomGroup) =>
        this.#measureIllumination(originM, normal, -1, geomGroup).intensity,
      );
    }
    return out;
  }
  #nearestRegion(pointM) {
    let best = this.#fixture.ecology.regions[0], distance = Infinity;
    for (const region of this.#fixture.ecology.regions) {
      const d = Math.hypot(...region.center_m.map((x, i) => x - pointM[i]));
      if (d < distance) { best = region; distance = d; }
    }
    return best.id;
  }
  #geomNormalAt(geom, pointMm) {
    const m = this.#model, d = this.#data;
    const center = arr(d.geom_xpos.subarray(geom * 3, geom * 3 + 3));
    const rotation = d.geom_xmat.subarray(geom * 9, geom * 9 + 9);
    const local = matTransposeVec3(rotation, pointMm.map((x, i) => x - center[i]));
    const size = m.geom_size.subarray(geom * 3, geom * 3 + 3), type = m.geom_type[geom];
    let normal;
    if (type === this.#mj.mjtGeom.mjGEOM_PLANE.value) normal = [0, 0, 1];
    else if (type === this.#mj.mjtGeom.mjGEOM_SPHERE.value)
      normal = normalize3(local);
    else if (type === this.#mj.mjtGeom.mjGEOM_ELLIPSOID.value)
      normal = normalize3(local.map((x, i) => x / (size[i] * size[i])));
    else if (type === this.#mj.mjtGeom.mjGEOM_BOX.value) {
      const axis = [0, 1, 2].reduce((a, i) => Math.abs(local[i] / size[i]) > Math.abs(local[a] / size[a]) ? i : a, 0);
      normal = [0, 0, 0]; normal[axis] = Math.sign(local[axis]) || 1;
    } else if (type === this.#mj.mjtGeom.mjGEOM_CAPSULE.value) {
      const cap = Math.max(-size[1], Math.min(size[1], local[2]));
      normal = normalize3([local[0], local[1], local[2] - cap]);
    } else if (type === this.#mj.mjtGeom.mjGEOM_CYLINDER.value) {
      const radialGap = Math.abs(Math.hypot(local[0], local[1]) - size[0]);
      const capGap = Math.abs(Math.abs(local[2]) - size[1]);
      normal = capGap < radialGap ? [0, 0, Math.sign(local[2]) || 1] : normalize3([local[0], local[1], 0]);
    } else return null;
    return normalize3(matVec3(rotation, normal));
  }
  #supportOnGeom(geom, direction) {
    const m = this.#model, d = this.#data;
    const center = arr(d.geom_xpos.subarray(geom * 3, geom * 3 + 3));
    const rotation = d.geom_xmat.subarray(geom * 9, geom * 9 + 9);
    const localDirection = matTransposeVec3(rotation, normalize3(direction));
    const size = m.geom_size.subarray(geom * 3, geom * 3 + 3), type = m.geom_type[geom];
    let local;
    if (type === this.#mj.mjtGeom.mjGEOM_ELLIPSOID.value) {
      const denominator = Math.sqrt(localDirection.reduce((s, x, i) => s + size[i] * size[i] * x * x, 0));
      local = localDirection.map((x, i) => size[i] * size[i] * x / denominator);
    } else if (type === this.#mj.mjtGeom.mjGEOM_SPHERE.value)
      local = localDirection.map((x) => x * size[0]);
    else if (type === this.#mj.mjtGeom.mjGEOM_BOX.value)
      local = localDirection.map((x, i) => (x < 0 ? -size[i] : size[i]));
    else if (type === this.#mj.mjtGeom.mjGEOM_CAPSULE.value) {
      local = localDirection.map((x) => x * size[0]);
      local[2] += (localDirection[2] < 0 ? -1 : 1) * size[1];
    } else return null;
    const point = matVec3(rotation, local).map((x, i) => x + center[i]);
    return { point, normal: this.#geomNormalAt(geom, point) };
  }
  #growthRay(originM, direction, excludedBody, maxDistanceM = 0.006, geomGroup = [1, 1, 1, 1, 1, 1]) {
    const originMm = originM.map((x) => x * 1000), vector = normalize3(direction);
    this.#buffers.hits.GetView()[0] = -1;
    const measuredMm = this.#mj.mj_ray(
      this.#model, this.#data, originMm, vector, geomGroup, true,
      excludedBody, this.#buffers.hits, this.#buffers.normals,
    );
    const geom = this.#buffers.hits.GetView()[0];
    const distanceM = measuredMm >= 0 ? Math.min(maxDistanceM, measuredMm * 0.001) : maxDistanceM;
    return { geom: measuredMm >= 0 && measuredMm * 0.001 <= maxDistanceM ? geom : -1, distanceM };
  }
  #measureIllumination(positionM, surfaceNormal, excludedBody, geomGroup) {
    const light = this.#fixture.illumination;
    const normal = normalize3(surfaceNormal);
    const originM = positionM.map((x, i) => x + normal[i] * 2e-6);
    const skyDirection = normalize3(light.sky_direction_world);
    const skyHit = this.#growthRay(originM, skyDirection, excludedBody, this.#fixture.ray_distance_mm * 0.001, geomGroup);
    const skyIncidence = Math.max(0, normal.reduce((sum, x, i) => sum + x * skyDirection[i], 0));
    const sky = skyHit.geom < 0 ? light.sky_intensity * skyIncidence : 0;

    const screen = this.#fixture.screen_geom;
    const screenPointM = arr(this.#data.geom_xpos.subarray(screen * 3, screen * 3 + 3)).map((x) => x * 0.001);
    const screenVector = screenPointM.map((x, i) => x - originM[i]);
    const screenDistance = Math.hypot(...screenVector);
    const screenDirection = normalize3(screenVector);
    const screenHit = this.#growthRay(originM, screenDirection, excludedBody, screenDistance + 1e-6, geomGroup);
    const screenIncidence = Math.max(0, normal.reduce((sum, x, i) => sum + x * screenDirection[i], 0));
    const luminance = this.#frame.reduce((sum, x) => sum + x, 0) / this.#frame.length;
    const screenLight = screenHit.geom === screen
      ? light.screen_intensity * luminance * screenIncidence
      : 0;
    const intensity = Math.min(1, sky + screenLight);
    const direction = normalize3(
      skyDirection.map((x, i) => x * sky + screenDirection[i] * screenLight),
      skyDirection,
    );
    return {
      intensity,
      direction,
      sky_intensity: sky,
      screen_intensity: screenLight,
      available_energy_per_s: intensity * light.photon_energy_per_second,
    };
  }
  #growthInput(dt) {
    const ecology = JSON.parse(this.#core.ecology_observe()), colonies = [], photon_exposures = [], illumination = [];
    for (const organism of ecology.organisms) {
      if (!organism.anchored_region || !organism.genotype?.development) continue;
      const entity = this.#fixture.entities.find((item) => item.id === organism.physics_binding || item.physics_binding === organism.physics_binding);
      if (!entity?.geoms?.length) throw new Error(`Anchored colony physical binding missing: ${organism.physics_binding}`);
      let support;
      for (const geom of entity.geoms) {
        support = this.#supportOnGeom(geom, [0, 0, 1]);
        if (support) break;
      }
      if (!support?.normal) throw new Error(`Anchored colony support geometry unsupported: ${organism.physics_binding}`);
      const positionM = support.point.map((x) => x * 0.001);
      const originM = support.point.map((x, i) => (x + support.normal[i] * 0.002) * 0.001);
      const clearance_samples = [], nearby = new Map();
      for (const direction of GROWTH_RAYS) {
        const hit = this.#growthRay(originM, direction, entity.body);
        clearance_samples.push({ origin_m: originM, direction, free_distance_m: hit.distanceM });
        if (hit.geom < 0 || nearby.has(hit.geom)) continue;
        const pointMm = originM.map((x, i) => (x + direction[i] * hit.distanceM) * 1000);
        let normal = this.#geomNormalAt(hit.geom, pointMm);
        if (!normal) continue;
        if (normal.reduce((s, x, i) => s - x * direction[i], 0) < 0) normal = normal.map((x) => -x);
        const point_m = pointMm.map((x) => x * 0.001);
        nearby.set(hit.geom, {
          surface_id: `surface-${hit.geom}`,
          region_id: this.#nearestRegion(point_m),
          point_m,
          normal,
          attachable: false,
        });
      }
      clearance_samples.push(...(this.#growthClearanceMemory.get(organism.id) ?? []));
      const measuredLight = this.#measureIllumination(positionM, support.normal, entity.body);
      colonies.push({
        organism_id: organism.id,
        position_m: positionM,
        orientation_xyzw: quatFromZ(support.normal),
        surface_normal: support.normal,
        light_direction: measuredLight.direction,
        light_intensity: measuredLight.intensity,
        nearby_surfaces: [...nearby.values()],
        clearance_samples: clearance_samples.slice(0, 256),
        host_template_id: "fiber-capsule",
        child_template_id: null,
      });
      photon_exposures.push({
        organism_id: organism.id,
        available_energy_per_s: measuredLight.available_energy_per_s,
      });
      illumination.push({ organism_id: organism.id, ...measuredLight });
    }
    return { input: { dt_s: dt, colonies }, photon_exposures, illumination };
  }
  #disposeClearanceScratch() {
    if (!this.#clearanceScratch) return;
    this.#clearanceScratch.data.delete();
    this.#clearanceScratch.model.delete();
    this.#clearanceScratch = null;
  }
  #ensureClearanceScratch() {
    if (this.#clearanceScratch?.revision === this.#fixture.source_mjcf_sha256)
      return this.#clearanceScratch;
    this.#disposeClearanceScratch();
    const proxies = [0, 1].map((i) =>
      `<body name="clearance-proxy-${i}" pos="0 0 0"><geom name="clearance-proxy-${i}:geom" type="capsule" size="0.001 0.001" contype="0" conaffinity="0"/></body>`,
    ).join("");
    const xml = this.#xml.replace("</worldbody>", proxies + "</worldbody>");
    const model = compileModel(this.#mj, xml, this.#fixture, this.#assets);
    const data = new this.#mj.MjData(model);
    if (model.nq !== this.#model.nq || model.nv !== this.#model.nv ||
      model.nbody !== this.#model.nbody + 2 || model.ngeom !== this.#model.ngeom + 2) {
      data.delete(); model.delete();
      throw new Error("Clearance scratch proxies changed prior compiled addresses");
    }
    const proxyGeoms = [0, 1].map((i) => {
      const id = this.#mj.mj_name2id(model, this.#mj.mjtObj.mjOBJ_GEOM.value, `clearance-proxy-${i}:geom`);
      if (id !== this.#model.ngeom + i) throw new Error("Clearance scratch proxy address differs");
      return id;
    });
    this.#clearanceScratch = { revision: this.#fixture.source_mjcf_sha256, model, data, proxyGeoms };
    this.#clearanceScratchBuilds++;
    return this.#clearanceScratch;
  }
  #syncClearanceScratch(scratch) {
    const { model, data } = scratch, sourceModel = this.#model, sourceData = this.#data;
    for (const field of ["qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata"])
      if (sourceData[field]?.length) data[field].set(sourceData[field]);
    data.time = sourceData.time;
    for (const [name, width] of [["geom_size", 3], ["geom_pos", 3], ["geom_quat", 4], ["geom_rgba", 4], ["geom_friction", 3]])
      model[name].set(sourceModel[name].subarray(0, sourceModel.ngeom * width), 0);
    model.geom_contype.set(sourceModel.geom_contype.subarray(0, sourceModel.ngeom), 0);
    model.geom_conaffinity.set(sourceModel.geom_conaffinity.subarray(0, sourceModel.ngeom), 0);
    model.actuator_forcerange.set(sourceModel.actuator_forcerange);
    model.actuator_gainprm.set(sourceModel.actuator_gainprm);
  }
  #placeClearanceProxies(scratch, entries) {
    const { model, data, proxyGeoms } = scratch;
    for (let i = 0; i < proxyGeoms.length; i++) {
      const geom = proxyGeoms[i], entry = entries[i];
      if (!entry) {
        model.geom_size.set([0.001, 0.001, 0], geom * 3);
        model.geom_pos.set([1e6, 1e6, 1e6], geom * 3);
        model.geom_quat.set([1, 0, 0, 0], geom * 4);
        continue;
      }
      const vector = entry.query.to_m.map((x, j) => x - entry.query.from_m[j]);
      const center = entry.query.from_m.map((x, j) => (x + entry.query.to_m[j]) * 500);
      const [x, y, z, w] = quatFromZ(vector);
      model.geom_size.set([entry.query.radius_m * 1000, entry.length_m * 500, 0], geom * 3);
      model.geom_pos.set(center, geom * 3);
      model.geom_quat.set([w, x, y, z], geom * 4);
    }
    // MuJoCo derives geom bounding radii from size. Refresh those constants
    // through the supported API before each proxy placement.
    this.#mj.mj_setConst(model, data);
    this.#syncClearanceScratch(scratch);
    this.#mj.mj_forward(model, data);
  }
  #verifyGrowthProposal(proposal) {
    if (proposal?.format !== "chreatures-ecology-growth-v2" || typeof proposal.token !== "string")
      throw new Error("Native growth proposal contract differs");
    this.#growthStats.offspringRejected += proposal.birth_sites?.length ?? 0;
    const sites = proposal.construction_sites ?? [];
    if (!sites.length) return [];
    const queries = new Map((proposal.clearance_queries ?? []).map((query) => [query.site_id, query]));
    const entries = sites.map((site) => {
      const query = queries.get(site.site_id);
      if (!query || query.kind === "colony_birth") throw new Error("Growth construction clearance query differs");
      const length_m = Math.hypot(...query.to_m.map((x, i) => x - query.from_m[i]));
      return { site, query, length_m };
    });
    const scratch = this.#ensureClearanceScratch();
    this.#syncClearanceScratch(scratch);
    const rejected = new Map();
    const remember = (entry, witnessMm) => {
      if (rejected.has(entry.site.site_id)) return;
      const toward = witnessMm.map((x, i) => x * 0.001 - entry.query.from_m[i]);
      rejected.set(entry.site.site_id, {
        origin_m: entry.query.from_m,
        direction: normalize3(toward, entry.query.to_m.map((x, i) => x - entry.query.from_m[i])),
        free_distance_m: Math.max(0, Math.hypot(...toward)),
      });
    };
    for (let offset = 0; offset < entries.length; offset += 2) {
      const batch = entries.slice(offset, offset + 2);
      this.#placeClearanceProxies(scratch, batch);
      for (let i = 0; i < batch.length; i++) {
        const entry = batch[i], candidate = scratch.proxyGeoms[i];
        const attachment = (this.#fixture.entities ?? []).find((entity) => entity.id === entry.query.attachment_binding || entity.physics_binding === entry.query.attachment_binding);
        const excluded = new Set(attachment?.geoms ?? []);
        for (let geom = 0; geom < this.#model.ngeom; geom++) {
          if (excluded.has(geom) || (this.#model.geom_contype[geom] === 0 && this.#model.geom_conaffinity[geom] === 0)) continue;
          this.#clearanceChecks++;
          const distance = this.#mj.mj_geomDistance(scratch.model, scratch.data, candidate, geom, 100, this.#buffers.geomDistance);
          if (distance < -1e-6) remember(entry, arr(this.#buffers.geomDistance.GetView().subarray(3, 6)));
        }
      }
      if (batch.length === 2) {
        this.#clearanceChecks++;
        const distance = this.#mj.mj_geomDistance(scratch.model, scratch.data, scratch.proxyGeoms[0], scratch.proxyGeoms[1], 100, this.#buffers.geomDistance);
        if (distance < -1e-6) {
          const points = this.#buffers.geomDistance.GetView();
          remember(batch[0], arr(points.subarray(3, 6)));
          remember(batch[1], arr(points.subarray(0, 3)));
        }
      }
    }
    // Native currently emits at most one candidate per colony (two in the
    // canonical world). Preserve exact pairwise checks if that grows later.
    if (entries.length > 2) for (let i = 0; i < entries.length; i++) for (let j = i + 1; j < entries.length; j++) {
      if (Math.floor(i / 2) === Math.floor(j / 2)) continue;
      const pair = [entries[i], entries[j]];
      this.#placeClearanceProxies(scratch, pair);
      this.#clearanceChecks++;
      const distance = this.#mj.mj_geomDistance(scratch.model, scratch.data, scratch.proxyGeoms[0], scratch.proxyGeoms[1], 100, this.#buffers.geomDistance);
      if (distance < -1e-6) {
        const points = this.#buffers.geomDistance.GetView();
        remember(pair[0], arr(points.subarray(3, 6)));
        remember(pair[1], arr(points.subarray(0, 3)));
      }
    }
    const accepted = [];
    for (const entry of entries) {
      const failure = rejected.get(entry.site.site_id);
      if (failure) {
        this.#growthStats.blocked++;
        const prior = this.#growthClearanceMemory.get(entry.query.organism_id) ?? [];
        this.#growthClearanceMemory.set(entry.query.organism_id, [...prior, failure].slice(-32));
      } else {
        this.#growthStats.clearanceAccepted++;
        this.#growthClearanceMemory.delete(entry.query.organism_id);
        accepted.push(entry.site);
      }
    }
    return accepted;
  }
  #prepareAutonomousGrowth(dt) {
    if (typeof this.#core.propose_growth !== "function" || typeof this.#core.discard_growth !== "function")
      throw new Error("Native autonomous growth API unavailable");
    const measured = this.#growthInput(dt);
    const proposal = JSON.parse(this.#core.propose_growth(JSON.stringify(measured.input)));
    this.#growthStats.proposed += proposal.construction_sites?.length ?? 0;
    try {
      const construction_sites = this.#verifyGrowthProposal(proposal);
      this.#core.set_ecology_sites(JSON.stringify({
        growth_token: proposal.token,
        construction_sites,
        birth_sites: [],
        photon_exposures: measured.photon_exposures,
      }));
      this.#lastIllumination = measured.illumination;
      return proposal.token;
    } catch (error) {
      this.#core.discard_growth(proposal.token);
      throw error;
    }
  }
  async #appendConstructions(items) {
    if (!items.length) return null;
    for (const item of items) {
      if (item.kind === "offspring_body")
        throw new Error("Offspring creation requires an authored complete fly template");
      if (item.kind !== "constructed_geometry" || item.host_template_id !== "fiber-capsule")
        throw new Error(`Unsupported physical construction template: ${item.host_template_id}`);
      checkNumbers(item.position_m, 3, "Construction position");
      checkNumbers(item.orientation_xyzw, 4, "Construction orientation");
      if (![item.nominal_radius_m, item.nominal_length_m].every((v) => Number.isFinite(v) && v > 0 && v <= 1))
        throw new Error("Invalid bounded construction dimensions");
    }
    const revision = this.#fixture.source_mjcf_sha256, time = this.time;
    const fragments = items.map((item) => capsuleXml({
      ...item, radius_m: item.nominal_radius_m, length_m: item.nominal_length_m,
    }));
    const xml = this.#xml.replace("</worldbody>", fragments.join("") + "</worldbody>");
    const fixture = structuredClone(this.#fixture);
    fixture.source_mjcf_sha256 = await sha256(new TextEncoder().encode(xml));
    if (revision !== this.#fixture.source_mjcf_sha256 || time !== this.time)
      throw new Error("Construction transaction stale after world advanced");
    const old = { model: this.#model, data: this.#data, fixture: this.#fixture, xml: this.#xml, state: this.#buffers.state, meshes: this.#meshes };
    let model, data, stateBuffer, meshes;
    try {
      model = compileModel(this.#mj, xml, fixture, this.#assets);
      data = new this.#mj.MjData(model);
      if (model.nq !== old.model.nq || model.nv !== old.model.nv || model.nbody !== old.model.nbody + items.length || model.ngeom !== old.model.ngeom + items.length)
        throw new Error("Construction append changed prior compiled addresses");
      for (const field of ["qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata"])
        if (old.data[field]?.length) data[field].set(old.data[field]);
      data.time = old.data.time;
      for (const [name, width] of [["geom_size", 3], ["geom_pos", 3], ["geom_quat", 4], ["geom_rgba", 4], ["geom_friction", 3]])
        model[name].set(old.model[name].subarray(0, old.model.ngeom * width));
      model.geom_contype.set(old.model.geom_contype.subarray(0, old.model.ngeom));
      model.geom_conaffinity.set(old.model.geom_conaffinity.subarray(0, old.model.ngeom));
      model.actuator_forcerange.set(old.model.actuator_forcerange);
      model.actuator_gainprm.set(old.model.actuator_gainprm);
      this.#mj.mj_forward(model, data);
      for (const item of items) {
        const name = `ecology:${item.physics_binding}`;
        const body = this.#mj.mj_name2id(model, this.#mj.mjtObj.mjOBJ_BODY.value, name);
        const geom = this.#mj.mj_name2id(model, this.#mj.mjtObj.mjOBJ_GEOM.value, `${name}:geom`);
        if (body < old.model.nbody || geom < old.model.ngeom) throw new Error("Construction compiled address is not append-only");
        fixture.geoms.push({ id: geom, body, size: arr(model.geom_size.subarray(geom * 3, geom * 3 + 3)) });
        fixture.entities.push({ id: item.physics_binding, physics_binding: item.physics_binding, body, free: false, geoms: [geom] });
        fixture.material_bindings.push({ body, geoms: [geom], store: item.material_store, exposed: true });
      }
      const contacts = data.contact;
      try {
        for (let i = 0; i < contacts.size(); i++) {
          const c = contacts.get(i);
          try {
            if ((c.geom1 >= old.model.ngeom || c.geom2 >= old.model.ngeom) && c.dist < -0.003)
              throw new Error("Constructed geometry penetrates existing physical geometry");
          } finally { c?.delete(); }
        }
      } finally { contacts.delete(); }
      fixture.compiled_counts = {
        nq: model.nq, nv: model.nv, nu: model.nu, nbody: model.nbody, njnt: model.njnt,
        ngeom: model.ngeom, nmesh: model.nmesh, nsite: model.nsite, nsensor: model.nsensor, nsensordata: model.nsensordata,
      };
      meshes = this.#compileMeshObserver(model);
      if (typeof this.#core.rebind_physics !== "function") throw new Error("Native core lacks transactional physical rebind");
      this.#core.rebind_physics(JSON.stringify(fixture));
      stateBuffer = new this.#mj.DoubleBuffer(this.#mj.mj_stateSize(model, this.#mj.mjtState.mjSTATE_INTEGRATION.value));
      this.#model = model; this.#data = data; this.#fixture = fixture; this.#xml = xml;
      this.#buffers.state = stateBuffer; this.#meshes = meshes;
      return {
        commit: () => {
          this.#disposeClearanceScratch();
          old.state.delete(); old.data.delete(); old.model.delete();
        },
        rollback: () => {
          this.#buffers.state.delete(); this.#data.delete(); this.#model.delete();
          this.#model = old.model; this.#data = old.data; this.#fixture = old.fixture;
          this.#xml = old.xml; this.#buffers.state = old.state; this.#meshes = old.meshes;
        },
      };
    } catch (error) {
      stateBuffer?.delete(); data?.delete(); model?.delete(); throw error;
    }
  }
  async #applyEcologyProposal(proposal) {
    const topology = await this.#appendConstructions(proposal.physical_creations ?? []);
    const m = this.#model;
    const removals = (proposal.physical_removals ?? []).flatMap((item) => {
      const binding = (this.#fixture.entities ?? []).find((v) => v.id === item.physics_binding || v.physics_binding === item.physics_binding);
      if (!binding?.geoms?.length) throw new Error(`Unknown physical removal binding: ${item.physics_binding}`);
      return binding.geoms.map((geom) => ({ ...item, geom, remove: true }));
    });
    const changes = [...removals, ...(proposal.geom_updates ?? [])];
    const saved = new Map();
    try {
      for (const change of changes) {
        const geom = change.geom ?? change.reserved_geom;
        if (!Number.isInteger(geom) || geom < 0 || geom >= m.ngeom)
          throw new Error("Ecology proposal references an invalid compiled geom");
        if (!saved.has(geom)) saved.set(geom, {
          size: arr(m.geom_size.subarray(geom * 3, geom * 3 + 3)),
          pos: arr(m.geom_pos.subarray(geom * 3, geom * 3 + 3)),
          rgba: arr(m.geom_rgba.subarray(geom * 4, geom * 4 + 4)),
          contype: m.geom_contype[geom], conaffinity: m.geom_conaffinity[geom],
        });
        if (change.remove) {
          m.geom_contype[geom] = 0; m.geom_conaffinity[geom] = 0; m.geom_rgba[geom * 4 + 3] = 0;
          continue;
        }
        if (change.size) { checkNumbers(change.size, 3, "Proposed geom size"); m.geom_size.set(change.size, geom * 3); }
        if (change.position) { checkNumbers(change.position, 3, "Proposed geom position"); m.geom_pos.set(change.position, geom * 3); }
        if (change.rgba) { checkNumbers(change.rgba, 4, "Proposed geom color"); m.geom_rgba.set(change.rgba, geom * 4); }
        if (change.contype !== undefined) m.geom_contype[geom] = change.contype;
        if (change.conaffinity !== undefined) m.geom_conaffinity[geom] = change.conaffinity;
      }
      this.#mj.mj_forward(m, this.#data);
      return {
        applied_geoms: [...saved.keys()],
        commit: () => topology?.commit(),
        rollback: () => {
          for (const [geom, old] of saved) {
            m.geom_size.set(old.size, geom * 3); m.geom_pos.set(old.pos, geom * 3);
            m.geom_rgba.set(old.rgba, geom * 4); m.geom_contype[geom] = old.contype;
            m.geom_conaffinity[geom] = old.conaffinity;
          }
          this.#mj.mj_forward(m, this.#data); topology?.rollback();
        },
      };
    } catch (error) {
      for (const [geom, old] of saved) {
        m.geom_size.set(old.size, geom * 3); m.geom_pos.set(old.pos, geom * 3);
        m.geom_rgba.set(old.rgba, geom * 4); m.geom_contype[geom] = old.contype;
        m.geom_conaffinity[geom] = old.conaffinity;
      }
      this.#mj.mj_forward(m, this.#data);
      topology?.rollback();
      throw error;
    }
  }
  async advance(actions, dt = CONTROL_DT) {
    this.#assertOpen();
    if (this.#paused)
      throw new Error(
        "World paused after incomplete physical mutation; restore coherent checkpoint",
      );
    checkNumbers(actions, this.residents * CNS_MOTOR, "CNS motor command");
    for (let row = 0; row < this.residents; row++) {
      const offset = row * CNS_MOTOR;
      for (let k = 0; k < CNS_MOTOR; k++) {
        const value = actions[offset + k];
        if (k < 84) {
          if (value < -1 || value > 1)
            throw new Error("Signed CNS motor command outside [-1,1]");
        } else if (value < 0 || value > 1) {
          throw new Error("Nonnegative CNS motor command outside [0,1]");
        }
      }
    }
    if (dt !== CONTROL_DT)
      throw new Error("Fly control tick is fixed at 0.01 seconds");
    this.#busy = true;
    const commands = Float64Array.from(actions);
    const m = this.#model,
      d = this.#data;
    try {
      this.#prepareAutonomousGrowth(dt);
      const controls = this.#core.actuation(commands);
      checkNumbers(controls, this.residents * PHYSICAL_CONTROL, "Physical fly control");
      this.#applyActuatorCapacity();
      d.ctrl.fill(0);
      for (let row = 0; row < this.residents; row++) {
        const body = this.#fixture.bodies[row];
        for (let k = 0; k < PHYSICAL_CONTROL; k++) d.ctrl[body.actuators[k]] = controls[row * PHYSICAL_CONTROL + k];
      }
      const samples = [];
      for (let step = 0; step < SUBSTEPS; step++) {
        d.qfrc_applied.fill(0);
        d.xfrc_applied.fill(0);
        for (const event of this.#visitorForces) {
          for (let k = 0; k < 3; k++)
            d.xfrc_applied[event.body * 6 + k] += event.force[k];
        }
        this.#mj.mj_step(m, d);
        if ((step + 1) % (SUBSTEPS / CONTACT_SAMPLES) === 0) samples.push(this.#capturePhysics());
      }
      this.#mj.mj_forward(m, d);
      if (
        !Array.from(d.qpos).every(Number.isFinite) ||
        Math.abs(d.time - this.#core.time() - dt) > 1e-8
      )
        throw new Error("Physical clock/finite-state violation");
      const packet = this.#sampledPhysics(samples);
      const proposalText = this.#core.prepare_advance(
        commands, d.qpos, d.qvel, this.#jointLoads(), packet.positions, packet.rotations,
        packet.velocities, packet.contacts, packet.offsets, this.#irradiance(), dt,
      );
      const proposal = JSON.parse(proposalText);
      let applied;
      try {
        applied = await this.#applyEcologyProposal(proposal);
        this.#core.commit_advance(JSON.stringify({
          token: proposal.token,
          created: (proposal.physical_creations ?? []).map((item) => item.proposal_id),
          removed: (proposal.physical_removals ?? []).map((item) => item.proposal_id),
        }));
        applied.commit();
      } catch (error) {
        applied?.rollback();
        this.#core.abort_advance(String(proposal.token));
        throw error;
      }
      this.#visitorForces = [];
      this.#physicsSensed = true;
      return { time: this.time };
    } catch (error) {
      this.#paused = true;
      throw error;
    } finally {
      this.#busy = false;
    }
  }
  /** Host video bytes are a physical emitting surface, never a CNS input array. */
  setScreenFrame(rgb, width, height) {
    this.#assertOpen();
    if (
      !Number.isInteger(width) ||
      !Number.isInteger(height) ||
      width < 1 ||
      height < 1 ||
      width > 2048 ||
      height > 2048
    )
      throw new Error("Invalid screen dimensions");
    checkNumbers(rgb, width * height * 3, "Screen RGB");
    if (Array.from(rgb).some((v) => v < 0 || v > 1))
      throw new Error("Screen RGB outside [0,1]");
    this.#frame = Float32Array.from(rgb);
    this.#width = width;
    this.#height = height;
  }
  #colors() {
    const m = this.#model;
    const out = new Float64Array(m.ngeom * 4);
    for (let g = 0; g < m.ngeom; g++) {
      const mat = m.geom_matid[g];
      out.set(
        mat >= 0
          ? m.mat_rgba.subarray(mat * 4, mat * 4 + 4)
          : m.geom_rgba.subarray(g * 4, g * 4 + 4),
        g * 4,
      );
    }
    return out;
  }
  #withResidentRayMask(row, callback) {
    const m = this.#model, resident = this.#fixture.bodies[row];
    const bodies = new Set(resident.segments), saved = [];
    // mj_multiRay accepts one body exclusion, while a fly spans 69 articulated
    // bodies. Reserve group 5 for this synchronous query and remap any external
    // group-5 geometry to visible group 0 for the duration of the call.
    for (let geom = 0; geom < m.ngeom; geom++) {
      const own = bodies.has(m.geom_bodyid[geom]) ||
        this.#fixture.geom_map?.[geom]?.name?.startsWith(`${resident.id}/`);
      if (!own && m.geom_group[geom] !== 5) continue;
      saved.push([geom, m.geom_group[geom]]);
      m.geom_group[geom] = own ? 5 : 0;
    }
    try {
      return callback([1, 1, 1, 1, 1, 0]);
    } finally {
      for (const [geom, group] of saved) m.geom_group[geom] = group;
    }
  }
  sample() {
    this.#assertOpen();
    if (this.#paused) throw new Error("Cannot sample incoherent world");
    const m = this.#model,
      d = this.#data;
    const optic = new Float32Array(this.residents * SITES * 3);
    const colors = this.#colors();
    for (let row = 0; row < this.residents; row++) {
      const b = this.#fixture.bodies[row];
      const rays = this.#core.rays(
        row,
        d.xpos,
        d.xmat,
      );
      const hits = new Int32Array(SITES * 2);
      const distances = new Float64Array(SITES * 2);
      hits.fill(-1);
      distances.fill(this.#fixture.ray_distance_mm);
      this.#withResidentRayMask(row, (geomGroup) => {
        for (let eye = 0; eye < 2; eye++) {
          const start = eye * RAY_STRIDE, sites = this.#retinalSiteIndices[eye], directions = [];
          const rayHits = this.#buffers[`retinalHits${eye}`];
          const rayDistances = this.#buffers[`retinalDistances${eye}`];
          const rayNormals = this.#buffers[`retinalNormals${eye}`];
          for (const site of sites)
            directions.push(...rays.subarray(start + 3 + site * 3, start + 6 + site * 3));
          this.#mj.mj_multiRay(
            m,
            d,
            arr(rays.subarray(start, start + 3)),
            directions,
            geomGroup,
            true,
            -1,
            rayHits,
            rayDistances,
            rayNormals,
            sites.length,
            this.#fixture.ray_distance_mm,
          );
          const eyeHits = rayHits.GetView(), eyeDistances = rayDistances.GetView();
          for (let i = 0; i < sites.length; i++) {
            hits[eye * SITES + sites[i]] = eyeHits[i];
            distances[eye * SITES + sites[i]] = eyeDistances[i];
          }
        }
      });
      const residentBodies = new Set(b.segments);
      let tracedHits = 0, screenHits = 0, selfHits = 0, nearestGeom = -1, nearestMm = this.#fixture.ray_distance_mm;
      for (let eye = 0; eye < 2; eye++) for (const site of this.#retinalSiteIndices[eye]) {
        const index = eye * SITES + site;
        if (hits[index] >= 0) {
          tracedHits++;
          if (distances[index] < nearestMm) { nearestMm = distances[index]; nearestGeom = hits[index]; }
          if (hits[index] === this.#fixture.screen_geom) screenHits++;
          if (residentBodies.has(m.geom_bodyid[hits[index]]) ||
            this.#fixture.geom_map?.[hits[index]]?.name?.startsWith(`${b.id}/`)) selfHits++;
        }
      }
      this.#lastRetinalTrace[row] = {
        supported_rays: this.#retinalSiteIndices[0].length + this.#retinalSiteIndices[1].length,
        hits: tracedHits,
        screen_hits: screenHits,
        self_hits: selfHits,
        nearest_distance_mm: nearestMm,
        nearest_geom: nearestGeom,
        nearest_body: nearestGeom >= 0 ? m.geom_bodyid[nearestGeom] : -1,
        cutoff_mm: this.#fixture.ray_distance_mm,
      };
      optic.set(
        this.#core.retina(
          row,
          rays,
          hits,
          distances,
          d.geom_xpos,
          d.geom_xmat,
          m.geom_size,
          colors,
          this.#frame,
          this.#width,
          this.#height,
        ),
        row * SITES * 3,
      );
    }
    if (!this.#physicsSensed) {
      const packet = this.#sampledPhysics([this.#capturePhysics()]);
      this.#core.sense_physics(
        d.qpos, d.qvel, this.#jointLoads(), packet.positions, packet.rotations,
        packet.velocities, packet.contacts, packet.offsets, this.#irradiance(),
      );
      this.#physicsSensed = true;
    }
    const body = this.#core.afferents();
    checkNumbers(body, this.residents * BODY_CHANNELS, "CNS BODY807 sample");
    return { optic, body };
  }
  /** Observer-only geometry: never pass this object to neural adapters or policy. */
  observe() {
    this.#assertOpen();
    const m = this.#model,
      d = this.#data;
    const residentForBody = new Map();
    for (const b of this.#fixture.bodies)
      for (const body of b.segments) residentForBody.set(body, b.id);
    const nameForGeom = (id) => {
      if (typeof this.#mj.mj_id2name === "function") {
        const name = this.#mj.mj_id2name(m, this.#mj.mjtObj.mjOBJ_GEOM.value, id);
        if (name) return name;
      }
      return this.#fixture.geom_map?.[id]?.name ?? `geom-${id}`;
    };
    return {
      engine: ENGINE,
      time: this.time,
      paused: this.#paused,
      residents: this.#fixture.bodies.map((b) => ({
        id: b.id,
        root: b.root,
        head: b.head,
        segmentBodyIds: Uint32Array.from(b.segments),
      })),
      screenGeom: this.#fixture.screen_geom,
      geometry: this.#fixture.geoms.map((g) => ({
        ...g,
        name: nameForGeom(g.id),
        type: m.geom_type[g.id],
        resident_id: residentForBody.get(g.body) ?? null,
        size: arr(m.geom_size.subarray(g.id * 3, g.id * 3 + 3)),
        mesh_id: m.geom_type[g.id] === this.#mj.mjtGeom.mjGEOM_MESH.value ? m.geom_dataid[g.id] : -1,
        position: Float32Array.from(d.geom_xpos.subarray(g.id * 3, g.id * 3 + 3)),
        rotation: Float32Array.from(d.geom_xmat.subarray(g.id * 9, g.id * 9 + 9)),
      })),
      meshes: this.#meshes,
      worldSizeMm: structuredClone(this.#fixture.world_size),
      positions: Float32Array.from(d.geom_xpos),
      rotations: Float32Array.from(d.geom_xmat),
      bodyPositions: Float32Array.from(d.xpos),
      bodyRotations: Float32Array.from(d.xmat),
      bodyVelocities: Float32Array.from(this.#velocities()),
      colors: Float32Array.from(this.#colors()),
      physicalControl: Float32Array.from(d.ctrl),
      actuatorForceRange: Float32Array.from(m.actuator_forcerange),
      actuatorGainParameters: Float32Array.from(m.actuator_gainprm),
      food: Float32Array.from(this.#core.food()),
      ecology: JSON.parse(this.#core.ecology_observe()),
      actuators: JSON.parse(this.#core.actuator_state()),
      growth: {
        ...structuredClone(this.#growthStats),
        clearanceScratchBuilds: this.#clearanceScratchBuilds,
        clearanceChecks: this.#clearanceChecks,
      },
      illumination: structuredClone(this.#lastIllumination),
      retinalTrace: structuredClone(this.#lastRetinalTrace),
      routeMeasurements: {
        open: Float32Array.from(this.#routeMeasurements.open),
        flowsM3S: Float32Array.from(this.#routeMeasurements.flows),
      },
    };
  }
  /** Raw observer/teacher targets. This object must never enter resident cognition. */
  researchObserve() {
    this.#assertOpen();
    const d = this.#data;
    const bodyAfferents = this.#core.afferents();
    checkNumbers(bodyAfferents, this.residents * BODY_CHANNELS, "CNS BODY807 research trace");
    return {
      qpos: Float64Array.from(d.qpos),
      qvel: Float64Array.from(d.qvel),
      bodyPositions: Float64Array.from(d.xpos),
      bodyQuaternions: Float64Array.from(d.xquat),
      bodyRotations: Float64Array.from(d.xmat),
      sensordata: Float64Array.from(d.sensordata),
      ctrl: Float64Array.from(d.ctrl),
      bodyAfferents: Float32Array.from(bodyAfferents),
      bodyMap: this.#researchBodyMap,
      ecology: JSON.parse(this.#core.ecology_observe()),
      actuatorState: JSON.parse(this.#core.actuator_state()),
    };
  }
  setMemoryCheckpoint(opaqueCnsMemory) {
    this.#core.set_memory(String(opaqueCnsMemory));
  }
  setRouteMeasurements(openFraction, advectionM3S) {
    this.#assertOpen();
    checkNumbers(openFraction, this.#fixture.ecology.routes.length, "Route openness");
    checkNumbers(advectionM3S, openFraction.length, "Route advection");
    this.#core.set_route_measurements(Float64Array.from(openFraction), Float64Array.from(advectionM3S));
    this.#routeMeasurements = {
      open: Float64Array.from(openFraction),
      flows: Float64Array.from(advectionM3S),
    };
  }
  setEcologySites({ construction_sites = [], birth_sites = [], photon_exposures = [] } = {}) {
    this.#assertOpen();
    this.#core.set_ecology_sites(JSON.stringify({ growth_token: null, construction_sites, birth_sites, photon_exposures }));
  }
  snapshot() {
    this.#assertOpen();
    if (this.#paused) throw new Error("Cannot checkpoint incoherent world");
    this.#mj.mj_getState(
      this.#model,
      this.#data,
      this.#buffers.state,
      this.#mj.mjtState.mjSTATE_INTEGRATION.value,
    );
    return {
      format: "chreatures-browser-fly-physical-snapshot-v3",
      engine: ENGINE,
      model: this.#fixture.source_mjcf_sha256,
      atlas: this.#fixture.atlas_sha256,
      physical: arr(this.#buffers.state.GetView()),
      core: this.#core.snapshot(),
      geomSize: arr(this.#model.geom_size),
      geomPos: arr(this.#model.geom_pos),
      geomRGBA: arr(this.#model.geom_rgba),
      geomContype: arr(this.#model.geom_contype),
      geomConaffinity: arr(this.#model.geom_conaffinity),
      ctrl: arr(this.#data.ctrl),
      actuatorForceRange: arr(this.#model.actuator_forcerange),
      actuatorGainParameters: arr(this.#model.actuator_gainprm),
      frame: arr(this.#frame),
      width: this.#width,
      height: this.#height,
      visitorCounter: this.#visitorCounter,
      visitorForces: structuredClone(this.#visitorForces),
      physicsSensed: this.#physicsSensed,
      growthClearanceMemory: structuredClone([...this.#growthClearanceMemory]),
      growthStats: structuredClone(this.#growthStats),
      lastIllumination: structuredClone(this.#lastIllumination),
      routeMeasurements: {
        open: arr(this.#routeMeasurements.open),
        flows: arr(this.#routeMeasurements.flows),
      },
      fixture: structuredClone(this.#fixture),
      xml: this.#xml,
    };
  }
  restore(snapshot) {
    this.#assertOpen();
    if (
      snapshot?.format !== "chreatures-browser-fly-physical-snapshot-v3" ||
      snapshot.engine !== ENGINE ||
      snapshot.model !== this.#fixture.source_mjcf_sha256 ||
      snapshot.atlas !== this.#fixture.atlas_sha256
    )
      throw new Error("Physical checkpoint identity differs");
    checkNumbers(
      snapshot.physical,
      this.#buffers.state.GetView().length,
      "Physical state",
    );
    checkNumbers(
      snapshot.geomSize,
      this.#model.geom_size.length,
      "Geometry sizes",
    );
    checkNumbers(
      snapshot.geomPos,
      this.#model.geom_pos.length,
      "Geometry positions",
    );
    checkNumbers(
      snapshot.geomRGBA,
      this.#model.geom_rgba.length,
      "Geometry colors",
    );
    checkNumbers(snapshot.geomContype, this.#model.geom_contype.length, "Geometry contact types");
    checkNumbers(snapshot.geomConaffinity, this.#model.geom_conaffinity.length, "Geometry contact affinities");
    checkNumbers(snapshot.ctrl, this.#data.ctrl.length, "Physical controls");
    checkNumbers(snapshot.actuatorForceRange, this.#model.actuator_forcerange.length, "Actuator force ranges");
    checkNumbers(snapshot.actuatorGainParameters, this.#model.actuator_gainprm.length, "Actuator gain parameters");
    checkNumbers(
      snapshot.frame,
      snapshot.width * snapshot.height * 3,
      "Screen state",
    );
    if (
      !Number.isInteger(snapshot.width) ||
      !Number.isInteger(snapshot.height) ||
      snapshot.width < 1 ||
      snapshot.height < 1 ||
      snapshot.width > 2048 ||
      snapshot.height > 2048 ||
      snapshot.frame.some((v) => v < 0 || v > 1)
    )
      throw new Error("Invalid saved physical screen");
    for (const event of snapshot.visitorForces ?? []) {
      checkNumbers(event.force, 3, "Saved visitor force");
      if (
        !this.#fixture.entities.some((e) => e.free && e.body === event.body) ||
        Math.hypot(...event.force) > 60
      )
        throw new Error("Invalid saved visitor force");
    }
    const routeCount = this.#fixture.ecology.routes.length;
    checkNumbers(snapshot.routeMeasurements?.open, routeCount, "Saved route openness");
    checkNumbers(snapshot.routeMeasurements?.flows, routeCount, "Saved route flows");
    if (!Array.isArray(snapshot.growthClearanceMemory) || snapshot.growthClearanceMemory.some(([id, samples]) =>
      typeof id !== "string" || !Array.isArray(samples) || samples.some((sample) => {
        try {
          checkNumbers(sample.origin_m, 3, "Saved growth clearance origin");
          checkNumbers(sample.direction, 3, "Saved growth clearance direction");
          return !Number.isFinite(sample.free_distance_m) || sample.free_distance_m < 0;
        } catch { return true; }
      }))) throw new Error("Saved growth clearance memory differs");
    const growthKeys = ["proposed", "clearanceAccepted", "blocked", "offspringRejected"];
    if (!snapshot.growthStats || growthKeys.some((key) =>
      !Number.isSafeInteger(snapshot.growthStats[key]) || snapshot.growthStats[key] < 0))
      throw new Error("Saved growth observer counters differ");
    const core = JSON.parse(snapshot.core);
    if (Math.abs(core.state?.time - snapshot.physical[0]) > 1e-8)
      throw new Error("Saved physical and ecological clocks differ");
    this.#disposeClearanceScratch();
    this.#core.restore(snapshot.core);
    try {
      this.#mj.mj_setState(
        this.#model,
        this.#data,
        snapshot.physical,
        this.#mj.mjtState.mjSTATE_INTEGRATION.value,
      );
      this.#model.geom_size.set(snapshot.geomSize);
      this.#model.geom_pos.set(snapshot.geomPos);
      this.#model.geom_rgba.set(snapshot.geomRGBA);
      this.#model.geom_contype.set(snapshot.geomContype);
      this.#model.geom_conaffinity.set(snapshot.geomConaffinity);
      this.#data.ctrl.set(snapshot.ctrl);
      this.#model.actuator_forcerange.set(snapshot.actuatorForceRange);
      this.#model.actuator_gainprm.set(snapshot.actuatorGainParameters);
      this.setScreenFrame(snapshot.frame, snapshot.width, snapshot.height);
      this.#visitorCounter = snapshot.visitorCounter ?? 0;
      this.#visitorForces = structuredClone(snapshot.visitorForces ?? []);
      this.#growthClearanceMemory = new Map(structuredClone(snapshot.growthClearanceMemory));
      this.#growthStats = structuredClone(snapshot.growthStats);
      this.#lastIllumination = structuredClone(snapshot.lastIllumination ?? []);
      this.#routeMeasurements = {
        open: Float64Array.from(snapshot.routeMeasurements.open),
        flows: Float64Array.from(snapshot.routeMeasurements.flows),
      };
      this.#mj.mj_forward(this.#model, this.#data);
      this.#physicsSensed = Boolean(snapshot.physicsSensed);
      this.#paused = false;
    } catch (e) {
      this.#paused = true;
      throw e;
    }
  }
  /** Append one physical object at an explicit observer/visitor position.
   * Builds and validates a private candidate, then swaps once. No running tick is retried.
   */
  async insertObject({
    position,
    size = [0.07, 0.07, 0.07],
    shape = "box",
    rgba = [0.65, 0.35, 0.16, 1],
    food = 0,
    odor = -1,
  } = {}) {
    this.#assertOpen();
    if (this.#paused) throw new Error("Cannot change incoherent topology");
    checkNumbers(position, 3, "Object position");
    checkNumbers(size, 3, "Object half-size");
    checkNumbers(rgba, 4, "Object color");
    if (
      !["box", "sphere", "capsule", "cylinder", "ellipsoid"].includes(shape) ||
      size.some((x) => x <= 0 || x > 0.5) ||
      position.some((x) => Math.abs(x) > 100) ||
      rgba.some((x) => x < 0 || x > 1) ||
      !Number.isFinite(food) ||
      food < 0 ||
      food > 1 ||
      ![-1, 0, 1, 2].includes(odor) ||
      this.#fixture.entities.length >= 96
    )
      throw new Error("Invalid bounded material insertion");
    this.#busy = true;
    try {
      const revision = this.#fixture.source_mjcf_sha256,
        time = this.time;
      const id = `visitor-${this.#visitorCounter + 1}`;
      const fragment = `<body name="entity:${id}" pos="${position.join(" ")}"><freejoint name="entity:${id}:free"/><geom name="entity:${id}:geom:0" type="${shape}" size="${size.join(" ")}" rgba="${rgba.join(" ")}" density="35" friction="0.9 0.02 0.004"/></body>`;
      const xml = this.#xml.replace("</worldbody>", fragment + "</worldbody>");
      const digest = await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(xml),
      );
      if (revision !== this.#fixture.source_mjcf_sha256 || time !== this.time)
        throw new Error("Topology transaction stale after world advanced");
      const fixture = structuredClone(this.#fixture);
      fixture.source_mjcf_sha256 = Array.from(new Uint8Array(digest), (b) =>
        b.toString(16).padStart(2, "0"),
      ).join("");
      let model, data, core, stateBuffer, meshes;
      try {
      model = compileModel(this.#mj, xml, fixture, this.#assets);
      data = new this.#mj.MjData(model);
      const body = this.#mj.mj_name2id(
        model,
        this.#mj.mjtObj.mjOBJ_BODY.value,
        `entity:${id}`,
      );
      const geom = this.#mj.mj_name2id(
        model,
        this.#mj.mjtObj.mjOBJ_GEOM.value,
        `entity:${id}:geom:0`,
      );
      if (
        model.nq !== this.#model.nq + 7 ||
        model.nv !== this.#model.nv + 6 ||
        geom !== this.#model.ngeom ||
        body !== this.#model.nbody
      )
        throw new Error("Append changed prior model addresses");
      // Append-only MJCF keeps prior addresses. Preserve all existing mutable integration arrays.
      for (const field of [
        "qpos",
        "qvel",
        "act",
        "qacc_warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
      ])
        if (this.#data[field]?.length) data[field].set(this.#data[field]);
      if (this.#model.neq) {
        const eq = new this.#mj.DoubleBuffer(this.#model.neq);
        try {
          const mask = this.#mj.mjtState.mjSTATE_EQ_ACTIVE.value;
          this.#mj.mj_getState(this.#model, this.#data, eq, mask);
          this.#mj.mj_setState(model, data, arr(eq.GetView()), mask);
        } finally {
          eq.delete();
        }
      }
      data.time = this.#data.time;
      for (const [name, width] of [["geom_size", 3], ["geom_pos", 3], ["geom_quat", 4], ["geom_rgba", 4], ["geom_friction", 3]])
        model[name].set(this.#model[name].subarray(0, this.#model.ngeom * width));
      model.geom_contype.set(this.#model.geom_contype.subarray(0, this.#model.ngeom));
      model.geom_conaffinity.set(this.#model.geom_conaffinity.subarray(0, this.#model.ngeom));
      model.actuator_forcerange.set(this.#model.actuator_forcerange);
      model.actuator_gainprm.set(this.#model.actuator_gainprm);
      this.#mj.mj_forward(model, data);
      const contacts = data.contact;
      try {
        for (let i = 0; i < contacts.size(); i++) {
          const c = contacts.get(i);
          try {
            if ((c.geom1 === geom || c.geom2 === geom) && c.dist < -0.003)
              throw new Error(
                `Insertion would penetrate existing physical geometry ${c.geom1 === geom ? c.geom2 : c.geom1}`,
              );
          } finally {
            c?.delete();
          }
        }
      } finally {
        contacts.delete();
      }
      fixture.geoms.push({
        id: geom,
        name: `entity:${id}:geom:0`,
        type: model.geom_type[geom],
        size: arr(model.geom_size.subarray(geom * 3, geom * 3 + 3)),
        rgba: arr(model.geom_rgba.subarray(geom * 4, geom * 4 + 4)),
        body,
      });
      fixture.entities.push({
        id,
        body,
        free: true,
        food,
        nutrition: 1,
        odor,
        strength: odor >= 0 ? 1 : 0,
        growth: food > 0 ? 0.002 : 0,
        geoms: [geom],
      });
      fixture.compiled_counts = {
        nq: model.nq, nv: model.nv, nu: model.nu, nbody: model.nbody,
        njnt: model.njnt, ngeom: model.ngeom, nmesh: model.nmesh,
        nsite: model.nsite, nsensor: model.nsensor, nsensordata: model.nsensordata,
      };
      meshes = this.#compileMeshObserver(model);
      core = new this.#Core(JSON.stringify(this.#fixture), 1);
      core.restore(this.#core.snapshot());
      if (typeof core.rebind_physics !== "function")
        throw new Error("Native core lacks transactional physical rebind");
      core.rebind_physics(JSON.stringify(fixture));
      stateBuffer = new this.#mj.DoubleBuffer(
        this.#mj.mj_stateSize(
          model,
          this.#mj.mjtState.mjSTATE_INTEGRATION.value,
        ),
      );
      this.#core.free();
      this.#data.delete();
      this.#model.delete();
      this.#buffers.state.delete();
      this.#core = core;
      this.#data = data;
      this.#model = model;
      this.#buffers.state = stateBuffer;
      this.#fixture = fixture;
      this.#xml = xml;
      this.#meshes = meshes;
      this.#disposeClearanceScratch();
      this.#physicsSensed = false;
      this.#visitorCounter++;
      return { id, body, geom, model: fixture.source_mjcf_sha256 };
      } catch (error) {
        stateBuffer?.delete();
        core?.free();
        data?.delete();
        model?.delete();
        throw error;
      }
    } finally {
      this.#busy = false;
    }
  }
  visitorSound(position, frequencyHz, envelope = 1, duration = 0.15) {
    this.#assertOpen();
    checkNumbers(position, 3, "Sound position");
    if (![frequencyHz, envelope, duration].every(Number.isFinite))
      throw new Error("Sound requires finite frequency, envelope and duration");
    this.#core.visitor_sound(Float64Array.from(position), frequencyHz, envelope, duration);
  }
  /** Human force is queued into the next ordinary physical tick. */
  queueVisitorForce(entityId, force) {
    this.#assertOpen();
    checkNumbers(force, 3, "Object force");
    if (Math.hypot(...force) > 60)
      throw new Error("Visitor force limit exceeded");
    const e = this.#fixture.entities.find((e) => e.id === entityId && e.free);
    if (!e) throw new Error("Expected movable physical entity");
    this.#visitorForces = this.#visitorForces.filter(
      (event) => event.body !== e.body,
    );
    this.#visitorForces.push({ body: e.body, force: Array.from(force) });
    this.#visitorCounter++;
  }
  dispose() {
    if (this.#closed) return;
    if (this.#busy) throw new Error("Cannot dispose while a world mutation is pending");
    this.#closed = true;
    this.#disposeClearanceScratch();
    for (const b of Object.values(this.#buffers)) b.delete();
    this.#core.free();
    this.#data.delete();
    this.#model.delete();
  }
}
