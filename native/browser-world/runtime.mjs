/** Thin host boundary over MuJoCo Wasm and Rust/Wasm. No browser/DOM dependency.
 * Policy clients receive only sample(): optic B*5313 and body B*110.
 * observe()/snapshot() belong exclusively to the observer/checkpoint owner.
 */
export const ACTION_NAMES = Object.freeze([
  ...["lf", "lm", "lh", "rf", "rm", "rh"].flatMap((leg) =>
    ["hip", "knee"].flatMap((joint) => [
      `${leg}_${joint}_positive`,
      `${leg}_${joint}_negative`,
    ]),
  ),
  "gaze_pitch",
  "posture",
  "grip",
  "signal_low",
  "signal_mid",
  "signal_high",
  "eat",
  "release",
  "secrete",
  "allocate",
]);
export const ENGINE = "mujoco-3.12.0-wasm-browser-epoch-2";
const SITES = 1771,
  RAY_STRIDE = 3 + SITES * 3;
const arr = (v) => Array.from(v);
function checkNumbers(v, length, name) {
  if (!v || v.length !== length || !Array.from(v).every(Number.isFinite))
    throw new Error(`${name} requires ${length} finite scalars`);
}
export async function createBrowserWorld({
  fixture,
  xml,
  seed = 7,
  mujocoFactory,
  coreModule,
  coreWasm,
  mujocoOptions = {},
} = {}) {
  if (!fixture || typeof xml !== "string" || fixture.engine !== ENGINE)
    throw new Error("Current browser fixture/XML required");
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(xml),
  );
  if (
    Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("") !== fixture.source_mjcf_sha256
  )
    throw new Error("MJCF fixture digest differs");
  if (!mujocoFactory) mujocoFactory = (await import("@mujoco/mujoco")).default;
  if (!coreModule)
    coreModule = await import("./pkg/chreatures_browser_world.js");
  await coreModule.default(coreWasm ? { module_or_path: coreWasm } : undefined);
  const mj = await mujocoFactory(mujocoOptions);
  return new BrowserWorld(mj, coreModule.WorldCore, fixture, xml, seed);
}
class BrowserWorld {
  #mj;
  #Core;
  #model;
  #data;
  #core;
  #fixture;
  #xml;
  #buffers;
  #frame;
  #width = 2;
  #height = 2;
  #paused = false;
  #closed = false;
  #visitorCounter = 0;
  #visitorForces = [];
  constructor(mj, Core, fixture, xml, seed) {
    if (mj.mj_versionString() !== "3.12.0")
      throw new Error("MuJoCo engine pin differs");
    this.#mj = mj;
    this.#Core = Core;
    this.#fixture = structuredClone(fixture);
    this.#xml = xml;
    this.#model = mj.MjModel.from_xml_string(xml);
    this.#data = new mj.MjData(this.#model);
    this.#core = new Core(JSON.stringify(fixture), seed);
    mj.mj_forward(this.#model, this.#data);
    this.#frame = new Float32Array(12);
    this.#buffers = {
      velocity: new mj.DoubleBuffer(6),
      hits: new mj.IntBuffer(SITES),
      distances: new mj.DoubleBuffer(SITES),
      normals: new mj.DoubleBuffer(SITES * 3),
      state: new mj.DoubleBuffer(
        mj.mj_stateSize(this.#model, mj.mjtState.mjSTATE_INTEGRATION.value),
      ),
    };
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
  }
  #velocities(local = false) {
    const out = new Float64Array(this.residents * 6);
    for (let row = 0; row < this.residents; row++) {
      this.#mj.mj_objectVelocity(
        this.#model,
        this.#data,
        this.#mj.mjtObj.mjOBJ_BODY.value,
        this.#fixture.bodies[row].root,
        this.#buffers.velocity,
        Number(local),
      );
      out.set(this.#buffers.velocity.GetView(), row * 6);
    }
    return out;
  }
  advance(actions, dt = 0.05) {
    this.#assertOpen();
    if (this.#paused)
      throw new Error(
        "World paused after incomplete physical mutation; restore coherent checkpoint",
      );
    checkNumbers(actions, this.residents * 34, "CNS motor command");
    for (let row = 0; row < this.residents; row++) {
      const offset = row * 34;
      for (let k = 0; k < 34; k++) {
        const value = actions[offset + k];
        if (k === 24 || k === 25) {
          if (value < -1 || value > 1)
            throw new Error("Signed CNS motor command outside [-1,1]");
        } else if (value < 0 || value > 1) {
          throw new Error("Nonnegative CNS motor command outside [0,1]");
        }
      }
    }
    if (dt !== 0.05)
      throw new Error("Browser control tick is fixed at 0.05 seconds");
    const commands = Float64Array.from(actions);
    const m = this.#model,
      d = this.#data;
    try {
      const steps = Math.round(dt / this.#fixture.physics_dt);
      for (let step = 0; step < steps; step++) {
        d.qfrc_applied.fill(0);
        d.xfrc_applied.fill(0);
        const forces = this.#core.actuation(
          commands,
          d.qpos,
          d.qvel,
          d.xpos,
          d.xmat,
          this.#velocities(),
          d.time,
          this.#fixture.physics_dt,
        );
        for (let row = 0; row < this.residents; row++) {
          const b = this.#fixture.bodies[row],
            offset = row * 19;
          for (let j = 0; j < 12; j++)
            d.qfrc_applied[b.dofs[j]] = forces[offset + j];
          for (let k = 0; k < 3; k++)
            d.xfrc_applied[b.root * 6 + 3 + k] = forces[offset + 12 + k];
          const entity = forces[offset + 15];
          if (entity >= 0) {
            const body = this.#fixture.entities[entity].body;
            for (let k = 0; k < 3; k++) {
              d.xfrc_applied[body * 6 + k] += forces[offset + 16 + k];
              d.xfrc_applied[b.root * 6 + k] -= forces[offset + 16 + k];
            }
          }
        }
        for (const event of this.#visitorForces) {
          for (let k = 0; k < 3; k++)
            d.xfrc_applied[event.body * 6 + k] += event.force[k];
        }
        this.#mj.mj_step(m, d);
      }
      this.#mj.mj_forward(m, d);
      if (
        !Array.from(d.qpos).every(Number.isFinite) ||
        Math.abs(d.time - this.#core.time() - dt) > 1e-8
      )
        throw new Error("Physical clock/finite-state violation");
      this.#core.advance(commands, d.xpos, dt);
      this.#visitorForces = [];
      // The Rust ecology owns resource quantity; MuJoCo owns resulting collision geometry.
      const food = this.#core.food(),
        scales = this.#core.growth_scales();
      for (let i = 0; i < food.length; i++) {
        const e = this.#fixture.entities[i];
        if (e.food > 0) {
          const scale = scales[i];
          for (const g of e.geoms)
            for (let k = 0; k < 3; k++)
              m.geom_size[g * 3 + k] = this.#fixture.geoms[g].size[k] * scale;
        }
      }
      this.#mj.mj_forward(m, d);
      return { time: this.time };
    } catch (error) {
      this.#paused = true;
      throw error;
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
  #contacts() {
    const aggregate = new Float64Array(this.residents * 25);
    const feet = new Float64Array(this.residents * 6);
    const legIndex = new Map(
      ["lf", "lm", "lh", "rf", "rm", "rh"].map((leg, i) => [leg, i]),
    );
    const contacts = this.#data.contact;
    try {
      for (let i = 0; i < contacts.size(); i++) {
        const c = contacts.get(i);
        try {
          for (let row = 0; row < this.residents; row++) {
            const root = this.#fixture.bodies[row].root;
            const a = this.#model.body_rootid[this.#model.geom_bodyid[c.geom1]],
              b = this.#model.body_rootid[this.#model.geom_bodyid[c.geom2]];
            if (a !== root && b !== root) continue;
            const count = aggregate[row * 25];
            if (count >= 8) continue;
            aggregate[row * 25]++;
            for (let k = 0; k < 3; k++)
              aggregate[row * 25 + 1 + count * 3 + k] =
                c.frame[k] * (a === root ? -1 : 1);
            for (const geom of [c.geom1, c.geom2]) {
              const name = this.#fixture.geoms[geom]?.name ?? "";
              const match = name.match(/^resident:[^:]+:geom:(lf|lm|lh|rf|rm|rh):tarsus$/);
              if (match) feet[row * 6 + legIndex.get(match[1])] = 1;
            }
          }
        } finally {
          c?.delete();
        }
      }
    } finally {
      contacts.delete();
    }
    return { aggregate, feet };
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
        d.geom_xpos.subarray(b.head * 3, b.head * 3 + 3),
        d.geom_xmat.subarray(b.head * 9, b.head * 9 + 9),
      );
      const hits = new Int32Array(SITES * 2);
      const distances = new Float64Array(SITES * 2);
      for (let eye = 0; eye < 2; eye++) {
        const start = eye * RAY_STRIDE;
        this.#mj.mj_multiRay(
          m,
          d,
          arr(rays.subarray(start, start + 3)),
          arr(rays.subarray(start + 3, start + RAY_STRIDE)),
          [1, 1, 1, 1, 1, 1],
          true,
          b.root,
          this.#buffers.hits,
          this.#buffers.distances,
          this.#buffers.normals,
          SITES,
          3.2,
        );
        hits.set(this.#buffers.hits.GetView(), eye * SITES);
        distances.set(this.#buffers.distances.GetView(), eye * SITES);
      }
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
    const shade = new Float64Array(this.residents);
    for (let row = 0; row < this.residents; row++) {
      const b = this.#fixture.bodies[row];
      shade[row] = this.#mj.mj_ray(
        m,
        d,
        arr(d.geom_xpos.subarray(b.head * 3, b.head * 3 + 3)),
        [0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        true,
        b.root,
        this.#buffers.hits,
        this.#buffers.normals,
      );
    }
    const contactState = this.#contacts();
    const jointLoads = new Float64Array(this.residents * 12);
    for (let row = 0; row < this.residents; row++)
      for (let j = 0; j < 12; j++)
        jointLoads[row * 12 + j] =
          d.qfrc_applied[this.#fixture.bodies[row].dofs[j]] +
          d.qfrc_constraint[this.#fixture.bodies[row].dofs[j]];
    const body = this.#core.afferents(
      d.xpos,
      d.xmat,
      this.#velocities(true),
      contactState.aggregate,
      shade,
      d.qpos,
      d.qvel,
      jointLoads,
      contactState.feet,
    );
    return { optic, body };
  }
  /** Observer-only geometry: never pass this object to neural adapters or policy. */
  observe() {
    this.#assertOpen();
    const m = this.#model,
      d = this.#data;
    return {
      engine: ENGINE,
      time: this.time,
      paused: this.#paused,
      residents: this.#fixture.bodies.map((b) => ({
        id: b.id,
        root: b.root,
        head: b.head,
      })),
      screenGeom: this.#fixture.screen_geom,
      geometry: this.#fixture.geoms.map((g) => ({
        ...g,
        size: arr(m.geom_size.subarray(g.id * 3, g.id * 3 + 3)),
      })),
      positions: Float32Array.from(d.geom_xpos),
      rotations: Float32Array.from(d.geom_xmat),
      bodyPositions: Float32Array.from(d.xpos),
      colors: Float32Array.from(this.#colors()),
      food: Float32Array.from(this.#core.food()),
    };
  }
  setMemoryCheckpoint(opaqueCnsMemory) {
    this.#core.set_memory(String(opaqueCnsMemory));
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
      format: "chreatures-browser-physical-snapshot-v2",
      engine: ENGINE,
      model: this.#fixture.source_mjcf_sha256,
      atlas: this.#fixture.atlas_sha256,
      physical: arr(this.#buffers.state.GetView()),
      core: this.#core.snapshot(),
      geomSize: arr(this.#model.geom_size),
      geomRGBA: arr(this.#model.geom_rgba),
      frame: arr(this.#frame),
      width: this.#width,
      height: this.#height,
      visitorCounter: this.#visitorCounter,
      visitorForces: structuredClone(this.#visitorForces),
      fixture: structuredClone(this.#fixture),
      xml: this.#xml,
    };
  }
  restore(snapshot) {
    this.#assertOpen();
    if (
      snapshot?.format !== "chreatures-browser-physical-snapshot-v2" ||
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
      snapshot.geomRGBA,
      this.#model.geom_rgba.length,
      "Geometry colors",
    );
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
    const core = JSON.parse(snapshot.core);
    if (Math.abs(core.time - snapshot.physical[0]) > 1e-8)
      throw new Error("Saved physical and ecological clocks differ");
    this.#core.restore(snapshot.core);
    try {
      this.#mj.mj_setState(
        this.#model,
        this.#data,
        snapshot.physical,
        this.#mj.mjtState.mjSTATE_INTEGRATION.value,
      );
      this.#model.geom_size.set(snapshot.geomSize);
      this.#model.geom_rgba.set(snapshot.geomRGBA);
      this.setScreenFrame(snapshot.frame, snapshot.width, snapshot.height);
      this.#visitorCounter = snapshot.visitorCounter ?? 0;
      this.#visitorForces = structuredClone(snapshot.visitorForces ?? []);
      this.#mj.mj_forward(this.#model, this.#data);
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
    let model, data, core, stateBuffer;
    try {
      model = this.#mj.MjModel.from_xml_string(xml);
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
      model.geom_size.set(this.#model.geom_size);
      model.geom_rgba.set(this.#model.geom_rgba);
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
      core = new this.#Core(JSON.stringify(this.#fixture), 1);
      core.restore(this.#core.snapshot());
      core.append_entity_config(JSON.stringify(fixture));
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
      this.#visitorCounter++;
      return { id, body, geom, model: fixture.source_mjcf_sha256 };
    } catch (error) {
      stateBuffer?.delete();
      core?.free();
      data?.delete();
      model?.delete();
      throw error;
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
    this.#closed = true;
    for (const b of Object.values(this.#buffers)) b.delete();
    this.#core.free();
    this.#data.delete();
    this.#model.delete();
  }
}
