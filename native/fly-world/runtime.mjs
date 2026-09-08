// SPDX-License-Identifier: AGPL-3.0-or-later
/** Thin typed-array host for the unified Rust + MuJoCo Emscripten world. */

const decoder = new TextDecoder();
const encoder = new TextEncoder();

export class UnifiedWasmFlyWorld {
  static async open({moduleFactory, moduleURL, wasmBinary, scenePath, seed, prepareFilesystem, locateFile}) {
    const factory = moduleFactory ?? (await import(moduleURL)).default;
    const module = await factory({locateFile, wasmBinary});
    await prepareFilesystem?.(module);
    const bytes = encoder.encode(scenePath);
    const pointer = module._malloc(bytes.byteLength);
    if (!pointer) throw new Error(`cannot allocate ${bytes.byteLength} scene-path bytes`);
    let handle;
    try {
      module.HEAPU8.set(bytes, pointer);
      handle = module._chreatures_fly_world_open(pointer, bytes.byteLength, seed >>> 0);
    } finally {
      module._free(pointer);
    }
    if (!handle) throw new Error(UnifiedWasmFlyWorld.error(module));
    const world = new UnifiedWasmFlyWorld(module, handle);
    try {
      return world.#initialize();
    } catch (error) {
      world.close();
      throw error;
    }
  }

  static error(module) {
    const length = module._chreatures_fly_world_last_error_length();
    if (!length) return 'unified fly world failed without an error message';
    const pointer = module._malloc(length);
    if (!pointer) return `unified fly world failed; cannot allocate ${length} error bytes`;
    try {
      module._chreatures_fly_world_last_error(pointer, length);
      return decoder.decode(module.HEAPU8.slice(pointer, pointer + length));
    } finally {
      module._free(pointer);
    }
  }

  constructor(module, handle) {
    this.module = module;
    this.handle = handle;
    this.buffers = new Map();
  }

  #initialize() {
    this.metadata = this.#json('metadata');
    this.residents = this.metadata.residents;
    this.optic = new Float32Array(this.residents * 5313);
    this.body = new Float32Array(this.residents * 807);
    this.#buffer('optic', this.optic.byteLength);
    this.#buffer('body', this.body.byteLength);
    this.#buffer('motor', this.residents * 92 * Float32Array.BYTES_PER_ELEMENT);
    return this;
  }

  #check(status) {
    if (status !== 0) throw new Error(UnifiedWasmFlyWorld.error(this.module));
  }

  #buffer(name, bytes) {
    const current = this.buffers.get(name);
    if (current && current.capacity >= bytes) return current.pointer;
    const pointer = this.module._malloc(Math.max(bytes, 1));
    if (!pointer) throw new Error(`cannot allocate ${bytes} bytes for ${name}`);
    if (current) this.module._free(current.pointer);
    this.buffers.set(name, {pointer, capacity: bytes});
    return pointer;
  }

  #json(kind) {
    const length = this.module[`_chreatures_fly_world_${kind}_length`](this.handle);
    if (!length) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    return this.#readJson(kind, length);
  }

  #readJson(kind, length) {
    const pointer = this.#buffer(`${kind}-json`, length);
    this.#check(this.module[`_chreatures_fly_world_${kind}`](this.handle, pointer, length));
    return JSON.parse(decoder.decode(this.module.HEAPU8.slice(pointer, pointer + length)));
  }

  setScreen(frame, width, height) {
    if (!(frame instanceof Float32Array)) throw new Error('screen must be Float32Array');
    const pointer = this.#buffer('screen', frame.byteLength);
    this.module.HEAPF32.set(frame, pointer >>> 2);
    this.#check(this.module._chreatures_fly_world_set_screen(
      this.handle, pointer, frame.length, width, height));
  }

  visitorSound(position, frequency, envelope = 1, duration = 0.15) {
    const values = Float64Array.from(position);
    if (values.length !== 3) throw new Error('sound position must contain three values');
    const pointer = this.#buffer('sound-position', values.byteLength);
    this.module.HEAPF64.set(values, pointer >>> 3);
    this.#check(this.module._chreatures_fly_world_visitor_sound(
      this.handle, pointer, values.length, frequency, envelope, duration));
  }

  queueVisitorForce(entityId, force) {
    const entity = encoder.encode(entityId);
    const values = Float64Array.from(force);
    if (values.length !== 3) throw new Error('visitor force must contain three values');
    const entityPointer = this.#buffer('visitor-entity', entity.byteLength);
    const forcePointer = this.#buffer('visitor-force', values.byteLength);
    this.module.HEAPU8.set(entity, entityPointer);
    this.module.HEAPF64.set(values, forcePointer >>> 3);
    this.#check(this.module._chreatures_fly_world_visitor_force(
      this.handle, entityPointer, entity.byteLength, forcePointer, values.length));
  }

  setRoutes(open, flow) {
    const openness = Float64Array.from(open), advection = Float64Array.from(flow);
    const openPointer = this.#buffer('route-open', openness.byteLength);
    const flowPointer = this.#buffer('route-flow', advection.byteLength);
    this.module.HEAPF64.set(openness, openPointer >>> 3);
    this.module.HEAPF64.set(advection, flowPointer >>> 3);
    this.#check(this.module._chreatures_fly_world_set_routes(
      this.handle, openPointer, openness.length, flowPointer, advection.length));
  }

  insertObject({position, size = [0.07, 0.07, 0.07], shape = 'box', rgba = [0.65, 0.35, 0.16, 1], food = 0, odor = -1}) {
    const shapes = ['box', 'sphere', 'capsule', 'cylinder', 'ellipsoid'];
    const positionValues = Float64Array.from(position), sizeValues = Float64Array.from(size), color = Float64Array.from(rgba);
    const positionPointer = this.#buffer('insert-position', positionValues.byteLength);
    const sizePointer = this.#buffer('insert-size', sizeValues.byteLength);
    const colorPointer = this.#buffer('insert-color', color.byteLength);
    this.module.HEAPF64.set(positionValues, positionPointer >>> 3);
    this.module.HEAPF64.set(sizeValues, sizePointer >>> 3);
    this.module.HEAPF64.set(color, colorPointer >>> 3);
    const length = this.module._chreatures_fly_world_insert_object(
      this.handle, positionPointer, positionValues.length, sizePointer, sizeValues.length,
      shapes.indexOf(shape), colorPointer, color.length, food, odor);
    if (!length) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    this.geometryCache = undefined;
    return this.#readJson('mutation_json', length);
  }

  sample() {
    const opticPointer = this.#buffer('optic', this.optic.byteLength);
    const bodyPointer = this.#buffer('body', this.body.byteLength);
    this.#check(this.module._chreatures_fly_world_sample(
      this.handle, opticPointer, this.optic.length, bodyPointer, this.body.length));
    // Reacquire heap views after every native call because memory may grow.
    this.optic.set(this.module.HEAPF32.subarray(opticPointer >>> 2, (opticPointer >>> 2) + this.optic.length));
    this.body.set(this.module.HEAPF32.subarray(bodyPointer >>> 2, (bodyPointer >>> 2) + this.body.length));
    return {optic: this.optic, body: this.body};
  }

  advance(motor, dt = 0.01) {
    if (!(motor instanceof Float32Array) || motor.length !== this.residents * 92)
      throw new Error(`motor must be Float32Array[${this.residents * 92}]`);
    const pointer = this.#buffer('motor', motor.byteLength);
    this.module.HEAPF32.set(motor, pointer >>> 2);
    this.#check(this.module._chreatures_fly_world_advance(this.handle, pointer, motor.length, dt));
  }

  snapshot() {
    const length = this.module._chreatures_fly_world_snapshot_length(this.handle);
    if (!length) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    const pointer = this.#buffer('snapshot', length);
    this.#check(this.module._chreatures_fly_world_snapshot(this.handle, pointer, length));
    return this.module.HEAPU8.slice(pointer, pointer + length);
  }

  restore(snapshot) {
    if (!(snapshot instanceof Uint8Array)) throw new Error('snapshot must be Uint8Array');
    const pointer = this.#buffer('restore', snapshot.byteLength);
    this.module.HEAPU8.set(snapshot, pointer);
    this.#check(this.module._chreatures_fly_world_restore(this.handle, pointer, snapshot.byteLength));
    this.geometryCache = undefined;
  }

  observe() {
    const length = this.module._chreatures_fly_world_observe(this.handle);
    if (!length) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    const pointer = this.#buffer('observation-json', length);
    this.#check(this.module._chreatures_fly_world_observation_json(this.handle, pointer, length));
    const result = JSON.parse(decoder.decode(this.module.HEAPU8.slice(pointer, pointer + length)));
    const f64Fields = [
      'qpos', 'qvel', 'body_positions', 'body_quaternions', 'body_rotations',
      'sensor_data', 'controls', 'geom_positions', 'geom_rotations', 'geom_sizes', 'geom_colors',
    ];
    for (let field = 0; field < f64Fields.length; field++) {
      const name = f64Fields[field], count = result.numeric_lengths[`${name}_f64`];
      const address = this.#buffer(`observation-f64-${field}`, count * 8);
      this.#check(this.module._chreatures_fly_world_observation_f64(this.handle, field, address, count));
      result[name] = this.module.HEAPF64.slice(address >>> 3, (address >>> 3) + count);
    }
    const f32Fields = ['entity_positions', 'optic', 'body'];
    for (let field = 0; field < f32Fields.length; field++) {
      const name = f32Fields[field], count = result.numeric_lengths[`${name}_f32`];
      const address = this.#buffer(`observation-f32-${field}`, count * 4);
      this.#check(this.module._chreatures_fly_world_observation_f32(this.handle, field, address, count));
      result[name] = this.module.HEAPF32.slice(address >>> 2, (address >>> 2) + count);
    }
    return result;
  }

  geometry() {
    const revision = this.module._chreatures_fly_world_topology_revision(this.handle);
    if (revision === 0xffffffff) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    if (this.geometryCache?.topology_revision === revision) return this.geometryCache;
    const length = this.module._chreatures_fly_world_geometry(this.handle);
    if (!length) throw new Error(UnifiedWasmFlyWorld.error(this.module));
    const result = this.#readJson('geometry_json', length);
    const f64Names = [
      'geom_size', 'geom_position', 'geom_quaternion', 'geom_rgba', 'material_rgba',
      'mesh_vertices', 'mesh_normals',
    ];
    for (let field = 0; field < f64Names.length; field++) {
      const count = result.f64_lengths[field];
      const pointer = this.#buffer(`geometry-f64-${field}`, count * 8);
      this.#check(this.module._chreatures_fly_world_geometry_f64(this.handle, field, pointer, count));
      result[f64Names[field]] = this.module.HEAPF64.slice(pointer >>> 3, (pointer >>> 3) + count);
    }
    const i32Names = ['geom_body_id', 'geom_type', 'geom_material_id', 'geom_data_id', 'mesh_vertex_address',
      'mesh_vertex_count', 'mesh_normal_address', 'mesh_normal_count', 'mesh_face_address',
      'mesh_face_count', 'mesh_faces'];
    for (let field = 0; field < i32Names.length; field++) {
      const count = result.i32_lengths[field];
      const pointer = this.#buffer(`geometry-i32-${field}`, count * 4);
      this.#check(this.module._chreatures_fly_world_geometry_i32(this.handle, field, pointer, count));
      result[i32Names[field]] = new Int32Array(this.module.HEAPU8.buffer)
        .slice(pointer >>> 2, (pointer >>> 2) + count);
    }
    this.geometryCache = result;
    return result;
  }

  close() {
    if (this.handle) {
      const handle = this.handle;
      this.handle = 0;
      try {
        this.#check(this.module._chreatures_fly_world_close(handle));
      } finally {
        for (const {pointer} of this.buffers.values()) this.module._free(pointer);
        this.buffers.clear();
      }
    }
  }
}
