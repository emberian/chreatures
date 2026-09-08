// SPDX-License-Identifier: AGPL-3.0-or-later
/** LiveEngine adapter for the canonical unified Rust + MuJoCo world. */

import {UnifiedWasmFlyWorld} from './world-runtime.mjs';

const decoder = new TextDecoder();
const MEMFS_ROOT = '/chreatures-world';

function bytes(value, label) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new Error(`${label} must contain bytes`);
}

function safeRelativePath(value, label) {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0')) {
    throw new Error(`${label} has an unsafe path`);
  }
  const parts = value.split('/');
  if (value.startsWith('/') || parts.some(part => !part || part === '.' || part === '..')) {
    throw new Error(`${label} has an unsafe path`);
  }
  return parts.join('/');
}

function assetEntries(assets) {
  if (assets instanceof Map) return [...assets.entries()];
  if (Array.isArray(assets)) return assets.map((asset, index) => {
    if (!asset || typeof asset !== 'object') throw new Error(`asset ${index} is invalid`);
    return [asset.path ?? asset.name, asset.bytes ?? asset.data];
  });
  if (assets && typeof assets === 'object') return Object.entries(assets);
  throw new Error('assets must be a Map, object, or array');
}

function ensureDirectory(module, path) {
  if (typeof module.FS.mkdirTree === 'function') module.FS.mkdirTree(path);
  else {
    let current = '';
    for (const part of path.split('/').filter(Boolean)) {
      current += `/${part}`;
      try { module.FS.mkdir(current); } catch (error) {
        if (!module.FS.analyzePath(current).exists) throw error;
      }
    }
  }
}

function writeExact(module, relativePath, value, label) {
  const path = safeRelativePath(relativePath, label);
  const slash = path.lastIndexOf('/');
  if (slash >= 0) ensureDirectory(module, `${MEMFS_ROOT}/${path.slice(0, slash)}`);
  module.FS.writeFile(`${MEMFS_ROOT}/${path}`, bytes(value, label));
}

function copyF32(source) {
  return Float32Array.from(source);
}

function copyU32(source) {
  return Uint32Array.from(source, value => value >>> 0);
}

class PhysicalWorld {
  constructor(nativeWorld, fixture) {
    this.native = nativeWorld;
    this.residents = nativeWorld.residents;
    this.engine = String(fixture.engine ?? nativeWorld.metadata.engine ?? 'unified-wasm-fly-world');
    this.worldSize = Array.from(fixture.world_size ?? nativeWorld.metadata.world_size ?? [50, 40, 16]);
    this.residentDescriptors = [];
    this.catalogRevision = -1;
    this.catalog = null;
    this.observation = null;
    this.sensory = null;
    this.sensoryCurrent = false;
    this.currentTime = 0;
  }

  get time() {
    return this.currentTime;
  }

  #refreshCatalog() {
    const source = this.native.geometry();
    const revision = Number(source.topology_revision);
    if (this.catalog && revision === this.catalogRevision) return this.catalog;

    const metadata = source.metadata;
    const residents = metadata.residents ?? [];
    this.residentDescriptors = residents.map(item => ({
      id: String(item.id),
      root: Number(item.root_body_id),
      head: Number(item.head_body_id),
    }));
    if (this.residentDescriptors.length !== this.residents) {
      throw new Error(`native geometry describes ${this.residentDescriptors.length} of ${this.residents} residents`);
    }
    this.worldSize = Array.from(metadata.world_size ?? this.worldSize, Number);

    const residentByBody = new Map();
    for (const resident of residents) {
      for (const segment of resident.segments69 ?? []) residentByBody.set(Number(segment.body_id), String(resident.id));
    }
    const geoms = metadata.geoms ?? [];
    const names = metadata.geom_names ?? [];
    if (source.geom_type.length !== geoms.length || source.geom_body_id.length !== geoms.length) {
      throw new Error('native geometry catalog extents differ');
    }
    const geometry = geoms.map((item, index) => {
      const body = Number(source.geom_body_id[index]);
      const descriptor = {
        id: Number(item.id ?? index),
        index,
        type: Number(source.geom_type[index]),
        name: String(names[index] ?? `geom:${item.id ?? index}`),
        body,
      };
      const resident = residentByBody.get(body);
      if (resident !== undefined) descriptor.resident_id = resident;
      const mesh = Number(source.geom_data_id[index]);
      if (descriptor.type === 7 && mesh >= 0) descriptor.mesh_id = mesh;
      return descriptor;
    });
    const meshes = Array.from(source.mesh_vertex_count, (vertexCount, meshId) => {
      const vertexAddress = Number(source.mesh_vertex_address[meshId]) * 3;
      const faceAddress = Number(source.mesh_face_address[meshId]) * 3;
      return {
        positions: copyF32(source.mesh_vertices.subarray(vertexAddress, vertexAddress + Number(vertexCount) * 3)),
        faces: copyU32(source.mesh_faces.subarray(faceAddress, faceAddress + Number(source.mesh_face_count[meshId]) * 3)),
      };
    });
    this.catalogRevision = revision;
    this.catalog = {
      geometry,
      meshes,
      screenGeom: Number(metadata.screen_geom ?? -1),
      meshRevision: revision,
    };
    return this.catalog;
  }

  sample() {
    if (this.sensoryCurrent && this.sensory) return this.sensory;
    this.sensory = this.native.sample();
    this.sensoryCurrent = true;
    return this.sensory;
  }

  advance(motor, dt = 0.01) {
    this.native.advance(motor, dt);
    this.currentTime += dt;
    this.observation = null;
    this.sensoryCurrent = false;
  }

  observe() {
    if (!this.observation) {
      this.observation = this.native.observe();
      this.currentTime = Number(this.observation.time);
      this.sensory = {optic: this.observation.optic, body: this.observation.body};
      this.sensoryCurrent = true;
    }
    const catalog = this.#refreshCatalog();
    const sizes = this.observation.geom_sizes;
    const geometry = catalog.geometry.map((item, index) => ({
      ...item,
      size: [Number(sizes[index * 3]), Number(sizes[index * 3 + 1]), Number(sizes[index * 3 + 2])],
    }));
    return {
      ...this.observation,
      geometry,
      positions: copyF32(this.observation.geom_positions),
      rotations: copyF32(this.observation.geom_rotations),
      colors: copyF32(this.observation.geom_colors),
      bodyPositions: copyF32(this.observation.body_positions),
      bodyRotations: copyF32(this.observation.body_rotations),
      residents: this.residentDescriptors.map(item => ({...item})),
      screenGeom: catalog.screenGeom,
      meshRevision: catalog.meshRevision,
      meshes: catalog.meshes,
    };
  }

  setScreenFrame(frame, width, height) {
    this.native.setScreen(frame, width, height);
    this.observation = null;
    this.sensoryCurrent = false;
  }

  visitorSound(position, frequency, envelope = 1, duration = 0.15) {
    this.native.visitorSound(position, frequency, envelope, duration);
    this.observation = null;
    this.sensoryCurrent = false;
  }

  queueVisitorForce(entityId, force) {
    this.native.queueVisitorForce(entityId, force);
  }

  insertObject(options) {
    const result = this.native.insertObject(options);
    this.catalog = null;
    this.observation = null;
    this.sensoryCurrent = false;
    return result;
  }

  snapshot() {
    return this.native.snapshot();
  }

  restore(snapshot) {
    this.native.restore(snapshot);
    this.catalog = null;
    this.observation = null;
    this.sensory = null;
    this.sensoryCurrent = false;
    this.observation = this.native.observe();
    this.sensory = {optic: this.observation.optic, body: this.observation.body};
    this.sensoryCurrent = true;
    this.currentTime = Number(this.observation.time);
  }

  dispose() {
    this.native.close();
    this.catalog = null;
    this.observation = null;
    this.sensory = null;
  }
}

export async function createPhysicalWorld({moduleFactory, wasmBinary, fixtureBytes, xmlBytes, assets, seed = 1}) {
  if (typeof moduleFactory !== 'function') throw new Error('moduleFactory must be a function');
  const exactFixture = bytes(fixtureBytes, 'fixtureBytes');
  let fixture;
  try {
    fixture = JSON.parse(decoder.decode(exactFixture));
  } catch (error) {
    throw new Error(`fixtureBytes are not valid JSON: ${error.message}`);
  }
  const sceneRelative = safeRelativePath(fixture.scene_xml, 'fixture scene_xml');
  if (sceneRelative === 'world.json') throw new Error('fixture scene_xml collides with world.json');
  const preparedAssets = assetEntries(assets).map(([path, value]) => [
    safeRelativePath(path, `asset ${path}`),
    bytes(value, `asset ${path}`),
  ]);
  const occupiedPaths = new Set(['world.json', sceneRelative]);
  for (const [path] of preparedAssets) {
    if (occupiedPaths.has(path)) throw new Error(`asset ${path} collides with another world file`);
    occupiedPaths.add(path);
  }
  let nativeWorld;
  try {
    nativeWorld = await UnifiedWasmFlyWorld.open({
      moduleFactory,
      wasmBinary: bytes(wasmBinary, 'wasmBinary'),
      scenePath: `${MEMFS_ROOT}/world.json`,
      seed,
      prepareFilesystem(module) {
        ensureDirectory(module, MEMFS_ROOT);
        module.FS.writeFile(`${MEMFS_ROOT}/world.json`, exactFixture);
        writeExact(module, sceneRelative, xmlBytes, 'xmlBytes');
        for (const [path, value] of preparedAssets) writeExact(module, path, value, `asset ${path}`);
      },
    });
    const world = new PhysicalWorld(nativeWorld, fixture);
    world.observe();
    return world;
  } catch (error) {
    nativeWorld?.close();
    throw error;
  }
}
