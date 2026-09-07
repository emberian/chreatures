// Chreatures full MaleCNS WebGPU V2 engine.
// Copyright (C) 2026 Chreatures contributors. AGPL-3.0-or-later.
//
// The only deployed information path is:
// optic RGB/body-local senses -> afferent drive -> full recurrent graph -> Z512.

import { loadBlob, loadJSON } from './assets.js';

export const CNS_WEBGPU_FORMAT = 'chreatures-cns-webgpu-v2';
export const CNS_WEBGPU_VERSION = 2;
export const CNS_COUNTS = Object.freeze({
  neurons: 165122,
  edges: 25563197,
  types: 11752,
  opticSites: 1771,
  opticValues: 5313,
  receptors: 4107,
  receptorTypes: 10,
  receptorSiteEdges: 4669,
  bodyChannels: 43,
  bodyHidden: 128,
  bodyTargets: 11233,
  afferents: 15340,
  rank: 64,
  latent: 512,
});

const MAX_CAPACITY = 4;
const DEFAULT_CAPACITY = 3;
const STATE_STRIDE = 12; // three vec4<f32>: rate, signed adaptation, support
const STATE_BYTES = CNS_COUNTS.neurons * STATE_STRIDE * 4;
const VECTOR_BYTES = CNS_COUNTS.neurons * 4 * 4;
const LATENT_BYTES = CNS_COUNTS.latent * 4 * 4;
const PROJECTION_BLOCKS = Math.ceil(CNS_COUNTS.neurons / 256);
const SNAPSHOT_MAGIC = new Uint8Array([67, 72, 87, 71, 50, 83, 84, 0]); // CHWG2ST\0
const decoder = new TextDecoder();
const encoder = new TextEncoder();

const SPECS = Object.freeze({
  'graph.crow': ['u32', [165123]],
  'graph.col': ['u32', [25563197]],
  'graph.weight': ['packed-f16', [25563197]],
  'atlas.receptor_rows': ['u32', [4107]],
  'atlas.receptor_type': ['u32', [4107]],
  'atlas.receptor_ptr': ['u32', [4108]],
  'atlas.site_indices': ['u32', [4669]],
  'atlas.site_weight': ['f32', [4669]],
  'atlas.body_rows': ['u32', [11233]],
  'atlas.neuron_type': ['u32', [165122]],
  'optic.spectral_logits': ['f32', [10, 3]],
  'optic.gain_raw': ['f32', [10]],
  'optic.bias': ['f32', [10]],
  'body.mean': ['f32', [43]],
  'body.scale': ['f32', [43]],
  'body.input.weight': ['f32', [128, 43]],
  'body.input.bias': ['f32', [128]],
  'body.output.weight': ['f32', [11233, 128]],
  'body.output.bias': ['f32', [11233]],
  'dynamics.baseline_raw': ['f32', [11752]],
  'dynamics.recurrent_gain_raw': ['f32', [11752]],
  'dynamics.tau_raw': ['f32', [11752]],
  'dynamics.adaptation_gain_raw': ['f32', [11752]],
  'dynamics.adaptation_tau_raw': ['f32', [11752]],
  'afferent.neutral_drive': ['f32', [165122]],
  'afferent.mask': ['u32', [165122]],
  'readout.projection.weight': ['packed-f16', [64, 165122]],
  'readout.output.weight': ['f32', [512, 64]],
  'readout.output.bias': ['f32', [512]],
});

const SHADER_FILES = Object.freeze(['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl']);

function product(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

function byteLength(dtype, shape) {
  const values = product(shape);
  if (dtype === 'packed-f16') return Math.ceil(values / 2) * 4;
  return values * 4;
}

function assertHexDigest(value, label) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest`);
  }
}

function sameShape(left, right) {
  return Array.isArray(left) && left.length === right.length && left.every((v, i) => v === right[i]);
}

function validateManifest(manifest) {
  if (!manifest || manifest.format !== CNS_WEBGPU_FORMAT || manifest.version !== CNS_WEBGPU_VERSION) {
    throw new Error(`CNS manifest must be ${CNS_WEBGPU_FORMAT} version ${CNS_WEBGPU_VERSION}`);
  }
  for (const [name, expected] of Object.entries(CNS_COUNTS)) {
    if (manifest.counts?.[name] !== expected) throw new Error(`CNS count differs: ${name}`);
  }
  for (const name of ['artifact', 'graph', 'atlas', 'mask']) {
    assertHexDigest(manifest.identity?.[name], `identity.${name}`);
  }
  assertHexDigest(manifest.serviceArtifactSha256, 'serviceArtifactSha256');
  if (typeof manifest.sourceRevision !== 'string' || manifest.sourceRevision.length === 0) {
    throw new Error('sourceRevision is required');
  }
  if (!manifest.buffers || Object.keys(manifest.buffers).length !== Object.keys(SPECS).length) {
    throw new Error('CNS manifest buffer set differs from the V2 contract');
  }
  for (const [name, [dtype, shape]] of Object.entries(SPECS)) {
    const entry = manifest.buffers[name];
    if (!entry || entry.dtype !== dtype || !sameShape(entry.shape, shape)) {
      throw new Error(`CNS buffer type/shape differs: ${name}`);
    }
    const expectedBytes = byteLength(dtype, shape);
    if (entry.byteLength !== expectedBytes) throw new Error(`CNS buffer byte length differs: ${name}`);
    assertHexDigest(entry.sha256, `${name}.sha256`);
    if (entry.encoding !== undefined && entry.encoding !== 'gzip') {
      throw new Error(`unsupported CNS transport encoding: ${name}`);
    }
    if (entry.encoding === 'gzip') {
      if (!Number.isSafeInteger(entry.transportByteLength) || entry.transportByteLength <= 0) {
        throw new Error(`missing transport byte length: ${name}`);
      }
      assertHexDigest(entry.transportSha256, `${name}.transportSha256`);
    }
  }
  return manifest;
}

function identityEnvelope(manifest) {
  return {
    format: manifest.format,
    version: manifest.version,
    artifact: manifest.identity.artifact,
    graph: manifest.identity.graph,
    atlas: manifest.identity.atlas,
    mask: manifest.identity.mask,
    serviceArtifactSha256: manifest.serviceArtifactSha256,
    sourceRevision: manifest.sourceRevision,
  };
}

function identityEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function asFloat32(value, size, name) {
  if (!(value instanceof Float32Array) || value.length !== size) {
    throw new TypeError(`${name} must be Float32Array[${size}]`);
  }
  return value;
}

function maskBits(value, capacity) {
  let bits = 0;
  if (Number.isInteger(value)) bits = value;
  else if (value instanceof Uint32Array && value.length === 1) bits = value[0];
  else if (Array.isArray(value) && value.length === capacity) {
    value.forEach((active, lane) => { if (active) bits |= 1 << lane; });
  } else throw new TypeError('activeMask must be a bit field, Uint32Array[1], or capacity booleans');
  const valid = (1 << capacity) - 1;
  if ((bits & ~valid) !== 0 || bits < 0) throw new RangeError('activeMask selects a lane outside capacity');
  return bits >>> 0;
}

function resetBits(slots, capacity) {
  if (!Array.isArray(slots) || slots.length === 0) throw new TypeError('reset slots must be a nonempty array');
  let bits = 0;
  for (const slot of slots) {
    if (!Number.isInteger(slot) || slot < 0 || slot >= capacity) throw new RangeError('reset slot outside capacity');
    bits |= 1 << slot;
  }
  return bits >>> 0;
}

function configBytes(capacity, active, reset, selected, dt) {
  const bytes = new ArrayBuffer(32);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = capacity;
  u32[1] = active;
  u32[2] = reset;
  u32[3] = selected ?? 0xffffffff;
  f32[4] = dt;
  u32[5] = CNS_COUNTS.neurons;
  u32[6] = CNS_COUNTS.edges;
  return bytes;
}

function createBuffer(device, label, size, usage, initial) {
  const aligned = Math.max(4, Math.ceil(size / 4) * 4);
  const buffer = device.createBuffer({ label, size: aligned, usage, mappedAtCreation: initial !== undefined });
  if (initial !== undefined) {
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(initial));
    buffer.unmap();
  }
  return buffer;
}

function combineFloatArrays(parts) {
  const count = parts.reduce((sum, part) => sum + part.byteLength / 4, 0);
  const result = new Float32Array(count);
  let offset = 0;
  for (const bytes of parts) {
    const input = new Float32Array(bytes);
    result.set(input, offset);
    offset += input.length;
  }
  return result;
}

function opticParameters(assets) {
  const logits = new Float32Array(assets.get('optic.spectral_logits'));
  const gain = new Float32Array(assets.get('optic.gain_raw'));
  const bias = new Float32Array(assets.get('optic.bias'));
  const result = new Float32Array(CNS_COUNTS.receptorTypes * 5);
  for (let type = 0; type < CNS_COUNTS.receptorTypes; type++) {
    result.set(logits.subarray(type * 3, type * 3 + 3), type * 5);
    result[type * 5 + 3] = gain[type];
    result[type * 5 + 4] = bias[type];
  }
  return result;
}

function bodyNormalizer(assets) {
  return combineFloatArrays([assets.get('body.mean'), assets.get('body.scale')]);
}

function computeBaseline(assets) {
  const raw = new Float32Array(assets.get('dynamics.baseline_raw'));
  const types = new Uint32Array(assets.get('atlas.neuron_type'));
  const baseline = new Float32Array(CNS_COUNTS.neurons);
  for (let neuron = 0; neuron < baseline.length; neuron++) {
    baseline[neuron] = Math.fround(0.05 + 0.4 / (1 + Math.exp(-raw[types[neuron]])));
  }
  return baseline;
}

function validateLoadedAssets(assets) {
  const crow = new Uint32Array(assets.get('graph.crow'));
  if (crow[0] !== 0 || crow.at(-1) !== CNS_COUNTS.edges) throw new Error('graph.crow endpoints differ');
  for (let index = 1; index < crow.length; index++) if (crow[index] < crow[index - 1]) throw new Error('graph.crow is not monotone');
  const col = new Uint32Array(assets.get('graph.col'));
  for (const source of col) if (source >= CNS_COUNTS.neurons) throw new Error('graph.col contains an out-of-range neuron');
  const types = new Uint32Array(assets.get('atlas.neuron_type'));
  for (const type of types) if (type >= CNS_COUNTS.types) throw new Error('atlas.neuron_type is out of range');
  const mask = new Uint32Array(assets.get('afferent.mask'));
  let masked = 0;
  for (const value of mask) {
    if (value > 1) throw new Error('afferent.mask must contain only zero or one');
    masked += value === 0;
  }
  if (masked !== CNS_COUNTS.afferents) throw new Error('afferent.mask does not mask exactly 15,340 injected neurons');
  for (const name of ['atlas.receptor_rows', 'atlas.body_rows']) {
    const rows = new Uint32Array(assets.get(name));
    for (let index = 0; index < rows.length; index++) {
      if (rows[index] >= CNS_COUNTS.neurons || mask[rows[index]] !== 0 || (index && rows[index] <= rows[index - 1])) {
        throw new Error(`${name} is out of order, range, or the readout mask`);
      }
    }
  }
  const scale = new Float32Array(assets.get('body.scale'));
  for (const value of scale) if (!(Number.isFinite(value) && value > 0)) throw new Error('body.scale must be finite and positive');
  for (const [name, [dtype]] of Object.entries(SPECS)) {
    if (dtype !== 'f32') continue;
    for (const value of new Float32Array(assets.get(name))) if (!Number.isFinite(value)) throw new Error(`${name} contains a nonfinite value`);
  }
  // Half exponent 31 represents Inf/NaN. Both packed tensors must remain finite.
  for (const name of ['graph.weight', 'readout.projection.weight']) {
    for (const pair of new Uint32Array(assets.get(name))) {
      if ((pair & 0x7c00) === 0x7c00 || ((pair >>> 16) & 0x7c00) === 0x7c00) {
        throw new Error(`${name} contains a nonfinite binary16 value`);
      }
    }
  }
}

async function shaderTexts(shaderBaseURL, supplied) {
  if (supplied) {
    const answer = {};
    for (const name of SHADER_FILES) {
      if (typeof supplied[name] !== 'string') throw new Error(`missing supplied shader: ${name}`);
      answer[name] = supplied[name];
    }
    return answer;
  }
  const base = new URL('./shaders/', shaderBaseURL ?? import.meta.url);
  const entries = await Promise.all(SHADER_FILES.map(async name => {
    const response = await fetch(new URL(name, base));
    if (!response.ok) throw new Error(`shader download ${response.status}: ${name}`);
    return [name, await response.text()];
  }));
  return Object.fromEntries(entries);
}

async function checkedModule(device, label, code) {
  const module = device.createShaderModule({ label, code });
  if (typeof module.getCompilationInfo === 'function') {
    const info = await module.getCompilationInfo();
    const failures = info.messages.filter(message => message.type === 'error');
    if (failures.length) {
      throw new Error(`${label} WGSL failed:\n${failures.map(v => `${v.lineNum}:${v.linePos} ${v.message}`).join('\n')}`);
    }
  }
  return module;
}

async function pipeline(device, module, entryPoint) {
  return device.createComputePipelineAsync({
    label: `MaleCNS V2 ${entryPoint}`,
    layout: 'auto',
    compute: { module, entryPoint },
  });
}

function bind(device, pipeline, resources) {
  return device.createBindGroup({
    label: `${pipeline.label} bindings`,
    layout: pipeline.getBindGroupLayout(0),
    entries: Object.entries(resources).map(([binding, buffer]) => ({
      binding: Number(binding), resource: { buffer },
    })),
  });
}

function dispatch(pass, pipeline, group, x, y = 1) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y);
}

/**
 * Full 165,122-neuron / 25,563,197-edge MaleCNS V2 WebGPU service.
 * Construct with `await MaleCNSWebGPU.load(...)`; direct construction is private.
 */
export class MaleCNSWebGPU {
  static async load({
    device,
    manifest,
    baseURL = import.meta.url,
    capacity = DEFAULT_CAPACITY,
    onProgress = () => {},
    progress,
    assetLoader = loadBlob,
    assetBuffers,
    shaderBaseURL,
    shaderSources,
  }) {
    if (!device) throw new TypeError('a WebGPU device is required');
    const reportProgress = progress === undefined
      ? onProgress
      : event => progress(event.bytes);
    if (!Number.isInteger(capacity) || capacity < 1 || capacity > MAX_CAPACITY) {
      throw new RangeError(`capacity must be between 1 and ${MAX_CAPACITY}`);
    }
    const manifestObject = typeof manifest === 'string' || manifest instanceof URL
      ? await loadJSON(new URL(manifest, baseURL))
      : manifest;
    validateManifest(manifestObject);
    const neededBinding = Math.max(...Object.values(manifestObject.buffers).map(value => value.byteLength));
    if (device.limits.maxStorageBufferBindingSize < neededBinding || device.limits.maxBufferSize < neededBinding) {
      throw new Error(`WebGPU device cannot bind the ${neededBinding}-byte full graph buffer`);
    }
    const totalBytes = Object.values(manifestObject.buffers)
      .reduce((sum, entry) => sum + (entry.transportByteLength ?? entry.byteLength), 0);
    let loadedBytes = 0;
    const assets = new Map();
    for (const [name, entry] of Object.entries(manifestObject.buffers)) {
      let bytes;
      if (assetBuffers?.has?.(name)) bytes = assetBuffers.get(name);
      else if (assetBuffers && Object.hasOwn(assetBuffers, name)) bytes = assetBuffers[name];
      else bytes = await assetLoader(entry, baseURL, () => {});
      if (!(bytes instanceof ArrayBuffer) || bytes.byteLength !== entry.byteLength) {
        throw new Error(`loaded CNS buffer has wrong size: ${name}`);
      }
      assets.set(name, bytes);
      loadedBytes += entry.transportByteLength ?? entry.byteLength;
      reportProgress({ name, bytes: entry.transportByteLength ?? entry.byteLength, loadedBytes, totalBytes });
    }
    const shaders = await shaderTexts(shaderBaseURL, shaderSources);
    const engine = new MaleCNSWebGPU(device, manifestObject, capacity);
    validateLoadedAssets(assets);
    await engine._initialize(assets, shaders);
    return engine;
  }

  constructor(device, manifest, capacity) {
    this.device = device;
    this.manifest = manifest;
    this.identity = Object.freeze(identityEnvelope(manifest));
    this.capacity = capacity;
    this.times = new Float64Array(capacity);
    this.identities = Array(capacity).fill(null);
    this.stepSerial = 0;
    this._currentState = 0;
    this._poisoned = null;
    this._destroyed = false;
    this._tail = Promise.resolve();
    this._buffers = {};
    this._pipelines = {};
    this._groups = {};
    this.device.lost.then(info => { this._poisoned = new Error(`WebGPU device lost: ${info.message || info.reason}`); });
  }

  get baselineRates() {
    return this._baselineRates.slice();
  }

  async _initialize(assets, shaders) {
    const D = this.device;
    const U = GPUBufferUsage;
    this._baselineRates = computeBaseline(assets);
    const immutable = (name, usage = U.STORAGE) => createBuffer(D, name, assets.get(name).byteLength, usage, assets.get(name));
    const B = this._buffers;
    B.config = createBuffer(D, 'CNS V2 config', 32, U.UNIFORM | U.COPY_DST);
    for (const name of ['graph.crow', 'graph.col', 'graph.weight', 'atlas.receptor_rows',
      'atlas.receptor_type', 'atlas.receptor_ptr', 'atlas.site_indices', 'atlas.site_weight',
      'atlas.body_rows', 'atlas.neuron_type', 'body.input.weight', 'body.input.bias',
      'body.output.weight', 'body.output.bias', 'afferent.neutral_drive', 'afferent.mask',
      'readout.projection.weight', 'readout.output.weight', 'readout.output.bias']) {
      B[name] = immutable(name);
    }
    const dynamicNames = ['dynamics.baseline_raw', 'dynamics.recurrent_gain_raw', 'dynamics.tau_raw',
      'dynamics.adaptation_gain_raw', 'dynamics.adaptation_tau_raw'];
    const dynamics = combineFloatArrays(dynamicNames.map(name => assets.get(name)));
    B.dynamics = createBuffer(D, 'V2 dynamics raw', dynamics.byteLength, U.STORAGE, dynamics.buffer);
    const optic = opticParameters(assets);
    B.opticParameters = createBuffer(D, 'optic parameters', optic.byteLength, U.STORAGE, optic.buffer);
    const normalizer = bodyNormalizer(assets);
    B.bodyNormalizer = createBuffer(D, 'body normalizer', normalizer.byteLength, U.STORAGE, normalizer.buffer);
    assets.clear();

    B.opticInput = createBuffer(D, 'optic RGB input', this.capacity * CNS_COUNTS.opticValues * 4, U.STORAGE | U.COPY_DST);
    B.bodyInput = createBuffer(D, 'body-local input', this.capacity * CNS_COUNTS.bodyChannels * 4, U.STORAGE | U.COPY_DST);
    B.drive = createBuffer(D, 'afferent drive vec4', VECTOR_BYTES, U.STORAGE);
    B.opticMixed = createBuffer(D, 'spectral site mixtures vec4', CNS_COUNTS.receptorTypes * CNS_COUNTS.opticSites * 16, U.STORAGE);
    B.bodyHidden = createBuffer(D, 'body hidden vec4', CNS_COUNTS.bodyHidden * 16, U.STORAGE);
    B.state = [
      createBuffer(D, 'CNS state A', STATE_BYTES, U.STORAGE | U.COPY_SRC | U.COPY_DST),
      createBuffer(D, 'CNS state B', STATE_BYTES, U.STORAGE | U.COPY_SRC | U.COPY_DST),
    ];
    B.deviation = createBuffer(D, 'CNS deviation vec4', VECTOR_BYTES, U.STORAGE);
    B.recurrence = createBuffer(D, 'CNS recurrence vec4', VECTOR_BYTES, U.STORAGE);
    B.projectionPartial = createBuffer(D, 'readout projection partial', CNS_COUNTS.rank * PROJECTION_BLOCKS * 16, U.STORAGE);
    B.projected = createBuffer(D, 'readout rank64 vec4', CNS_COUNTS.rank * 16, U.STORAGE);
    B.latent = createBuffer(D, 'CNS latent vec4', LATENT_BYTES, U.STORAGE | U.COPY_SRC);
    B.latentRead = createBuffer(D, 'CNS latent readback', LATENT_BYTES, U.MAP_READ | U.COPY_DST);
    B.stateRead = createBuffer(D, 'CNS state readback', STATE_BYTES, U.MAP_READ | U.COPY_DST);

    const [afferent, dynamicsModule, readout] = await Promise.all([
      checkedModule(D, 'afferent', shaders['afferent.wgsl']),
      checkedModule(D, 'dynamics', shaders['dynamics.wgsl']),
      checkedModule(D, 'readout', shaders['readout.wgsl']),
    ]);
    const definitions = {
      clearDrive: [afferent, 'clear_drive'], mixOptic: [afferent, 'mix_optic'],
      scatterOptic: [afferent, 'scatter_optic'], encodeBody: [afferent, 'encode_body'],
      scatterBody: [afferent, 'scatter_body'], resetState: [dynamicsModule, 'reset_state'],
      deriveDeviation: [dynamicsModule, 'derive_deviation'], recurrentSum: [dynamicsModule, 'recurrent_sum'],
      jacobiUpdate: [dynamicsModule, 'jacobi_update'], finalizeTick: [dynamicsModule, 'finalize_tick'],
      projectPartial: [readout, 'project_partial'], reduceProjection: [readout, 'reduce_projection'],
      outputLatent: [readout, 'output_latent'],
    };
    const built = await Promise.all(Object.entries(definitions).map(async ([name, [module, entry]]) => [name, await pipeline(D, module, entry)]));
    this._pipelines = Object.fromEntries(built);
    D.pushErrorScope('validation');
    this._buildStaticGroups();
    const bindingError = await D.popErrorScope();
    if (bindingError) throw new Error(`MaleCNS V2 binding layout failed: ${bindingError.message}`);
    await this._reset(Array.from({ length: this.capacity }, (_, index) => index), Array(this.capacity).fill(null), true);
  }

  _buildStaticGroups() {
    const D = this.device, B = this._buffers, P = this._pipelines, G = this._groups;
    G.clearDrive = bind(D, P.clearDrive, { 0: B.config, 3: B.drive });
    G.mixOptic = bind(D, P.mixOptic, { 0: B.config, 1: B.opticInput, 9: B.opticParameters, 10: B.opticMixed });
    G.scatterOptic = bind(D, P.scatterOptic, { 3: B.drive, 4: B['atlas.receptor_rows'],
      5: B['atlas.receptor_type'], 6: B['atlas.receptor_ptr'], 7: B['atlas.site_indices'],
      8: B['atlas.site_weight'], 9: B.opticParameters, 10: B.opticMixed });
    G.encodeBody = bind(D, P.encodeBody, { 0: B.config, 2: B.bodyInput, 11: B.bodyNormalizer,
      12: B['body.input.weight'], 13: B['body.input.bias'], 14: B.bodyHidden });
    G.scatterBody = bind(D, P.scatterBody, { 0: B.config, 3: B.drive, 14: B.bodyHidden,
      15: B['body.output.weight'], 16: B['body.output.bias'], 17: B['atlas.body_rows'] });
    G.recurrentSum = bind(D, P.recurrentSum, { 1: B['graph.crow'], 2: B['graph.col'],
      3: B['graph.weight'], 8: B.deviation, 9: B.recurrence });
    G.reduceProjection = bind(D, P.reduceProjection, { 6: B.projectionPartial, 7: B.projected });
    G.outputLatent = bind(D, P.outputLatent, { 0: B.config, 7: B.projected,
      8: B['readout.output.weight'], 9: B['readout.output.bias'], 10: B.latent });
    G.resetState = B.state.map(state => bind(D, P.resetState, { 0: B.config, 4: B['atlas.neuron_type'],
      5: B.dynamics, 7: state }));
    G.deriveDeviation = B.state.map(state => bind(D, P.deriveDeviation, { 4: B['atlas.neuron_type'],
      5: B.dynamics, 6: state, 8: B.deviation }));
    G.jacobiUpdate = B.state.map((stateIn, index) => bind(D, P.jacobiUpdate, { 0: B.config,
      4: B['atlas.neuron_type'], 5: B.dynamics, 6: stateIn, 7: B.state[1 - index],
      9: B.recurrence, 10: B.drive, 11: B['afferent.neutral_drive'] }));
    G.finalizeTick = B.state.map(state => bind(D, P.finalizeTick, { 0: B.config,
      4: B['atlas.neuron_type'], 5: B.dynamics, 7: state }));
    G.projectPartial = B.state.map(state => bind(D, P.projectPartial, { 1: state,
      2: B['atlas.neuron_type'], 3: B.dynamics, 4: B['afferent.mask'],
      5: B['readout.projection.weight'], 6: B.projectionPartial }));
  }

  _enqueue(operation) {
    const result = this._tail.then(() => operation());
    this._tail = result.catch(() => {});
    return result;
  }

  _assertUsable(allowPoisoned = false) {
    if (this._destroyed) throw new Error('MaleCNSWebGPU is destroyed');
    if (this._poisoned && !allowPoisoned) throw this._poisoned;
  }

  _writeConfig(active = 0, reset = 0, selected = undefined, dt = 0) {
    this.device.queue.writeBuffer(this._buffers.config, 0, configBytes(this.capacity, active, reset, selected, dt));
  }

  reset(slots, identities = []) {
    return this._enqueue(() => this._reset(slots, identities, false));
  }

  async _reset(slots, identities, initializing) {
    const bits = initializing ? 0xf : resetBits(slots, this.capacity);
    const full = bits === (1 << this.capacity) - 1;
    this._assertUsable(full);
    if (!Array.isArray(identities) || identities.length !== slots.length) {
      throw new TypeError('identities must correspond one-for-one with reset slots');
    }
    this._writeConfig(0, bits, undefined, 0);
    try {
      this.device.pushErrorScope('validation');
      const command = this.device.createCommandEncoder({ label: 'MaleCNS V2 reset' });
      const pass = command.beginComputePass();
      for (let index = 0; index < 2; index++) {
        dispatch(pass, this._pipelines.resetState, this._groups.resetState[index], Math.ceil(CNS_COUNTS.neurons / 256));
      }
      pass.end();
      this.device.queue.submit([command.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError) throw new Error(`MaleCNS V2 reset validation failed: ${validationError.message}`);
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      this._poisoned = error instanceof Error ? error : new Error(String(error));
      throw this._poisoned;
    }
    slots.forEach((slot, index) => {
      this.times[slot] = 0;
      this.identities[slot] = identities[index] ?? null;
    });
    if (full) this._poisoned = null;
    if (!initializing) this.stepSerial++;
  }

  step({ dt, activeMask, opticRGB, body, selectedResident }) {
    const active = maskBits(activeMask, this.capacity);
    if (!(Number.isFinite(dt) && dt > 0 && dt <= 1)) throw new RangeError('dt must be finite in (0,1] seconds');
    asFloat32(opticRGB, this.capacity * CNS_COUNTS.opticValues, 'opticRGB');
    asFloat32(body, this.capacity * CNS_COUNTS.bodyChannels, 'body');
    for (const [values, name] of [[opticRGB, 'opticRGB'], [body, 'body']]) {
      for (const value of values) if (!Number.isFinite(value)) throw new RangeError(`${name} contains a nonfinite value`);
    }
    if (selectedResident !== undefined && (!Number.isInteger(selectedResident) || selectedResident < 0 || selectedResident >= this.capacity)) {
      throw new RangeError('selectedResident outside capacity');
    }
    return this._enqueue(() => this._step(dt, active, opticRGB, body, selectedResident));
  }

  async _step(dt, active, opticRGB, body, selectedResident) {
    this._assertUsable();
    const D = this.device, B = this._buffers, P = this._pipelines, G = this._groups;
    this._writeConfig(active, 0, selectedResident, dt);
    D.queue.writeBuffer(B.opticInput, 0, opticRGB);
    D.queue.writeBuffer(B.bodyInput, 0, body);
    D.pushErrorScope('validation');
    const command = D.createCommandEncoder({ label: `MaleCNS V2 tick ${this.stepSerial}` });
    const pass = command.beginComputePass({ label: 'full MaleCNS V2' });
    dispatch(pass, P.clearDrive, G.clearDrive, Math.ceil(CNS_COUNTS.neurons / 256));
    dispatch(pass, P.mixOptic, G.mixOptic, Math.ceil(CNS_COUNTS.receptorTypes * CNS_COUNTS.opticSites / 128));
    dispatch(pass, P.scatterOptic, G.scatterOptic, Math.ceil(CNS_COUNTS.receptors / 128));
    dispatch(pass, P.encodeBody, G.encodeBody, 1);
    dispatch(pass, P.scatterBody, G.scatterBody, Math.ceil(CNS_COUNTS.bodyTargets / 128));
    let state = this._currentState;
    for (let substep = 0; substep < 2; substep++) {
      dispatch(pass, P.deriveDeviation, G.deriveDeviation[state], Math.ceil(CNS_COUNTS.neurons / 256));
      dispatch(pass, P.recurrentSum, G.recurrentSum, Math.ceil(CNS_COUNTS.neurons / 256));
      dispatch(pass, P.jacobiUpdate, G.jacobiUpdate[state], Math.ceil(CNS_COUNTS.neurons / 256));
      state = 1 - state;
    }
    dispatch(pass, P.finalizeTick, G.finalizeTick[state], Math.ceil(CNS_COUNTS.neurons / 256));
    dispatch(pass, P.projectPartial, G.projectPartial[state], PROJECTION_BLOCKS, CNS_COUNTS.rank);
    dispatch(pass, P.reduceProjection, G.reduceProjection, CNS_COUNTS.rank);
    dispatch(pass, P.outputLatent, G.outputLatent, Math.ceil(CNS_COUNTS.latent / 128));
    pass.end();
    command.copyBufferToBuffer(B.latent, 0, B.latentRead, 0, LATENT_BYTES);
    if (selectedResident !== undefined) command.copyBufferToBuffer(B.state[state], 0, B.stateRead, 0, STATE_BYTES);
    try {
      D.queue.submit([command.finish()]);
      const validationError = await D.popErrorScope();
      if (validationError) throw new Error(`MaleCNS V2 tick validation failed: ${validationError.message}`);
      const maps = [B.latentRead.mapAsync(GPUMapMode.READ)];
      if (selectedResident !== undefined) maps.push(B.stateRead.mapAsync(GPUMapMode.READ));
      await Promise.all(maps);
      const gpuLatent = new Float32Array(B.latentRead.getMappedRange());
      const latent = new Float32Array(this.capacity * CNS_COUNTS.latent);
      for (let output = 0; output < CNS_COUNTS.latent; output++) {
        for (let lane = 0; lane < this.capacity; lane++) latent[lane * CNS_COUNTS.latent + output] = gpuLatent[output * 4 + lane];
      }
      B.latentRead.unmap();
      let selectedRates;
      if (selectedResident !== undefined) {
        const gpuState = new Float32Array(B.stateRead.getMappedRange());
        selectedRates = new Float32Array(CNS_COUNTS.neurons);
        for (let neuron = 0; neuron < CNS_COUNTS.neurons; neuron++) {
          selectedRates[neuron] = gpuState[neuron * STATE_STRIDE + selectedResident];
        }
        B.stateRead.unmap();
      }
      this._currentState = state;
      for (let lane = 0; lane < this.capacity; lane++) if ((active & (1 << lane)) !== 0) this.times[lane] += dt;
      this.stepSerial++;
      return { latent, selectedRates, times: this.times.slice(), stepSerial: this.stepSerial };
    } catch (error) {
      if (B.latentRead.mapState === 'mapped') B.latentRead.unmap();
      if (B.stateRead.mapState === 'mapped') B.stateRead.unmap();
      this._poisoned = error instanceof Error ? error : new Error(String(error));
      throw this._poisoned;
    }
  }

  snapshot() {
    return this._enqueue(() => this._snapshot());
  }

  async _snapshot() {
    this._assertUsable();
    const B = this._buffers;
    const command = this.device.createCommandEncoder({ label: 'MaleCNS V2 snapshot' });
    command.copyBufferToBuffer(B.state[this._currentState], 0, B.stateRead, 0, STATE_BYTES);
    this.device.queue.submit([command.finish()]);
    await B.stateRead.mapAsync(GPUMapMode.READ);
    const state = new Uint8Array(B.stateRead.getMappedRange()).slice();
    B.stateRead.unmap();
    const metadata = encoder.encode(JSON.stringify({
      format: 'chreatures-cns-webgpu-state-v2',
      identity: this.identity,
      capacity: this.capacity,
      times: Array.from(this.times),
      identities: this.identities,
      stepSerial: this.stepSerial,
      stateBytes: STATE_BYTES,
      stateLayout: 'neuron-major rate-vec4/adaptation-vec4/support-vec4 f32-le',
    }));
    const paddedMetadataLength = Math.ceil(metadata.length / 4) * 4;
    const result = new ArrayBuffer(12 + paddedMetadataLength + state.length);
    const bytes = new Uint8Array(result);
    bytes.set(SNAPSHOT_MAGIC, 0);
    new DataView(result).setUint32(8, metadata.length, true);
    bytes.set(metadata, 12);
    bytes.set(state, 12 + paddedMetadataLength);
    return result;
  }

  restore(snapshot) {
    return this._enqueue(() => this._restore(snapshot));
  }

  async _restore(snapshot) {
    this._assertUsable(true);
    if (!(snapshot instanceof ArrayBuffer) || snapshot.byteLength < 12) throw new TypeError('snapshot must be an ArrayBuffer');
    const bytes = new Uint8Array(snapshot);
    if (!SNAPSHOT_MAGIC.every((value, index) => bytes[index] === value)) throw new Error('snapshot magic differs');
    const metadataLength = new DataView(snapshot).getUint32(8, true);
    const paddedMetadataLength = Math.ceil(metadataLength / 4) * 4;
    if (12 + paddedMetadataLength + STATE_BYTES !== snapshot.byteLength) throw new Error('snapshot length differs');
    let metadata;
    try { metadata = JSON.parse(decoder.decode(bytes.subarray(12, 12 + metadataLength))); }
    catch { throw new Error('snapshot metadata is invalid JSON'); }
    if (metadata.format !== 'chreatures-cns-webgpu-state-v2' || metadata.capacity !== this.capacity ||
        metadata.stateBytes !== STATE_BYTES || !identityEqual(metadata.identity, this.identity)) {
      throw new Error('snapshot identity or dimensions differ');
    }
    if (!Array.isArray(metadata.times) || metadata.times.length !== this.capacity ||
        metadata.times.some(value => !Number.isFinite(value) || value < 0) ||
        !Array.isArray(metadata.identities) || metadata.identities.length !== this.capacity ||
        !Number.isSafeInteger(metadata.stepSerial) || metadata.stepSerial < 0) {
      throw new Error('snapshot host state is invalid');
    }
    const stateOffset = 12 + paddedMetadataLength;
    const floats = new Float32Array(snapshot, stateOffset, STATE_BYTES / 4);
    for (let neuron = 0; neuron < CNS_COUNTS.neurons; neuron++) {
      const first = neuron * STATE_STRIDE;
      for (let lane = 0; lane < MAX_CAPACITY; lane++) {
        const rate = floats[first + lane], adaptation = floats[first + 4 + lane], support = floats[first + 8 + lane];
        if (!Number.isFinite(rate) || rate < 0 || rate > 1 || !Number.isFinite(adaptation) || Math.abs(adaptation) > 1 ||
            !Number.isFinite(support) || support < 0.65 || support > 1) throw new Error('snapshot contains invalid neural state');
      }
    }
    const stateBytes = bytes.subarray(stateOffset);
    try {
      this.device.queue.writeBuffer(this._buffers.state[0], 0, stateBytes);
      this.device.queue.writeBuffer(this._buffers.state[1], 0, stateBytes);
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      this._poisoned = error instanceof Error ? error : new Error(String(error));
      throw this._poisoned;
    }
    this._currentState = 0;
    this.times.set(metadata.times);
    this.identities = structuredClone(metadata.identities);
    this.stepSerial = metadata.stepSerial;
    this._poisoned = null;
  }

  destroy() {
    this._destroyed = true;
    for (const value of Object.values(this._buffers)) {
      if (Array.isArray(value)) value.forEach(buffer => buffer.destroy());
      else value.destroy?.();
    }
    this._buffers = {};
  }
}
