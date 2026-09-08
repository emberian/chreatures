// Chreatures full MaleCNS WebGPU V4 engine.
// Copyright (C) 2026 Chreatures contributors. AGPL-3.0-or-later.
//
// The only deployed information path is:
// optic RGB/body-local senses/context12 -> CNS recurrence -> Z512 + motor92.

import { digest, loadBlob, loadJSON } from './assets.js';

export const CNS_WEBGPU_FORMAT = 'chreatures-cns-webgpu-v4';
export const CNS_WEBGPU_VERSION = 4;
export const CNS_COUNTS = Object.freeze({
  neurons: 165122,
  edges: 25563197,
  types: 11752,
  opticSites: 1771,
  opticValues: 5313,
  receptors: 4107,
  receptorTypes: 10,
  receptorSiteEdges: 4669,
  bodyChannels: 807,
  contextChannels: 12,
  contextTargets: 1314,
  motor: 92,
  motorTargets: 815,
  bodyTargets: 11798,
  rank: 64,
  latent: 512,
});

const MAX_CAPACITY = 4;
const DEFAULT_CAPACITY = 3;
const STATE_STRIDE = 28; // seven vec4: rate, adaptation, support, release, DA, OA, 5HT
const STATE_BYTES = CNS_COUNTS.neurons * STATE_STRIDE * 4;
const VECTOR_BYTES = CNS_COUNTS.neurons * 4 * 4;
const LATENT_BYTES = CNS_COUNTS.latent * 4 * 4;
const PROJECTION_BLOCKS = Math.ceil(CNS_COUNTS.neurons / 256);
const MOTOR_BYTES = CNS_COUNTS.motor * 4 * 4;
const SNAPSHOT_MAGIC = new Uint8Array([67, 72, 87, 71, 52, 83, 84, 0]); // CHWG4ST\0
const decoder = new TextDecoder();
const encoder = new TextEncoder();
const STATE_FIELDS = Object.freeze({rate:0,adaptation:1,support:2,release:3,dopamine:4,octopamine:5,serotonin:6});

const SPECS = Object.freeze({
  'graph.crow': ['u32', [165123]],
  'graph.col': ['u32', [25563197]],
  'graph.weight_bits': ['u16', [25563197]],
  'graph.channel': ['u32', [165122]],
  'atlas.receptor_rows': ['u32', [4107]],
  'atlas.receptor_type': ['u32', [4107]],
  'atlas.receptor_ptr': ['u32', [4108]],
  'atlas.site_indices': ['u32', [4669]],
  'atlas.site_weight': ['f32', [4669]],
  'atlas.body_rows': ['u32', [11798]],
  'atlas.body_mask': ['f32', [11798, 807]],
  'atlas.context_rows': ['u32', [1314]],
  'atlas.motor_rows': ['u32', [815]],
  'atlas.motor_mask': ['f32', [92, 815]],
  'atlas.neuron_type': ['u32', [165122]],
  'optic.spectral_logits': ['f32', [10, 3]],
  'optic.gain_raw': ['f32', [10]],
  'optic.bias': ['f32', [10]],
  'body.mean': ['f32', [807]],
  'body.scale': ['f32', [807]],
  'body.weight': ['f32', [11798, 807]],
  'body.bias': ['f32', [11798]],
  'context.weight': ['f32', [1314, 12]],
  'context.bias': ['f32', [1314]],
  'dynamics.baseline_raw': ['f32', [11752]],
  'dynamics.recurrent_gain_raw': ['f32', [11752]],
  'dynamics.tau_raw': ['f32', [11752]],
  'dynamics.adaptation_gain_raw': ['f32', [11752]],
  'dynamics.adaptation_tau_raw': ['f32', [11752]],
  'dynamics.release_tau_raw': ['f32', [11752]],
  'dynamics.release_use_raw': ['f32', [11752]],
  'dynamics.mod_gain_raw': ['f32', [11752, 3]],
  'dynamics.mod_adaptation_raw': ['f32', [11752, 3]],
  'dynamics.modulation_tau_raw': ['f32', [3]],
  'afferent.neutral_drive': ['f32', [165122]],
  'afferent.mask': ['u32', [165122]],
  'readout.projection.weight': ['f32', [64, 165122]],
  'readout.output.weight': ['f32', [512, 64]],
  'readout.output.bias': ['f32', [512]],
  'motor.reference_rate': ['f32', [815]],
  'motor.rate_scale': ['f32', [815]],
  'motor.weight': ['f32', [92, 815]],
  'motor.intercept': ['f32', [92]],
});

const SHADER_FILES = Object.freeze(['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl', 'motor.wgsl','observe.wgsl']);

function product(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

function byteLength(dtype, shape) {
  const values = product(shape);
  if (dtype === 'u16') return values * 2;
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
  for (const name of ['artifact', 'graph', 'atlas', 'anatomy', 'mask', 'morphology', 'sensorySchema', 'actuatorSchema', 'motorCalibration', 'graphSourceWeight']) {
    assertHexDigest(manifest.identity?.[name], `identity.${name}`);
  }
  if (!Number.isSafeInteger(manifest.counts.afferents) || manifest.counts.afferents <= 0 || manifest.counts.afferents > CNS_COUNTS.neurons) throw new Error('Invalid injected row count');
  if (manifest.controlDt !== 0.01 || manifest.substeps !== 2) throw new Error('V4 requires two 0.005-second CNS substeps');
  const quantization = manifest.graphQuantization;
  if (quantization?.storage !== 'ieee-754-binary16-bits-little-endian' || quantization.rounding !== 'round-to-nearest-ties-to-even' || quantization.compute !== 'decode-once-to-float32') throw new Error('Canonical graph quantization differs');
  assertHexDigest(manifest.serviceArtifactSha256, 'serviceArtifactSha256');
  if (typeof manifest.sourceRevision !== 'string' || manifest.sourceRevision.length === 0) {
    throw new Error('sourceRevision is required');
  }
  if (!manifest.buffers || Object.keys(manifest.buffers).length !== Object.keys(SPECS).length) {
    throw new Error('CNS manifest buffer set differs from the V4 contract');
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
    anatomy: manifest.identity.anatomy,
    mask: manifest.identity.mask,
    morphology: manifest.identity.morphology,
    sensorySchema: manifest.identity.sensorySchema,
    actuatorSchema: manifest.identity.actuatorSchema,
    motorCalibration: manifest.identity.motorCalibration,
    graphSourceWeight: manifest.identity.graphSourceWeight,
    graphQuantization: manifest.graphQuantization,
    controlDt: manifest.controlDt,
    substeps: manifest.substeps,
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

function configBytes(capacity, active, reset, selected, dt, selectedField=0) {
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
  u32[7] = selectedField;
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

function dynamicsParameters(assets) {
  const scalar = ['dynamics.baseline_raw','dynamics.recurrent_gain_raw','dynamics.tau_raw',
    'dynamics.adaptation_gain_raw','dynamics.adaptation_tau_raw','dynamics.release_tau_raw',
    'dynamics.release_use_raw'];
  const out = new Float32Array(13 * CNS_COUNTS.types + 3);
  let field=0;
  for(const name of scalar) out.set(new Float32Array(assets.get(name)), field++ * CNS_COUNTS.types);
  for(const name of ['dynamics.mod_gain_raw','dynamics.mod_adaptation_raw']){
    const source=new Float32Array(assets.get(name));
    for(let family=0;family<3;family++,field++) for(let type=0;type<CNS_COUNTS.types;type++)
      out[field*CNS_COUNTS.types+type]=source[type*3+family];
  }
  out.set(new Float32Array(assets.get('dynamics.modulation_tau_raw')),13*CNS_COUNTS.types);
  return out;
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

function bodyParameters(assets) {
  return combineFloatArrays(['body.mean','body.scale','body.weight','body.bias'].map(name => assets.get(name)));
}
function opticAuxiliary(assets) {
  const types=new Uint32Array(assets.get('atlas.receptor_type')), sites=new Uint32Array(assets.get('atlas.site_indices'));
  const weightBits=new Uint32Array(assets.get('atlas.site_weight'));
  const out=new Uint32Array(types.length+sites.length+weightBits.length);out.set(types);out.set(sites,types.length);
  out.set(weightBits,types.length+sites.length);return out;
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

function validateLoadedAssets(assets, manifest) {
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
  if (masked !== manifest.counts.afferents) throw new Error('Injected row count differs');
  const injected = new Uint8Array(CNS_COUNTS.neurons);
  for (const name of ['atlas.receptor_rows', 'atlas.body_rows', 'atlas.context_rows']) {
    const rows = new Uint32Array(assets.get(name));
    for (let index = 0; index < rows.length; index++) {
      injected[rows[index]] = 1;
      if (rows[index] >= CNS_COUNTS.neurons || mask[rows[index]] !== 0 || (index && rows[index] <= rows[index - 1])) {
        throw new Error(`${name} is out of order, range, or the readout mask`);
      }
    }
  }
  for (let row = 0; row < mask.length; row++) if (mask[row] !== 1 - injected[row]) throw new Error('Readout mask differs from exact injected union');
  for (const row of new Uint32Array(assets.get('atlas.motor_rows'))) if (row >= CNS_COUNTS.neurons) throw new Error('Motor row out of range');
  for (const value of new Float32Array(assets.get('motor.rate_scale'))) if (!(value > 0)) throw new Error('Motor rate scale must be positive');
  for (const value of new Float32Array(assets.get('motor.reference_rate'))) if (value < 0 || value > 1) throw new Error('Motor reference rate must be in [0,1]');
  for (const name of ['atlas.body_mask', 'atlas.motor_mask']) for (const value of new Float32Array(assets.get(name))) if (value !== 0 && value !== 1) throw new Error(`${name} must be binary`);
  const scale = new Float32Array(assets.get('body.scale'));
  for (const value of scale) if (!(Number.isFinite(value) && value > 0)) throw new Error('body.scale must be finite and positive');
  for (const [name, [dtype]] of Object.entries(SPECS)) {
    if (dtype !== 'f32') continue;
    for (const value of new Float32Array(assets.get(name))) if (!Number.isFinite(value)) throw new Error(`${name} contains a nonfinite value`);
  }
  // Canonical IEEE binary16 bits are validated before one-time float32 decoding.
  for (const bits of new Uint16Array(assets.get('graph.weight_bits'))) {
    if ((bits & 0x7c00) === 0x7c00) throw new Error('Graph contains a nonfinite binary16 value');
  }
}

function decodeGraphWeights(buffer) {
  const bits = new Uint16Array(buffer);
  const decoded = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const sign = bits[i] & 0x8000 ? -1 : 1;
    const exponent = (bits[i] >>> 10) & 31;
    const fraction = bits[i] & 1023;
    decoded[i] = sign * (exponent === 0 ? fraction * 2 ** -24 : (1024 + fraction) * 2 ** (exponent - 25));
  }
  return decoded;
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
    label: `MaleCNS V4 ${entryPoint}`,
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
 * Full 165,122-neuron / 25,563,197-edge MaleCNS V4 WebGPU service.
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
      if ((assetBuffers || assetLoader !== loadBlob) && await digest(bytes) !== entry.sha256) throw new Error(`Loaded CNS buffer checksum differs: ${name}`);
      assets.set(name, bytes);
      loadedBytes += entry.transportByteLength ?? entry.byteLength;
      reportProgress({ name, bytes: entry.transportByteLength ?? entry.byteLength, loadedBytes, totalBytes });
    }
    const shaders = await shaderTexts(shaderBaseURL, shaderSources);
    const engine = new MaleCNSWebGPU(device, manifestObject, capacity);
    validateLoadedAssets(assets, manifestObject);
    const maskBytes = Uint8Array.from(new Uint32Array(assets.get('afferent.mask')));
    if (await digest(maskBytes) !== manifestObject.identity.mask) throw new Error('Readout mask identity differs');
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
    B.config = createBuffer(D, 'CNS V4 config', 32, U.UNIFORM | U.COPY_DST);
    for (const name of ['graph.crow', 'graph.col', 'atlas.receptor_rows',
      'atlas.receptor_ptr', 'atlas.body_rows', 'atlas.body_mask', 'atlas.context_rows',
      'atlas.motor_rows', 'atlas.motor_mask', 'atlas.neuron_type', 'graph.channel',
      'afferent.neutral_drive', 'afferent.mask', 'readout.projection.weight',
      'readout.output.weight', 'readout.output.bias', 'motor.weight', 'motor.intercept', 'motor.reference_rate', 'motor.rate_scale']) {
      B[name] = immutable(name);
    }
    const graphWeights = decodeGraphWeights(assets.get('graph.weight_bits'));
    B['graph.weight'] = createBuffer(D, 'decoded canonical graph weights', graphWeights.byteLength, U.STORAGE, graphWeights.buffer);
    const dynamics = dynamicsParameters(assets);
    B.dynamics = createBuffer(D, 'V4 dynamics raw', dynamics.byteLength, U.STORAGE, dynamics.buffer);
    B.baseline = createBuffer(D, 'baseline by neuron', this._baselineRates.byteLength, U.STORAGE, this._baselineRates.buffer);
    const optic = opticParameters(assets);
    B.opticParameters = createBuffer(D, 'optic parameters', optic.byteLength, U.STORAGE, optic.buffer);
    const opticAux = opticAuxiliary(assets);
    B.opticAux = createBuffer(D, 'optic indices', opticAux.byteLength, U.STORAGE, opticAux.buffer);
    const body = bodyParameters(assets);
    B.bodyParameters = createBuffer(D, 'masked body parameters', body.byteLength, U.STORAGE, body.buffer);
    const context = combineFloatArrays([assets.get('context.weight'), assets.get('context.bias')]);
    B.contextParameters = createBuffer(D, 'context parameters', context.byteLength, U.STORAGE, context.buffer);
    assets.clear();

    B.opticInput = createBuffer(D, 'optic RGB input', this.capacity * CNS_COUNTS.opticValues * 4, U.STORAGE | U.COPY_DST);
    B.bodyInput = createBuffer(D, 'body-local input', this.capacity * CNS_COUNTS.bodyChannels * 4, U.STORAGE | U.COPY_DST);
    B.contextInput = createBuffer(D, 'context12 input', this.capacity * CNS_COUNTS.contextChannels * 4, U.STORAGE | U.COPY_DST);
    B.drive = createBuffer(D, 'afferent drive vec4', VECTOR_BYTES, U.STORAGE);
    B.opticMixed = createBuffer(D, 'spectral site mixtures vec4', CNS_COUNTS.receptorTypes * CNS_COUNTS.opticSites * 16, U.STORAGE);
    B.state = [
      createBuffer(D, 'CNS state A', STATE_BYTES, U.STORAGE | U.COPY_SRC | U.COPY_DST),
      createBuffer(D, 'CNS state B', STATE_BYTES, U.STORAGE | U.COPY_SRC | U.COPY_DST),
    ];
    B.recurrence = createBuffer(D, 'CNS fast/mod recurrence', VECTOR_BYTES * 4, U.STORAGE);
    B.projectionPartial = createBuffer(D, 'readout projection partial', CNS_COUNTS.rank * PROJECTION_BLOCKS * 16, U.STORAGE);
    B.projected = createBuffer(D, 'readout rank64 vec4', CNS_COUNTS.rank * 16, U.STORAGE);
    B.latent = createBuffer(D, 'CNS latent vec4', LATENT_BYTES, U.STORAGE | U.COPY_SRC);
    B.latentRead = createBuffer(D, 'CNS outputs readback', LATENT_BYTES+MOTOR_BYTES, U.MAP_READ | U.COPY_DST);
    B.motor = createBuffer(D, 'CNS motor92 vec4', MOTOR_BYTES, U.STORAGE | U.COPY_SRC);
    B.stateRead = createBuffer(D, 'CNS state readback', STATE_BYTES, U.MAP_READ | U.COPY_DST);
    B.snapshotRead = createBuffer(D, 'CNS snapshot readback', STATE_BYTES, U.MAP_READ | U.COPY_DST);
    B.observe = createBuffer(D, 'selected neural observation', CNS_COUNTS.neurons*8, U.STORAGE | U.COPY_SRC);
    B.observeRead=createBuffer(D,'selected neural observation readback',CNS_COUNTS.neurons*8,U.MAP_READ|U.COPY_DST);

    const [afferent, dynamicsModule, readout, motorModule, observeModule] = await Promise.all([
      checkedModule(D, 'afferent', shaders['afferent.wgsl']),
      checkedModule(D, 'dynamics', shaders['dynamics.wgsl']),
      checkedModule(D, 'readout', shaders['readout.wgsl']),
      checkedModule(D, 'motor', shaders['motor.wgsl']),
      checkedModule(D, 'observe', shaders['observe.wgsl']),
    ]);
    const definitions = {
      clearDrive: [afferent, 'clear_drive'], mixOptic: [afferent, 'mix_optic'],
      scatterOptic: [afferent, 'scatter_optic'], encodeBody: [afferent, 'encode_body'],
      injectContext: [afferent, 'inject_context'], resetState: [dynamicsModule, 'reset_state'],
      recurrentSum: [dynamicsModule, 'recurrent_sum'],
      jacobiUpdate: [dynamicsModule, 'jacobi_update'], finalizeTick: [dynamicsModule, 'finalize_tick'],
      gatherObserve: [observeModule, 'gather_observe'],
      projectPartial: [readout, 'project_partial'], reduceProjection: [readout, 'reduce_projection'],
      outputLatent: [readout, 'output_latent'],
      outputMotor: [motorModule, 'output_motor'],
    };
    const built = await Promise.all(Object.entries(definitions).map(async ([name, [module, entry]]) => [name, await pipeline(D, module, entry)]));
    this._pipelines = Object.fromEntries(built);
    D.pushErrorScope('validation');
    this._buildStaticGroups();
    const bindingError = await D.popErrorScope();
    if (bindingError) throw new Error(`MaleCNS V4 binding layout failed: ${bindingError.message}`);
    await this._reset(Array.from({ length: this.capacity }, (_, index) => index), Array(this.capacity).fill(null), true);
  }

  _buildStaticGroups() {
    const D = this.device, B = this._buffers, P = this._pipelines, G = this._groups;
    G.clearDrive = bind(D, P.clearDrive, { 0:B.config, 2:B.drive });
    G.mixOptic = bind(D,P.mixOptic,{0:B.config,1:B.opticInput,4:B.opticParameters,7:B.opticMixed});
    G.scatterOptic = bind(D,P.scatterOptic,{2:B.drive,3:B['atlas.receptor_rows'],4:B.opticParameters,5:B['atlas.receptor_ptr'],6:B.opticAux,7:B.opticMixed});
    G.encodeBody = bind(D,P.encodeBody,{0:B.config,1:B.bodyInput,2:B.drive,3:B['atlas.body_rows'],4:B.bodyParameters,5:B['atlas.body_mask']});
    G.injectContext = bind(D,P.injectContext,{0:B.config,1:B.contextInput,2:B.drive,3:B['atlas.context_rows'],4:B.contextParameters});
    G.reduceProjection = bind(D, P.reduceProjection, { 6: B.projectionPartial, 7: B.projected });
    G.outputLatent = bind(D, P.outputLatent, { 0: B.config, 7: B.projected,
      8: B['readout.output.weight'], 9: B['readout.output.bias'], 10: B.latent });
    G.resetState = B.state.map(state => bind(D, P.resetState, { 0: B.config, 4: B['atlas.neuron_type'],
      5: B.dynamics, 7: state }));
    G.recurrentSum = B.state.map(state => bind(D,P.recurrentSum,{1:B['graph.crow'],2:B['graph.col'],3:B['graph.weight'],4:B['graph.channel'],5:B.baseline,6:state,8:B.recurrence}));
    G.jacobiUpdate = B.state.map((stateIn, index) => bind(D, P.jacobiUpdate, { 0: B.config,
      4: B['atlas.neuron_type'], 5: B.dynamics, 6: stateIn, 7: B.state[1 - index],
      8: B.recurrence, 9: B.drive, 10: B['afferent.neutral_drive'] }));
    G.finalizeTick = B.state.map(state => bind(D, P.finalizeTick, { 0: B.config,
      4: B['atlas.neuron_type'], 5: B.dynamics, 7: state }));
    G.outputMotor = B.state.map(state=>bind(D,P.outputMotor,{0:state,1:B['atlas.motor_rows'],2:B['motor.weight'],3:B['atlas.motor_mask'],4:B['motor.intercept'],5:B.motor,6:B['motor.reference_rate'],7:B['motor.rate_scale']}));
    G.gatherObserve = B.state.map(state=>bind(D,P.gatherObserve,{0:B.config,1:state,2:B.observe}));
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

  _writeConfig(active = 0, reset = 0, selected = undefined, dt = 0, selectedField=0) {
    this.device.queue.writeBuffer(this._buffers.config, 0, configBytes(this.capacity, active, reset, selected, dt, selectedField));
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
      const command = this.device.createCommandEncoder({ label: 'MaleCNS V4 reset' });
      const pass = command.beginComputePass();
      for (let index = 0; index < 2; index++) {
        dispatch(pass, this._pipelines.resetState, this._groups.resetState[index], Math.ceil(CNS_COUNTS.neurons / 256));
      }
      pass.end();
      this.device.queue.submit([command.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError) throw new Error(`MaleCNS V4 reset validation failed: ${validationError.message}`);
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

  step({ dt, activeMask, opticRGB, body, context, selectedResident, selectedField='rate' }) {
    const active = maskBits(activeMask, this.capacity);
    if (dt !== 0.01) throw new RangeError('V4 requires dt=0.01 seconds (two 0.005-second CNS substeps)');
    asFloat32(opticRGB, this.capacity * CNS_COUNTS.opticValues, 'opticRGB');
    asFloat32(body, this.capacity * CNS_COUNTS.bodyChannels, 'body');
    asFloat32(context, this.capacity * CNS_COUNTS.contextChannels, 'context');
    for (const [values, name] of [[opticRGB, 'opticRGB'], [body, 'body'], [context, 'context']]) {
      for (const value of values) if (!Number.isFinite(value)) throw new RangeError(`${name} contains a nonfinite value`);
    }
    for(const value of opticRGB) if(value<0||value>1) throw new RangeError('opticRGB must be bounded in [0,1]');
    for(const value of context) if(value < -1 || value > 1) throw new RangeError('context must be signed and bounded in [-1,1]');
    if (selectedResident !== undefined && (!Number.isInteger(selectedResident) || selectedResident < 0 || selectedResident >= this.capacity)) {
      throw new RangeError('selectedResident outside capacity');
    }
    if (!Object.hasOwn(STATE_FIELDS, selectedField)) throw new RangeError('selectedField is not a V4 neural state field');
    return this._enqueue(() => this._step(dt, active, opticRGB, body, context, selectedResident, selectedField));
  }

  async _step(dt, active, opticRGB, body, context, selectedResident, selectedField) {
    this._assertUsable();
    const D = this.device, B = this._buffers, P = this._pipelines, G = this._groups;
    this._writeConfig(active, 0, selectedResident, dt, STATE_FIELDS[selectedField]);
    D.queue.writeBuffer(B.opticInput, 0, opticRGB);
    D.queue.writeBuffer(B.bodyInput, 0, body);
    D.queue.writeBuffer(B.contextInput, 0, context);
    D.pushErrorScope('validation');
    const command = D.createCommandEncoder({ label: `MaleCNS V4 tick ${this.stepSerial}` });
    let pass = command.beginComputePass({ label: 'full MaleCNS V4' });
    dispatch(pass, P.clearDrive, G.clearDrive, Math.ceil(CNS_COUNTS.neurons / 256));
    dispatch(pass, P.mixOptic, G.mixOptic, Math.ceil(CNS_COUNTS.receptorTypes * CNS_COUNTS.opticSites / 128));
    dispatch(pass, P.scatterOptic, G.scatterOptic, Math.ceil(CNS_COUNTS.receptors / 128));
    dispatch(pass, P.encodeBody, G.encodeBody, Math.ceil(CNS_COUNTS.bodyTargets / 128));
    dispatch(pass, P.injectContext, G.injectContext, Math.ceil(CNS_COUNTS.contextTargets / 128));
    let state = this._currentState;
    for (let substep = 0; substep < 2; substep++) {
      dispatch(pass, P.recurrentSum, G.recurrentSum[state], Math.ceil(CNS_COUNTS.neurons / 256));
      dispatch(pass, P.jacobiUpdate, G.jacobiUpdate[state], Math.ceil(CNS_COUNTS.neurons / 256));
      state = 1 - state;
    }
    dispatch(pass, P.finalizeTick, G.finalizeTick[state], Math.ceil(CNS_COUNTS.neurons / 256));
    dispatch(pass, P.projectPartial, G.projectPartial[state], PROJECTION_BLOCKS, CNS_COUNTS.rank);
    dispatch(pass, P.reduceProjection, G.reduceProjection, CNS_COUNTS.rank);
    dispatch(pass, P.outputLatent, G.outputLatent, Math.ceil(CNS_COUNTS.latent / 128));
    dispatch(pass, P.outputMotor, G.outputMotor[state], Math.ceil(CNS_COUNTS.motor / 64));
    pass.end();
    if(selectedResident!==undefined){
      pass=command.beginComputePass({label:'CNS V4 observer gather'});
      dispatch(pass,P.gatherObserve,G.gatherObserve[state],Math.ceil(CNS_COUNTS.neurons/128));
      pass.end();
    }
    command.copyBufferToBuffer(B.latent, 0, B.latentRead, 0, LATENT_BYTES);
    command.copyBufferToBuffer(B.motor, 0, B.latentRead, LATENT_BYTES, MOTOR_BYTES);
    if (selectedResident !== undefined) command.copyBufferToBuffer(B.observe,0,B.observeRead,0,CNS_COUNTS.neurons*8);
    try {
      D.queue.submit([command.finish()]);
      const validationError = await D.popErrorScope();
      if (validationError) throw new Error(`MaleCNS V4 tick validation failed: ${validationError.message}`);
      await B.latentRead.mapAsync(GPUMapMode.READ);
      const mappedOutputs=B.latentRead.getMappedRange();
      const gpuLatent = new Float32Array(mappedOutputs,0,LATENT_BYTES/4).slice();
      const gpuMotor = new Float32Array(mappedOutputs,LATENT_BYTES,MOTOR_BYTES/4).slice();
      B.latentRead.unmap();
      const latent = new Float32Array(this.capacity * CNS_COUNTS.latent);
      for (let output = 0; output < CNS_COUNTS.latent; output++) {
        for (let lane = 0; lane < this.capacity; lane++) latent[lane * CNS_COUNTS.latent + output] = gpuLatent[output * 4 + lane];
      }
      const motor = new Float32Array(this.capacity * CNS_COUNTS.motor);
      for(let output=0;output<CNS_COUNTS.motor;output++) for(let lane=0;lane<this.capacity;lane++) motor[lane*CNS_COUNTS.motor+output]=gpuMotor[output*4+lane];
      for(const value of latent) if(!Number.isFinite(value)) throw new Error('CNS latent output is nonfinite');
      for(const value of motor) if(!Number.isFinite(value)) throw new Error('CNS motor output is nonfinite');
      let selectedRates;
      let selectedSignal;
      if (selectedResident !== undefined) {
        await B.observeRead.mapAsync(GPUMapMode.READ);
        const gpuState=new Float32Array(B.observeRead.getMappedRange()).slice();
        B.observeRead.unmap();
        selectedRates = new Float32Array(CNS_COUNTS.neurons);
        selectedSignal = selectedField === 'rate' ? selectedRates : new Float32Array(CNS_COUNTS.neurons);
        for (let neuron = 0; neuron < CNS_COUNTS.neurons; neuron++) {
          selectedRates[neuron] = gpuState[neuron*2];
          if(selectedSignal!==selectedRates) selectedSignal[neuron]=gpuState[neuron*2+1];
        }
      }
      this._currentState = state;
      for (let lane = 0; lane < this.capacity; lane++) if ((active & (1 << lane)) !== 0) this.times[lane] += dt;
      this.stepSerial++;
      return { latent, motor, selectedRates, selectedSignal, selectedField, times: this.times.slice(), stepSerial: this.stepSerial };
    } catch (error) {
      if (B.latentRead.mapState === 'mapped') B.latentRead.unmap();
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
    const command = this.device.createCommandEncoder({ label: 'MaleCNS V4 snapshot' });
    command.copyBufferToBuffer(B.state[this._currentState], 0, B.snapshotRead, 0, STATE_BYTES);
    this.device.queue.submit([command.finish()]);
    await B.snapshotRead.mapAsync(GPUMapMode.READ);
    const state = new Uint8Array(B.snapshotRead.getMappedRange()).slice();
    B.snapshotRead.unmap();
    const metadata = encoder.encode(JSON.stringify({
      format: 'chreatures-cns-webgpu-state-v4',
      identity: this.identity,
      capacity: this.capacity,
      times: Array.from(this.times),
      identities: this.identities,
      stepSerial: this.stepSerial,
      stateBytes: STATE_BYTES,
      stateLayout: 'neuron-major rate/adaptation/support/release/mDA/mOA/mHT vec4 f32-le',
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
    if (metadata.format !== 'chreatures-cns-webgpu-state-v4' || metadata.capacity !== this.capacity ||
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
        const rate = floats[first + lane], adaptation = floats[first + 4 + lane], support = floats[first + 8 + lane],
          release = floats[first + 12 + lane], da=floats[first+16+lane], oa=floats[first+20+lane], ht=floats[first+24+lane];
        if (!Number.isFinite(rate) || rate < 0 || rate > 1 || !Number.isFinite(adaptation) || Math.abs(adaptation) > 1 ||
            !Number.isFinite(support) || support < 0.65 || support > 1 || !Number.isFinite(release) || release<0.2 || release>1 ||
            !Number.isFinite(da)||!Number.isFinite(oa)||!Number.isFinite(ht)) throw new Error(`snapshot contains invalid neural state at ${neuron}/${lane}: ${rate},${adaptation},${support},${release},${da},${oa},${ht}`);
      }
    }
    const stateBytes = bytes.subarray(stateOffset);
    let upload;
    try {
      upload=createBuffer(this.device,'CNS V4 restore staging',STATE_BYTES,GPUBufferUsage.COPY_SRC,stateBytes);
      const command=this.device.createCommandEncoder({label:'CNS V4 restore'});
      command.copyBufferToBuffer(upload,0,this._buffers.state[0],0,STATE_BYTES);
      command.copyBufferToBuffer(upload,0,this._buffers.state[1],0,STATE_BYTES);
      this.device.queue.submit([command.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      this._poisoned = error instanceof Error ? error : new Error(String(error));
      throw this._poisoned;
    } finally {
      upload?.destroy();
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
