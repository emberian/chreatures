#!/usr/bin/env node
// Headless production-shader probe using Dawn's official `webgpu` Node package.
// It runs the packed full MaleCNS graph; no browser automation is involved.

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync, inflateRawSync } from 'node:zlib';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const here = dirname(new URL(import.meta.url).pathname);
const liveSource = resolve(here, '../../site/live');
const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), all[index + 1]]);
  return pairs;
}, []));
if (!args.model || !args.fixture) {
  console.error('usage: node probe-cns-webgpu.mjs --model PACK_DIRECTORY --fixture torch-v4-fixture.npz [--report receipt.json]');
  process.exit(2);
}

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const toArrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

function unzipEntries(bytes) {
  // Minimal ZIP central-directory reader, sufficient for NumPy's compressed NPZ.
  let eocd = -1;
  for (let at = bytes.length - 22; at >= Math.max(0, bytes.length - 65557); at--) {
    if (bytes.readUInt32LE(at) === 0x06054b50) { eocd = at; break; }
  }
  if (eocd < 0) throw new Error('NPZ end-of-central-directory missing');
  const count = bytes.readUInt16LE(eocd + 10);
  let cursor = bytes.readUInt32LE(eocd + 16);
  const entries = new Map();
  for (let index = 0; index < count; index++) {
    if (bytes.readUInt32LE(cursor) !== 0x02014b50) throw new Error('invalid NPZ central directory');
    const method = bytes.readUInt16LE(cursor + 10);
    const compressedSize = bytes.readUInt32LE(cursor + 20);
    const nameLength = bytes.readUInt16LE(cursor + 28);
    const extraLength = bytes.readUInt16LE(cursor + 30);
    const commentLength = bytes.readUInt16LE(cursor + 32);
    const local = bytes.readUInt32LE(cursor + 42);
    const name = bytes.subarray(cursor + 46, cursor + 46 + nameLength).toString('utf8');
    if (bytes.readUInt32LE(local) !== 0x04034b50) throw new Error('invalid NPZ local header');
    const localName = bytes.readUInt16LE(local + 26);
    const localExtra = bytes.readUInt16LE(local + 28);
    const start = local + 30 + localName + localExtra;
    const packed = bytes.subarray(start, start + compressedSize);
    entries.set(name, method === 8 ? inflateRawSync(packed) : Buffer.from(packed));
    cursor += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

function parseNpy(bytes) {
  if (bytes.subarray(0, 6).toString('latin1') !== '\x93NUMPY') throw new Error('invalid NPY member');
  const major = bytes[6];
  const headerLength = major === 1 ? bytes.readUInt16LE(8) : bytes.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = bytes.subarray(headerStart, headerStart + headerLength).toString('latin1');
  const descr = /['"]descr['"]:\s*['"]([^'"]+)/.exec(header)?.[1];
  const shapeText = /['"]shape['"]:\s*\(([^)]*)\)/.exec(header)?.[1];
  const fortran = /['"]fortran_order['"]:\s*(True|False)/.exec(header)?.[1];
  if ((descr !== '<f4' && !/^<U[0-9]+$/.test(descr)) || fortran !== 'False' || shapeText === undefined) throw new Error(`unsupported NPY layout: ${header}`);
  const shape = shapeText.split(',').map(value => value.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((total, value) => total * value, 1);
  const payload = bytes.subarray(headerStart + headerLength);
  if (descr.startsWith('<U')) {
    const width = Number(descr.slice(2));
    if (count !== 1 || payload.byteLength !== width * 4) throw new Error('Expected scalar Unicode metadata');
    let value = '';
    for (let offset=0;offset<payload.byteLength;offset+=4) { const code=payload.readUInt32LE(offset); if(code) value += String.fromCodePoint(code); }
    return {shape,value};
  }
  if (payload.byteLength !== count * 4) throw new Error('NPY payload length differs');
  return { shape, values: new Float32Array(toArrayBuffer(payload)) };
}

function maximumError(actual, expected, expectedIndex = index => index) {
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    const value=actual[index], reference=expected[expectedIndex(index)];
    if (!Number.isFinite(value) || !Number.isFinite(reference)) throw new Error('Nonfinite parity value');
    maximum = Math.max(maximum, Math.abs(value - reference));
  }
  return maximum;
}

function splitSensory(values, tick, batch) {
  const stride = 6120;
  const optic = new Float32Array(batch * 5313);
  const body = new Float32Array(batch * 807);
  for (let resident = 0; resident < batch; resident++) {
    const start = (tick * batch + resident) * stride;
    optic.set(values.subarray(start, start + 5313), resident * 5313);
    body.set(values.subarray(start + 5313, start + stride), resident * 807);
  }
  return { optic, body };
}

function snapshotState(snapshot) {
  const metadataLength = new DataView(snapshot).getUint32(8, true);
  const offset = 12 + Math.ceil(metadataLength / 4) * 4;
  return new Float32Array(snapshot, offset);
}

const modelDirectory = resolve(args.model);
const manifestBytes = await readFile(resolve(modelDirectory, 'cns-manifest.json'));
const manifest = JSON.parse(manifestBytes);
const fixtureBytes = await readFile(resolve(args.fixture));
const npz = unzipEntries(fixtureBytes);
const fixtureMetadata = JSON.parse(parseNpy(npz.get('metadata_json.npy')).value);
if (fixtureMetadata.format !== 'chreatures-cns-v4-parity-v1' || fixtureMetadata.service_file_sha256 !== manifest.serviceArtifactSha256 || fixtureMetadata.adapter_sha256 !== manifest.identity.artifact || fixtureMetadata.dt !== .01 || fixtureMetadata.substeps !== 2 || fixtureMetadata.substep_dt !== .005 || fixtureMetadata.ticks !== 3 || fixtureMetadata.batch !== 1) throw new Error('V4 fixture identity or clock differs');
const fixture = Object.fromEntries([...npz].filter(([name])=>name!=='metadata_json.npy').map(([name, bytes]) => [name.replace(/\.npy$/, ''), parseNpy(bytes)]));
const stateKeys=['rates','adaptation','support','release','mod_da','mod_oa','mod_ht'];
for(const key of stateKeys) if(fixture['initial.'+key]?.shape.join(',') !== '165122,1' || fixture['state.'+key]?.shape.join(',') !== '3,165122,1') throw new Error('V4 private state shape differs: '+key);
function stateErrors(snapshot,tick){
 const state=snapshotState(snapshot);
 return Object.fromEntries(stateKeys.map((key,field)=>{
  const expected=fixture[(tick===null?'initial.':'state.')+key].values;
  let maximum=0;
  for(let neuron=0;neuron<165122;neuron++){const actual=state[neuron*28+field*4],reference=expected[(tick??0)*165122+neuron];if(!Number.isFinite(actual)||!Number.isFinite(reference))throw Error('Nonfinite state '+key);maximum=Math.max(maximum,Math.abs(actual-reference));}
  return[key,maximum];
 }));
}
const assets = new Map();
for (const [name, entry] of Object.entries(manifest.buffers)) {
  const transport = await readFile(resolve(modelDirectory, entry.url));
  if (transport.byteLength !== entry.transportByteLength || sha256(transport) !== entry.transportSha256) {
    throw new Error(`packed asset authentication failed: ${name}`);
  }
  const raw = entry.encoding === 'gzip' ? gunzipSync(transport) : transport;
  if (raw.byteLength !== entry.byteLength || sha256(raw) !== entry.sha256) {
    throw new Error(`raw asset authentication failed: ${name}`);
  }
  assets.set(name, toArrayBuffer(raw));
}
const bodyMean = new Float32Array(assets.get('body.mean')).slice();
const shaderSources = Object.fromEntries(await Promise.all(['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl', 'motor.wgsl','observe.wgsl'].map(async name => [
  name, await readFile(resolve(liveSource, 'shaders', name), 'utf8'),
])));

const dawn = create(['backend=metal']);
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('Dawn found no Metal WebGPU adapter');
const needed = Math.max(...Object.values(manifest.buffers).map(entry => entry.byteLength));
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: Math.max(needed, 128 * 1024 * 1024),
    maxStorageBufferBindingSize: Math.max(needed, 128 * 1024 * 1024),
  },
});
const { MaleCNSWebGPU } = await import(pathToFileURL(resolve(liveSource, 'cns-webgpu.js')));
const engine = await MaleCNSWebGPU.load({ device, manifest, capacity: 1, assetBuffers: assets, shaderSources });
console.error('probe: engine loaded');
assets.clear();

// The sealed neutral drive must cancel the adapter at the defined operating input.
const neutralOptic = new Float32Array(5313).fill(0.5);
const neutralBody = bodyMean.slice();
const neutralContext = new Float32Array(12);
const neutral = await engine.step({ dt: 0.01, activeMask: 1, opticRGB: neutralOptic, body: neutralBody, context:neutralContext, selectedResident: 0 });
console.error('probe: neutral tick complete');
const neutralRateError = maximumError(neutral.selectedRates, engine.baselineRates);
await engine.reset([0], ['fixture-0']);

const initialStateErrors = stateErrors(await engine.snapshot(),null);
if (fixture.sensory.shape.join(',') !== '3,1,6120' || fixture.context.shape.join(',') !== '3,1,12' ||
    fixture['state.rates'].shape.join(',') !== '3,165122,1' || fixture.latent.shape.join(',') !== '3,1,512' ||
    fixture.motor.shape.join(',') !== '3,1,92') throw new Error('unexpected canonical V4 fixture dimensions');
const tickRateErrors = [], tickLatentErrors = [];
const tickMotorErrors=[], tickStateErrors=[];
let beforeLast, firstLast, firstMotor, firstFinal;
for (let tick = 0; tick < 3; tick++) {
  if (tick === 2) { beforeLast = await engine.snapshot(); console.error('probe: pre-final snapshot complete'); }
  const input = splitSensory(fixture.sensory.values, tick, 1);
  const context=fixture.context.values.slice(tick*12,(tick+1)*12);
  const output = await engine.step({ dt: 0.01, activeMask: 1, opticRGB: input.optic, body: input.body, context, selectedResident:0 });
  console.error(`probe: fixture tick ${tick} complete`);
  for(let neuron=0;neuron<output.selectedRates.length;neuron++) if(!Number.isFinite(output.selectedRates[neuron])) throw new Error(`nonfinite GPU rate at tick ${tick}, neuron ${neuron}`);
  tickLatentErrors.push(maximumError(output.latent, fixture.latent.values,
    index => tick * 512 + index));
  tickRateErrors.push(maximumError(output.selectedRates, fixture['state.rates'].values, neuron => tick * 165122 + neuron));
  tickMotorErrors.push(maximumError(output.motor,fixture.motor.values,index=>tick*92+index));
  tickStateErrors.push(stateErrors(await engine.snapshot(),tick));
  if (tick === 2) { firstLast = output.latent.slice(); firstMotor=output.motor.slice(); firstFinal = await engine.snapshot(); console.error('probe: final snapshot complete'); }
}

const finalStateErrors=tickStateErrors.at(-1);
console.error('probe: seven-state comparison complete');

await engine.restore(beforeLast);
console.error('probe: restore complete');
const lastInput = splitSensory(fixture.sensory.values, 2, 1);
const lastContext=fixture.context.values.slice(2*12,3*12);
const secondLast = await engine.step({ dt: 0.01, activeMask: 1, opticRGB: lastInput.optic, body: lastInput.body, context:lastContext });
const secondFinal = await engine.snapshot();
const restoreLatentExact = maximumError(firstLast, secondLast.latent) === 0;
const restoreMotorExact = maximumError(firstMotor, secondLast.motor) === 0;
const restoreStateExact = Buffer.from(firstFinal).equals(Buffer.from(secondFinal));

const report = {
  format: 'chreatures-cns-webgpu-v4-dawn-probe-v1',
  manifestArtifact: manifest.identity.artifact,
  graph: manifest.identity.graph,
  serviceArtifactSha256: manifest.serviceArtifactSha256,
  fixtureSha256: sha256(fixtureBytes),
  fixtureMetadata,
  manifestSha256: sha256(manifestBytes),
  identity: manifest.identity,
  graphQuantization: manifest.graphQuantization,
  readoutProjectionDtype: manifest.buffers['readout.projection.weight'].dtype,
  sourceRevision: manifest.sourceRevision,
  sourceSha256: Object.fromEntries(await Promise.all(['cns-webgpu.js','assets.js',...Object.keys(shaderSources).map(name=>'shaders/'+name)].map(async name=>[name,sha256(await readFile(resolve(liveSource,name)))]))),
  probeSha256:sha256(await readFile(new URL(import.meta.url))),
  adapter: { name: adapter.info?.device || adapter.info?.description || 'Dawn Metal adapter' },
  capacity: 1,
  ticks: 3,
  neutralRateMaxAbs: neutralRateError,
  rateMaxAbsByTick: tickRateErrors,
  latentMaxAbsByTick: tickLatentErrors,
  motorMaxAbsByTick: tickMotorErrors,
  initialMaxAbs:initialStateErrors,
  stateMaxAbsByTick:tickStateErrors,
  finalMaxAbs: finalStateErrors,
  byteExactRestore: { latent: restoreLatentExact, motor:restoreMotorExact, snapshot: restoreStateExact },
  limits: { neutralRate: 2e-6, state: 2e-5, latent: 2e-5, motor: 2e-5 },
  limitBasis: 'Predeclared absolute cross-backend regression gate 2e-5; same canonical graph bits and float32 learned tensors. No biological tolerance or competence claim.',
  note: 'Full V4 graph decoded from identical canonical binary16 bits; readout float32; explicit centered/scaled signed MN decoder; seven private states compared every tick.',
  nodeDawnLifetime: {
    requirement: 'The object returned by create() remains strongly referenced for the full GPU lifetime.',
    upstream: 'https://github.com/dawn-gpu/node-webgpu#lifetime',
    issue: 'https://issues.chromium.org/issues/387965810',
    diagnosis: 'Without the strong reference, lldb stopped in dawn::native::InstanceBase::ProcessEvents at std::mutex::lock after successful ticks.',
  },
};
report.passed = neutralRateError < 2e-6 &&
  Math.max(...Object.values(initialStateErrors), ...tickStateErrors.flatMap(x=>Object.values(x))) < 2e-5 &&
  Math.max(...tickLatentErrors) < 2e-5 && Math.max(...tickMotorErrors) < 2e-5 &&
  restoreLatentExact && restoreMotorExact && restoreStateExact;
console.log(JSON.stringify(report, null, 2));
if (args.report) await writeFile(resolve(args.report), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
engine.destroy();
device.destroy();
if (!report.passed) process.exitCode = 1;
