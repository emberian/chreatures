#!/usr/bin/env node
// Dawn probe for the private V5 selected-edge state and dispatch chronology.

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
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
if (!args.model) {
  console.error('usage: node probe-cns-v5-plasticity.mjs --model V5_PACK [--report receipt.json]');
  process.exit(2);
}

const N = 165122, E = 4184, STATE_STRIDE = 28, STATE_FLOATS = N * STATE_STRIDE;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const toArrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const exact = (left, right) => Buffer.from(left).equals(Buffer.from(right));

function snapshotViews(snapshot) {
  const metadataLength = new DataView(snapshot).getUint32(8, true);
  const stateOffset = 12 + Math.ceil(metadataLength / 4) * 4;
  const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(snapshot, 12, metadataLength)));
  const state = new Float32Array(snapshot, stateOffset, STATE_FLOATS);
  const efficacy = new Float32Array(snapshot, stateOffset + STATE_FLOATS * 4, E * 4);
  const eligibility = new Float32Array(snapshot, stateOffset + (STATE_FLOATS + E * 4) * 4, E * 4);
  return { metadata, state, efficacy, eligibility };
}

function inactiveLaneEqual(before, after, lane) {
  const left = snapshotViews(before), right = snapshotViews(after);
  for (let neuron = 0; neuron < N; neuron++) {
    for (let field = 0; field < 7; field++) {
      const index = neuron * STATE_STRIDE + field * 4 + lane;
      if (left.state[index] !== right.state[index]) return false;
    }
  }
  for (let edge = 0; edge < E; edge++) {
    if (left.efficacy[edge * 4 + lane] !== right.efficacy[edge * 4 + lane] ||
        left.eligibility[edge * 4 + lane] !== right.eligibility[edge * 4 + lane]) return false;
  }
  return true;
}

const modelDirectory = resolve(args.model);
const manifestBytes = await readFile(resolve(modelDirectory, 'cns-manifest.json'));
const manifest = JSON.parse(manifestBytes);
if (manifest.format !== 'chreatures-cns-webgpu-v5' || manifest.version !== 5 ||
    manifest.identity?.plasticity !== '1eb37790b406573b191e4493a5d1ddb0e5ece8bb5edd2a377f2921adab971b87') {
  throw new Error('probe requires the frozen initial V5 plasticity identity');
}
const assets = new Map();
for (const [name, entry] of Object.entries(manifest.buffers)) {
  const transport = await readFile(resolve(modelDirectory, entry.url));
  if (transport.byteLength !== (entry.transportByteLength ?? entry.byteLength) ||
      sha256(transport) !== (entry.transportSha256 ?? entry.sha256)) {
    throw new Error(`packed asset authentication failed: ${name}`);
  }
  const raw = entry.encoding === 'gzip' ? gunzipSync(transport) : transport;
  if (raw.byteLength !== entry.byteLength || sha256(raw) !== entry.sha256) {
    throw new Error(`raw asset authentication failed: ${name}`);
  }
  assets.set(name, toArrayBuffer(raw));
}
const bodyMean = new Float32Array(assets.get('body.mean')).slice();
const positions = new Uint32Array(assets.get('plasticity.edge_positions'));
const graphCol = new Uint32Array(assets.get('graph.col'));
const target0Sources = new Set(Array.from(positions.subarray(0, 2048), position => graphCol[position]));
const shaderNames = ['afferent.wgsl', 'dynamics.wgsl', 'plasticity.wgsl', 'readout.wgsl', 'motor.wgsl', 'observe.wgsl'];
const shaderSources = Object.fromEntries(await Promise.all(shaderNames.map(async name => [
  name, await readFile(resolve(liveSource, 'shaders', name), 'utf8'),
])));

const dawn = create(['backend=metal']);
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('Dawn found no Metal WebGPU adapter');
const needed = Math.max(...Object.values(manifest.buffers).map(entry => entry.byteLength));
const device = await adapter.requestDevice({ requiredLimits: {
  maxBufferSize: Math.max(needed, 128 * 1024 * 1024),
  maxStorageBufferBindingSize: Math.max(needed, 128 * 1024 * 1024),
}});
const { MaleCNSWebGPU } = await import(pathToFileURL(resolve(liveSource, 'cns-webgpu.js')));
const brain = await MaleCNSWebGPU.load({ device, manifest, capacity: 2, assetBuffers: assets, shaderSources });
assets.clear();
await brain.reset([0, 1], ['plastic-probe-0', 'plastic-probe-1']);

const optic = new Float32Array(2 * 5313).fill(0.5);
const body = new Float32Array(2 * 807); body.set(bodyMean); body.set(bodyMean, 807);
const context = new Float32Array(24);
const inputs = { dt: 0.01, activeMask: 1, opticRGB: optic, body, context, selectedResident: 0 };
const initialPrivate = await brain.inspectPlasticity(0);
if (initialPrivate.efficacy.some(value => value !== 0) || initialPrivate.eligibility.some(value => value !== 0)) {
  throw new Error('private plastic state is nonzero after reset');
}

// Research-only state intervention: both branches receive identical elevated
// target-0 KC rates/release; only the efficacy deviation differs.
const initial = await brain.snapshot();
const branchZero = initial.slice(0), branchPlastic = initial.slice(0);
const baselineRates = brain.baselineRates;
for (const snapshot of [branchZero, branchPlastic]) {
  const view = snapshotViews(snapshot);
  for (const source of target0Sources) {
    view.state[source * STATE_STRIDE] = Math.min(0.99, baselineRates[source] + 0.4);
    view.state[source * STATE_STRIDE + 12] = 1;
  }
}
const plasticView = snapshotViews(branchPlastic);
for (let edge = 0; edge < 2048; edge++) plasticView.efficacy[edge * 4] = -0.8;

await brain.restore(branchZero);
const zeroOutput = await brain.step(inputs);
await brain.restore(branchPlastic);
const beforePlasticStep = await brain.snapshot();
const plasticOutput = await brain.step(inputs);
const afterPlasticStep = await brain.snapshot();
const targetRateDelta = plasticOutput.selectedRates[655] - zeroOutput.selectedRates[655];
if (targetRateDelta === 0 || !Number.isFinite(targetRateDelta)) {
  throw new Error('nonzero private efficacy did not alter its measured-edge MBON target recurrence');
}
if (!inactiveLaneEqual(beforePlasticStep, afterPlasticStep, 1)) {
  throw new Error('inactive resident lane changed neuronal or plastic state');
}

// Same-runtime continuation must include both private arrays byte exactly.
const continuationStart = await brain.snapshot();
const firstOutput = await brain.step(inputs);
const firstEnd = await brain.snapshot();
await brain.restore(continuationStart);
const secondOutput = await brain.step(inputs);
const secondEnd = await brain.snapshot();
if (!exact(firstOutput.latent.buffer, secondOutput.latent.buffer) ||
    !exact(firstOutput.motor.buffer, secondOutput.motor.buffer) || !exact(firstEnd, secondEnd)) {
  throw new Error('V5 private-state snapshot continuation differs');
}

// Exact-cell chronology assay: a simultaneous cue and PPL gate enrolls
// eligibility but cannot depress on that first tick because learning consumes
// only the old decayed trace. The retained trace permits depression next tick.
const cueGate = initial.slice(0);
const cueGateView = snapshotViews(cueGate);
for (const source of target0Sources) {
  cueGateView.state[source * STATE_STRIDE] = Math.min(0.99, baselineRates[source] + 0.4);
  cueGateView.state[source * STATE_STRIDE + 12] = 1;
}
cueGateView.state[1774 * STATE_STRIDE] = Math.min(0.99, baselineRates[1774] + 0.4);
cueGateView.state[1774 * STATE_STRIDE + 12] = 1;
await brain.restore(cueGate);
await brain.step(inputs);
const cueGateFirst = await brain.inspectPlasticity(0);
const firstTickMaximumEligibility = Math.max(...cueGateFirst.eligibility.subarray(0, 2048));
const firstTickMinimumEfficacy = Math.min(...cueGateFirst.efficacy.subarray(0, 2048));
await brain.step(inputs);
const cueGateSecond = await brain.inspectPlasticity(0);
const secondTickMinimumEfficacy = Math.min(...cueGateSecond.efficacy.subarray(0, 2048));
if (!(firstTickMaximumEligibility > 0) || firstTickMinimumEfficacy !== 0 ||
    !(secondTickMinimumEfficacy < 0)) {
  throw new Error('cue-before-gate eligibility chronology differs');
}

// Invalid private bounds are rejected before GPU state mutation and do not
// poison a coherent life because no mutation was submitted.
const coherent = await brain.snapshot();
const invalid = coherent.slice(0);
snapshotViews(invalid).efficacy[0] = 0.01;
let invalidRejected = false;
try { await brain.restore(invalid); } catch { invalidRejected = true; }
const afterRejected = await brain.snapshot();
if (!invalidRejected || !exact(coherent, afterRejected)) {
  throw new Error('invalid private snapshot was not rejected before mutation');
}

await brain.restore(branchPlastic);
await brain.reset([0], ['plastic-probe-reset']);
const resetPrivate = await brain.inspectPlasticity(0);
const resetCleared = !resetPrivate.efficacy.some(value => value !== 0) &&
  !resetPrivate.eligibility.some(value => value !== 0);
if (!resetCleared) throw new Error('resident reset did not clear private plastic state');

const report = {
  format: 'chreatures-cns-webgpu-v5-plasticity-dawn-probe-v1',
  status: 'executed',
  manifestSha256: sha256(manifestBytes),
  serviceArtifactSha256: manifest.serviceArtifactSha256,
  adapterIdentity: manifest.identity.artifact,
  plasticityIdentity: manifest.identity.plasticity,
  sourceRevision: manifest.sourceRevision,
  adapter: adapter.info?.device || adapter.info?.description || 'Dawn Metal adapter',
  selectedEdges: E,
  target0DistinctSources: target0Sources.size,
  privateStateBytesPerFourLaneEngine: E * 4 * 4 * 2,
  intervention: {
    scope: 'research-only exact-cell state intervention; not a sensory-learning result',
    targetRow: 655,
    targetRateZeroDeviation: zeroOutput.selectedRates[655],
    targetRateDepressedDeviation: plasticOutput.selectedRates[655],
    targetRateDifference: targetRateDelta,
  },
  cueBeforeGateIntervention: {
    scope: 'research-only exact KC/PPL state intervention; not a sensory-learning result',
    firstTickMaximumEligibility,
    firstTickMinimumEfficacy,
    secondTickMinimumEfficacy,
  },
  checks: {
    zeroInitialPrivateState: true,
    nonzeroDeviationChangesMeasuredTargetRecurrence: true,
    oldEligibilityUsedBeforeCurrentCueEnrollment: true,
    inactiveLaneByteExact: true,
    exactSameRuntimeSnapshotContinuation: true,
    invalidPrivateStateRejectedBeforeMutation: invalidRejected,
    resetClearsPrivateState: resetCleared,
  },
  sourceSha256: Object.fromEntries(await Promise.all([
    'cns-webgpu.js', ...shaderNames.map(name => `shaders/${name}`),
  ].map(async name => [name, sha256(await readFile(resolve(liveSource, name)))]))),
  probeSha256: sha256(await readFile(new URL(import.meta.url))),
};
console.log(JSON.stringify(report, null, 2));
if (args.report) await writeFile(resolve(args.report), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
brain.destroy(); device.destroy(); delete globalThis.__chreaturesDawnInstance;
