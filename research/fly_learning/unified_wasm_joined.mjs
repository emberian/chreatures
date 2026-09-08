#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
/** Headless joined check for the unified native fly-world Wasm and full CNS V4. */

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {dirname, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {openUnifiedWasmFlyWorld} from './unified_wasm_node.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), all[index + 1]]);
  return pairs;
}, []));
if (!args.model || !args.module || !args.scene) {
  throw new Error('usage: unified_wasm_joined.mjs --model CNS_PACK --module FLY_WORLD_MJS --scene WORLD_JSON [--ticks 8]');
}
const ticks = Number(args.ticks ?? 8);
if (!Number.isInteger(ticks) || ticks < 1) throw new Error('--ticks must be a positive integer');

const here = dirname(new URL(import.meta.url).pathname);
const require = createRequire(pathToFileURL(resolve(here, '../../native/webgpu-probe/package.json')));
const {create, globals} = require('webgpu');
Object.assign(globalThis, globals);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const toArrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const exact = (a, b) => Buffer.from(a.buffer, a.byteOffset, a.byteLength)
  .equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength));

const modelDirectory = resolve(args.model);
const manifestBytes = await readFile(resolve(modelDirectory, 'cns-manifest.json'));
const manifest = JSON.parse(manifestBytes);
assert.equal(manifest.format, 'chreatures-cns-webgpu-v4');
const assets = new Map();
for (const [name, entry] of Object.entries(manifest.buffers)) {
  const transported = await readFile(resolve(modelDirectory, entry.url));
  assert.equal(sha256(transported), entry.transportSha256, `${name} transport identity`);
  const raw = entry.encoding === 'gzip' ? gunzipSync(transported) : transported;
  assert.equal(sha256(raw), entry.sha256, `${name} raw identity`);
  assets.set(name, toArrayBuffer(raw));
}
const live = resolve(here, '../../site/live');
const shaderNames = ['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl', 'motor.wgsl', 'observe.wgsl'];
const shaderSources = Object.fromEntries(await Promise.all(shaderNames.map(async name => [
  name, await readFile(resolve(live, 'shaders', name), 'utf8'),
])));

const dawn = create(['backend=metal']);
globalThis.__chreaturesUnifiedDawnInstance = dawn;
const adapter = await dawn.requestAdapter({powerPreference: 'high-performance'});
assert(adapter, 'Dawn found no Metal adapter');
const largest = Math.max(...Object.values(manifest.buffers).map(entry => entry.byteLength));
const device = await adapter.requestDevice({requiredLimits: {
  maxBufferSize: Math.max(largest, 128 * 1024 * 1024),
  maxStorageBufferBindingSize: Math.max(largest, 128 * 1024 * 1024),
}});
const {MaleCNSWebGPU} = await import(pathToFileURL(resolve(live, 'cns-webgpu.js')));
const brain = await MaleCNSWebGPU.load({device, manifest, capacity: 4, assetBuffers: assets, shaderSources});
assets.clear();
await brain.reset([0, 1, 2, 3], ['fly-0', 'fly-1', 'fly-2', 'fly-3']);
const world = await openUnifiedWasmFlyWorld({modulePath: args.module, scene: args.scene, seed: 23});
assert.equal(world.residents, 4);

const context = new Float32Array(4 * 12);
const timings = {sample: [], cns: [], advance: []};
let output;
try {
  for (let tick = 0; tick < ticks; tick++) {
    let began = performance.now();
    const sensory = world.sample();
    timings.sample.push(performance.now() - began);
    began = performance.now();
    output = await brain.step({dt: 0.01, activeMask: 0b1111, opticRGB: sensory.optic, body: sensory.body, context});
    timings.cns.push(performance.now() - began);
    assert(output.latent.every(Number.isFinite) && output.motor.every(Number.isFinite));
    began = performance.now();
    world.advance(output.motor, 0.01);
    timings.advance.push(performance.now() - began);
  }
  const worldSnapshot = world.snapshot();
  const brainSnapshot = await brain.snapshot();
  const sensory = world.sample();
  const first = await brain.step({dt: 0.01, activeMask: 0b1111, opticRGB: sensory.optic, body: sensory.body, context});
  world.advance(first.motor, 0.01);
  const firstState = world.observe();
  world.restore(worldSnapshot);
  await brain.restore(brainSnapshot);
  const sensoryReplay = world.sample();
  const replay = await brain.step({dt: 0.01, activeMask: 0b1111, opticRGB: sensoryReplay.optic, body: sensoryReplay.body, context});
  world.advance(replay.motor, 0.01);
  const replayState = world.observe();
  assert(exact(first.motor, replay.motor), 'CNS motor replay differs');
  assert(exact(first.latent, replay.latent), 'CNS latent replay differs');
  assert(exact(firstState.qpos, replayState.qpos), 'physical qpos replay differs');
  assert(exact(firstState.qvel, replayState.qvel), 'physical qvel replay differs');
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  console.log(JSON.stringify({
    format: 'chreatures-unified-native-wasm-cns-v4-joined-v1',
    ticks,
    residents: world.residents,
    modelSeconds: firstState.time,
    cnsManifestSha256: sha256(manifestBytes),
    cnsServiceSha256: manifest.serviceArtifactSha256,
    unifiedModuleSha256: sha256(await readFile(resolve(args.module))),
    unifiedWasmSha256: sha256(await readFile(resolve(dirname(args.module), 'chreatures-fly-world.wasm'))),
    sceneFixtureSha256: world.metadata.fixture_sha256,
    finiteSensoryMotorLatent: true,
    cnsReplayExact: true,
    physicalReplayExact: true,
    meanMilliseconds: Object.fromEntries(Object.entries(timings).map(([name, values]) => [name, mean(values)])),
    scope: 'Actual MuJoCo 3.12 + Rust world/core in one Emscripten module, actual Dawn Metal full CNS V4; initialized state, no competence claim',
  }, null, 2));
} finally {
  world.close();
  brain.destroy();
  device.destroy();
}
