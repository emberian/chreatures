#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
/** Matched native-process versus unified-Emscripten fly-world chronology. */

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {openUnifiedWasmFlyWorld} from './unified_wasm_node.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), all[index + 1]]);
  return pairs;
}, []));
if (!args.native || !args.module || !args.scene) {
  throw new Error('usage: compare_unified_wasm.mjs --native BINARY --module MJS --scene WORLD_JSON [--output RECEIPT]');
}
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const exact = (a, b) => Buffer.from(a).equals(Buffer.from(b));
const decode = value => {
  const bytes = Buffer.from(value.base64, 'base64');
  const view = value.dtype === '<f8'
    ? new Float64Array(bytes.buffer, bytes.byteOffset, value.length)
    : new Float32Array(bytes.buffer, bytes.byteOffset, value.length);
  return view.slice();
};
const encodeF32 = values => Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString('base64');

class NativeProcess {
  constructor(binary, scene) {
    this.process = spawn(resolve(binary), ['--scene', resolve(scene), '--seed', '20260908'], {
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    this.lines = createInterface({input: this.process.stdout})[Symbol.asyncIterator]();
    this.id = 0;
  }
  async read() {
    const next = await this.lines.next();
    if (next.done) throw new Error(`native host exited ${this.process.exitCode}`);
    return JSON.parse(next.value);
  }
  async request(command, payload = {}) {
    const id = ++this.id;
    this.process.stdin.write(`${JSON.stringify({id, command, ...payload})}\n`);
    const response = await this.read();
    if (response.id !== id || !response.ok) throw new Error(`native ${command}: ${response.error}`);
    return response;
  }
  async close() {
    if (this.process.exitCode === null) await this.request('close');
  }
}

class ErrorStats {
  constructor() { this.maximum = 0; this.square = 0; this.count = 0; }
  add(a, b) {
    assert.equal(a.length, b.length);
    for (let index = 0; index < a.length; index++) {
      const error = Math.abs(a[index] - b[index]);
      assert(Number.isFinite(error));
      this.maximum = Math.max(this.maximum, error);
      this.square += error * error;
      this.count++;
    }
  }
  json() { return {max_abs: this.maximum, rms: Math.sqrt(this.square / this.count), values: this.count}; }
}

const native = new NativeProcess(args.native, args.scene);
const ready = await native.read();
assert(ready.ok && ready.event === 'ready');
const wasm = await openUnifiedWasmFlyWorld({modulePath: args.module, scene: args.scene, seed: 20260908});
assert.equal(ready.fixture_sha256, wasm.metadata.fixture_sha256);
const stats = Object.fromEntries(['optic', 'body', 'qpos', 'qvel', 'bodyPositions', 'bodyQuaternions',
  'bodyRotations', 'sensordata', 'ctrl', 'entityPosition'].map(name => [name, new ErrorStats()]));
const mapping = {optic: 'optic', body: 'body', qpos: 'qpos', qvel: 'qvel', bodyPositions: 'body_positions',
  bodyQuaternions: 'body_quaternions', bodyRotations: 'body_rotations', sensordata: 'sensor_data',
  ctrl: 'controls', entityPosition: 'entity_positions'};
let nativeInsertion, wasmInsertion;
const timing = {nativeAdvance: [], wasmAdvance: [], nativeObserve: [], wasmObserve: []};

function motorAt(tick) {
  const motor = new Float32Array(4 * 92);
  for (let lane = 0; lane < 4; lane++) for (let channel = 0; channel < 92; channel++) {
    motor[lane * 92 + channel] = channel < 84
      ? 0.025 * Math.sin((tick + channel * 3 + lane * 11) * 0.03125)
      : 0.05;
  }
  return motor;
}
const frameAt = tick => Float32Array.from({length: 12}, (_, index) => ((tick / 64 + index) % 4) / 3);
async function events(tick) {
  if (tick % 64 === 0) {
    const frame = frameAt(tick), screenBytes = Buffer.from(frame.buffer);
    await native.request('stimulus', {screen: {width: 2, height: 2, rgb_f32_base64: screenBytes.toString('base64'), sha256: sha256(screenBytes)},
      sound: {position_mm: [0, 0, 8], frequency_hz: [80, 200, 500, 1250][tick / 64], envelope: .4, duration_s: .15}});
    wasm.setScreen(frame, 2, 2);
    wasm.visitorSound([0, 0, 8], [80, 200, 500, 1250][tick / 64], .4, .15);
  }
  if (tick === 8) {
    const request = {position: [12, -12, 8], size: [.07, .07, .07], shape: 'box', rgba: [.6, .3, .1, 1], food: .2, odor: 1};
    nativeInsertion = (await native.request('insert_object', request)).receipt;
    wasmInsertion = wasm.insertObject(request);
    assert.equal(nativeInsertion.id, wasmInsertion.id);
    assert.equal(nativeInsertion.model, wasmInsertion.model);
  }
  if (tick === 32) {
    const count = wasm.observe().ecology.native_host.route_open_fraction.length;
    const open = Array.from({length: count}, (_, index) => ((index % 17) + 1) / 17);
    const flow = Array(count).fill(0);
    await native.request('routes', {open_fraction: open, advection_m3_s: flow});
    wasm.setRoutes(open, flow);
  }
  if (tick === 80) {
    await native.request('visitor_force', {entity_id: nativeInsertion.id, force: [.1, 0, 0]});
    wasm.queueVisitorForce(wasmInsertion.id, [.1, 0, 0]);
  }
}

async function step(tick, accumulate) {
  await events(tick);
  const motor = motorAt(tick);
  let began = performance.now();
  await native.request('advance', {motor92_base64: encodeF32(motor)});
  if (accumulate) timing.nativeAdvance.push(performance.now() - began);
  began = performance.now();
  wasm.advance(motor, .01);
  if (accumulate) timing.wasmAdvance.push(performance.now() - began);
  began = performance.now();
  const nativePacket = await native.request('sample');
  if (accumulate) timing.nativeObserve.push(performance.now() - began);
  began = performance.now();
  const wasmPacket = wasm.observe();
  if (accumulate) timing.wasmObserve.push(performance.now() - began);
  const sample = nativePacket.sample;
  if (accumulate) for (const [nativeName, wasmName] of Object.entries(mapping)) {
    stats[nativeName].add(decode(sample[nativeName]), wasmPacket[wasmName]);
  }
  assert.equal(sample.ecology.native_host.topology_revision, wasmPacket.ecology.native_host.topology_revision);
  assert.deepEqual(sample.ecology.native_host.growth, wasmPacket.ecology.native_host.growth);
  return {native: sample, wasm: wasmPacket};
}

let nativeSnapshot, wasmSnapshot, replayExact = false, final;
try {
  for (let tick = 0; tick < 200; tick++) {
    if (tick === 100) {
      nativeSnapshot = Buffer.from((await native.request('snapshot')).snapshot_base64, 'base64');
      wasmSnapshot = wasm.snapshot();
      for (let branch = 0; branch < 20; branch++) await step(tick + branch, false);
      const nativeFuture = Buffer.from((await native.request('snapshot')).snapshot_base64, 'base64');
      const wasmFuture = wasm.snapshot();
      await native.request('restore', {snapshot_base64: nativeSnapshot.toString('base64')});
      wasm.restore(wasmSnapshot);
      for (let branch = 0; branch < 20; branch++) await step(tick + branch, false);
      const nativeReplay = Buffer.from((await native.request('snapshot')).snapshot_base64, 'base64');
      const wasmReplay = wasm.snapshot();
      replayExact = exact(nativeFuture, nativeReplay) && exact(wasmFuture, wasmReplay);
      assert(replayExact, 'whole-world continuation differs after restore');
      await native.request('restore', {snapshot_base64: nativeSnapshot.toString('base64')});
      wasm.restore(wasmSnapshot);
    }
    final = await step(tick, true);
  }
  const geometry = wasm.geometry(), geomCount = geometry.geom_body_id.length;
  assert.equal(geometry.topology_revision, final.wasm.ecology.native_host.topology_revision);
  assert.equal(geometry.geom_type.length, geomCount);
  assert.equal(geometry.geom_material_id.length, geomCount);
  assert.equal(geometry.geom_size.length, geomCount * 3);
  assert.equal(geometry.geom_position.length, geomCount * 3);
  assert.equal(geometry.geom_quaternion.length, geomCount * 4);
  assert.equal(geometry.geom_rgba.length, geomCount * 4);
  assert(Math.max(...geometry.geom_body_id) < final.wasm.body_positions.length / 3);
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const report = {
    format: 'chreatures-native-unified-wasm-physical-comparison-v1', ticks: 200, seed: 20260908,
    fixtureSha256: ready.fixture_sha256, sceneSha256: ready.scene_xml_sha256,
    nativeBinarySha256: sha256(await readFile(resolve(args.native))),
    unifiedMjsSha256: sha256(await readFile(resolve(args.module))),
    unifiedWasmSha256: sha256(await readFile(resolve(args.module).replace(/\.mjs$/, '.wasm'))),
    buildReceiptSha256: sha256(await readFile(resolve(args.module).replace(/chreatures-fly-world\.mjs$/, 'build-receipt.json'))),
    numericalComparison: Object.fromEntries(Object.entries(stats).map(([name, value]) => [name, value.json()])),
    insertion: {native: nativeInsertion, unifiedWasm: wasmInsertion},
    growth: final.wasm.ecology.native_host.growth,
    topologyRevision: final.wasm.ecology.native_host.topology_revision,
    routeCount: final.wasm.ecology.native_host.route_open_fraction.length,
    finalObserverCounts: {geoms: geomCount, bodies: final.wasm.body_positions.length / 3,
      meshes: geometry.mesh_vertex_address.length, meshVertices: geometry.mesh_vertices.length / 3,
      meshNormals: geometry.mesh_normals.length / 3, meshFaces: geometry.mesh_faces.length / 3},
    meanMilliseconds: Object.fromEntries(Object.entries(timing).map(([name, values]) => [name, mean(values)])),
    exactWholeWorldRestoreContinuation: replayExact,
    finalTimeSeconds: final.wasm.time,
    scope: 'Matched current native host and unified Rust+MuJoCo Emscripten host; identical B4 scene, f32 controls, screen/sound/routes/visitor insertion/force chronology',
  };
  const text = `${JSON.stringify(report, null, 2)}\n`;
  process.stdout.write(text);
  if (args.output) await writeFile(resolve(args.output), text, {flag: 'wx'});
} finally {
  wasm.close();
  await native.close();
}
