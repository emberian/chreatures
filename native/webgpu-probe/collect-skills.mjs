#!/usr/bin/env node
// Offline privileged-teacher collection over the actual browser MuJoCo/Wasm
// body and full V2 WebGPU CNS. Geometry chooses labels here; only CNS Z512 and
// delivered command history enter the resulting learner episodes.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { create, globals } from 'webgpu';
import initResident from '../../site/live/pkg/resident_runtime.js';
import { LiveEngine } from '../../site/live/engine.js';

const CNS_SERVICE = 'ca66b6e5c07a09d55fc7e29552f504bd86b3807a6298425b9b933edf2cde9378';
const CNS_ADAPTER = '6875314d15f46806abb0fab04802856d4427d3a6ef29b19b12c70bb79706806d';
const RESIDENT_FILE = 'bc11d35aa0d828cdcb6b19a4fd5e9215c5d98e841ffa1ce765d17d85acdb44c1';
const RESIDENT_ARTIFACT = 'c63c9f2db4bf62ffade3265dd1e162a2c26693b0ba27ba3d92e94fcc3fe7d255';
const ACTIONS = ['thrust', 'yaw', 'gaze_pitch', 'posture', 'grip', 'signal_low',
  'signal_mid', 'signal_high', 'eat', 'release', 'secrete', 'allocate'];
const BATCH = 3, LATENT = 512, ACTION_DIM = 12, DT = 0.05;

const pairs = process.argv.slice(2);
const args = {};
for (let index = 0; index < pairs.length; index += 2) {
  if (!pairs[index]?.startsWith('--') || pairs[index + 1] === undefined) throw new Error('arguments must be --name value pairs');
  args[pairs[index].slice(2)] = pairs[index + 1];
}
if (!args.model || !args.output) {
  console.error('usage: node collect-skills.mjs --model PACK_DIRECTORY --output NEW_DIRECTORY [--site dist/site] [--episode 0..7] [--backend metal|vulkan]');
  process.exit(2);
}
const siteDirectory = resolve(args.site ?? '../../dist/site');
const modelDirectory = resolve(args.model);
const outputDirectory = resolve(args.output);
const ticks = Number(args.ticks ?? 512);
if (ticks !== 512) throw new Error('the current curriculum contract requires exactly 512 ticks');
const selectedEpisode = args.episode === undefined ? undefined : Number(args.episode);
if (selectedEpisode !== undefined && (!Number.isInteger(selectedEpisode) || selectedEpisode < 0 || selectedEpisode > 7)) {
  throw new Error('episode must be an integer in [0,7]');
}
const backend = args.backend ?? (process.platform === 'darwin' ? 'metal' : 'vulkan');
if (!['metal', 'vulkan'].includes(backend)) throw new Error('backend must be metal or vulkan');

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const mime = { '.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream' };

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function numpy(descr, shape, payload) {
  const tuple = shape.length === 0 ? '()' : `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  const source = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${tuple}, }`;
  const padding = (64 - ((10 + Buffer.byteLength(source) + 1) % 64)) % 64;
  const headerText = source + ' '.repeat(padding) + '\n';
  const header = Buffer.from(headerText, 'latin1');
  const prefix = Buffer.alloc(10);
  prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  prefix.writeUInt16LE(header.length, 8);
  return Buffer.concat([prefix, header, Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength)]);
}

function unicodeScalar(text) {
  const points = Array.from(text, character => character.codePointAt(0));
  const payload = Buffer.alloc(points.length * 4);
  points.forEach((point, index) => payload.writeUInt32LE(point, index * 4));
  return numpy(`<U${points.length}`, [], payload);
}

function npz(members) {
  const locals = [], centrals = [];
  let offset = 0;
  for (const [member, payload] of Object.entries(members)) {
    const name = Buffer.from(`${member}.npy`);
    const crc = crc32(payload);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4);
    local.writeUInt32LE(crc, 14); local.writeUInt32LE(payload.length, 18); local.writeUInt32LE(payload.length, 22);
    local.writeUInt16LE(name.length, 26);
    locals.push(local, name, payload);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6);
    central.writeUInt32LE(crc, 16); central.writeUInt32LE(payload.length, 20); central.writeUInt32LE(payload.length, 24);
    central.writeUInt16LE(name.length, 28); central.writeUInt32LE(offset, 42);
    centrals.push(central, name);
    offset += local.length + name.length + payload.length;
  }
  const centralBytes = centrals.reduce((sum, value) => sum + value.length, 0);
  const end = Buffer.alloc(22);
  const count = Object.keys(members).length;
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(count, 8); end.writeUInt16LE(count, 10);
  end.writeUInt32LE(centralBytes, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, ...centrals, end]);
}

function targetPosition(observation, name) {
  const geometry = observation.geometry.find(item => item.name === name);
  if (!geometry) throw new Error(`teacher target is missing: ${name}`);
  return observation.positions.subarray(geometry.id * 3, geometry.id * 3 + 3);
}

function residentPose(observation, resident) {
  const root = observation.residents[resident].root;
  const p = observation.bodyPositions.subarray(root * 3, root * 3 + 3);
  const rotation = observation.rotations.subarray(root * 9, root * 9 + 9);
  return { position: Array.from(p), heading: Math.atan2(rotation[3], rotation[0]),
    forward: [rotation[0], rotation[3]], rotation: Array.from(rotation) };
}

const SKILLS = Object.freeze(['approach', 'heading-correction', 'stop', 'withdraw', 'contact-recovery']);
const TEACHER_BOUTS = Object.freeze([
  { start_tick: 0, end_tick: 64, skill: 'contact-recovery' },
  { start_tick: 64, end_tick: 112, skill: 'withdraw' },
  { start_tick: 112, end_tick: 136, skill: 'stop' },
  { start_tick: 136, end_tick: 168, skill: 'heading-correction' },
  { start_tick: 168, end_tick: 224, skill: 'approach' },
  { start_tick: 224, end_tick: 256, skill: 'stop' },
  { start_tick: 256, end_tick: 320, skill: 'contact-recovery' },
  { start_tick: 320, end_tick: 368, skill: 'withdraw' },
  { start_tick: 368, end_tick: 392, skill: 'stop' },
  { start_tick: 392, end_tick: 424, skill: 'heading-correction' },
  { start_tick: 424, end_tick: 480, skill: 'approach' },
  { start_tick: 480, end_tick: 512, skill: 'stop' },
]);
assert.equal(TEACHER_BOUTS[0].start_tick, 0);
assert.equal(TEACHER_BOUTS.at(-1).end_tick, ticks);
for (let index = 0; index < TEACHER_BOUTS.length; index++) {
  assert(SKILLS.includes(TEACHER_BOUTS[index].skill));
  if (index) assert.equal(TEACHER_BOUTS[index - 1].end_tick, TEACHER_BOUTS[index].start_tick);
}

const CURRICULUM_CONTRACT = Object.freeze({
  format: 'chreatures-privileged-teacher-curriculum-contract-v1',
  resident_episode_members: {
    metadata: { dtype: 'unicode-json-scalar', shape: [] },
    cns_latent: { dtype: '<f4', shape: [513, BATCH, LATENT] },
    delivered_command: { dtype: '<f4', shape: [512, BATCH, ACTION_DIM] },
    physical_reward: { dtype: '<f4', shape: [512, BATCH] },
    reset: { dtype: '|b1', shape: [513, BATCH] },
    terminal: { dtype: '|b1', shape: [512, BATCH] },
  },
  action_names: ACTIONS,
  action_bounds: ACTIONS.map((_, index) => index < 4 ? [-1, 1] : [0, 1]),
  teacher_bouts: TEACHER_BOUTS,
  teacher_reward_algorithm: 'privileged-physical-teacher-v3-heading-sign-fixed-finite-distance-goals',
  dt_seconds: DT,
  cns_service_artifact_sha256: CNS_SERVICE,
  adapter_sha256: CNS_ADAPTER,
  resident_file_sha256: RESIDENT_FILE,
  resident_artifact_sha256: RESIDENT_ARTIFACT,
});
const CURRICULUM_CONTRACT_SHA256 = sha256(Buffer.from(JSON.stringify(CURRICULUM_CONTRACT)));

const LAYOUTS = Object.freeze([
  [[2.0, .78], [6.0, .78], [10.1, .78]],
  [[5.35, 4.0], [7.15, 4.05], [10.9, 4.8]],
  [[2.0, 7.15], [5.45, 7.12], [8.0, 7.12]],
  [[10.85, 7.08], [7.35, 7.12], [4.95, 4.0]],
  [[1.05, 3.55], [4.95, 4.02], [10.95, 4.75]],
  [[3.0, .78], [7.2, 4.05], [10.8, 7.08]],
  [[11.0, .78], [5.5, 4.08], [2.2, 7.18]],
  [[1.08, 3.35], [8.1, .78], [7.55, 7.18]],
]);
const TARGET_SPECS = Object.freeze([
  { role: 'berry-resource', size: [.08, .08, .08], rgba: [.9, .08, .18, 1], food: .75, odor: 0 },
  { role: 'nectar-resource', size: [.07, .07, .07], rgba: [.96, .68, .08, 1], food: .68, odor: 1 },
  { role: 'movable-object', size: [.12, .12, .12], rgba: [.52, .2, .88, 1], food: 0, odor: 2 },
]);
// Bounded open-ground candidates used only after the requested seeded point
// fails the world's atomic physical-overlap check.
const LEGAL_PLACEMENT_FALLBACKS = Object.freeze([
  [1.0, .58], [3.0, .58], [5.0, .58], [7.0, .58], [9.0, .58], [11.0, .58],
  [.65, 7.42], [3.7, 7.42], [6.6, 7.42], [11.3, 7.42],
]);
const WORLD_SEEDS = Object.freeze([20261011, 20261103, 20261207, 20261309, 20261413, 20261517, 20270119, 20270223]);
const VARIATION_SEEDS = Object.freeze([77141, 99233, 114229, 131267, 155291, 177313, 220351, 244399]);

function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let x = value; x = Math.imul(x ^ x >>> 15, x | 1);
    x ^= x + Math.imul(x ^ x >>> 7, x | 61);
    return ((x ^ x >>> 14) >>> 0) / 4294967296;
  };
}

function screenFrame(episode, tick) {
  const palettes = [
    [[.96, .26, .08], [.04, .18, .88]], [[.1, .88, .34], [.82, .08, .72]],
    [[.95, .72, .08], [.08, .66, .92]], [[.7, .12, .94], [.12, .9, .68]],
  ];
  const colors = palettes[episode % palettes.length], phase = Math.floor(tick / 16) % 2;
  return Float32Array.from([...colors[phase], ...colors[1 - phase], ...colors[1 - phase], ...colors[phase]]);
}

async function configureEpisode(world, episode) {
  const random = randomGenerator(VARIATION_SEEDS[episode]), targets = [];
  for (let resident = 0; resident < BATCH; resident++) {
    const spec = TARGET_SPECS[resident], base = LAYOUTS[episode][resident];
    const requestedPosition = [base[0] + (random() - .5) * .12, base[1] + (random() - .5) * .12];
    const candidates = [requestedPosition];
    for (let offset = 0; offset < LEGAL_PLACEMENT_FALLBACKS.length; offset++) {
      candidates.push(LEGAL_PLACEMENT_FALLBACKS[(episode * BATCH + resident + offset) % LEGAL_PLACEMENT_FALLBACKS.length]);
    }
    let inserted, chosenPosition, placementCandidateIndex = -1, lastError;
    for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex++) {
      chosenPosition = [...candidates[candidateIndex], spec.size[0] + .012];
      try {
        inserted = await world.insertObject({ position: chosenPosition, size: spec.size, shape: 'sphere', rgba: spec.rgba,
          food: spec.food, odor: spec.odor });
        placementCandidateIndex = candidateIndex;
        break;
      } catch (error) {
        if (!String(error?.message ?? error).startsWith('Insertion would penetrate existing physical geometry')) throw error;
        lastError = error;
      }
    }
    if (!inserted) throw new Error(`No legal bounded placement for episode ${episode} resident ${resident}: ${lastError}`);
    targets.push({ ...spec, id: inserted.id, geom: `entity:${inserted.id}:geom:0`, requestedPosition,
      chosenPosition, placementCandidateIndex });
  }
  const observation = world.observe(), snapshot = world.snapshot();
  const core = JSON.parse(snapshot.core); core.rng = WORLD_SEEDS[episode]; snapshot.core = JSON.stringify(core);
  const starts = [];
  for (let resident = 0; resident < BATCH; resident++) {
    const target = targetPosition(observation, targets[resident].geom);
    const radial = random() * Math.PI * 2;
    const distance = targets[resident].food > 0 ? .235 + random() * .02 : .235 + random() * .02;
    const dx = Math.cos(radial) * distance, dy = Math.sin(radial) * distance;
    const body = snapshot.fixture.bodies[resident], offset = 1 + body.qpos[0] - 7;
    const desiredHeading = Math.atan2(-dy, -dx);
    const headingOffset = (resident % 2 ? -1 : 1) * (.15 + random() * .15);
    snapshot.physical[offset] = target[0] + dx; snapshot.physical[offset + 1] = target[1] + dy;
    snapshot.physical[offset + 2] = .09;
    const yaw = desiredHeading + headingOffset;
    snapshot.physical[offset + 3] = Math.cos(yaw / 2); snapshot.physical[offset + 4] = 0;
    snapshot.physical[offset + 5] = 0; snapshot.physical[offset + 6] = Math.sin(yaw / 2);
    starts.push({ distance, radial, heading_offset: headingOffset });
  }
  world.restore(snapshot);
  const finalSnapshot = world.snapshot();
  for (const target of targets) {
    const entityIndex = finalSnapshot.fixture.entities.findIndex(entity => entity.id === target.id);
    assert(entityIndex >= 0); target.foodIndex = entityIndex;
  }
  const descriptor = { episode, world_seed: WORLD_SEEDS[episode], variation_seed: VARIATION_SEEDS[episode],
    model: finalSnapshot.model, targets: targets.map((target, resident) => ({ role: target.role,
      geom: target.geom, requested_position: target.requestedPosition, chosen_position: target.chosenPosition,
      placement_candidate_index: target.placementCandidateIndex,
      position: Array.from(targetPosition(world.observe(), target.geom)), start: starts[resident] })) };
  return { targets, starts, layoutIdentity: sha256(Buffer.from(JSON.stringify(descriptor))), model: finalSnapshot.model,
    atlas: finalSnapshot.atlas };
}

function measure(observation, resident, target, previous) {
  const pose = residentPose(observation, resident), point = targetPosition(observation, target.geom);
  const dx = point[0] - pose.position[0], dy = point[1] - pose.position[1], dz = point[2] - pose.position[2];
  const forward = pose.forward, rotation = pose.rotation;
  const localX = rotation[0] * dx + rotation[3] * dy + rotation[6] * dz;
  const localY = rotation[1] * dx + rotation[4] * dy + rotation[7] * dz;
  const localZ = rotation[2] * dx + rotation[5] * dy + rotation[8] * dz;
  const headingError = Math.atan2(localY, localX), centerDistance = Math.hypot(dx, dy, dz);
  const velocity = previous ? pose.position.map((value, axis) => (value - previous.pose.position[axis]) / DT) : [0, 0, 0];
  const targetVelocity = previous ? Array.from(point, (value, axis) => (value - previous.target[axis]) / DT) : [0, 0, 0];
  return { pose, target: Array.from(point), headingError, centerDistance,
    speed: Math.hypot(...velocity), forwardSpeed: forward[0] * velocity[0] + forward[1] * velocity[1],
    yawRate: previous ? Math.atan2(Math.sin(pose.heading - previous.pose.heading), Math.cos(pose.heading - previous.pose.heading)) / DT : 0,
    gripReachError: Math.hypot(localX - .17, localY, localZ - .04), targetSpeed: Math.hypot(...targetVelocity),
    food: observation.food[target.foodIndex] };
}

function currentBout(tick) {
  const index = TEACHER_BOUTS.findIndex(bout => tick >= bout.start_tick && tick < bout.end_tick);
  assert(index >= 0); return { ...TEACHER_BOUTS[index], index, age: tick - TEACHER_BOUTS[index].start_tick };
}

function headingControl(value) {
  // Positive body yaw rotates clockwise in the Rust gait convention, while
  // the observer's world-frame heading error is positive counter-clockwise.
  return clamp(-1.65 * value.headingError + .09 * value.yawRate, -1, 1);
}

function teacher(observation, targets, previous, tick, episode) {
  const command = new Float32Array(BATCH * ACTION_DIM), bout = currentBout(tick), measurements = [];
  for (let resident = 0; resident < BATCH; resident++) {
    const value = measure(observation, resident, targets[resident], previous?.[resident]);
    const first = resident * ACTION_DIM, desired = targets[resident].food > 0 ? .19 : .22;
    measurements.push(value);
    command[first + 1] = headingControl(value);
    if (bout.skill === 'heading-correction') {
      command[first] = Math.abs(value.headingError) < .18 ? clamp(-3 * value.forwardSpeed, -.3, .3) : .08;
    } else if (bout.skill === 'approach') {
      const desiredSpeed = clamp((value.centerDistance - desired) * .75, 0, .16);
      command[first] = clamp(.20 + 3.6 * (desiredSpeed - value.forwardSpeed), 0, 1) * clamp(Math.cos(value.headingError), 0, 1);
    } else if (bout.skill === 'stop') {
      command[first] = clamp(-4.5 * value.forwardSpeed, -.65, .65);
      command[first + 1] = clamp(.11 * value.yawRate, -.45, .45); command[first + 9] = 1;
    } else if (bout.skill === 'withdraw') {
      const desiredSpeed = value.centerDistance < .49 ? -.11 : 0;
      command[first] = clamp(3.4 * (desiredSpeed - value.forwardSpeed), -.82, .25);
      command[first + 9] = 1;
    } else {
      const duration = bout.end_tick - bout.start_tick, recovery = bout.age >= Math.floor(duration * .75);
      const hold = bout.age >= Math.floor(duration * .50) && !recovery;
      if (recovery) {
        const desiredSpeed = value.centerDistance < .46 ? -.09 : 0;
        command[first] = clamp(3.8 * (desiredSpeed - value.forwardSpeed), -.75, .25); command[first + 9] = 1;
      } else {
        const desiredSpeed = hold ? 0 : clamp((value.centerDistance - desired) * .7, 0, .09);
        command[first] = clamp((hold ? 0 : .12) + 4 * (desiredSpeed - value.forwardSpeed), -.25, .62) * clamp(Math.cos(value.headingError), 0, 1);
        const aligned = Math.abs(value.headingError) < .55;
        if (targetIsFood(targets[resident]) && value.centerDistance < .285 && aligned) command[first + 8] = 1;
        if (!targetIsFood(targets[resident]) && value.centerDistance < .34) command[first + 4] = 1;
      }
    }
    command[first + 3] = bout.skill === 'contact-recovery' ? .10 : .04;
    if (bout.skill !== 'stop') command[first + 5 + ((resident + episode) % 3)] = .16;
  }
  return { command, measurements, bout };
}

function targetIsFood(target) { return target.food > 0; }
const potential = (before, after, goal) => Math.abs(before - goal) - Math.abs(after - goal);
function teacherReward(label, after, target, command, foodConsumed) {
  const before = label.measurement, desired = target.food > 0 ? .19 : .22;
  const effort = .025 * (command[0] ** 2 + command[1] ** 2) + .006 * (command[4] + command[8]);
  let value = 0;
  if (label.skill === 'heading-correction') {
    value = Math.abs(before.headingError) > .10 ? 2.2 * (Math.abs(before.headingError) - Math.abs(after.headingError)) : 0;
    if (Math.abs(after.headingError) < .10 && Math.abs(before.headingError) >= .10) value += .18;
  } else if (label.skill === 'approach') {
    value = 8 * potential(before.centerDistance, after.centerDistance, desired);
    if (Math.abs(after.centerDistance - desired) < .045 && after.speed < .04) value += .16;
  } else if (label.skill === 'stop') {
    value = (after.speed < .015 ? .28 : after.speed < .04 ? .10 : -.8 * Math.min(after.speed, .25));
  } else if (label.skill === 'withdraw') {
    value = 7 * potential(before.centerDistance, after.centerDistance, .49);
    if (Math.abs(after.centerDistance - .49) < .05 && after.speed < .035) value += .14;
  } else if (label.recovery) {
    value = 7 * potential(before.centerDistance, after.centerDistance, .46);
    if (Math.abs(after.centerDistance - .46) < .05 && after.speed < .035) value += .20;
  } else {
    value = 6 * potential(before.centerDistance, after.centerDistance, desired);
    if (Math.abs(after.headingError) < .28 && after.centerDistance < .34) value += .08;
    if (target.food > 0 && after.centerDistance < .28) value += .34;
    if (target.food === 0 && after.gripReachError < .16) value += .34;
    if (foodConsumed > 0) value += .5;
  }
  return clamp(value - effort, -2, 2);
}

await mkdir(outputDirectory, { recursive: selectedEpisode !== undefined });
const collectorSource = await readFile(new URL(import.meta.url));
await writeFile(resolve(outputDirectory, 'collector-source.mjs'), collectorSource, { flag: 'wx' });
const cnsManifest = JSON.parse(await readFile(resolve(modelDirectory, 'cns-manifest.json'), 'utf8'));
const residentManifest = JSON.parse(await readFile(resolve(modelDirectory, 'resident-manifest.json'), 'utf8'));
const runtimeManifest = JSON.parse(await readFile(resolve(siteDirectory, 'live/runtime-manifest.json'), 'utf8'));
const collectorFileSha256 = sha256(collectorSource);
assert.equal(cnsManifest.serviceArtifactSha256, CNS_SERVICE);
assert.equal(cnsManifest.identity.artifact, CNS_ADAPTER);
assert.equal(residentManifest.cnsServiceArtifactSha256, CNS_SERVICE);
assert.equal(residentManifest.artifactSha256, RESIDENT_ARTIFACT);

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const modelRequest = pathname.startsWith('/live/model/');
    const root = modelRequest ? modelDirectory : siteDirectory;
    const relative = modelRequest ? pathname.slice('/live/model/'.length) : pathname.slice(1);
    const file = resolve(root, relative);
    if (file !== root && !file.startsWith(root + sep)) throw new Error('path escapes root');
    const info = await stat(file); if (!info.isFile()) throw new Error('not a file');
    response.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
    response.setHeader('Content-Length', info.size); createReadStream(file).pipe(response);
  } catch { response.statusCode = 404; response.end('missing'); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));

Object.assign(globalThis, globals);
const dawn = create([`backend=${backend}`]);
// webgpu's native implementation lives only while create()'s return value is
// strongly reachable; pin it across the long asynchronous collection.
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' });
assert(adapter, `actual Dawn ${backend} adapter required`);
const device = await adapter.requestDevice();
const episodeReceipts = [];
let engine, activeEpisode = null, residentInitialized = false;
try {
  const residentWasm = await readFile(resolve(siteDirectory, 'live/pkg/resident_runtime_bg.wasm'));
  const episodeIndexes = selectedEpisode === undefined ? Array.from({ length: 8 }, (_, index) => index) : [selectedEpisode];
  for (const episode of episodeIndexes) {
    activeEpisode = episode;
    engine = await LiveEngine.create({ baseURL: `http://127.0.0.1:${server.address().port}/live/`, device,
      modules: { initResident: residentInitialized ? async () => {} : async () => {
        await initResident({ module_or_path: residentWasm }); residentInitialized = true;
      } } });
    assert.equal(engine.batch, BATCH);
    const layout = await configureEpisode(engine.world, episode);
    const lifeIds = engine.world.observe().residents.map(item => ({ id: item.id, teacherEpisode: episode,
      worldSeed: WORLD_SEEDS[episode], variationSeed: VARIATION_SEEDS[episode], layoutIdentity: layout.layoutIdentity }));
    await engine.brain.reset([0, 1, 2], lifeIds);
    const cnsLatent = new Float32Array((ticks + 1) * BATCH * LATENT);
    const delivered = new Float32Array(ticks * BATCH * ACTION_DIM);
    const reward = new Float32Array(ticks * BATCH);
    const reset = new Uint8Array((ticks + 1) * BATCH); reset.fill(1, 0, BATCH);
    const terminal = new Uint8Array(ticks * BATCH); terminal.fill(1, (ticks - 1) * BATCH);
    const afferent = new Float32Array((ticks + 1) * BATCH * (5313 + 43));
    const inputTime = new Float64Array(ticks + 1), tickIndex = new Uint32Array(ticks + 1);
    const activeMask = new Uint8Array(ticks + 1); activeMask.fill(7);
    const initial = engine.world.observe();
    const origins = initial.residents.map((_, resident) => residentPose(initial, resident).position);
    const eventCounts = { headingAlignedTicks: 0, approachArrivalTicks: 0, stoppedTicks: 0,
      headingNearTargetTicks: 0, envelopeProximityTicks: 0, eatLawEligibleTicks: 0, gripCaptureVolumeProxyTicks: 0,
      eatCommandTicks: 0, gripCommandTicks: 0, foodConsumedTicks: 0, gripCommandTargetMotionTicks: 0,
      withdrawGoalTicks: 0, recoveryStableTicks: 0, forwardThrustTicks: 0, reverseThrustTicks: 0,
      positiveYawTicks: 0, negativeYawTicks: 0 };
    const boutAggregates = TEACHER_BOUTS.map(bout => ({ ...bout, samples: 0, reward_sum: 0,
      effort_sum: 0, heading_aligned_ticks: 0, stopped_ticks: 0, heading_near_target_ticks: 0,
      envelope_proximity_ticks: 0, start_distance: Array(BATCH).fill(null), end_distance: Array(BATCH).fill(null),
      start_speed: Array(BATCH).fill(null), end_speed: Array(BATCH).fill(null),
      start_heading_error: Array(BATCH).fill(null), end_heading_error: Array(BATCH).fill(null) }));
    const diagnostic = { minCenterDistance: Array(BATCH).fill(Infinity), minGripReachError: Array(BATCH).fill(Infinity),
      finalCenterDistance: Array(BATCH).fill(0) };
    let maxMotion = 0, rewardSum = 0, previous;
    for (let tick = 0; tick <= ticks; tick++) {
      engine.world.setScreenFrame(screenFrame(episode, tick), 2, 2);
      const sensory = engine.world.sample();
      for (let resident = 0; resident < BATCH; resident++) {
        const offset = (tick * BATCH + resident) * 5356;
        afferent.set(sensory.optic.subarray(resident * 5313, (resident + 1) * 5313), offset);
        afferent.set(sensory.body.subarray(resident * 43, (resident + 1) * 43), offset + 5313);
      }
      inputTime[tick] = engine.world.time; tickIndex[tick] = tick;
      const neural = await engine.brain.step({ dt: DT, activeMask: 7, opticRGB: sensory.optic, body: sensory.body });
      cnsLatent.set(neural.latent, tick * BATCH * LATENT);
      if (tick === ticks) break;
      const before = engine.world.observe();
      const label = teacher(before, layout.targets, previous, tick, episode);
      delivered.set(label.command, tick * BATCH * ACTION_DIM);
      engine.world.advance(label.command, DT);
      const after = engine.world.observe();
      for (let resident = 0; resident < BATCH; resident++) {
        const first = resident * ACTION_DIM, beforeValue = label.measurements[resident];
        const afterValue = measure(after, resident, layout.targets[resident], beforeValue);
        const foodConsumed = Math.max(0, beforeValue.food - afterValue.food);
        const duration = label.bout.end_tick - label.bout.start_tick;
        const recovery = label.bout.skill === 'contact-recovery' && label.bout.age >= Math.floor(duration * .75);
        const value = teacherReward({ measurement: beforeValue, skill: label.bout.skill, recovery },
          afterValue, layout.targets[resident], label.command.subarray(first, first + ACTION_DIM), foodConsumed);
        reward[tick * BATCH + resident] = value; rewardSum += value;
        const command = label.command.subarray(first, first + ACTION_DIM);
        const aligned = Math.abs(afterValue.headingError) < .10, stopped = afterValue.speed < .015;
        const headingNearTarget = Math.abs(afterValue.headingError) < .28 && afterValue.centerDistance < .34;
        // Root-center distance includes the resident's 0.18 m anterior body
        // envelope, so this threshold denotes overlapping physical envelopes.
        const contact = afterValue.centerDistance < layout.targets[resident].size[0] + .18;
        diagnostic.minCenterDistance[resident] = Math.min(diagnostic.minCenterDistance[resident], afterValue.centerDistance);
        diagnostic.minGripReachError[resident] = Math.min(diagnostic.minGripReachError[resident], afterValue.gripReachError);
        diagnostic.finalCenterDistance[resident] = afterValue.centerDistance;
        const aggregate = boutAggregates[label.bout.index]; aggregate.samples++; aggregate.reward_sum += value;
        aggregate.effort_sum += command[0] ** 2 + command[1] ** 2 + .2 * (command[4] + command[8]);
        if (aggregate.start_distance[resident] === null) {
          aggregate.start_distance[resident] = beforeValue.centerDistance;
          aggregate.start_speed[resident] = beforeValue.speed;
          aggregate.start_heading_error[resident] = beforeValue.headingError;
        }
        aggregate.end_distance[resident] = afterValue.centerDistance;
        aggregate.end_speed[resident] = afterValue.speed;
        aggregate.end_heading_error[resident] = afterValue.headingError;
        if (aligned) { eventCounts.headingAlignedTicks++; aggregate.heading_aligned_ticks++; }
        if (stopped) { eventCounts.stoppedTicks++; aggregate.stopped_ticks++; }
        if (headingNearTarget) { eventCounts.headingNearTargetTicks++; aggregate.heading_near_target_ticks++; }
        if (contact) { eventCounts.envelopeProximityTicks++; aggregate.envelope_proximity_ticks++; }
        if (label.bout.skill === 'approach' && Math.abs(afterValue.centerDistance - (layout.targets[resident].food > 0 ? .19 : .22)) < .045) eventCounts.approachArrivalTicks++;
        if (afterValue.centerDistance < .28 && layout.targets[resident].food > 0) eventCounts.eatLawEligibleTicks++;
        if (afterValue.gripReachError < .16 && layout.targets[resident].food === 0) eventCounts.gripCaptureVolumeProxyTicks++;
        if (command[8] > .5) eventCounts.eatCommandTicks++;
        if (command[4] > .5) eventCounts.gripCommandTicks++;
        if (foodConsumed > 0) eventCounts.foodConsumedTicks++;
        if (command[4] > .5 && afterValue.targetSpeed > .01) eventCounts.gripCommandTargetMotionTicks++;
        if (label.bout.skill === 'withdraw' && Math.abs(afterValue.centerDistance - .49) < .05) eventCounts.withdrawGoalTicks++;
        if (recovery && Math.abs(afterValue.centerDistance - .46) < .05 && afterValue.speed < .035) eventCounts.recoveryStableTicks++;
        if (command[0] > .08) eventCounts.forwardThrustTicks++;
        if (command[0] < -.08) eventCounts.reverseThrustTicks++;
        if (command[1] > .08) eventCounts.positiveYawTicks++;
        if (command[1] < -.08) eventCounts.negativeYawTicks++;
        const current = afterValue.pose.position, origin = origins[resident];
        maxMotion = Math.max(maxMotion, Math.hypot(current[0] - origin[0], current[1] - origin[1], current[2] - origin[2]));
      }
      previous = label.measurements;
    }
    for (const values of [cnsLatent, delivered, reward, afferent, inputTime]) for (const value of values) assert(Number.isFinite(value));
    console.error(`episode ${episode} coverage ${JSON.stringify(eventCounts)} diagnostic ${JSON.stringify(diagnostic)} bouts ${JSON.stringify(boutAggregates)}`);
    const requiredCoverage = ['stoppedTicks', 'headingNearTargetTicks', 'envelopeProximityTicks', 'eatLawEligibleTicks',
      'eatCommandTicks', 'gripCommandTicks', 'foodConsumedTicks', 'forwardThrustTicks', 'reverseThrustTicks',
      'positiveYawTicks', 'negativeYawTicks'];
    const missingCoverage = requiredCoverage.filter(name => eventCounts[name] === 0);
    const split = episode < 6 ? 'train' : 'heldout-worlds';
    const metadata = {
      format: 'chreatures-privileged-teacher-episode-v1', action_source: 'privileged-physical-teacher',
      cns_service_artifact_sha256: CNS_SERVICE, adapter_sha256: CNS_ADAPTER,
      resident_file_sha256: RESIDENT_FILE, resident_artifact_sha256: RESIDENT_ARTIFACT,
      resident_parent_role: 'identity-bound initialization only; it did not choose teacher commands',
      cns_source_revision: cnsManifest.sourceRevision, execution_source_revision: runtimeManifest.sourceRevision,
      collector_file_sha256: collectorFileSha256, curriculum_contract_sha256: CURRICULUM_CONTRACT_SHA256,
      episode_index: episode, split,
      world_seed: WORLD_SEEDS[episode], variation_seed: VARIATION_SEEDS[episode], layout_identity: layout.layoutIdentity,
      world_identity: { engine: engine.world.engine, source_mjcf_sha256: layout.model, atlas_sha256: layout.atlas },
      teacher_bouts: TEACHER_BOUTS, bout_aggregates: boutAggregates, event_counts: eventCounts,
      target_roles: layout.targets.map(target => target.role),
      action_names: ACTIONS, dt_seconds: DT, transitions: ticks, residents: BATCH,
      latent_alignment: 'cns_latent[t] follows sensing physical state t; delivered_command[t] advances state t to t+1',
      reward_formula: 'phase-specific bounded potential improvement or low-speed/contact success minus action effort; approach and withdrawal potentials have finite desired-distance goals',
      teacher_privileged_fields_used: ['resident root bodyPositions', 'resident root rotations', 'target geom positions', 'target food quantity', 'teacher bout clock'],
      raw_geometry_retained: false, raw_senses_retained: false, controller_inputs_allowed: ['cns_latent', 'previous_delivered_command'],
    };
    const archive = npz({ metadata: unicodeScalar(JSON.stringify(metadata)),
      cns_latent: numpy('<f4', [ticks + 1, BATCH, LATENT], cnsLatent),
      delivered_command: numpy('<f4', [ticks, BATCH, ACTION_DIM], delivered),
      physical_reward: numpy('<f4', [ticks, BATCH], reward),
      reset: numpy('|b1', [ticks + 1, BATCH], reset), terminal: numpy('|b1', [ticks, BATCH], terminal) });
    const filename = `episode-${episode.toString().padStart(2, '0')}-${episode < 6 ? 'train' : 'heldout-world'}.npz`;
    const archiveHash = sha256(archive);
    const sidecarMetadata = { format: 'chreatures-cns-afferent-sidecar-v1', source_episode_file: filename,
      source_episode_sha256: archiveHash, episode_index: episode, split, world_seed: WORLD_SEEDS[episode],
      variation_seed: VARIATION_SEEDS[episode], layout_identity: layout.layoutIdentity,
      cns_service_artifact_sha256: CNS_SERVICE, adapter_sha256: CNS_ADAPTER,
      graph_sha256: cnsManifest.identity.graph, atlas_sha256: cnsManifest.identity.atlas,
      afferent_mask_sha256: cnsManifest.identity.mask, afferent_mask_buffer_sha256: cnsManifest.buffers['afferent.mask'].sha256,
      input_layout: 'resident-major per tick: optic RGB[5313] then body[43]', input_time_alignment: 'pre-CNS/pre-action world time',
      active_mask_semantics: 'bit i marks resident i active', raw_geometry_retained: false,
      current_training_use: 'archival only; excluded from resident corpus and current learning objective' };
    const sidecarDirectory = `episode-${episode.toString().padStart(2, '0')}-afferent-sidecar`;
    const sidecarPath = resolve(outputDirectory, sidecarDirectory);
    const rawBuffers = {
      'afferent-input.f32': { values: afferent, dtype: '<f4', shape: [ticks + 1, BATCH, 5356] },
      'input-time-seconds.f64': { values: inputTime, dtype: '<f8', shape: [ticks + 1] },
      'tick-index.u32': { values: tickIndex, dtype: '<u4', shape: [ticks + 1] },
      'active-mask.u8': { values: activeMask, dtype: '|u1', shape: [ticks + 1] },
      'delivered-command.f32': { values: delivered, dtype: '<f4', shape: [ticks, BATCH, ACTION_DIM] },
    };
    sidecarMetadata.execution_source_revision = runtimeManifest.sourceRevision;
    sidecarMetadata.collector_file_sha256 = collectorFileSha256;
    sidecarMetadata.buffers = {};
    for (const [name, entry] of Object.entries(rawBuffers)) {
      const bytes = Buffer.from(entry.values.buffer, entry.values.byteOffset, entry.values.byteLength);
      sidecarMetadata.buffers[name] = { dtype: entry.dtype, shape: entry.shape, byteLength: bytes.length, sha256: sha256(bytes) };
    }
    await writeFile(resolve(outputDirectory, filename), archive, { flag: 'wx' });
    await mkdir(sidecarPath);
    for (const [name, entry] of Object.entries(rawBuffers)) {
      await writeFile(resolve(sidecarPath, name), new Uint8Array(entry.values.buffer, entry.values.byteOffset, entry.values.byteLength), { flag: 'wx' });
    }
    const sidecarManifest = Buffer.from(JSON.stringify(sidecarMetadata, null, 2) + '\n');
    await writeFile(resolve(sidecarPath, 'manifest.json'), sidecarManifest, { flag: 'wx' });
    const sidecarBytes = Object.values(sidecarMetadata.buffers).reduce((sum, entry) => sum + entry.byteLength, 0) + sidecarManifest.length;
    const episodeReceipt = { file: filename, bytes: archive.length, sha256: archiveHash,
      sidecarDirectory, sidecarManifestSha256: sha256(sidecarManifest), sidecarBytes, split, episodeIndex: episode,
      worldSeed: WORLD_SEEDS[episode], variationSeed: VARIATION_SEEDS[episode], layoutIdentity: layout.layoutIdentity,
      curriculumContractSha256: CURRICULUM_CONTRACT_SHA256,
      ticks, maxMotionMeters: maxMotion, meanPhysicalReward: rewardSum / (ticks * BATCH), eventCounts,
      coveragePassed: missingCoverage.length === 0, missingCoverage };
    episodeReceipts.push(episodeReceipt);
    await writeFile(resolve(outputDirectory, `episode-${episode.toString().padStart(2, '0')}.receipt.json`),
      JSON.stringify(episodeReceipt, null, 2) + '\n', { flag: 'wx' });
    console.error(`${filename}: ${ticks} ticks, max motion ${maxMotion.toFixed(3)} m, envelope ${eventCounts.envelopeProximityTicks}, near-target heading ${eventCounts.headingNearTargetTicks}, actual food ${eventCounts.foodConsumedTicks}, grip+motion ${eventCounts.gripCommandTargetMotionTicks}`);
    engine.destroy(); engine = null; activeEpisode = null;
  }
  const receipt = { format: 'chreatures-browser-teacher-collection-v2', createdBy: `actual Dawn ${backend} + MuJoCo 3.12 Wasm + Rust/Wasm body`,
    cnsServiceArtifactSha256: CNS_SERVICE, adapterSha256: CNS_ADAPTER,
    residentFileSha256: RESIDENT_FILE, residentArtifactSha256: RESIDENT_ARTIFACT,
    curriculumContractSha256: CURRICULUM_CONTRACT_SHA256, curriculumContract: CURRICULUM_CONTRACT,
    actionSource: 'privileged-physical-teacher', rawGeometryRetained: false, rawSensesRetainedByResidentEpisodes: false,
    afferentSidecarsExcludedFromResidentTraining: true,
    backend, adapter: adapter.info?.device || adapter.info?.description || `Dawn ${backend}`,
    adapterLimits: {maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize},
    episodes: episodeReceipts };
  const receiptName = selectedEpisode === undefined ? 'receipt.json' : `receipt-episode-${selectedEpisode.toString().padStart(2, '0')}.json`;
  await writeFile(resolve(outputDirectory, receiptName), JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(receipt, null, 2));
} catch (error) {
  const failure = { format: 'chreatures-browser-teacher-collection-failure-v2', failedEpisode: activeEpisode,
    completedEpisodes: episodeReceipts.map(item => item.episodeIndex), error: String(error?.stack ?? error) };
  await writeFile(resolve(outputDirectory, `failure-${Date.now()}.json`), JSON.stringify(failure, null, 2) + '\n', { flag: 'wx' });
  throw error;
} finally {
  engine?.destroy(); device.destroy(); delete globalThis.__chreaturesDawnInstance;
  server.closeAllConnections(); server.close();
}
