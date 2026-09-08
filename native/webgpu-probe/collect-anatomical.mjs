#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Actual MuJoCo/Wasm physical bootstrap data for anatomical MaleCNS V3.
// Observer pose chooses teacher targets and diagnostics only. The eventual CNS
// receives optic RGB, BODY110, and the delivered context12 recorded here.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const TICKS = 512, BATCH = 3, DT = .05, OPTIC = 5313, BODY = 110, MOTOR = 34, CONTEXT = 12;
const SKILLS = Object.freeze(['pose-hold', 'turn', 'stop', 'reach', 'babble']);
const TONES_HZ = Object.freeze([80, 200, 500, 1250]);
const WORLD_SEEDS = Object.freeze([3031101, 3031207, 3031313, 3031419, 3031523, 3031601, 3031709, 3031817]);
const VARIATION_SEEDS = Object.freeze([71053, 81071, 91081, 101089, 111103, 121117, 131129, 141143]);
const BOUTS = Object.freeze([
  { start_tick: 0, end_tick: 96, skill: 'pose-hold', tone_hz: TONES_HZ[0] },
  { start_tick: 96, end_tick: 192, skill: 'turn', tone_hz: TONES_HZ[1] },
  { start_tick: 192, end_tick: 272, skill: 'stop', tone_hz: TONES_HZ[2] },
  { start_tick: 272, end_tick: 400, skill: 'reach', tone_hz: TONES_HZ[3] },
  { start_tick: 400, end_tick: 512, skill: 'babble', tone_hz: null },
]);
const MOTOR_NAMES = Object.freeze([
  ...['lf', 'lm', 'lh', 'rf', 'rm', 'rh'].flatMap(leg =>
    ['hip', 'knee'].flatMap(joint => [`${leg}_${joint}_positive`, `${leg}_${joint}_negative`])),
  'gaze_pitch', 'posture', 'grip', 'signal_low', 'signal_mid', 'signal_high',
  'eat', 'release', 'secrete', 'allocate',
]);

const argv = process.argv.slice(2), args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) throw new Error('arguments must be --name value pairs');
  args[argv[index].slice(2)] = argv[index + 1];
}
if (!args.output || args.episode === undefined) {
  console.error('usage: node collect-anatomical.mjs --output NEW_DIRECTORY --episode 0..7 [--world native/browser-world]');
  process.exit(2);
}
const episode = Number(args.episode);
if (!Number.isInteger(episode) || episode < 0 || episode > 7) throw new Error('episode must be an integer in [0,7]');
const here = dirname(fileURLToPath(import.meta.url));
const worldDirectory = resolve(args.world ?? resolve(here, '../browser-world'));
const outputDirectory = resolve(args.output);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

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
  const header = Buffer.from(source + ' '.repeat(padding) + '\n', 'latin1');
  const prefix = Buffer.alloc(10); prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
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
  const locals = [], centrals = []; let offset = 0;
  for (const [member, payload] of Object.entries(members)) {
    const name = Buffer.from(`${member}.npy`), crc = crc32(payload), local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(payload.length, 18); local.writeUInt32LE(payload.length, 22); local.writeUInt16LE(name.length, 26);
    locals.push(local, name, payload);
    const central = Buffer.alloc(46); central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6); central.writeUInt32LE(crc, 16); central.writeUInt32LE(payload.length, 20);
    central.writeUInt32LE(payload.length, 24); central.writeUInt16LE(name.length, 28); central.writeUInt32LE(offset, 42);
    centrals.push(central, name); offset += local.length + name.length + payload.length;
  }
  const centralBytes = centrals.reduce((sum, value) => sum + value.length, 0), end = Buffer.alloc(22);
  const count = Object.keys(members).length; end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(count, 8);
  end.writeUInt16LE(count, 10); end.writeUInt32LE(centralBytes, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, ...centrals, end]);
}

function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5; let x = value; x = Math.imul(x ^ x >>> 15, x | 1);
    x ^= x + Math.imul(x ^ x >>> 7, x | 61); return ((x ^ x >>> 14) >>> 0) / 4294967296;
  };
}

function currentBout(tick) {
  const value = BOUTS.find(bout => tick >= bout.start_tick && tick < bout.end_tick);
  assert(value); return value;
}

function rootPose(observation, resident) {
  const root = observation.residents[resident].root;
  const position = observation.bodyPositions.subarray(root * 3, root * 3 + 3);
  const rotation = observation.rotations.subarray(root * 9, root * 9 + 9);
  return Float32Array.from([...position, ...rotation]);
}

function targetFor(skill, resident, tick, random) {
  const target = new Float32Array(12);
  const side = resident % 2 ? -1 : 1;
  if (skill === 'pose-hold') {
    for (let joint = 0; joint < 12; joint++) target[joint] = (joint % 2 ? .26 : -.12) + side * ((joint % 4) - 1.5) * .025;
  } else if (skill === 'turn') {
    const phase = Math.sin((tick - 96) * Math.PI / 24), direction = ((episode + resident) % 2 ? -1 : 1);
    for (let leg = 0; leg < 6; leg++) {
      const legSide = leg < 3 ? -1 : 1, swing = direction * legSide * (.22 + .10 * phase);
      target[leg * 2] = swing; target[leg * 2 + 1] = .28 - .08 * phase * (leg % 2 ? -1 : 1);
    }
  } else if (skill === 'stop') {
    for (let joint = 0; joint < 12; joint++) target[joint] = joint % 2 ? .32 : 0;
  } else if (skill === 'reach') {
    for (let joint = 0; joint < 12; joint++) target[joint] = joint % 2 ? .30 : 0;
    const reachSide = (episode + resident) % 2 ? 0 : 3;
    target[reachSide * 2] = -.48; target[reachSide * 2 + 1] = -.22;
    target[(reachSide + 1) * 2] = -.30; target[(reachSide + 1) * 2 + 1] = .08;
  } else {
    for (let joint = 0; joint < 12; joint++) target[joint] = (random() * 2 - 1) * (joint % 2 ? .48 : .38);
  }
  return target;
}

function teacherMotor(body, target, skill, resident, tick, random) {
  const action = new Float32Array(MOTOR), first = resident * BODY;
  const cocontraction = skill === 'stop' ? .24 : skill === 'pose-hold' ? .13 : .08;
  for (let joint = 0; joint < 12; joint++) {
    const angle = body[first + 56 + joint], velocity = body[first + 68 + joint];
    let drive = 1.35 * (target[joint] - angle) - .20 * velocity;
    if (skill === 'babble') drive += (random() * 2 - 1) * .18;
    drive = clamp(drive, -.78, .78);
    action[joint * 2] = clamp(cocontraction + Math.max(0, drive), 0, 1);
    action[joint * 2 + 1] = clamp(cocontraction + Math.max(0, -drive), 0, 1);
  }
  action[24] = skill === 'reach' ? ((episode + resident) % 2 ? -.25 : .25) : 0;
  action[25] = skill === 'stop' || skill === 'pose-hold' ? .75 : .45;
  if (skill === 'reach' && tick % 32 >= 22) action[26] = .7;
  if (skill === 'babble') {
    action[27 + ((episode + resident + Math.floor((tick - 400) / 24)) % 3)] = .2;
    action[31] = tick % 48 < 8 ? 1 : 0;
  }
  return action;
}

function initializePose(world, random) {
  const snapshot = world.snapshot(), descriptor = [];
  for (let resident = 0; resident < BATCH; resident++) {
    const body = snapshot.fixture.bodies[resident], root = 1 + body.qpos[0] - 7;
    const dx = (random() - .5) * .12, dy = (random() - .5) * .12;
    snapshot.physical[root] += dx; snapshot.physical[root + 1] += dy;
    const yaw = (random() - .5) * .45;
    snapshot.physical[root + 3] = Math.cos(yaw / 2); snapshot.physical[root + 4] = 0;
    snapshot.physical[root + 5] = 0; snapshot.physical[root + 6] = Math.sin(yaw / 2);
    const joints = [];
    for (let joint = 0; joint < 12; joint++) {
      const angle = (random() - .5) * (joint % 2 ? .18 : .12);
      snapshot.physical[1 + body.qpos[joint]] = angle; joints.push(angle);
    }
    descriptor.push({ dx, dy, yaw, joints });
  }
  world.restore(snapshot); return descriptor;
}

const collectorSource = await readFile(fileURLToPath(import.meta.url));
const collectorSha = sha256(collectorSource), random = randomGenerator(VARIATION_SEEDS[episode]);
const fixturePath = resolve(worldDirectory, 'fixtures/garden.json');
const xmlPath = resolve(worldDirectory, 'fixtures/garden.xml');
const wasmPath = resolve(worldDirectory, 'pkg/chreatures_browser_world_bg.wasm');
const [fixtureBytes, xmlBytes, coreWasm] = await Promise.all([readFile(fixturePath), readFile(xmlPath), readFile(wasmPath)]);
const fixture = JSON.parse(fixtureBytes), xml = xmlBytes.toString('utf8');
const { createBrowserWorld, ACTION_NAMES, ENGINE } = await import(pathToFileURL(resolve(worldDirectory, 'runtime.mjs')));
assert.deepEqual(ACTION_NAMES, MOTOR_NAMES); assert.equal(ENGINE, 'mujoco-3.12.0-wasm-browser-epoch-2');
await mkdir(outputDirectory, { recursive: true });
const world = await createBrowserWorld({ fixture, xml, seed: WORLD_SEEDS[episode], coreWasm });
try {
  assert.equal(world.residents, BATCH);
  const initialPose = initializePose(world, random), initialSnapshot = world.snapshot();
  const layoutIdentity = sha256(Buffer.from(JSON.stringify({ model: initialSnapshot.model, atlas: initialSnapshot.atlas,
    world_seed: WORLD_SEEDS[episode], variation_seed: VARIATION_SEEDS[episode], initial_pose: initialPose })));
  const optic = new Float32Array((TICKS + 1) * BATCH * OPTIC);
  const body = new Float32Array((TICKS + 1) * BATCH * BODY);
  const context = new Float32Array(TICKS * BATCH * CONTEXT);
  const motor = new Float32Array(TICKS * BATCH * MOTOR);
  const jointTarget = new Float32Array(TICKS * BATCH * 12);
  const pose = new Float32Array((TICKS + 1) * BATCH * 12);
  const skillId = new Uint8Array(TICKS * BATCH);
  const reset = new Uint8Array((TICKS + 1) * BATCH); reset.fill(1, 0, BATCH);
  const terminal = new Uint8Array(TICKS * BATCH); terminal.fill(1, (TICKS - 1) * BATCH);
  const firstRoots = [], previousRoots = [], path = new Float64Array(BATCH), maximumJointError = new Float64Array(SKILLS.length);
  let currentToneBout = null;
  for (let tick = 0; tick <= TICKS; tick++) {
    const sensory = world.sample(), observation = world.observe();
    optic.set(sensory.optic, tick * BATCH * OPTIC); body.set(sensory.body, tick * BATCH * BODY);
    for (let resident = 0; resident < BATCH; resident++) {
      const root = rootPose(observation, resident); pose.set(root, (tick * BATCH + resident) * 12);
      if (tick === 0) { firstRoots.push(Array.from(root.subarray(0, 3))); previousRoots.push(Array.from(root.subarray(0, 3))); }
      else {
        const current = root.subarray(0, 3), previous = previousRoots[resident];
        path[resident] += Math.hypot(...Array.from(current, (value, axis) => value - previous[axis]));
        previousRoots[resident] = Array.from(current);
      }
    }
    if (tick === TICKS) break;
    const bout = currentBout(tick), skill = SKILLS.indexOf(bout.skill);
    if (bout !== currentToneBout) {
      currentToneBout = bout;
      if (bout.tone_hz !== null) {
        const duration = (bout.end_tick - bout.start_tick) * DT;
        for (let resident = 0; resident < BATCH; resident++) {
          const root = rootPose(observation, resident);
          world.visitorSound([root[0], root[1], root[2] + .12], bout.tone_hz, .8, duration);
        }
      }
    }
    const command = new Float32Array(BATCH * MOTOR);
    for (let resident = 0; resident < BATCH; resident++) {
      const target = targetFor(bout.skill, resident, tick, random);
      const action = teacherMotor(sensory.body, target, bout.skill, resident, tick, random);
      command.set(action, resident * MOTOR); jointTarget.set(target, (tick * BATCH + resident) * 12);
      skillId[tick * BATCH + resident] = skill;
      const first = resident * BODY;
      for (let joint = 0; joint < 12; joint++) maximumJointError[skill] = Math.max(maximumJointError[skill],
        Math.abs(target[joint] - sensory.body[first + 56 + joint]));
    }
    motor.set(command, tick * BATCH * MOTOR); world.advance(command, DT);
  }
  for (const values of [optic, body, context, motor, jointTarget, pose, path, maximumJointError]) {
    assert(Array.from(values).every(Number.isFinite));
  }
  const finalObservation = world.observe(), finalRoots = finalObservation.residents.map((_, resident) =>
    Array.from(rootPose(finalObservation, resident).subarray(0, 3)));
  const metadata = {
    format: 'chreatures-anatomical-cns-physical-episode-v1', episode_index: episode,
    split: episode < 6 ? 'train' : 'heldout-worlds', world_seed: WORLD_SEEDS[episode],
    variation_seed: VARIATION_SEEDS[episode], layout_identity: layoutIdentity,
    collector_sha256: collectorSha, curriculum_contract_sha256: sha256(Buffer.from(JSON.stringify({
      format: 'chreatures-anatomical-cns-physical-curriculum-v1', ticks: TICKS, residents: BATCH, dt: DT,
      skills: SKILLS, bouts: BOUTS, tones_hz: TONES_HZ, motor_names: MOTOR_NAMES,
      sensory: { optic_rgb: OPTIC, body: BODY }, context: CONTEXT,
    }))),
    source_mjcf_sha256: initialSnapshot.model, atlas_sha256: initialSnapshot.atlas,
    physical_engine: world.engine, dt_seconds: DT, transitions: TICKS, residents: BATCH,
    context_source: 'zero-initial-motor-bootstrap', old_abstract_action_relabelled_as_context: false,
    optic_body_are_actual_pre_action_samples: true,
    chronology: 'optic_rgb[t],body[t],delivered_context[t] precede delivered_motor[t] and physical state t+1',
    teacher_bouts: BOUTS, skill_names: SKILLS, tone_frequencies_hz: TONES_HZ,
    teacher_source: 'joint-target PD plus bounded exploration over actual BODY110 joint feedback',
    teacher_privileged_fields_retained: true, teacher_privileged_fields_are_model_inputs: false,
    privileged_target_members: ['teacher_joint_target', 'root_pose', 'skill_id'],
    model_input_members: ['optic_rgb', 'body', 'delivered_context'],
    motor_target_member: 'delivered_motor', raw_world_geometry_retained: false,
    body_order: 'ANATOMICAL_CNS_V3 BODY110', motor_order: MOTOR_NAMES,
    initial_pose: initialPose, diagnostics: { first_roots: firstRoots, final_roots: finalRoots,
      path_length_meters: Array.from(path), maximum_joint_tracking_error: Array.from(maximumJointError) },
  };
  const archive = npz({
    metadata: unicodeScalar(JSON.stringify(metadata)), optic_rgb: numpy('<f4', [TICKS + 1, BATCH, OPTIC], optic),
    body: numpy('<f4', [TICKS + 1, BATCH, BODY], body),
    delivered_context: numpy('<f4', [TICKS, BATCH, CONTEXT], context),
    delivered_motor: numpy('<f4', [TICKS, BATCH, MOTOR], motor),
    teacher_joint_target: numpy('<f4', [TICKS, BATCH, 12], jointTarget),
    root_pose: numpy('<f4', [TICKS + 1, BATCH, 12], pose), skill_id: numpy('|u1', [TICKS, BATCH], skillId),
    reset: numpy('|b1', [TICKS + 1, BATCH], reset), terminal: numpy('|b1', [TICKS, BATCH], terminal),
  });
  const stem = `episode-${String(episode).padStart(2, '0')}-${episode < 6 ? 'train' : 'heldout-world'}`;
  const episodePath = resolve(outputDirectory, `${stem}.npz`), receiptPath = resolve(outputDirectory, `${stem}.receipt.json`);
  await writeFile(episodePath, archive, { flag: 'wx' });
  const receipt = { format: 'chreatures-anatomical-cns-physical-collection-receipt-v1', completed: true,
    episode_file: `${stem}.npz`, episode_sha256: sha256(archive), metadata, source: {
      collector_sha256: collectorSha, runtime_sha256: sha256(await readFile(resolve(worldDirectory, 'runtime.mjs'))),
      core_wasm_sha256: sha256(coreWasm), fixture_sha256: sha256(fixtureBytes), xml_sha256: sha256(xmlBytes),
    } };
  await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ episode: episodePath, receipt: receiptPath, sha256: receipt.episode_sha256,
    split: metadata.split, diagnostics: metadata.diagnostics }));
} finally {
  world.dispose();
}

