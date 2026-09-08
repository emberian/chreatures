#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Matched actual-world assay for initialized versus physical-bootstrap CHCNS3.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { create, globals } from 'webgpu';

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2), args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) throw new Error('arguments must be --name value pairs');
  args[argv[index].slice(2)] = argv[index + 1];
}
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const BATCH = 3, TICKS = 512, DT = .05, MOTOR = 34, BODY = 110;
const TONE_SKILLS = Object.freeze([
  { skill: 'pose-hold', tone_hz: 80 }, { skill: 'turn', tone_hz: 200 },
  { skill: 'stop', tone_hz: 500 }, { skill: 'reach', tone_hz: 1250 },
]);
const LATIN_ORDERS = Object.freeze([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]);
const BOUTS = Object.freeze([
  ...LATIN_ORDERS.flatMap((order, block) => order.map((skillIndex, position) => ({
    start_tick: (block * 4 + position) * 28, end_tick: (block * 4 + position + 1) * 28,
    block, position, skill_index: skillIndex, ...TONE_SKILLS[skillIndex],
  }))),
  { start_tick: 448, end_tick: 512, block: 4, position: 0, skill_index: 4, skill: 'babble', tone_hz: null },
]);

function child(command) {
  return new Promise((done, fail) => {
    const childProcess = spawn(process.execPath, command, { stdio: ['ignore', 'inherit', 'inherit'] });
    childProcess.on('error', fail); childProcess.on('exit', code => code === 0 ? done() : fail(new Error(`assay arm exited ${code}`)));
  });
}

if (!args.arm) {
  if (!args.initialized || !args.trained || !args.output) {
    console.error('usage: node assay-anatomical-bootstrap.mjs --initialized PACK --trained PACK --output NEW_DIR [--world native/browser-world] [--backend metal|vulkan]');
    process.exit(2);
  }
  const output = resolve(args.output); await mkdir(output, { recursive: false });
  const common = ['--world', resolve(args.world ?? resolve(here, '../browser-world')), '--backend', args.backend ?? (process.platform === 'darwin' ? 'metal' : 'vulkan')];
  for (const [arm, model] of [['initialized', args.initialized], ['trained', args.trained]]) {
    await child([fileURLToPath(import.meta.url), '--arm', arm, '--model', resolve(model), '--report', resolve(output, `${arm}.json`), ...common]);
  }
  const initialized = JSON.parse(await readFile(resolve(output, 'initialized.json')));
  const trained = JSON.parse(await readFile(resolve(output, 'trained.json')));
  assert.equal(initialized.initial_world_sha256, trained.initial_world_sha256, 'initial physical worlds differ');
  assert.equal(initialized.world_source.model, trained.world_source.model, 'physical model differs');
  assert.equal(initialized.world_source.runtime_sha256, trained.world_source.runtime_sha256, 'world runtime differs');
  const result = {
    format: 'chreatures-anatomical-cns-physical-bootstrap-assay-v1', completed: true,
    source_sha256: sha256(await readFile(fileURLToPath(import.meta.url))),
    policy_input_contract: ['actual optic_rgb[5313]', 'actual BODY110', 'delivered context12'],
    context_source: 'exact zero; initialized context organ does not participate in this motor bootstrap assay',
    teacher_or_observer_fields_used_by_policy: false,
    matched: { initial_world_sha256: initialized.initial_world_sha256, world_seed: initialized.world_seed,
      schedule: BOUTS, ticks: TICKS, batch: BATCH },
    initialized, trained,
    deltas: {
      mean_squared_motor_effort: trained.action.mean_squared_effort - initialized.action.mean_squared_effort,
      stopped_ticks: trained.physical.stopped_ticks - initialized.physical.stopped_ticks,
      mean_joint_target_error: trained.physical.mean_joint_target_error - initialized.physical.mean_joint_target_error,
      total_path_meters: trained.physical.path_meters.reduce((a, b) => a + b, 0) - initialized.physical.path_meters.reduce((a, b) => a + b, 0),
    },
    interpretation: 'Matched actual tone/proprioceptive closed loop. Differences establish behavior of the physical-bootstrap CNS artifact, not autonomous goals or biological muscle homology.',
  };
  const path = resolve(output, 'comparison.json'); await writeFile(path, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ report: path, sha256: sha256(await readFile(path)), deltas: result.deltas }));
  process.exit(0);
}

if (!['initialized', 'trained'].includes(args.arm) || !args.model || !args.report || !args.world) throw new Error('invalid internal assay arm');
Object.assign(globalThis, globals);
const modelDirectory = resolve(args.model), worldDirectory = resolve(args.world);
const manifestBytes = await readFile(resolve(modelDirectory, 'cns-manifest.json'));
const manifest = JSON.parse(manifestBytes), assets = new Map();
for (const [name, entry] of Object.entries(manifest.buffers)) {
  const transport = await readFile(resolve(modelDirectory, entry.url));
  if (transport.byteLength !== entry.transportByteLength || sha256(transport) !== entry.transportSha256) throw new Error(`transport differs: ${name}`);
  const raw = entry.encoding === 'gzip' ? gunzipSync(transport) : transport;
  if (raw.byteLength !== entry.byteLength || sha256(raw) !== entry.sha256) throw new Error(`asset differs: ${name}`);
  assets.set(name, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}
const motorRows = new Uint32Array(assets.get('atlas.motor_rows')).slice();
const motorMask = new Float32Array(assets.get('atlas.motor_mask')).slice();
const motorWeightRaw = new Float32Array(assets.get('motor.weight_raw')).slice();
const motorBias = new Float32Array(assets.get('motor.bias')).slice();
const [fixtureBytes, xmlBytes, coreWasm] = await Promise.all([
  readFile(resolve(worldDirectory, 'fixtures/garden.json')),
  readFile(resolve(worldDirectory, 'fixtures/garden.xml')),
  readFile(resolve(worldDirectory, 'pkg/chreatures_browser_world_bg.wasm')),
]);
const fixture = JSON.parse(fixtureBytes), xml = xmlBytes.toString('utf8');
const [{ MaleCNSWebGPU }, { createBrowserWorld }] = await Promise.all([
  import(pathToFileURL(resolve(here, '../../site/live/cns-webgpu.js'))),
  import(pathToFileURL(resolve(worldDirectory, 'runtime.mjs'))),
]);
const shaderSources = Object.fromEntries(await Promise.all(['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl', 'motor.wgsl'].map(async name =>
  [name, await readFile(resolve(here, '../../site/live/shaders', name), 'utf8')])));
const backend = args.backend ?? (process.platform === 'darwin' ? 'metal' : 'vulkan');
const dawn = create([`backend=${backend}`]); globalThis.__chreaturesAnatomicalAssayDawn = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' }); assert(adapter);
const needed = Math.max(...Object.values(manifest.buffers).map(entry => entry.byteLength));
const device = await adapter.requestDevice({ requiredLimits: {
  maxBufferSize: Math.max(needed, 128 * 1024 * 1024), maxStorageBufferBindingSize: Math.max(needed, 128 * 1024 * 1024),
} });
const brain = await MaleCNSWebGPU.load({ device, manifest, capacity: BATCH, assetBuffers: assets, shaderSources });
const world = await createBrowserWorld({ fixture, xml, seed: 3031923, coreWasm });
try {
  await brain.reset([0, 1, 2], ['assay-resident-0', 'assay-resident-1', 'assay-resident-2']);
  const initialWorld = Buffer.from(JSON.stringify(world.snapshot()));
  const initialObservation = world.observe();
  const soundOrigins = initialObservation.residents.map(resident => {
    const at = resident.root * 3; return Array.from(initialObservation.bodyPositions.subarray(at, at + 3));
  });
  const previousRoots = soundOrigins.map(row => row.slice()), path = new Float64Array(BATCH);
  const actionSum = new Float64Array(MOTOR), actionSquare = new Float64Array(MOTOR), actionCount = TICKS * BATCH;
  const skillMotorSum = Array.from({ length: 4 }, () => new Float64Array(MOTOR)), skillMotorCount = new Uint32Array(4);
  const mnSum = new Float64Array(motorRows.length), mnSquare = new Float64Array(motorRows.length);
  const dynamicSum = new Float64Array(MOTOR), dynamicSquare = new Float64Array(MOTOR);
  const baselineRates = brain.baselineRates;
  const effectiveMotorWeight = Float64Array.from(motorWeightRaw, (value, index) =>
    motorMask[index] ? Math.log1p(Math.exp(-Math.abs(value))) + Math.max(value, 0) : 0);
  const baselinePreactivation = Float64Array.from(motorBias);
  for (let channel = 0; channel < MOTOR; channel++) for (let neuron = 0; neuron < motorRows.length; neuron++)
    baselinePreactivation[channel] += effectiveMotorWeight[channel * motorRows.length + neuron] * baselineRates[motorRows[neuron]];
  let effort = 0, stopped = 0, jointError = 0, jointSamples = 0, currentBout = null, failures = 0;
  for (let tick = 0; tick < TICKS; tick++) {
    const bout = BOUTS.find(value => tick >= value.start_tick && tick < value.end_tick); assert(bout);
    if (bout !== currentBout) {
      currentBout = bout;
      if (bout.tone_hz !== null) for (const origin of soundOrigins) world.visitorSound([origin[0], origin[1], origin[2] + .12], bout.tone_hz, .8, (bout.end_tick - bout.start_tick) * DT);
    }
    const sensory = world.sample(), context = new Float32Array(BATCH * 12);
    const neural = await brain.step({ dt: DT, activeMask: 7, opticRGB: sensory.optic, body: sensory.body, context,
      selectedResident: tick % BATCH });
    const motor = neural.motor;
    const deviation = new Float64Array(motorRows.length);
    for (let neuron = 0; neuron < motorRows.length; neuron++) {
      const value = neural.selectedRates[motorRows[neuron]] - baselineRates[motorRows[neuron]];
      deviation[neuron] = value; mnSum[neuron] += value; mnSquare[neuron] += value * value;
    }
    for (let channel = 0; channel < MOTOR; channel++) {
      let value = 0;
      for (let neuron = 0; neuron < motorRows.length; neuron++)
        value += effectiveMotorWeight[channel * motorRows.length + neuron] * deviation[neuron];
      dynamicSum[channel] += value; dynamicSquare[channel] += value * value;
    }
    for (let resident = 0; resident < BATCH; resident++) for (let channel = 0; channel < MOTOR; channel++) {
      const value = motor[resident * MOTOR + channel], signed = channel === 24 || channel === 25;
      if (!Number.isFinite(value) || value < (signed ? -1 : 0) || value > 1) failures++;
      actionSum[channel] += value; actionSquare[channel] += value * value; effort += value * value;
      if (bout.skill_index < 4) skillMotorSum[bout.skill_index][channel] += value;
    }
    if (bout.skill_index < 4) skillMotorCount[bout.skill_index] += BATCH;
    if (failures) throw new Error('CNS motor output violated physical bounds');
    world.advance(motor, DT);
    const after = world.observe(), afterSensory = world.sample();
    for (let resident = 0; resident < BATCH; resident++) {
      const root = after.residents[resident].root, at = root * 3;
      const current = Array.from(after.bodyPositions.subarray(at, at + 3)), previous = previousRoots[resident];
      const distance = Math.hypot(...current.map((value, axis) => value - previous[axis]));
      path[resident] += distance; if (distance / DT < .01) stopped++; previousRoots[resident] = current;
      if (bout.skill !== 'babble') {
        const target = new Float32Array(12), side = resident % 2 ? -1 : 1;
        if (bout.skill === 'pose-hold') for (let joint = 0; joint < 12; joint++) target[joint] = (joint % 2 ? .26 : -.12) + side * ((joint % 4) - 1.5) * .025;
        else if (bout.skill === 'turn') {
          const phase = Math.sin((tick - bout.start_tick) * Math.PI / 14), direction = ((8 + resident) % 2 ? -1 : 1);
          for (let leg = 0; leg < 6; leg++) { const legSide = leg < 3 ? -1 : 1;
            target[leg * 2] = direction * legSide * (.22 + .10 * phase); target[leg * 2 + 1] = .28 - .08 * phase * (leg % 2 ? -1 : 1); }
        } else if (bout.skill === 'stop') for (let joint = 0; joint < 12; joint++) target[joint] = joint % 2 ? .32 : 0;
        else { for (let joint = 0; joint < 12; joint++) target[joint] = joint % 2 ? .30 : 0;
          const reachSide = (8 + resident) % 2 ? 0 : 3; target[reachSide * 2] = -.48; target[reachSide * 2 + 1] = -.22;
          target[(reachSide + 1) * 2] = -.30; target[(reachSide + 1) * 2 + 1] = .08; }
        for (let joint = 0; joint < 12; joint++) { const difference = afterSensory.body[resident * BODY + 56 + joint] - target[joint]; jointError += difference * difference; jointSamples++; }
      }
    }
  }
  const channelStd = Array.from(actionSum, (sum, channel) => Math.sqrt(Math.max(0, actionSquare[channel] / actionCount - (sum / actionCount) ** 2)));
  const motorMeanBySkill = Object.fromEntries(TONE_SKILLS.map((value, skill) => [value.skill,
    Array.from(skillMotorSum[skill], sum => sum / skillMotorCount[skill])]));
  const responseDistances = [];
  for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) responseDistances.push(Math.sqrt(
    motorMeanBySkill[TONE_SKILLS[a].skill].reduce((sum, value, channel) => sum + (value - motorMeanBySkill[TONE_SKILLS[b].skill][channel]) ** 2, 0) / MOTOR));
  const mnStd = Array.from(mnSum, (sum, neuron) => Math.sqrt(Math.max(0, mnSquare[neuron] / TICKS - (sum / TICKS) ** 2)));
  const dynamicStd = Array.from(dynamicSum, (sum, channel) => Math.sqrt(Math.max(0, dynamicSquare[channel] / TICKS - (sum / TICKS) ** 2)));
  const sortedMnStd = mnStd.slice().sort((a, b) => a - b);
  const report = {
    format: 'chreatures-anatomical-cns-physical-bootstrap-arm-v1', arm: args.arm, completed: true,
    model: { manifest_sha256: sha256(manifestBytes), service_artifact_sha256: manifest.serviceArtifactSha256,
      adapter_sha256: manifest.identity.artifact, training_status: manifest.trainingStatus },
    world_source: { model: fixture.source_mjcf_sha256, atlas: fixture.atlas_sha256,
      runtime_sha256: sha256(await readFile(resolve(worldDirectory, 'runtime.mjs'))), core_wasm_sha256: sha256(coreWasm) },
    backend, adapter: adapter.info?.device || adapter.info?.description || 'Dawn', world_seed: 3031923,
    initial_world_sha256: sha256(initialWorld), ticks: TICKS, batch: BATCH, context_nonzero: 0,
    policy_inputs: ['optic_rgb', 'body110', 'zero delivered context12'], execution_failures: failures,
    action: { mean_squared_effort: effort / actionCount / MOTOR, mean_channel_std: channelStd.reduce((a, b) => a + b, 0) / MOTOR,
      channel_std: channelStd, motor_mean_by_tone_skill: motorMeanBySkill,
      counterbalanced_tone_response_pairwise_rms_mean: responseDistances.reduce((a, b) => a + b, 0) / responseDistances.length,
      counterbalanced_tone_response_pairwise_rms_min: Math.min(...responseDistances) },
    physical: { path_meters: Array.from(path), stopped_ticks: stopped, mean_joint_target_error: jointError / jointSamples },
    motor_neuron_activity: {
      sampled_resident_schedule: 'tick modulo 3; rates are never policy inputs', samples: TICKS,
      rate_deviation_temporal_std_median: sortedMnStd[Math.floor(sortedMnStd.length / 2)],
      rate_deviation_temporal_std_p95: sortedMnStd[Math.floor(sortedMnStd.length * .95)],
      rate_deviation_temporal_std_max: sortedMnStd.at(-1),
      neurons_std_above_1e_5: mnStd.filter(value => value > 1e-5).length,
      neurons_std_above_1e_4: mnStd.filter(value => value > 1e-4).length,
      neurons_std_above_1e_3: mnStd.filter(value => value > 1e-3).length,
      motor_neurons: motorRows.length,
      decoder_baseline_preactivation_min: Math.min(...baselinePreactivation),
      decoder_baseline_preactivation_max: Math.max(...baselinePreactivation),
      activity_dependent_preactivation_std_mean: dynamicStd.reduce((a, b) => a + b, 0) / MOTOR,
      activity_dependent_preactivation_std_max: Math.max(...dynamicStd),
    },
  };
  await writeFile(resolve(args.report), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ arm: args.arm, report: resolve(args.report), physical: report.physical, action: report.action }));
} finally {
  world.dispose(); brain.destroy(); device.destroy(); delete globalThis.__chreaturesAnatomicalAssayDawn;
}
