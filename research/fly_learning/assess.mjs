#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
/** Physical three-arm assessment for a trained full-CNS V4 child. */

import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {access, mkdir, readFile, stat, writeFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {extname, resolve, sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const require = createRequire(new URL('../../native/webgpu-probe/package.json', import.meta.url));
const {create, globals} = require('webgpu');

const argv = process.argv.slice(2);
const args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined)
    throw new Error('arguments must be --name value pairs');
  args[argv[index].slice(2)] = argv[index + 1];
}
for (const required of ['site-10', 'site-11', 'initialized-model', 'trained-model', 'learned-model', 'output'])
  if (!args[required]) throw new Error(`missing --${required}`);
const diagnostic = args.diagnostic === 'true';
const observerCapture = diagnostic && args.observer === 'true';
const ticks = Number(args.ticks ?? (diagnostic ? 25 : 1024));
if (!Number.isSafeInteger(ticks) || (diagnostic ? ticks !== 25 : ticks < 800 || ticks > 1024))
  throw new Error('--ticks must be 25 for diagnosis or a fixed 8–10.24 second assessment window');

const output = resolve(args.output);
await mkdir(output, {recursive: true});
Object.assign(globalThis, globals);
const dawn = create(['backend=metal']);
globalThis.__chreaturesAssessmentDawnInstance = dawn;
const adapter = await dawn.requestAdapter({powerPreference: 'high-performance'});
assert(adapter, 'Actual Dawn Metal adapter required');
const device = await adapter.requestDevice();
const sha256 = bytes => createHash('sha256').update(Buffer.from(bytes)).digest('hex');
const hashFile = async path => sha256(await readFile(path));
const finite = (value, label) => {
  if (!Number.isFinite(value)) throw new Error(`${label} is nonfinite`);
  return value;
};
const mime = {'.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream'};

function startServer(siteDirectory, modelDirectory) {
  const site = resolve(siteDirectory), model = resolve(modelDirectory);
  const server = createServer(async (request, response) => {
    try {
      const pathname = new URL(request.url, 'http://localhost').pathname;
      const root = pathname.startsWith('/live/model/') ? model : site;
      const suffix = pathname.startsWith('/live/model/') ? pathname.slice('/live/model/'.length) : `.${pathname}`;
      const file = resolve(root, suffix);
      if (!file.startsWith(root + sep)) throw new Error('path escapes root');
      const info = await stat(file);
      if (!info.isFile()) throw new Error('not a file');
      response.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
      response.setHeader('Content-Length', info.size);
      createReadStream(file).pipe(response);
    } catch {
      response.statusCode = 404;
      response.end('missing');
    }
  });
  return new Promise(resolveServer => server.listen(0, '127.0.0.1', () => resolveServer(server)));
}

function reserve(organism) {
  return finite(organism.atp, 'ATP') + organism.material.quantity.reduce((sum, value) => sum + finite(value, 'material'), 0);
}

function summarizeResident(accumulator, ticksRun) {
  const elapsed = ticksRun * .01;
  return {
    id: accumulator.id,
    root_displacement_mm: Math.hypot(...accumulator.lastPosition.map((v, i) => v - accumulator.firstPosition[i])),
    root_path_mm: accumulator.path,
    mean_path_speed_mm_s: accumulator.path / elapsed,
    upright_threshold: .5,
    upright_duration_s: accumulator.uprightTicks * .01,
    foot_contact_duration_s: accumulator.footTicks.map(value => value * .01),
    foot_force_integral_model_units_s: accumulator.footForce.map(value => value * .01),
    mouth_contact_duration_s: accumulator.mouthTicks * .01,
    mouth_contact_bouts: accumulator.mouthBouts,
    mouth_contact_intervals_s: accumulator.mouthIntervals.map(([start, stop]) => [start * .01, stop * .01]),
    measured_oral_transfer_model_units: accumulator.pumpTransfer,
    measured_salivary_transfer_model_units: accumulator.salivaryTransfer,
    mechanical_work_model_units: accumulator.work,
    initial_resource_model_units: accumulator.initialReserve,
    final_resource_model_units: accumulator.finalReserve,
    resource_change_model_units: accumulator.finalReserve - accumulator.initialReserve,
    antenna_joint_path_rad: accumulator.antennaPath,
    proboscis_joint_path_rad: accumulator.proboscisPath,
    motor_min: accumulator.motorLow,
    motor_max: accumulator.motorHigh,
    motor_range: accumulator.motorHigh.map((value, i) => value - accumulator.motorLow[i]),
    motor_mean_absolute_effort: accumulator.motorAbs.map(value => value / ticksRun),
    declared_windows: Object.fromEntries(Object.entries(accumulator.windows).map(([name, window]) =>
      [name, {ticks: window.ticks, root_path_mm: window.path, mean_path_speed_mm_s: window.ticks ? window.path / (window.ticks * .01) : 0}])),
    tone_responses: accumulator.tones,
  };
}

const schedule = Array.from({length: 16}, (_, event) => ({
  tick: event * 64,
  frequency_hz: [100, 630, 250, 1000][event % 4],
  duration_s: .64,
  amplitude: .65,
  screen_rgb_2x2: Array.from({length: 12}, (_, channel) => ((event + channel * 3) % 5) / 4),
}));
const declaredWindows = {locomotion_early: [64, 192], stop_early: [192, 256],
  locomotion_late: [384, 512], stop_late: [704, 768]};

async function assessCondition(layout, arm) {
  const server = await startServer(layout.site, arm.model);
  let engine;
  try {
    const [{default: initResident}, {LiveEngine}] = await Promise.all([
      import(pathToFileURL(resolve(layout.site, 'live/pkg/resident_runtime.js'))),
      import(pathToFileURL(resolve(layout.site, 'live/engine.js'))),
    ]);
    const residentWasm = await readFile(resolve(layout.site, 'live/pkg/resident_runtime_bg.wasm'));
    engine = await LiveEngine.create({
      baseURL: `http://127.0.0.1:${server.address().port}/live/`, device,
      modules: {
        worldWasm: await readFile(resolve(layout.site, 'live/pkg/chreatures_browser_world_bg.wasm')),
        initResident: () => initResident({module_or_path: residentWasm}),
      },
    });
    assert.equal(engine.batch, 4);
    const initialWorld = engine.world.snapshot();
    const initialWorldSha256 = sha256(Buffer.from(JSON.stringify(initialWorld)));
    if (layout.initialWorldSha256 && layout.initialWorldSha256 !== initialWorldSha256)
      throw new Error('condition did not start from the identical physical world state');
    layout.initialWorldSha256 = initialWorldSha256;
    const firstRaw = engine.world.researchObserve();
    const bodies = firstRaw.bodyMap.bodies;
    const organisms = new Map(firstRaw.ecology.organisms.map(item => [item.id, item]));
    const accumulators = bodies.map((body, row) => {
      const root = body.root * 3;
      const semantic = firstRaw.bodyMap.residents[row].actuators90.map(item => item.semantic_id);
      const antenna = semantic.flatMap((name, i) => /pedicel/.test(name) ? [body.qpos[i]] : []);
      const proboscis = semantic.flatMap((name, i) => /(rostrum|haustellum)/.test(name) ? [body.qpos[i]] : []);
      return {id: body.id, ecologyId: body.ecology_id, firstPosition: Array.from(firstRaw.bodyPositions.slice(root, root + 3)),
        lastPosition: Array.from(firstRaw.bodyPositions.slice(root, root + 3)), path: 0, uprightTicks: 0,
        footTicks: Array(6).fill(0), footForce: Array(6).fill(0), mouthTicks: 0, mouthBouts: 0,
        mouthActive: false, mouthStart: null, mouthIntervals: [], pumpTransfer: 0, salivaryTransfer: 0, work: 0,
        initialReserve: reserve(organisms.get(body.ecology_id)), finalReserve: 0,
        motorLow: Array(92).fill(Infinity), motorHigh: Array(92).fill(-Infinity), motorAbs: Array(92).fill(0),
        lastAntenna: antenna.map(address => firstRaw.qpos[address]), lastProboscis: proboscis.map(address => firstRaw.qpos[address]),
        antennaAddresses: antenna, proboscisAddresses: proboscis, antennaPath: 0, proboscisPath: 0,
        windows: Object.fromEntries(Object.keys(declaredWindows).map(name => [name, {ticks: 0, path: 0}])),
        tones: [], trace: []};
    });
    const began = performance.now();
    const observerFrames = [];
    for (let tick = 0; tick < ticks; tick++) {
      const event = schedule.find(item => item.tick === tick);
      if (event) {
        engine.screen(Float32Array.from(event.screen_rgb_2x2), 2, 2, tick * .01);
        engine.tone(event.frequency_hz, event.duration_s, event.amplitude);
      }
      if (arm.zeroContext) {
        while (engine.externalEvents.length && engine.externalEvents[0].tick <= engine.tick) {
          const sound = engine.externalEvents.shift();
          engine.world.visitorSound(sound.position, sound.frequency, sound.amplitude, sound.duration);
        }
        const {optic, body: bodyInput} = engine.world.sample();
        const context = new Float32Array(4 * 12);
        const neural = await engine.brain.step({dt: .01, activeMask: 0b1111,
          opticRGB: optic, body: bodyInput, context,
          ...(observerCapture ? {selectedResident: 0, selectedField: 'rate'} : {})});
        if (neural.motor.length !== 4 * 92 || !neural.motor.every(Number.isFinite))
          throw new Error('zero-context CNS motor output differs');
        await engine.world.advance(neural.motor, .01);
        engine.lastMotor = neural.motor.slice();
        engine.deliveredContext = context;
        engine.pendingContext.fill(0);
        // Keep the life envelope clock coherent. This receipt is never sent to
        // the private resident in the zero-context arm.
        engine.pendingTick = engine.tick;
        engine.tick++;
        if (observerCapture && [0, 10, 24].includes(tick)) observerFrames.push({
          tick, world_snapshot: engine.world.snapshot(),
          rate_state_f32_base64: Buffer.from(neural.selectedSignal.buffer, neural.selectedSignal.byteOffset, neural.selectedSignal.byteLength).toString('base64'),
        });
      } else {
        const frame = await engine.advance(observerCapture);
        if (observerCapture && [0, 10, 24].includes(tick)) observerFrames.push({
          tick, world_snapshot: engine.world.snapshot(),
          rate_state_f32_base64: Buffer.from(frame.neuralSignal.buffer, frame.neuralSignal.byteOffset, frame.neuralSignal.byteLength).toString('base64'),
        });
      }
      const raw = engine.world.researchObserve();
      const byId = new Map(raw.ecology.organisms.map(item => [item.id, item]));
      for (let row = 0; row < 4; row++) {
        const body = raw.bodyMap.bodies[row], acc = accumulators[row], root = body.root * 3;
        const position = Array.from(raw.bodyPositions.slice(root, root + 3));
        const stepDistance = Math.hypot(...position.map((value, i) => value - acc.lastPosition[i]));
        acc.path += stepDistance;
        for (const [name, [start, stop]] of Object.entries(declaredWindows)) if (tick >= start && tick < stop) {
          acc.windows[name].ticks++;
          acc.windows[name].path += stepDistance;
        }
        acc.lastPosition = position;
        if (raw.bodyRotations[body.root * 9 + 8] > .5) acc.uprightTicks++;
        for (let foot = 0; foot < 6; foot++) {
          const address = row * 807 + 459 + foot * 6;
          const force = Math.hypot(raw.bodyAfferents[address], raw.bodyAfferents[address + 1], raw.bodyAfferents[address + 2]);
          if (force > 1e-12) acc.footTicks[foot]++;
          acc.footForce[foot] += force;
        }
        const contact = raw.bodyAfferents[row * 807 + 711] > 0;
        if (contact) acc.mouthTicks++;
        if (contact && !acc.mouthActive) { acc.mouthBouts++; acc.mouthStart = tick; }
        if (!contact && acc.mouthActive) { acc.mouthIntervals.push([acc.mouthStart, tick]); acc.mouthStart = null; }
        acc.mouthActive = contact;
        const actuator = raw.actuatorState[row];
        acc.pumpTransfer += finite(actuator.pump_flow, 'pump flow') * .01;
        acc.salivaryTransfer += finite(actuator.salivary_flow, 'salivary flow') * .01;
        acc.work += finite(actuator.work, 'mechanical work');
        const motor = engine.lastMotor.subarray(row * 92, (row + 1) * 92);
        for (let channel = 0; channel < 92; channel++) {
          acc.motorLow[channel] = Math.min(acc.motorLow[channel], motor[channel]);
          acc.motorHigh[channel] = Math.max(acc.motorHigh[channel], motor[channel]);
          acc.motorAbs[channel] += Math.abs(motor[channel]);
        }
        if (diagnostic) acc.trace.push({
          tick,
          motor92: Array.from(motor),
          applied_ctrl90: body.actuators.map(address => raw.ctrl[address]),
          q126_rad: body.qpos.map(address => raw.qpos[address]),
          qdot126_rad_s: body.dofs.map(address => raw.qvel[address]),
          root_position_mm: position,
          upright_r22: raw.bodyRotations[body.root * 9 + 8],
          foot_contact: Array.from({length: 6}, (_, foot) => {
            const address = row * 807 + 459 + foot * 6;
            const vector = Array.from(raw.bodyAfferents.slice(address, address + 3));
            return {force_model_units: vector, active: Math.hypot(...vector) > 1e-12};
          }),
          adhesion_command6: Array.from(motor.slice(84, 90)),
          delivered_context12: Array.from(engine.deliveredContext.slice(row * 12, (row + 1) * 12)),
        });
        for (const [key, addresses, lastKey, pathKey] of [
          ['antenna', acc.antennaAddresses, 'lastAntenna', 'antennaPath'],
          ['proboscis', acc.proboscisAddresses, 'lastProboscis', 'proboscisPath'],
        ]) for (let j = 0; j < addresses.length; j++) {
          const value = raw.qpos[addresses[j]];
          acc[pathKey] += Math.abs(value - acc[lastKey][j]);
          acc[lastKey][j] = value;
        }
        if (event) acc.tones.push({tick, frequency_hz: event.frequency_hz, start_position_mm: position,
          start_upright: raw.bodyRotations[body.root * 9 + 8], antenna_joint_path_rad_next_32_ticks: 0,
          proboscis_joint_path_rad_next_32_ticks: 0, root_path_mm_next_32_ticks: 0,
          _antennaStart: acc.antennaPath, _proboscisStart: acc.proboscisPath});
        for (const response of acc.tones) if (tick > response.tick && tick <= response.tick + 32) {
          response.root_path_mm_next_32_ticks += stepDistance;
          // These totals are differences over observed physical joint coordinates.
          response.antenna_joint_path_rad_next_32_ticks = acc.antennaPath - response._antennaStart;
          response.proboscis_joint_path_rad_next_32_ticks = acc.proboscisPath - response._proboscisStart;
        }
        acc.finalReserve = reserve(byId.get(acc.ecologyId));
      }
    }
    const checkpoint = new Uint8Array(await engine.save());
    for (const acc of accumulators) if (acc.mouthActive) acc.mouthIntervals.push([acc.mouthStart, ticks]);
    const report = {
      format: 'chreatures-fly-cns-v4-physical-assessment-condition-v1', layout: layout.name, arm: arm.name,
      ticks, seconds: ticks * .01, controller_inputs: ['CNS latent512', 'private context12 feedback'],
      context_intervention: arm.zeroContext ? 'exact f32 zeros[4,12] delivered each .01 s tick' : 'artifact-bound private context12',
      service_artifact_sha256: engine.brain.manifest.serviceArtifactSha256,
      adapter_sha256: engine.brain.manifest.identity.artifact, engine_identity: engine.identity,
      model_release_sha256: await hashFile(resolve(arm.model, 'release.json')),
      initial_world_sha256: initialWorldSha256, final_life_sha256: sha256(checkpoint), final_life_bytes: checkpoint.byteLength,
      stimulus_schedule: schedule.filter(item => item.tick < ticks), declared_windows: declaredWindows,
      ...(observerCapture ? {observer_capture: {
        resident: 0, field: 'rate', units: 'dimensionless model rate state; baseline-subtracted only in the rendered diagnostic',
        soma_positions_f32_base64: Buffer.from(engine.brainPositions.buffer, engine.brainPositions.byteOffset, engine.brainPositions.byteLength).toString('base64'),
        soma_valid_u8_base64: Buffer.from(engine.brainValid).toString('base64'),
        baseline_rate_f32_base64: Buffer.from(engine.neuralBaseline.buffer, engine.neuralBaseline.byteOffset, engine.neuralBaseline.byteLength).toString('base64'),
        frames: observerFrames,
      }} : {}),
      residents: accumulators.map(item => {
        const summary = summarizeResident(item, ticks);
        if (diagnostic) {
          summary.first_25_tick_trace = item.trace;
          const body = firstRaw.bodyMap.bodies.find(candidate => candidate.id === item.id);
          summary.engineered_neutral_servo_targets84_rad = body.neutral.slice(0, 84);
          summary.engineered_servo_control_ranges84_rad = body.control_ranges.slice(0, 84);
        }
        for (const response of summary.tone_responses) {
          delete response._antennaStart;
          delete response._proboscisStart;
        }
        return summary;
      }), wall_seconds: (performance.now() - began) / 1000,
      scope: 'Actual Dawn Metal full MaleCNS V4, private resident Wasm, MuJoCo fly bodies and native finite ecology. Physical outcomes only; no independent-tick or competence claim.',
    };
    return report;
  } finally {
    engine?.destroy();
    server.closeAllConnections();
    await new Promise(done => server.close(done));
  }
}

try {
  const layoutNumbers = diagnostic ? [10] : [10, 11];
  const layouts = layoutNumbers.map(number => ({name: `heldout-${number}`, site: resolve(args[`site-${number}`]), initialWorldSha256: null}));
  const arms = [
    {name: 'initialized-cns_initialized-private', model: resolve(args['initialized-model']), zeroContext: false},
    {name: 'trained-cns_zero-context', model: resolve(args['trained-model']), zeroContext: true},
    {name: 'trained-cns_learned-private', model: resolve(args['learned-model']), zeroContext: false},
  ];
  const conditions = [];
  for (const layout of layouts) for (const arm of arms) {
    const filename = `${layout.name}--${arm.name}.json`;
    const path = resolve(output, filename);
    let report;
    try {
      await access(path);
      report = JSON.parse(await readFile(path, 'utf8'));
      if (report.format !== 'chreatures-fly-cns-v4-physical-assessment-condition-v1' ||
          report.layout !== layout.name || report.arm !== arm.name || report.ticks !== ticks)
        throw new Error(`existing condition receipt differs: ${filename}`);
      if (layout.initialWorldSha256 && layout.initialWorldSha256 !== report.initial_world_sha256)
        throw new Error(`existing condition physical start differs: ${filename}`);
      layout.initialWorldSha256 = report.initial_world_sha256;
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      report = await assessCondition(layout, arm);
      await writeFile(path, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
    }
    conditions.push({file: filename, sha256: await hashFile(resolve(output, filename)),
      layout: layout.name, arm: arm.name, initial_world_sha256: report.initial_world_sha256});
  }
  const receipt = {format: diagnostic ? 'chreatures-fly-cns-v4-instability-diagnosis-v1' : 'chreatures-fly-cns-v4-physical-assessment-v1', ticks, conditions,
    identical_physical_starts_by_layout: Object.fromEntries(layouts.map(item => [item.name, item.initialWorldSha256])),
    stimulus_schedule_sha256: sha256(Buffer.from(JSON.stringify(schedule.filter(item => item.tick < ticks)))),
    adapter: adapter.info?.device || adapter.info?.description || 'Dawn Metal'};
  await writeFile(resolve(output, 'assessment-receipt.json'), JSON.stringify(receipt, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  device.destroy();
}
