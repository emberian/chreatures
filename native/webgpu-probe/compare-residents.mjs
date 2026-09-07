#!/usr/bin/env node
// Matched, fresh-life comparison of two browser resident artifacts. Policy
// inputs stay inside LiveEngine's Z512 + previous A12/reset contract; physical
// state is read only after each committed action for evaluator metrics.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { create, globals } from 'webgpu';
import initResident from '../../site/live/pkg/resident_runtime.js';
import { LiveEngine } from '../../site/live/engine.js';

const argv = process.argv.slice(2), args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) throw new Error('arguments must be --name value pairs');
  args[argv[index].slice(2)] = argv[index + 1];
}
if (!args.parent || !args.trained || !args.report) {
  console.error('usage: node compare-residents.mjs --parent PACK --trained PACK --report NEW_JSON [--first-trained PACK] [--research-trained PACK] [--single-trained-label LABEL --reference-report JSON] [--site dist/site] [--ticks 512] [--backend metal|vulkan] [--world-seed N] [--variation-seed N]');
  process.exit(2);
}
const siteDirectory = resolve(args.site ?? '../../dist/site');
const packs = { parent: resolve(args.parent),
  ...(args['first-trained'] ? { 'first-trained': resolve(args['first-trained']) } : {}),
  ...(args['research-trained'] ? { 'research-trained': resolve(args['research-trained']) } : {}),
  trained: resolve(args.trained) };
const reportPath = resolve(args.report);
const singleTrainedLabel = args['single-trained-label'];
if (singleTrainedLabel && !args['reference-report']) throw new Error('--single-trained-label requires --reference-report');
const referenceReportBytes = args['reference-report'] ? await readFile(resolve(args['reference-report'])) : null;
const referenceReport = referenceReportBytes ? JSON.parse(referenceReportBytes) : null;
const ticks = Number(args.ticks ?? 512);
if (!Number.isInteger(ticks) || ticks < 96 || ticks > 512) throw new Error('ticks must be an integer in [96,512]');
const worldSeed = Number(args['world-seed'] ?? 20270331);
const variationSeed = Number(args['variation-seed'] ?? 271828);
if (![worldSeed, variationSeed].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff)) {
  throw new Error('world-seed and variation-seed must be uint32 integers');
}
const backend = args.backend ?? (process.platform === 'darwin' ? 'metal' : 'vulkan');
if (!['metal', 'vulkan'].includes(backend)) throw new Error('backend must be metal or vulkan');
const mime = { '.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream' };
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const rms = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
const wrapAngle = value => Math.atan2(Math.sin(value), Math.cos(value));
function summarizeCnsSnapshot(snapshot) {
  const bytes = new Uint8Array(snapshot), metadataLength = new DataView(snapshot).getUint32(8, true);
  const stateOffset = 12 + Math.ceil(metadataLength / 4) * 4;
  const metadata = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + metadataLength)));
  const { sourceRevision, ...stableIdentity } = metadata.identity;
  const canonicalMetadata = { ...metadata, identity: stableIdentity };
  return { snapshotSha256: sha256(bytes), stateSha256: sha256(bytes.subarray(stateOffset)),
    canonicalMetadataSha256: sha256(Buffer.from(JSON.stringify(canonicalMetadata))),
    sourceRevision, stateBytes: bytes.length - stateOffset };
}

const manifests = {};
for (const [name, directory] of Object.entries(packs)) {
  manifests[name] = {
    cns: JSON.parse(await readFile(resolve(directory, 'cns-manifest.json'), 'utf8')),
    resident: JSON.parse(await readFile(resolve(directory, 'resident-manifest.json'), 'utf8')),
  };
}
for (const [packName, manifest] of Object.entries(manifests)) {
  assert.equal(manifests.parent.cns.serviceArtifactSha256, manifest.cns.serviceArtifactSha256, `CNS service differs: ${packName}`);
  assert.equal(manifests.parent.cns.identity.artifact, manifest.cns.identity.artifact, `CNS adapter differs: ${packName}`);
  assert.equal(manifests.parent.resident.config.action_seed, manifest.resident.config.action_seed, `action RNG seed differs: ${packName}`);
  assert.equal(manifests.parent.resident.config.suffix_seed, manifest.resident.config.suffix_seed, `suffix RNG seed differs: ${packName}`);
  for (const [name, buffer] of Object.entries(manifests.parent.cns.buffers)) {
    assert.equal(buffer.sha256, manifest.cns.buffers[name]?.sha256, `CNS numeric buffer differs: ${packName}/${name}`);
  }
}

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const match = /^\/(parent|first-trained|research-trained|trained)(\/.*)$/.exec(pathname);
    if (!match) throw new Error('run prefix missing');
    const modelPrefix = '/live/model/';
    const modelRequest = match[2].startsWith(modelPrefix);
    const root = modelRequest ? packs[match[1]] : siteDirectory;
    const relative = modelRequest ? match[2].slice(modelPrefix.length) : match[2].slice(1);
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
// Keep Dawn's implementation alive across all asynchronous rollouts.
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' });
assert(adapter, `actual Dawn ${backend} adapter required`);
const device = await adapter.requestDevice();
const residentWasm = await readFile(resolve(siteDirectory, 'live/pkg/resident_runtime_bg.wasm'));
let residentInitialized = false;

const frame = tick => {
  const a = tick % 32 < 16 ? [0.92, 0.18, 0.06] : [0.04, 0.72, 0.95];
  const b = tick % 24 < 12 ? [0.12, 0.88, 0.22] : [0.78, 0.10, 0.82];
  return Float32Array.from([...a, ...b, ...b, ...a]);
};
const roots = observation => observation.residents.map(resident => {
  const p = resident.root * 3, r = resident.root * 9;
  return { position: Array.from(observation.bodyPositions.subarray(p, p + 3)),
    heading: Math.atan2(observation.rotations[r + 3], observation.rotations[r]) };
});

function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let x = value; x = Math.imul(x ^ x >>> 15, x | 1);
    x ^= x + Math.imul(x ^ x >>> 7, x | 61);
    return ((x ^ x >>> 14) >>> 0) / 4294967296;
  };
}

function geometryPosition(observation, name) {
  const geometry = observation.geometry.find(item => item.name === name);
  assert(geometry, `assay target is missing: ${name}`);
  return observation.positions.subarray(geometry.id * 3, geometry.id * 3 + 3);
}

const ASSAY_TARGETS = Object.freeze([
  { role: 'berry-resource', position: [3, .58], size: [.08, .08, .08], rgba: [.92, .07, .16, 1], food: .75, odor: 0 },
  { role: 'nectar-resource', position: [7, .58], size: [.07, .07, .07], rgba: [.98, .66, .06, 1], food: .68, odor: 1 },
  { role: 'movable-object', position: [11, .58], size: [.12, .12, .12], rgba: [.5, .18, .9, 1], food: 0, odor: 2 },
]);

async function configureAssay(world) {
  const random = randomGenerator(variationSeed), targets = [];
  for (const spec of ASSAY_TARGETS) {
    const position = [spec.position[0] + (random() - .5) * .14,
      spec.position[1] + (random() - .5) * .14, spec.size[0] + .012];
    const inserted = await world.insertObject({ position, size: spec.size, shape: 'sphere', rgba: spec.rgba,
      food: spec.food, odor: spec.odor });
    targets.push({ ...spec, id: inserted.id, geom: `entity:${inserted.id}:geom:0` });
  }
  const observation = world.observe(), snapshot = world.snapshot();
  const core = JSON.parse(snapshot.core); core.rng = worldSeed; snapshot.core = JSON.stringify(core);
  const starts = [];
  for (let resident = 0; resident < targets.length; resident++) {
    const target = geometryPosition(observation, targets[resident].geom);
    const radial = random() * Math.PI * 2, distance = .31 + random() * .08;
    const dx = Math.cos(radial) * distance, dy = Math.sin(radial) * distance;
    const body = snapshot.fixture.bodies[resident], offset = 1 + body.qpos[0] - 7;
    const desiredHeading = Math.atan2(-dy, -dx), headingOffset = (random() - .5) * 1.2;
    snapshot.physical[offset] = target[0] + dx; snapshot.physical[offset + 1] = target[1] + dy;
    snapshot.physical[offset + 2] = .09;
    const yaw = desiredHeading + headingOffset;
    snapshot.physical[offset + 3] = Math.cos(yaw / 2); snapshot.physical[offset + 4] = 0;
    snapshot.physical[offset + 5] = 0; snapshot.physical[offset + 6] = Math.sin(yaw / 2);
    starts.push({ distance, radial, headingOffset });
  }
  world.restore(snapshot);
  const finalSnapshot = world.snapshot();
  for (const target of targets) {
    target.foodIndex = finalSnapshot.fixture.entities.findIndex(entity => entity.id === target.id);
    assert(target.foodIndex >= 0);
  }
  const descriptor = { worldSeed, variationSeed, model: finalSnapshot.model,
    targets: targets.map((target, resident) => ({ role: target.role, geom: target.geom,
      position: Array.from(geometryPosition(world.observe(), target.geom)), start: starts[resident] })) };
  return { targets, layoutIdentity: sha256(Buffer.from(JSON.stringify(descriptor))) };
}

function assayMeasurements(observation, targets) {
  return roots(observation).map((root, resident) => {
    const target = geometryPosition(observation, targets[resident].geom);
    return { root, target: Array.from(target), distance: Math.hypot(target[0] - root.position[0],
      target[1] - root.position[1], target[2] - root.position[2]), food: observation.food[targets[resident].foodIndex] };
  });
}

function physiology(world) {
  const state = JSON.parse(world.snapshot().core);
  return state.residents.map(resident => ({ channels: resident.physiology,
    assimilableReserveEnergyPlus0_8Gut: resident.physiology[0] + .8 * resident.physiology[1],
    energy: resident.physiology[0], gut: resident.physiology[1], fatigue: resident.physiology[2],
    workAtSnapshot: resident.work }));
}

async function rollout(label, packName, controller) {
  let engine;
  const failures = [], started = performance.now();
  try {
    engine = await LiveEngine.create({ baseURL: `http://127.0.0.1:${server.address().port}/${packName}/live/`, device,
      modules: { initResident: residentInitialized ? async () => {} : async () => {
        await initResident({ module_or_path: residentWasm }); residentInitialized = true;
      } } });
    const assay = await configureAssay(engine.world);
    const initialWorld = engine.world.snapshot();
    const initialCns = summarizeCnsSnapshot(await engine.brain.snapshot());
    const firstMeasurement = assayMeasurements(engine.observe(), assay.targets);
    const initialPhysiology = physiology(engine.world);
    const first = firstMeasurement.map(item => item.root), previous = first.map(item => item.position);
    const path = new Float64Array(engine.batch), turn = new Float64Array(engine.batch);
    const netTurn = new Float64Array(engine.batch), physicalStops = new Uint32Array(engine.batch);
    const slowTicks = new Uint32Array(engine.batch), nearTargetTicks = new Uint32Array(engine.batch);
    const envelopeProximityTicks = new Uint32Array(engine.batch), longestStopRun = new Uint32Array(engine.batch);
    const currentStopRun = new Uint32Array(engine.batch), minTargetDistance = Float64Array.from(firstMeasurement, item => item.distance);
    const initialFood = Float64Array.from(firstMeasurement, item => item.food);
    const commands = new Float32Array(ticks * engine.batch * 12);
    const timing = [];
    for (let tick = 0; tick < ticks; tick++) {
      engine.screen(frame(tick), 2, 2, tick * 0.05);
      let observation;
      const tickStarted = performance.now();
      if (controller === 'resident') {
        observation = await engine.advance(false);
        commands.set(engine.previous, tick * engine.batch * 12);
      } else {
        engine.world.advance(new Float32Array(engine.batch * 12), .05);
        observation = engine.world.observe();
      }
      timing.push(controller === 'resident' ? observation.wallMilliseconds : performance.now() - tickStarted);
      const current = roots(observation);
      const measurement = assayMeasurements(observation, assay.targets);
      for (let resident = 0; resident < engine.batch; resident++) {
        const dx = current[resident].position[0] - previous[resident][0];
        const dy = current[resident].position[1] - previous[resident][1];
        const dz = current[resident].position[2] - previous[resident][2];
        const distance = Math.hypot(dx, dy, dz), delta = wrapAngle(current[resident].heading - first[resident].heading - netTurn[resident]);
        path[resident] += distance; turn[resident] += Math.abs(delta); netTurn[resident] += delta;
        if (distance <= 0.0005) physicalStops[resident]++;
        if (distance <= 0.0015) slowTicks[resident]++;
        if (distance <= 0.0005) currentStopRun[resident]++; else currentStopRun[resident] = 0;
        longestStopRun[resident] = Math.max(longestStopRun[resident], currentStopRun[resident]);
        minTargetDistance[resident] = Math.min(minTargetDistance[resident], measurement[resident].distance);
        if (measurement[resident].distance < .34) nearTargetTicks[resident]++;
        if (measurement[resident].distance < assay.targets[resident].size[0] + .18) envelopeProximityTicks[resident]++;
        previous[resident] = current[resident].position;
      }
      if (![...observation.bodyPositions, ...commands.subarray(tick * engine.batch * 12, (tick + 1) * engine.batch * 12)].every(Number.isFinite)) {
        throw new Error(`non-finite physics or command at tick ${tick}`);
      }
    }
    const finalMeasurement = assayMeasurements(engine.observe(), assay.targets);
    const finalPhysiology = physiology(engine.world);
    const last = finalMeasurement.map(item => item.root), channelRms = [], channelStd = [];
    for (let channel = 0; channel < 12; channel++) {
      const values = [];
      for (let index = channel; index < commands.length; index += 12) values.push(commands[index]);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      channelRms.push(rms(values));
      channelStd.push(Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length));
    }
    let stepDeltaSquared = 0, stepDeltaCount = 0;
    for (let tick = 1; tick < ticks; tick++) for (let resident = 0; resident < engine.batch; resident++) {
      let squared = 0;
      for (let channel = 0; channel < 12; channel++) {
        const index = (tick * engine.batch + resident) * 12 + channel;
        squared += (commands[index] - commands[index - engine.batch * 12]) ** 2;
      }
      stepDeltaSquared += squared; stepDeltaCount++;
    }
    const distinct = new Set();
    for (let index = 0; index < commands.length; index += 12) {
      distinct.add(Array.from(commands.subarray(index, index + 12), value => Math.round(value * 1000)).join(','));
    }
    let meanSquaredEffort = 0;
    for (let index = 0; index < commands.length; index += 12) {
      meanSquaredEffort += commands[index] ** 2 + commands[index + 1] ** 2;
      for (let channel = 2; channel < 12; channel++) meanSquaredEffort += .2 * commands[index + channel] ** 2;
    }
    meanSquaredEffort /= commands.length / 12;
    return {
      label, controller, residentArtifactSha256: manifests[packName].resident.artifactSha256,
      trainingStatus: controller === 'resident' ? manifests[packName].resident.trainingStatus : 'zero-command physical baseline',
      transitions: ticks, batch: engine.batch, layoutIdentity: assay.layoutIdentity,
      initialWorldSha256: sha256(Buffer.from(JSON.stringify(initialWorld))), initialCns,
      rng: { worldSeed, variationSeed, actionSeed: manifests[packName].resident.config.action_seed,
        suffixSeed: manifests[packName].resident.config.suffix_seed },
      action: { overallRms: rms(commands), channelRms, channelStd,
        meanChannelStd: channelStd.reduce((a, b) => a + b, 0) / channelStd.length,
        distinctRowsRounded1e3: distinct.size, totalRows: ticks * engine.batch,
        rootMeanSquareStepDeltaL2: Math.sqrt(stepDeltaSquared / stepDeltaCount), meanSquaredEffort },
      physical: {
        netDisplacementMeters: last.map((item, resident) => Math.hypot(...item.position.map((value, axis) => value - first[resident].position[axis]))),
        pathLengthMeters: Array.from(path), cumulativeAbsoluteTurnRadians: Array.from(turn),
        netTurnRadians: Array.from(netTurn), physicalStopTicksAt1cmPerSecond: Array.from(physicalStops),
        slowTicksAt3cmPerSecond: Array.from(slowTicks), longestConsecutiveStopTicks: Array.from(longestStopRun),
        initialTargetDistanceMeters: firstMeasurement.map(item => item.distance),
        minimumTargetDistanceMeters: Array.from(minTargetDistance),
        finalTargetDistanceMeters: finalMeasurement.map(item => item.distance),
        targetProximityTicksBelow34cm: Array.from(nearTargetTicks),
        physicalEnvelopeProximityTicks: Array.from(envelopeProximityTicks),
        initialTargetFood: Array.from(initialFood), finalTargetFood: finalMeasurement.map(item => item.food),
        netTargetFoodStockLoss: finalMeasurement.map((item, resident) => Math.max(0, initialFood[resident] - item.food)),
        foodStockInterpretation: 'Net target stock loss after physical ingestion and concurrent resource growth; it is not gross ingestion.',
        initialPhysiology, finalPhysiology,
        assimilableReserveChange: finalPhysiology.map((item, resident) => item.assimilableReserveEnergyPlus0_8Gut -
          initialPhysiology[resident].assimilableReserveEnergyPlus0_8Gut),
        fatigueChange: finalPhysiology.map((item, resident) => item.fatigue - initialPhysiology[resident].fatigue),
      },
      execution: { failed: false, failures, meanTickMs: timing.reduce((a, b) => a + b, 0) / timing.length,
        maxTickMs: Math.max(...timing), wallSeconds: (performance.now() - started) / 1000 },
    };
  } catch (error) {
    failures.push(String(error?.stack ?? error));
    return { label, controller, residentArtifactSha256: manifests[packName].resident.artifactSha256,
      trainingStatus: controller === 'resident' ? manifests[packName].resident.trainingStatus : 'zero-command physical baseline',
      transitionsAttempted: engine?.tick ?? 0,
      execution: { failed: true, failures, wallSeconds: (performance.now() - started) / 1000 } };
  } finally { engine?.destroy(); }
}

let report;
try {
  const results = [];
  const arms = singleTrainedLabel ? [[singleTrainedLabel, 'trained', 'resident']] : [['parent', 'parent', 'resident']];
  if (!singleTrainedLabel) {
    if (packs['first-trained']) arms.push(['first-trained', 'first-trained', 'resident']);
    if (packs['research-trained']) arms.push(['candidate01-future-key-contaminated', 'research-trained', 'resident']);
    arms.push(['trained', 'trained', 'resident'], ['zero-command', 'parent', 'zero']);
  }
  for (const [label, packName, controller] of arms) {
    const result = await rollout(label, packName, controller); results.push(result);
    console.error(`${label}: ${result.execution.failed ? 'FAILED' : `${ticks} committed transitions`}`);
  }
  const residentResults = results.filter(result => result.controller === 'resident');
  const referenceParent = singleTrainedLabel ? referenceReport?.results?.find(result => result.label === 'parent') : null;
  const matched = !results.some(result => result.execution.failed) && (singleTrainedLabel ?
    referenceParent && results[0].initialWorldSha256 === referenceParent.initialWorldSha256 &&
      results[0].layoutIdentity === referenceParent.layoutIdentity &&
      results[0].initialCns.stateSha256 === referenceParent.initialCns.stateSha256 &&
      results[0].initialCns.canonicalMetadataSha256 === referenceParent.initialCns.canonicalMetadataSha256 :
    residentResults.every(result => result.initialCns.stateSha256 === residentResults[0].initialCns.stateSha256 &&
      result.initialCns.canonicalMetadataSha256 === residentResults[0].initialCns.canonicalMetadataSha256) &&
      results.every(result => result.layoutIdentity === results[0].layoutIdentity && result.initialWorldSha256 === results[0].initialWorldSha256));
  report = { format: 'chreatures-autonomous-physical-outcome-assay-v1', backend,
    adapter: adapter.info?.device || adapter.info?.description || `Dawn ${backend}`,
    adapterLimits: { maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
    policyInputContract: ['cns_latent[512]', 'previous_delivered_command[12]', 'reset'],
    evaluatorOnlyFields: ['resident root bodyPositions', 'resident root rotations', 'assay target geometry positions', 'target food quantity'],
    teacherGeometryUsedByPolicy: false, cnsServiceArtifactSha256: manifests.parent.cns.serviceArtifactSha256,
    cnsAdapterSha256: manifests.parent.cns.identity.artifact, cnsNumericBuffersMatched: true,
    worldSeedMatched: true, actionAndSuffixSeedsMatched: true, initialWorldAndCnsStateMatched: matched,
    initialCnsComparison: 'decoded GPU state bytes and host metadata after removing only identity.sourceRevision; full snapshot hashes remain in each result',
    worldSeed, variationSeed, zeroCommandBaselineIncluded: results.some(result => result.controller === 'zero'),
    ...(referenceReportBytes ? {referenceReport: resolve(args['reference-report']), referenceReportSha256: sha256(referenceReportBytes)} : {}),
    screenSchedule: 'shared deterministic 2x2 color sequence; no geometry-derived policy input', results,
    interpretation: 'Matched fresh-life autonomous outcomes. There is no hidden teacher clock or requested goal; proximity, stopping, food transfer, motion, turning and effort are descriptive outcomes and do not establish goal conditioning or general embodied competence.' };
  await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report, null, 2));
  if (!matched || results.some(result => result.execution.failed)) process.exitCode = 1;
} finally {
  device.destroy(); delete globalThis.__chreaturesDawnInstance;
  server.closeAllConnections(); server.close();
}
