#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Paired full-engine response to a decoded physical film versus a black screen.

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, open, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync, gunzipSync } from 'node:zlib';

const argv = process.argv.slice(2), args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) throw new Error('arguments must be --name value pairs');
  args[argv[index].slice(2)] = argv[index + 1];
}
if (!args.video || !args.report) {
  console.error('usage: node screen-response.mjs --video MP4 --report NEW_JSON [--traces NEW_JSON_GZ] [--site dist/site] [--ticks 600] [--width 160] [--height 120]');
  process.exit(2);
}
const site = resolve(args.site ?? '../../dist/site'), video = resolve(args.video), reportPath = resolve(args.report);
const tracesPath = args.traces ? resolve(args.traces) : null;
const ticks = Number(args.ticks ?? 600), width = Number(args.width ?? 160), height = Number(args.height ?? 120);
for (const [name, value, low, high] of [['ticks', ticks, 8, 1200], ['width', width, 2, 2048], ['height', height, 2, 2048]]) {
  if (!Number.isInteger(value) || value < low || value > high) throw new Error(`${name} must be an integer in [${low},${high}]`);
}
const DT = .05, NEURONS = 165122, OPTIC = 5313, ACTIONS = 12, EPSILON = 1e-6;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const temporary = await mkdtemp(join(tmpdir(), 'chreatures-screen-response-'));
const runner = resolve(dirname(fileURLToPath(import.meta.url)), 'screen-response-rollout.mjs');
const prefix = condition => join(temporary, condition);
const decodedFilm = join(temporary, 'film.rgb');
const ffmpegCommand = ['ffmpeg', '-v', 'error', '-i', video, '-an', '-vf',
  `fps=${1 / DT},scale=${width}:${height}:flags=lanczos`, '-frames:v', String(ticks),
  '-f', 'rawvideo', '-pix_fmt', 'rgb24', decodedFilm];

async function decodeFilm() {
  const child = spawn(ffmpegCommand[0], ffmpegCommand.slice(1), { stdio: ['ignore', 'ignore', 'inherit'] });
  const result = await new Promise((done, reject) => {
    child.once('error', reject); child.once('close', (code, signal) => done({ code, signal }));
  });
  if (result.code !== 0) throw new Error(`ffmpeg exited ${result.code ?? result.signal}`);
  assert.equal((await stat(decodedFilm)).size, ticks * width * height * 3);
}

async function run(condition) {
  const command = [runner, '--condition', condition, '--site', site, '--decoded', decodedFilm, '--output', prefix(condition),
    '--ticks', String(ticks), '--width', String(width), '--height', String(height)];
  if (args.backend) command.push('--backend', args.backend);
  const child = spawn(process.execPath, command, { stdio: ['ignore', 'ignore', 'pipe'] });
  child.stderr.pipe(process.stderr);
  const result = await new Promise((done, reject) => {
    child.once('error', reject); child.once('close', (code, signal) => done({ code, signal }));
  });
  if (result.code !== 0) throw new Error(`${condition} rollout exited ${result.code ?? result.signal}`);
}
const freshStats = () => ({ samples: 0, sum: 0, squares: 0, min: Infinity, max: -Infinity,
  maxAbs: 0, changedSamples: 0, positiveSamples: 0, negativeSamples: 0 });
function add(stats, value) {
  stats.samples++; stats.sum += value; stats.squares += value * value; stats.min = Math.min(stats.min, value);
  stats.max = Math.max(stats.max, value); stats.maxAbs = Math.max(stats.maxAbs, Math.abs(value));
  if (Math.abs(value) > EPSILON) stats.changedSamples++;
  if (value > EPSILON) stats.positiveSamples++; else if (value < -EPSILON) stats.negativeSamples++;
}
const finish = stats => ({ samples: stats.samples, signedMean: stats.sum / stats.samples,
  rms: Math.sqrt(stats.squares / stats.samples), min: stats.min, max: stats.max, maxAbs: stats.maxAbs,
  changedSamples: stats.changedSamples, positiveSamples: stats.positiveSamples, negativeSamples: stats.negativeSamples });
async function readAt(handle, buffer, position) {
  let offset = 0;
  while (offset < buffer.length) {
    const { bytesRead } = await handle.read(buffer, offset, buffer.length - offset, position + offset);
    if (!bytesRead) throw new Error('truncated rollout capture'); offset += bytesRead;
  }
}
const asF32 = buffer => new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);

try {
  await decodeFilm(); await run('film'); await run('blank');
  const [film, blank, cns, resident, runtime, fixture, videoBytes, decodedBytes] = await Promise.all([
    readFile(`${prefix('film')}.json`, 'utf8').then(JSON.parse), readFile(`${prefix('blank')}.json`, 'utf8').then(JSON.parse),
    readFile(resolve(site, 'live/model/cns-manifest.json'), 'utf8').then(JSON.parse),
    readFile(resolve(site, 'live/model/resident-manifest.json'), 'utf8').then(JSON.parse),
    readFile(resolve(site, 'live/runtime-manifest.json'), 'utf8').then(JSON.parse),
    readFile(resolve(site, 'live/fixtures/garden.json'), 'utf8').then(JSON.parse), readFile(video), readFile(decodedFilm),
  ]);
  assert.equal(cns.format, 'chreatures-cns-webgpu-v2'); assert.equal(cns.trainingStatus, 'trained');
  assert.equal(resident.trainingStatus, 'initialized-untrained'); assert.equal(cns.counts.neurons, NEURONS);
  assert.equal(film.engineIdentity, blank.engineIdentity); assert.deepEqual(film.initial, blank.initial);
  assert.equal(film.batch, blank.batch); assert.equal(film.bodyValues, blank.bodyValues);
  const maskSpec = cns.buffers['afferent.mask'];
  const maskTransport = await readFile(resolve(site, 'live/model', maskSpec.url));
  assert.equal(sha256(maskTransport), maskSpec.transportSha256);
  const maskBytes = gunzipSync(maskTransport); assert.equal(sha256(maskBytes), maskSpec.sha256);
  const mask = new Uint32Array(maskBytes.buffer, maskBytes.byteOffset, maskBytes.byteLength / 4);
  const groups = { afferent: [], nonafferent: [] };
  for (let neuron = 0; neuron < mask.length; neuron++) groups[mask[neuron] === 0 ? 'afferent' : 'nonafferent'].push(neuron);
  assert.equal(groups.afferent.length, cns.counts.afferents);
  const baseline = Float32Array.from(film.baseline), supported = Uint8Array.from(fixture.supported_sites, Number);
  assert.equal(baseline.length, NEURONS); assert.equal(supported.length, OPTIC / 3);
  const neural = Object.fromEntries(Object.keys(groups).map(name => [name, { film: freshStats(), blank: freshStats(),
    delta: freshStats(), responsive: new Uint8Array(NEURONS) }]));
  const retina = { delta: freshStats(), ever: new Uint8Array(supported.length), unsupported: new Uint8Array(supported.length) };
  const action = { film: freshStats(), blank: freshStats(), delta: freshStats(), changedTicks: 0, changedRows: 0,
    channels: Array.from({ length: ACTIONS }, freshStats) };
  const body = freshStats(), maxRoot = new Float64Array(film.batch);
  const trace = { format: 'chreatures-screen-response-trace-v1', dt: DT, filmTimeSeconds: [],
    afferentDeltaRms: [], nonafferentDeltaRms: [], afferentDeltaSignedMean: [], nonafferentDeltaSignedMean: [],
    retinalChangedSites: [], actionDeltaRms: [], bodyDeltaRms: [], filmRootPositions: film.rootTrace,
    blankRootPositions: blank.rootTrace, filmActions: [], blankActions: [] };
  const suffixes = ['rates.f32', 'retina.f32', 'actions.f32', 'body.f32'];
  const handles = await Promise.all(['film', 'blank'].flatMap(condition => suffixes.map(name => open(`${prefix(condition)}.${name}`, 'r'))));
  const sizes = [NEURONS * 4, OPTIC * 4, film.batch * ACTIONS * 4, film.bodyValues * 4];
  const buffers = Array.from({ length: 8 }, (_, index) => Buffer.allocUnsafe(sizes[index % 4]));
  try {
    for (let tick = 0; tick < ticks; tick++) {
      await Promise.all(handles.map((handle, index) => readAt(handle, buffers[index], tick * sizes[index % 4])));
      const filmRates = asF32(buffers[0]), filmRetina = asF32(buffers[1]), filmAction = asF32(buffers[2]), filmBody = asF32(buffers[3]);
      const blankRates = asF32(buffers[4]), blankRetina = asF32(buffers[5]), blankAction = asF32(buffers[6]), blankBody = asF32(buffers[7]);
      for (const name of Object.keys(groups)) {
        let sum = 0, squares = 0;
        for (const neuron of groups[name]) {
          const fv = filmRates[neuron] - baseline[neuron], bv = blankRates[neuron] - baseline[neuron];
          const difference = filmRates[neuron] - blankRates[neuron];
          add(neural[name].film, fv); add(neural[name].blank, bv); add(neural[name].delta, difference);
          if (Math.abs(difference) > EPSILON) neural[name].responsive[neuron] = 1;
          sum += difference; squares += difference * difference;
        }
        trace[`${name}DeltaSignedMean`].push(sum / groups[name].length);
        trace[`${name}DeltaRms`].push(Math.sqrt(squares / groups[name].length));
      }
      let changedSites = 0;
      for (let siteIndex = 0; siteIndex < supported.length; siteIndex++) {
        let changed = false;
        for (let channel = 0; channel < 3; channel++) {
          const index = siteIndex * 3 + channel, difference = filmRetina[index] - blankRetina[index];
          add(retina.delta, difference); if (Math.abs(difference) > EPSILON) changed = true;
          if (!supported[siteIndex] && (Math.abs(filmRetina[index]) > EPSILON || Math.abs(blankRetina[index]) > EPSILON)) retina.unsupported[siteIndex] = 1;
        }
        if (changed) { changedSites++; retina.ever[siteIndex] = 1; }
      }
      trace.retinalChangedSites.push(changedSites);
      let actionSquares = 0, tickChanged = false;
      for (let row = 0; row < film.batch; row++) {
        let rowChanged = false;
        for (let channel = 0; channel < ACTIONS; channel++) {
          const index = row * ACTIONS + channel, difference = filmAction[index] - blankAction[index];
          add(action.film, filmAction[index]); add(action.blank, blankAction[index]); add(action.delta, difference);
          add(action.channels[channel], difference); actionSquares += difference * difference;
          if (Math.abs(difference) > EPSILON) rowChanged = true;
        }
        if (rowChanged) { action.changedRows++; tickChanged = true; }
      }
      if (tickChanged) action.changedTicks++;
      trace.actionDeltaRms.push(Math.sqrt(actionSquares / filmAction.length));
      trace.filmActions.push(Array.from(filmAction)); trace.blankActions.push(Array.from(blankAction));
      let bodySquares = 0;
      for (let index = 0; index < filmBody.length; index++) {
        const difference = filmBody[index] - blankBody[index]; add(body, difference); bodySquares += difference * difference;
      }
      trace.bodyDeltaRms.push(Math.sqrt(bodySquares / filmBody.length)); trace.filmTimeSeconds.push(tick * DT);
      for (let row = 0; row < film.batch; row++) maxRoot[row] = Math.max(maxRoot[row],
        Math.hypot(...film.rootTrace[tick][row].map((value, axis) => value - blank.rootTrace[tick][row][axis])));
    }
  } finally { await Promise.all(handles.map(handle => handle.close())); }
  const timing = (values, threshold = EPSILON) => {
    const firstTick = values.findIndex(value => Math.abs(value) > threshold);
    const peakTick = values.reduce((best, value, tick) => Math.abs(value) > Math.abs(values[best]) ? tick : best, 0);
    return { firstTick: firstTick < 0 ? null : firstTick, firstModelSeconds: firstTick < 0 ? null : firstTick * DT,
      peakTick, peakModelSeconds: peakTick * DT, peakValue: values[peakTick] };
  };
  const neuralReport = name => ({ neurons: groups[name].length, filmFromBaseline: finish(neural[name].film),
    blankFromBaseline: finish(neural[name].blank), filmMinusBlank: finish(neural[name].delta),
    responsiveNeurons: neural[name].responsive.reduce((sum, value) => sum + value, 0) });
  const report = { format: 'chreatures-screen-response-v1', engineIdentity: film.engineIdentity,
    sourceRevision: runtime.sourceRevision, adapter: film.adapter,
    model: { cnsFormat: cns.format, cnsTrainingStatus: cns.trainingStatus, cnsArtifactSha256: cns.identity.artifact,
      cnsServiceArtifactSha256: cns.serviceArtifactSha256, residentTrainingStatus: resident.trainingStatus,
      residentArtifactSha256: resident.artifactSha256 },
    stimulus: { source: video, sourceSha256: sha256(videoBytes), sourceDurationSeconds: 30.03, decodedWidth: width,
      decodedHeight: height, decodedFrames: ticks, decodedRgbSha256: sha256(decodedBytes),
      ffmpegCommand, comparison: 'decoded film on physical emitting screen versus black physical screen' },
    execution: { ticks, batch: film.batch, modelSeconds: ticks * DT, dt: DT,
      filmMeanTickMs: film.wallMilliseconds.reduce((a, b) => a + b, 0) / ticks,
      blankMeanTickMs: blank.wallMilliseconds.reduce((a, b) => a + b, 0) / ticks },
    matchedInitialState: { ...film.initial, exactAcrossConditions: true }, threshold: { absoluteDifference: EPSILON },
    temporalResponse: { retina: timing(trace.retinalChangedSites, 0), afferentRates: timing(trace.afferentDeltaRms),
      nonafferentRates: timing(trace.nonafferentDeltaRms), actions: timing(trace.actionDeltaRms),
      bodyPositions: timing(trace.bodyDeltaRms) },
    retina: { anatomicalSites: supported.length, supportedSites: supported.reduce((sum, value) => sum + value, 0),
      capturedResident: 0, capturesPerCondition: ticks, filmMinusBlank: finish(retina.delta),
      everChangedSites: retina.ever.reduce((sum, value) => sum + value, 0),
      meanChangedSitesPerTick: trace.retinalChangedSites.reduce((a, b) => a + b, 0) / ticks,
      maxChangedSitesPerTick: Math.max(...trace.retinalChangedSites),
      unsupportedSitesEverNonzero: retina.unsupported.reduce((sum, value) => sum + value, 0) },
    neural: { capturedResident: 0, capturesPerCondition: ticks,
      afferent: neuralReport('afferent'), nonafferent: neuralReport('nonafferent') },
    action: { valuesPerCondition: ticks * film.batch * ACTIONS, changedTicks: action.changedTicks,
      changedResidentRows: action.changedRows, film: finish(action.film), blank: finish(action.blank),
      filmMinusBlank: finish(action.delta), filmMinusBlankByChannel: action.channels.map(finish) },
    physical: { bodyPositionFilmMinusBlank: finish(body), filmPathLengthMeters: film.pathLengthMeters,
      blankPathLengthMeters: blank.pathLengthMeters, maxRootDivergenceMeters: Array.from(maxRoot),
      finalRootDivergenceMeters: film.finalRoots.map((row, residentIndex) =>
        Math.hypot(...row.map((value, axis) => value - blank.finalRoots[residentIndex][axis]))) },
    scope: 'Actual current V2 browser engine on Node Dawn; film enters only through the physical screen. Initialized resident behavior is not a competence claim.' };
  const reportBytes = Buffer.from(JSON.stringify(report, null, 2) + '\n');
  await writeFile(reportPath, reportBytes, { flag: 'wx' });
  if (tracesPath) await writeFile(tracesPath, gzipSync(Buffer.from(JSON.stringify(trace))), { flag: 'wx' });
  console.log(JSON.stringify({ ...report, artifacts: { report: reportPath, reportSha256: sha256(reportBytes),
    traces: tracesPath, tracesSha256: tracesPath ? sha256(await readFile(tracesPath)) : null } }, null, 2));
} finally {
  await rm(temporary, { recursive: true, force: true });
}
