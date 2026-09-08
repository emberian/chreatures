// Actual headless Wasm body + WebGPU full CNS + Wasm private resident integration.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {resolve, extname, sep} from 'node:path';
import {createHash} from 'node:crypto';
import {create, globals} from 'webgpu';
import initResident from '../../site/live/pkg/resident_runtime.js';
import {LiveEngine} from '../../site/live/engine.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, item, index, values) => index % 2 ? pairs : [...pairs, [item.slice(2), values[index + 1]]], []));
const directory = resolve(args.site ?? '../../dist/site');
const mime = {'.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream'};
const server = createServer(async (req, res) => {
  try {
    const file = resolve(directory, '.' + new URL(req.url, 'http://localhost').pathname);
    if (!file.startsWith(directory + sep)) throw new Error('path');
    const info = await stat(file); if (!info.isFile()) throw new Error('file');
    res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Content-Length', info.size); createReadStream(file).pipe(res);
  } catch {res.statusCode = 404; res.end('missing');}
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
Object.assign(globalThis, globals);
const dawn = create(['backend=metal']);
// Dawn's native owner must outlive every WebGPU object.
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({powerPreference: 'high-performance'});
assert(adapter, 'Actual GPU adapter required');
const device = await adapter.requestDevice();
let engine;
const began = performance.now();
const exact = (a, b) => Buffer.from(a).equals(Buffer.from(b));
const sha = bytes => createHash('sha256').update(Buffer.from(bytes)).digest('hex');
try {
  const residentWasm = await readFile(resolve(directory, 'live/pkg/resident_runtime_bg.wasm'));
  engine = await LiveEngine.create({baseURL: `http://127.0.0.1:${server.address().port}/live/`, device,
    modules: {worldWasm: await readFile(resolve(directory, 'live/pkg/chreatures_browser_world_bg.wasm')),
      initResident: () => initResident({module_or_path: residentWasm})}});
  const atlas = engine.describe();
  assert.equal(atlas.retinalSites.length, 1771 * 3);
  assert.equal(atlas.retinalSupported.reduce((sum, value) => sum + value, 0), 1486);
  const first = engine.observe(); const timings = [];
  const motorLow = new Float32Array(34).fill(Infinity), motorHigh = new Float32Array(34).fill(-Infinity);
  let contextMagnitude = 0;
  const observedFields = new Set(), fields = ['rate', 'adaptation', 'support', 'release', 'dopamine', 'octopamine', 'serotonin'];
  for (let tick = 0; tick < 64; tick++) {
    engine.neuralField = fields[Math.min(6, Math.floor(tick / 9))];
    if (tick === 4) engine.greet([0, 1, 2]);
    if ([16, 28, 40, 52].includes(tick)) engine.tone([80, 200, 500, 1250][(tick - 16) / 12], .4);
    engine.screen(new Float32Array(12).fill(tick % 4 < 2 ? 1 : 0), 2, 2, null);
    const frame = await engine.advance(true); timings.push(frame.wallMilliseconds);
    assert(frame.positions.every(Number.isFinite));
    assert.equal(frame.neuralField, engine.neuralField);
    assert.equal(frame.neuralSignal.length, 165122);
    assert(frame.neuralSignal.every(Number.isFinite));
    if (!observedFields.has(frame.neuralField)) {
      const snapshot = await engine.brain.snapshot();
      const headerBytes = new DataView(snapshot).getUint32(8, true);
      const state = new Float32Array(snapshot, 12 + Math.ceil(headerBytes / 4) * 4);
      const field = fields.indexOf(frame.neuralField);
      for (let neuron = 0; neuron < 165122; neuron++) {
        assert.equal(frame.neuralSignal[neuron], state[neuron * 28 + field * 4 + engine.selected]);
      }
      observedFields.add(frame.neuralField);
    }
    assert.equal(frame.motorActivation.length, 34);
    assert.equal(frame.deliveredContext.length, 12);
    for (let i = 0; i < 34; i++) {
      assert(Number.isFinite(frame.motorActivation[i]));
      motorLow[i] = Math.min(motorLow[i], frame.motorActivation[i]);
      motorHigh[i] = Math.max(motorHigh[i], frame.motorActivation[i]);
    }
    contextMagnitude = Math.max(contextMagnitude, ...frame.deliveredContext.map(Math.abs));
    if (frame.neuralRates) {
      assert(frame.neuralRates.every(Number.isFinite));
      assert.equal(frame.retinalRGB.length, 5313);
      assert(frame.retinalRGB.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    }
  }
  const checkpoint = await engine.save();
  const next = await engine.advance(true); const future = await engine.save();
  await engine.load(checkpoint);
  const replay = await engine.advance(true); const restoredFuture = await engine.save();
  assert(exact(next.positions.buffer, replay.positions.buffer), 'Physical continuation differs');
  assert(exact(next.neuralRates.buffer, replay.neuralRates.buffer), 'Full CNS continuation differs');
  assert(exact(next.motorActivation.buffer, replay.motorActivation.buffer), 'Motor recruitment continuation differs');
  assert(exact(next.deliveredContext.buffer, replay.deliveredContext.buffer), 'Context delivery continuation differs');
  assert(exact(future, restoredFuture), 'Complete life replay differs');
  const beforeInsert = engine.observe().geometry.length;
  const inserted = await engine.insertToy();
  engine.shove(inserted.id, [.1, 0, 0]);
  const grown = await engine.save(); await engine.advance(); await engine.load(grown);
  assert.equal(engine.observe().geometry.length, beforeInsert + 1);
  const after = engine.observe();
  let maxTravel = 0;
  for (const resident of after.residents) {
    const index = resident.root * 3;
    maxTravel = Math.max(maxTravel, Math.hypot(...after.bodyPositions.slice(index, index + 3).map((x, i) => x - first.bodyPositions[index + i])));
  }
  const report = {format: 'chreatures-anatomical-cns-v3-joined-headless-v1', engineIdentity: engine.identity,
    serviceArtifactSha256: engine.brain.manifest.serviceArtifactSha256,
    adapterSha256: engine.brain.manifest.identity.artifact,
    adapter: adapter.info?.device || adapter.info?.description || 'Dawn Metal', neurons: 165122, edges: 25563197,
    residents: engine.batch, physicalStepCalls: 67, retainedModelTicks: engine.tick, checkpointReplayCalls: 1, modelSeconds: engine.world.time,
    meanCompleteTickMs: timings.reduce((a,b) => a+b, 0) / timings.length, maxCompleteTickMs: Math.max(...timings),
    maxRootTravelMeters: maxTravel, snapshotBytes: checkpoint.byteLength, snapshotSHA256: sha(checkpoint),
    physicalReplayExact: true, fullNeuralReplayExact: true, wholeLifeReplayExact: true,
    observedNeuralFields: [...observedFields], observerMatchesPrivateStateExactly: true,
    contextOutputs: 12, anatomicalMotorOutputs: 34, physicalBodyInputs: 110, privateNeuralFields: 7,
    motorDynamicRanges: Array.from(motorHigh, (x, i) => x - motorLow[i]), maximumDeliveredContextMagnitude: contextMagnitude,
    neuralCapture: 'every physical tick', retinalInputSites: 1771, supportedRetinalSites: 1486, capturedRetinalRGB: true,
    restoredGrownWorld: true, geometryCount: after.geometry.length, wallSeconds: (performance.now() - began) / 1000,
    modelStatus: engine.modelStatus, controllerStatus: engine.controllerStatus,
    scope: 'Actual Node Dawn Metal + same browser Wasm/WGSL; no browser UI performance or learned motor competence claim'};
  console.log(JSON.stringify(report, null, 2));
  if (args.report) await writeFile(args.report, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
} finally {
  engine?.destroy(); device.destroy(); server.closeAllConnections(); server.close();
}
