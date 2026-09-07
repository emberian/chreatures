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
  const first = engine.observe(); const timings = [];
  for (let tick = 0; tick < 16; tick++) {
    if (tick === 4) engine.greet([0, 1, 2]);
    engine.screen(new Float32Array(12).fill(tick % 4 < 2 ? 1 : 0), 2, 2, null);
    const frame = await engine.advance(tick % 4 === 0); timings.push(frame.wallMilliseconds);
    assert(frame.positions.every(Number.isFinite));
    if (frame.neuralRates) assert(frame.neuralRates.every(Number.isFinite));
  }
  const checkpoint = await engine.save();
  const next = await engine.advance(true); const future = await engine.save();
  await engine.load(checkpoint);
  const replay = await engine.advance(true); const restoredFuture = await engine.save();
  assert(exact(next.positions.buffer, replay.positions.buffer), 'Physical continuation differs');
  assert(exact(next.neuralRates.buffer, replay.neuralRates.buffer), 'Full CNS continuation differs');
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
  const report = {format: 'chreatures-live-joined-headless-v1', engineIdentity: engine.identity,
    adapter: adapter.info?.device || adapter.info?.description || 'Dawn Metal', neurons: 165122, edges: 25563197,
    residents: engine.batch, physicalStepCalls: 19, retainedModelTicks: engine.tick, checkpointReplayCalls: 1, modelSeconds: engine.world.time,
    meanCompleteTickMs: timings.reduce((a,b) => a+b, 0) / timings.length, maxCompleteTickMs: Math.max(...timings),
    maxRootTravelMeters: maxTravel, snapshotBytes: checkpoint.byteLength, snapshotSHA256: sha(checkpoint),
    physicalReplayExact: true, fullNeuralReplayExact: true, wholeLifeReplayExact: true,
    restoredGrownWorld: true, geometryCount: after.geometry.length, wallSeconds: (performance.now() - began) / 1000,
    modelStatus: engine.modelStatus, controllerStatus: engine.controllerStatus,
    scope: 'Actual Node Dawn Metal + same browser Wasm/WGSL; no browser UI performance or learned motor competence claim'};
  console.log(JSON.stringify(report, null, 2));
  if (args.report) await writeFile(args.report, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
} finally {
  engine?.destroy(); device.destroy(); server.closeAllConnections(); server.close();
}
