// SPDX-License-Identifier: AGPL-3.0-or-later
// Phase profile for the actual headless Rust/Wasm + WebGPU + MuJoCo engine.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {readFile, stat, writeFile} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {create, globals} from 'webgpu';

const args = Object.fromEntries(process.argv.slice(2).reduce(
  (pairs, item, index, values) => index % 2 ? pairs : [...pairs, [item.slice(2), values[index + 1]]], []));
const directory = resolve(args.site ?? '../../dist/site');
const [{default: initResident}, {LiveEngine}] = await Promise.all([
  import(pathToFileURL(resolve(directory, 'live/pkg/resident_runtime.js'))),
  import(pathToFileURL(resolve(directory, 'live/engine.js'))),
]);
const ticks = Number(args.ticks ?? 32);
if (!Number.isInteger(ticks) || ticks < 8 || ticks > 512) throw new Error('ticks must be an integer in 8..512');
const mime = {'.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream'};
const server = createServer(async (req, res) => {
  try {
    const file = resolve(directory, '.' + new URL(req.url, 'http://localhost').pathname);
    if (!file.startsWith(directory + sep)) throw new Error('path');
    const info = await stat(file); if (!info.isFile()) throw new Error('file');
    res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Content-Length', info.size); createReadStream(file).pipe(res);
  } catch { res.statusCode = 404; res.end('missing'); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
Object.assign(globalThis, globals);
const dawn = create(['backend=metal']);
const adapter = await dawn.requestAdapter({powerPreference: 'high-performance'});
assert(adapter, 'Actual GPU adapter required');
const device = await adapter.requestDevice();
let engine;
const samples = Object.fromEntries(['worldSample', 'brainStep', 'residentStep', 'worldAdvance',
  'residentAcknowledge', 'worldObserve', 'completeTick'].map(name => [name, []]));
const timedSync = (object, method, key) => {
  const original = object[method].bind(object);
  object[method] = (...values) => {
    const began = performance.now();
    try { return original(...values); }
    finally { samples[key].push(performance.now() - began); }
  };
};
const timedAsync = (object, method, key) => {
  const original = object[method].bind(object);
  object[method] = async (...values) => {
    const began = performance.now();
    try { return await original(...values); }
    finally { samples[key].push(performance.now() - began); }
  };
};
const summarize = values => {
  const ordered = values.slice().sort((a, b) => a - b);
  const percentile = p => ordered[Math.min(ordered.length - 1, Math.floor(p * ordered.length))];
  return {meanMs: values.reduce((sum, value) => sum + value, 0) / values.length,
    p50Ms: percentile(.5), p95Ms: percentile(.95), maxMs: ordered.at(-1), calls: values.length};
};
try {
  const residentWasm = await readFile(resolve(directory, 'live/pkg/resident_runtime_bg.wasm'));
  engine = await LiveEngine.create({baseURL: `http://127.0.0.1:${server.address().port}/live/`, device,
    modules: {worldWasm: await readFile(resolve(directory, 'live/pkg/chreatures_browser_world_bg.wasm')),
      initResident: () => initResident({module_or_path: residentWasm})}});
  timedSync(engine.world, 'sample', 'worldSample');
  timedAsync(engine.brain, 'step', 'brainStep');
  timedSync(engine.resident, 'stepFlat', 'residentStep');
  timedSync(engine.world, 'advance', 'worldAdvance');
  timedSync(engine.resident, 'acknowledge', 'residentAcknowledge');
  timedSync(engine.world, 'observe', 'worldObserve');
  for (let tick = 0; tick < ticks; tick++) {
    engine.screen(new Float32Array(12).fill(tick % 4 < 2 ? 1 : 0), 2, 2, null);
    const began = performance.now();
    await engine.advance(tick % 4 === 0);
    samples.completeTick.push(performance.now() - began);
  }
  const phaseMean = Object.fromEntries(Object.entries(samples).map(([name, values]) =>
    [name, values.reduce((sum, value) => sum + value, 0) / values.length]));
  const accounted = phaseMean.worldSample + phaseMean.brainStep + phaseMean.residentStep +
    phaseMean.worldAdvance + phaseMean.residentAcknowledge + 2 * phaseMean.worldObserve;
  const report = {format: 'chreatures-live-phase-profile-v1', engineIdentity: engine.identity,
    adapter: adapter.info?.device || adapter.info?.description || 'Dawn Metal', residents: engine.batch,
    ticks, phases: Object.fromEntries(Object.entries(samples).map(([name, values]) => [name, summarize(values)])),
    accountedMeanMs: accounted, unaccountedMeanMs: phaseMean.completeTick - accounted,
    scope: 'Actual Node Dawn Metal, deployed browser Wasm and MuJoCo; steady ticks after model load'};
  console.log(JSON.stringify(report, null, 2));
  if (args.report) await writeFile(args.report, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
} finally {
  engine?.destroy(); device.destroy(); server.closeAllConnections(); server.close();
}
