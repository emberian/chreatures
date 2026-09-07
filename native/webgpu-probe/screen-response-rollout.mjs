#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Internal one-engine capture process for screen-response.mjs.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { open, readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { create, globals } from 'webgpu';

const argv = process.argv.slice(2), args = {};
for (let i = 0; i < argv.length; i += 2) args[argv[i].slice(2)] = argv[i + 1];
if (!['film', 'blank'].includes(args.condition) || !args.site || !args.output || (args.condition === 'film' && !args.decoded)) {
  throw new Error('internal usage: --condition film|blank --site DIR --decoded RGB --output PREFIX --ticks N --width N --height N');
}
const site = resolve(args.site), output = resolve(args.output);
const ticks = Number(args.ticks), width = Number(args.width), height = Number(args.height), dt = 0.05;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const mime = { '.json': 'application/json', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.gz': 'application/octet-stream' };
const server = createServer(async (request, response) => {
  try {
    const file = resolve(site, '.' + new URL(request.url, 'http://localhost').pathname);
    if (!file.startsWith(site + sep) || !(await stat(file)).isFile()) throw new Error('invalid file');
    response.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
    createReadStream(file).pipe(response);
  } catch { response.statusCode = 404; response.end('missing'); }
});
async function writeAll(handle, values) {
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  let offset = 0;
  while (offset < bytes.length) offset += (await handle.write(bytes, offset, bytes.length - offset)).bytesWritten;
}
const roots = observation => observation.residents.map(resident => {
  const at = resident.root * 3; return Array.from(observation.bodyPositions.subarray(at, at + 3));
});

async function loadFrames() {
  if (args.condition !== 'film') return null;
  const encoded = await readFile(resolve(args.decoded));
  assert.equal(encoded.length, ticks * width * height * 3);
  const values = new Float32Array(encoded.length);
  for (let index = 0; index < encoded.length; index++) values[index] = encoded[index] / 255;
  return values;
}
const decoded = await loadFrames();
await new Promise(done => server.listen(0, '127.0.0.1', done));
Object.assign(globalThis, globals);
const dawn = create([`backend=${args.backend ?? (process.platform === 'darwin' ? 'metal' : 'vulkan')}`]);
// webgpu's native implementation is owned by the object returned from create().
// Long async I/O below can otherwise leave no observable live reference for GC.
globalThis.__chreaturesScreenResponseDawn = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' }); assert(adapter);
const device = await adapter.requestDevice();
const [{ default: initResident }, { LiveEngine }] = await Promise.all([
  import(pathToFileURL(resolve(site, 'live/pkg/resident_runtime.js'))), import(pathToFileURL(resolve(site, 'live/engine.js'))),
]);
let engine;
const handles = await Promise.all(['rates.f32', 'retina.f32', 'actions.f32', 'body.f32'].map(name => open(`${output}.${name}`, 'wx')));
try {
  const residentWasm = await readFile(resolve(site, 'live/pkg/resident_runtime_bg.wasm'));
  engine = await LiveEngine.create({ baseURL: `http://127.0.0.1:${server.address().port}/live/`, device,
    modules: { initResident: () => initResident({ module_or_path: residentWasm }) } });
  const initialWorld = Buffer.from(JSON.stringify(engine.world.snapshot()));
  const initialCns = await engine.brain.snapshot(), initialResident = engine.resident.saveBytes();
  const firstRoots = roots(engine.observe()), previous = firstRoots.map(row => row.slice());
  const path = new Float64Array(engine.batch), wall = [], rootTrace = [];
  const frame = new Float32Array(width * height * 3), frameBytes = frame.length;
  for (let tick = 0; tick < ticks; tick++) {
    if (decoded) frame.set(decoded.subarray(tick * frameBytes, (tick + 1) * frameBytes));
    engine.screen(frame, width, height, tick * dt);
    const result = await engine.advance(true); wall.push(result.wallMilliseconds);
    await writeAll(handles[0], result.neuralRates); await writeAll(handles[1], result.retinalRGB);
    await writeAll(handles[2], engine.previous); await writeAll(handles[3], result.bodyPositions);
    const current = roots(result); rootTrace.push(current);
    for (let resident = 0; resident < engine.batch; resident++) {
      path[resident] += Math.hypot(...current[resident].map((value, axis) => value - previous[resident][axis]));
      previous[resident] = current[resident];
    }
    if (ticks <= 32 || (tick + 1) % 100 === 0 || tick + 1 === ticks) process.stderr.write(`[${args.condition}] ${tick + 1}/${ticks}\n`);
  }
  const metadata = { format: 'chreatures-screen-response-rollout-v1', condition: args.condition,
    engineIdentity: engine.identity, adapter: adapter.info?.device || adapter.info?.description || 'Dawn',
    ticks, batch: engine.batch, dt, width, height, bodyValues: engine.observe().bodyPositions.length,
    initial: { physicalSha256: sha256(initialWorld), cnsSha256: sha256(new Uint8Array(initialCns)),
      residentSha256: sha256(initialResident) },
    baseline: args.condition === 'film' ? Array.from(engine.neuralBaseline) : null,
    wallMilliseconds: wall, initialRoots: firstRoots, rootTrace, finalRoots: roots(engine.observe()),
    pathLengthMeters: Array.from(path) };
  await writeFile(`${output}.json`, JSON.stringify(metadata), { flag: 'wx' });
} finally {
  await Promise.all(handles.map(handle => handle.close()));
  engine?.destroy(); device.destroy(); delete globalThis.__chreaturesScreenResponseDawn;
  server.closeAllConnections(); server.close();
}
