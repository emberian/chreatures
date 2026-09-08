#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Newline-framed Dawn host for the authenticated full MaleCNS WebGPU V4 model.

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';

const here = dirname(new URL(import.meta.url).pathname);
const probeRequire = createRequire(pathToFileURL(resolve(here, '../../native/webgpu-probe/package.json')));
const { create, globals } = probeRequire('webgpu');
Object.assign(globalThis, globals);

const argv = process.argv.slice(2);
const args = {};
for (let index = 0; index < argv.length; index += 2) {
  if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) {
    throw new Error('arguments must be --name value pairs');
  }
  args[argv[index].slice(2)] = argv[index + 1];
}
if (!args.model) throw new Error('missing --model');

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const asArrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const response = value => process.stdout.write(`${JSON.stringify(value)}\n`);
const modelDirectory = resolve(args.model);
const manifestBytes = await readFile(resolve(modelDirectory, 'cns-manifest.json'));
const manifest = JSON.parse(manifestBytes);

const assets = new Map();
for (const [name, entry] of Object.entries(manifest.buffers)) {
  const transported = await readFile(resolve(modelDirectory, entry.url));
  if (transported.byteLength !== entry.transportByteLength || sha256(transported) !== entry.transportSha256) {
    throw new Error(`packed asset transport identity differs: ${name}`);
  }
  const raw = entry.encoding === 'gzip' ? gunzipSync(transported) : transported;
  if (raw.byteLength !== entry.byteLength || sha256(raw) !== entry.sha256) {
    throw new Error(`packed asset identity differs: ${name}`);
  }
  assets.set(name, asArrayBuffer(raw));
}

const liveDirectory = resolve(here, '../../site/live');
const shaderNames = ['afferent.wgsl', 'dynamics.wgsl', 'readout.wgsl', 'motor.wgsl', 'observe.wgsl'];
const shaderSources = Object.fromEntries(await Promise.all(shaderNames.map(async name => [
  name,
  await readFile(resolve(liveDirectory, 'shaders', name), 'utf8'),
])));

// Dawn requires its Instance wrapper to remain strongly referenced for the
// complete GPU lifetime.
const dawn = create(['backend=metal']);
globalThis.__chreaturesDawnInstance = dawn;
const adapter = await dawn.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('Dawn found no Metal WebGPU adapter');
const largestBuffer = Math.max(...Object.values(manifest.buffers).map(entry => entry.byteLength));
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: Math.max(largestBuffer, 128 * 1024 * 1024),
    maxStorageBufferBindingSize: Math.max(largestBuffer, 128 * 1024 * 1024),
  },
});
const { MaleCNSWebGPU } = await import(pathToFileURL(resolve(liveDirectory, 'cns-webgpu.js')));
const brain = await MaleCNSWebGPU.load({
  device,
  manifest,
  capacity: 4,
  assetBuffers: assets,
  shaderSources,
});
assets.clear();
await brain.reset([0, 1, 2, 3], ['fly-0', 'fly-1', 'fly-2', 'fly-3']);

const decodeF32 = (encoded, length, name) => {
  const bytes = Buffer.from(encoded, 'base64');
  if (bytes.byteLength !== length * 4) throw new Error(`${name} byte length differs`);
  return new Float32Array(asArrayBuffer(bytes));
};
const encodeF32 = value => Buffer.from(value.buffer, value.byteOffset, value.byteLength).toString('base64');

response({
  ok: true,
  event: 'ready',
  capacity: 4,
  manifest_sha256: sha256(manifestBytes),
  service_sha256: manifest.serviceArtifactSha256,
  adapter_sha256: manifest.identity.artifact,
  manifest,
  backend: 'dawn-metal-webgpu',
});

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
try {
  for await (const line of lines) {
    const request = JSON.parse(line);
    try {
      if (request.command === 'step') {
        const output = await brain.step({
          dt: 0.01,
          activeMask: 0b1111,
          opticRGB: decodeF32(request.optic_base64, 4 * 5313, 'optic'),
          body: decodeF32(request.body_base64, 4 * 807, 'body'),
          context: decodeF32(request.context_base64, 4 * 12, 'context'),
        });
        response({
          id: request.id,
          ok: true,
          latent_base64: encodeF32(output.latent),
          motor_base64: encodeF32(output.motor),
        });
      } else if (request.command === 'reset') {
        await brain.reset([0, 1, 2, 3], ['fly-0', 'fly-1', 'fly-2', 'fly-3']);
        response({ id: request.id, ok: true });
      } else if (request.command === 'snapshot') {
        response({ id: request.id, ok: true, snapshot_base64: Buffer.from(await brain.snapshot()).toString('base64') });
      } else if (request.command === 'restore') {
        const bytes = Buffer.from(request.snapshot_base64, 'base64');
        await brain.restore(asArrayBuffer(bytes));
        response({ id: request.id, ok: true });
      } else if (request.command === 'close') {
        response({ id: request.id, ok: true });
        break;
      } else {
        throw new Error('unknown CNS bridge command');
      }
    } catch (error) {
      response({ id: request.id, ok: false, error: String(error?.stack ?? error) });
    }
  }
} finally {
  brain.destroy();
  device.destroy();
}
