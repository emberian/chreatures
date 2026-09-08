// SPDX-License-Identifier: AGPL-3.0-or-later
// Transport only: all recurrent/private computations use production Rust Wasm.
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createInterface} from 'node:readline';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const runtimePath = resolve(process.argv[2]);
const packPath = resolve(process.argv[3]);
const runtime = JSON.parse(await readFile(runtimePath, 'utf8'));
const pack = JSON.parse(await readFile(packPath, 'utf8'));
if (runtime.format !== 'chreatures-resident-runtime-boundary-v1' ||
    pack.format !== 'chreatures-research-resident-pack-v1' || pack.config.batch !== 4)
  throw new Error('Current B4 resident boundary required');
async function authenticated(root, item) {
  const bytes = await readFile(resolve(root, item.file));
  if (hash(bytes) !== item.sha256) throw new Error('Resident boundary bytes differ');
  return bytes;
}
const moduleBytes = await authenticated(dirname(runtimePath), runtime.module);
const wasm = await authenticated(dirname(runtimePath), runtime.wasm);
const moduleURL = pathToFileURL(resolve(dirname(runtimePath), runtime.module.file));
const {default: init, ResidentRuntime} = await import(moduleURL);
await init({module_or_path: wasm});
const buffers = await Promise.all(['core', 'predictor', 'sequence'].map(async name => {
  const bytes = await authenticated(dirname(packPath), pack.buffers[name]);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}));
const resident = new ResidentRuntime(JSON.stringify(pack.config), ...buffers);
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
const decode = (value, length) => {
  const bytes = Buffer.from(value, 'base64');
  if (bytes.length !== length * 4) throw new Error('Resident input shape differs');
  const result = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  if (!result.every(Number.isFinite)) throw new Error('Nonfinite resident input');
  return result;
};
send({ready: true, pid: process.pid, runtime_module_sha256: hash(moduleBytes),
  runtime_wasm_sha256: hash(wasm), resident_artifact_sha256: pack.artifact_sha256});
const lines = createInterface({input: process.stdin, crlfDelay: Infinity});
try {
  for await (const line of lines) {
    let request;
    try {
      request = JSON.parse(line);
      let result;
      if (request.op === 'acknowledge') {
        const ticks = new BigUint64Array(4).fill(BigInt(request.tick));
        result = {accepted: Array.from(resident.acknowledge(ticks, decode(request.context, 48)))};
      } else if (request.op === 'step') {
        const step = resident.stepFlat(decode(request.latent, 2048), decode(request.context, 48),
          new BigUint64Array(4).fill(BigInt(request.tick)), new Uint8Array(4).fill(request.tick === 0 ? 1 : 0));
        try {
          const context = step.proposedContext;
          result = {context: Buffer.from(context.buffer, context.byteOffset, context.byteLength).toString('base64'),
            diagnostics: JSON.parse(step.diagnosticsJson)};
        } finally { step.free(); }
      } else if (request.op === 'snapshot') {
        const state = Buffer.from(resident.saveBytes());
        if (request.verify) {
          const copy = new ResidentRuntime(JSON.stringify(pack.config), ...buffers);
          try {
            copy.loadBytes(state);
            if (!Buffer.from(copy.saveBytes()).equals(state)) throw new Error('Resident checkpoint changed on exact restore');
          } finally { copy.free(); }
        }
        result = {state: state.toString('base64'), exact_restore_verified: Boolean(request.verify)};
      } else if (request.op === 'restore') {
        resident.loadBytes(Buffer.from(request.state, 'base64'));
        result = {restored: true};
      } else if (request.op === 'close') {
        send({id: request.id, ok: true});
        break;
      } else throw new Error('Unknown resident boundary operation');
      send({id: request.id, ok: true, ...result});
    } catch (error) {
      send({id: request?.id, ok: false, error: String(error)});
      process.exitCode = 1;
      break; // A failed mutation is terminal; never retry a decision.
    }
  }
} finally {
  resident.free();
  lines.close();
}
