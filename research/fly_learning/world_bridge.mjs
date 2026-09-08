#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Newline-framed research host for the canonical MuJoCo/Wasm fly world.
// Raw researchObserve values leave this process only for offline targets.

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';

const argv = process.argv.slice(2), args = {};
for (let i = 0; i < argv.length; i += 2) {
  if (!argv[i]?.startsWith('--') || argv[i + 1] === undefined)
    throw new Error('arguments must be --name value pairs');
  args[argv[i].slice(2)] = argv[i + 1];
}
for (const required of ['scene', 'runtime', 'core-wasm', 'seed'])
  if (!(required in args)) throw new Error(`missing --${required}`);

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const scenePath = resolve(args.scene), sceneDirectory = dirname(scenePath);
const fixtureBytes = await readFile(scenePath), fixture = JSON.parse(fixtureBytes);
const xmlPath = resolve(sceneDirectory, fixture.scene_xml), xmlBytes = await readFile(xmlPath);
const assets = new Map();
for (const item of fixture.mesh_assets) assets.set(item.path, await readFile(resolve(sceneDirectory, item.path)));
const coreWasm = await readFile(resolve(args['core-wasm']));
const { createBrowserWorld } = await import(pathToFileURL(resolve(args.runtime)));
const world = await createBrowserWorld({
  fixture, xml: xmlBytes.toString('utf8'), assets, seed: Number(args.seed), coreWasm,
});

const encode = value => {
  const bytes = Buffer.from(value.buffer, value.byteOffset, value.byteLength);
  return { dtype: value instanceof Float64Array ? '<f8' : '<f4', length: value.length, base64: bytes.toString('base64') };
};
const response = value => process.stdout.write(`${JSON.stringify(value)}\n`);
const snapshot = world.snapshot();
response({
  ok: true,
  event: 'ready',
  residents: world.residents,
  engine: fixture.engine,
  fixture_sha256: sha256(fixtureBytes),
  scene_xml_sha256: sha256(xmlBytes),
  core_wasm_sha256: sha256(coreWasm),
  initial_snapshot_sha256: sha256(Buffer.from(JSON.stringify(snapshot))),
  fixture: {
    source_revision: fixture.source_revision,
    source_mjcf_sha256: fixture.source_mjcf_sha256,
    atlas_sha256: fixture.atlas_sha256,
    morphology_sha256: fixture.morphology_sha256,
    sensory_schema_sha256: fixture.sensory_schema_sha256,
    actuator_schema_sha256: fixture.actuator_schema_sha256,
    bodies: fixture.bodies,
    entities: fixture.entities,
  },
});

function researchPacket() {
  const sensory = world.sample(), raw = world.researchObserve(), observer = world.observe();
  if (raw.bodyAfferents.length !== sensory.body.length)
    throw new Error('research observer BODY807 cache differs from policy sample');
  for (let i = 0; i < sensory.body.length; i++)
    if (raw.bodyAfferents[i] !== sensory.body[i]) throw new Error('research observer BODY807 values differ');
  const entityPosition = new Float32Array(fixture.entities.length * 3);
  for (let i = 0; i < fixture.entities.length; i++) {
    const entity = fixture.entities[i];
    const geometry = observer.geometry.find(item => entity.geoms.includes(item.id));
    if (geometry) entityPosition.set(geometry.position, i * 3);
  }
  return {
    optic: encode(sensory.optic), body: encode(sensory.body),
    qpos: encode(raw.qpos), qvel: encode(raw.qvel),
    bodyPositions: encode(raw.bodyPositions), bodyQuaternions: encode(raw.bodyQuaternions),
    bodyRotations: encode(raw.bodyRotations),
    sensordata: encode(raw.sensordata), ctrl: encode(raw.ctrl),
    entityPosition: encode(entityPosition),
    bodyMap: raw.bodyMap, ecology: raw.ecology, actuatorState: raw.actuatorState,
    time: observer.time,
  };
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
try {
  for await (const line of lines) {
    const request = JSON.parse(line);
    try {
      if (request.command === 'sample') {
        response({ id: request.id, ok: true, sample: researchPacket() });
      } else if (request.command === 'advance') {
        const bytes = Buffer.from(request.motor92_base64, 'base64');
        if (bytes.byteLength !== world.residents * 92 * 4)
          throw new Error('advance requires exact BxM92 float32 bytes');
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        const motor = new Float32Array(copy);
        await world.advance(motor, .01);
        response({ id: request.id, ok: true, time: world.time });
      } else if (request.command === 'close') {
        response({ id: request.id, ok: true });
        break;
      } else {
        throw new Error('unknown bridge command');
      }
    } catch (error) {
      response({ id: request.id, ok: false, error: String(error?.stack ?? error) });
    }
  }
} finally {
  world.dispose();
  // MuJoCo's Emscripten runtime intentionally keeps Node alive. This bridge
  // owns the whole process, so a requested close is also the process boundary.
  await new Promise(resolve => process.stdout.write('', resolve));
  process.exit(0);
}
