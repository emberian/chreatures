// An actual physical lineage run. The flies receive diagnostic neutral targets;
// this observes inherited colony development, not learned animal competence.
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {resolve, dirname} from 'node:path';
import {createHash} from 'node:crypto';
import {createBrowserWorld} from './runtime.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((rows, value, i, all) =>
  i % 2 ? rows : [...rows, [value.slice(2), all[i + 1]]], []));
if (!args.output) throw new Error('--output must name a fresh research receipt');
const ticks = Number(args.ticks ?? 2400);
assert(Number.isSafeInteger(ticks) && ticks >= 100 && ticks <= 100000);
const output = resolve(args.output), base = new URL('./fixtures/fly-ecology/', import.meta.url);
const hash = value => createHash('sha256').update(value).digest('hex');
const hostHash = hash(await readFile(new URL('./runtime.mjs', import.meta.url)));
const runnerHash = hash(await readFile(new URL(import.meta.url)));
const fixtureBytes = await readFile(new URL('world.json', base));
const fixture = JSON.parse(fixtureBytes), xml = await readFile(new URL('scene.xml', base), 'utf8');
const coreWasm = await readFile(new URL('./pkg/chreatures_browser_world_bg.wasm', import.meta.url));
const assets = new Map(await Promise.all(fixture.mesh_assets.map(async item =>
  [item.path, new Uint8Array(await readFile(new URL(item.path, base)))])));
const world = await createBrowserWorld({fixture, xml, assets, coreWasm, seed: 20260908});
const began = performance.now(), checkpoints = [], births = [], known = new Set();
let initial, final, replayExact = false, failure = null, completedTicks = 0;
try {
  initial = world.observe();
  for (const organism of initial.ecology.organisms) known.add(organism.id);
  world.setScreenFrame(new Float32Array(12).fill(.8), 2, 2);
  const command = new Float64Array(world.residents * 92);
  for (let tick = 1; tick <= ticks; tick++) {
    await world.advance(command);
    completedTicks = tick;
    if (tick % 25 && tick !== ticks) continue;
    const observation = world.observe();
    for (const organism of observation.ecology.organisms) {
      if (known.has(organism.id)) continue;
      const physical = observation.geometry.filter(g => fixture.entities.every(e => e.id !== organism.physics_binding) &&
        g.name === `ecology:${organism.physics_binding}:geom`);
      assert(organism.anchored_region && physical.length === 1, 'New colony must have one physical anchored body');
      births.push({observed_tick: tick, time: observation.time, organism, geometry: physical[0]});
      known.add(organism.id);
    }
    if (tick % 100 === 0 || tick === ticks) {
      checkpoints.push({tick, time: observation.time, geometry_count: observation.geometry.length,
        colonies: observation.ecology.organisms.filter(o => o.anchored_region).map(o => ({id: o.id,
          generation: o.generation, descendants: o.descendant_count, atp: o.atp,
          material: o.material.quantity, physics_binding: o.physics_binding})),
        accounting: observation.ecology.accounting, growth: observation.growth});
      console.log(JSON.stringify({tick, colonies: checkpoints.at(-1).colonies.length, births: births.length,
        geoms: observation.geometry.length, elapsed_s: (performance.now() - began) / 1000}));
    }
  }
  const checkpoint = world.snapshot();
  await world.advance(command);
  const future = world.snapshot();
  world.restore(checkpoint);
  await world.advance(command);
  assert.deepEqual(world.snapshot(), future, 'Full descendant world continuation must replay exactly');
  replayExact = true;
  world.restore(checkpoint);
  final = world.observe();
  await mkdir(dirname(output), {recursive: true});
  await writeFile(`${output}.snapshot.json`, JSON.stringify(checkpoint), {flag: 'wx'});
} catch (error) {
  failure = {message: error?.message ?? String(error), stack: error?.stack, tick: completedTicks};
} finally {
  world.dispose();
}
const report = {format: 'chreatures-physical-colony-lineage-v1', requested_ticks: ticks,
  completed_ticks: completedTicks, failure, whole_world_replay_exact: replayExact,
  fixture_sha256: hash(fixtureBytes), scene_sha256: hash(xml), core_wasm_sha256: hash(coreWasm),
  host_sha256: hostHash, runner_sha256: runnerHash,
  elapsed_s: (performance.now() - began) / 1000, births, checkpoints,
  initial_accounting: initial?.ecology.accounting, final_accounting: final?.ecology.accounting,
  fly_control: 'constant zero MOTOR92 diagnostic target; no CNS loaded in this ecological experiment',
  interpretation: births.length ? 'Resource-funded inherited colony bodies physically realized; no fly reproduction or behavioral competence claim.' :
    'No colony birth was realized during the recorded interval; preserve the unsuccessful outcome.'};
await mkdir(dirname(output), {recursive: true});
await writeFile(output, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
if (failure) { console.error(failure.message); process.exitCode = 1; }
