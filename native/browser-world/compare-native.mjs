// Matched long physical chronology across the native and Wasm MuJoCo hosts.
// Raw geometry is an evaluator input only. No CNS/policy runs in this assay.
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createBrowserWorld} from './runtime.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((rows, v, i, all) =>
  i % 2 ? rows : [...rows, [v.slice(2), all[i + 1]]], []));
assert(args.reference && args.output, '--reference and fresh --output required');
const sha = v => createHash('sha256').update(v).digest('hex');
const referenceBytes = await readFile(args.reference), reference = JSON.parse(referenceBytes);
assert.equal(reference.format, 'chreatures-native-browser-aero-comparison-input-v1');
const base = pathToFileURL(resolve(args.scene ?? './fixtures/fly-ecology') + '/');
const fixtureBytes = await readFile(new URL('world.json', base));
assert.equal(sha(fixtureBytes), reference.ready.fixture_sha256);
const fixture = JSON.parse(fixtureBytes), xml = await readFile(new URL('scene.xml', base), 'utf8');
assert.equal(sha(xml), reference.ready.scene_xml_sha256);
const coreWasm = await readFile(new URL('./pkg/chreatures_browser_world_bg.wasm', import.meta.url));
const assets = new Map(await Promise.all(fixture.mesh_assets.map(async item =>
  [item.path, new Uint8Array(await readFile(new URL(item.path, base)))])));
const world = await createBrowserWorld({fixture, xml, assets, coreWasm, seed: reference.seed});
// Native research bridge receives the same float32 neural command packet;
// both physical hosts widen those exact values internally.
const command = new Float32Array(reference.commands.shape.reduce((a,b) => a*b));
for (const [index, value] of Object.entries(reference.commands.nonzero)) command[Number(index)] = value;
const screen = new Float32Array(reference.screen.shape.reduce((a,b) => a*b)).fill(reference.screen.value);
assert.equal(sha(Buffer.from(command.buffer)), reference.commands.sha256);
assert.equal(sha(Buffer.from(screen.buffer)), reference.screen.sha256);
const tone = reference.silent_sound_required_by_bridge;
const decode = packet => {
  const raw = Buffer.from(packet.base64, 'base64');
  const bytes = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const values = packet.dtype === '<f8' ? new Float64Array(bytes) : new Float32Array(bytes);
  assert.equal(values.length, packet.length);
  return values;
};
function differences(actual, expected) {
  assert.equal(actual.length, expected.length);
  let maximum = 0, sum = 0, failed = 0;
  // Declared before execution: strict numeric comparison, not behavioral equivalence.
  for (let i = 0; i < actual.length; i++) {
    assert(Number.isFinite(actual[i]) && Number.isFinite(expected[i]));
    const delta = Math.abs(actual[i] - expected[i]);
    maximum = Math.max(maximum, delta); sum += delta * delta;
    if (delta > 1e-6 + 1e-6 * Math.abs(expected[i])) failed++;
  }
  return {length: actual.length, max_abs: maximum, rms: Math.sqrt(sum / actual.length),
    outside_atol_1e_6_rtol_1e_6: failed};
}
const began = performance.now(), timings = [];
let report;
try {
  world.setScreenFrame(screen, 2, 2); world.sample();
  for (let tick = 0; tick < reference.ticks; tick++) {
    world.visitorSound(tone.position_mm, tone.frequency_hz, tone.envelope, tone.duration_s);
    const t = performance.now(); await world.advance(command); world.sample();
    timings.push(performance.now() - t);
  }
  const sample = world.sample(), physical = world.researchObserve(), observer = world.observe();
  const arrays = {optic: sample.optic, body: sample.body, ...physical};
  const comparison = {};
  for (const name of ['optic', 'body', 'qpos', 'qvel', 'bodyPositions', 'bodyQuaternions', 'bodyRotations', 'sensordata', 'ctrl']) {
    comparison[name] = differences(arrays[name], decode(reference.final_sample[name]));
  }
  const checkpoint = world.snapshot();
  world.visitorSound(tone.position_mm, tone.frequency_hz, tone.envelope, tone.duration_s);
  await world.advance(command); world.sample(); const future = world.snapshot();
  world.restore(checkpoint);
  world.visitorSound(tone.position_mm, tone.frequency_hz, tone.envelope, tone.duration_s);
  await world.advance(command); world.sample();
  assert.deepEqual(world.snapshot(), future);
  report = {format: 'chreatures-native-wasm-physical-comparison-v1', reference_sha256: sha(referenceBytes),
    fixture_sha256: sha(fixtureBytes), core_wasm_sha256: sha(coreWasm),
    runtime_sha256: sha(await readFile(new URL('./runtime.mjs', import.meta.url))),
    runner_sha256: sha(await readFile(new URL(import.meta.url))), ticks: reference.ticks,
    strict_numerical_match: Object.values(comparison).every(v => v.outside_atol_1e_6_rtol_1e_6 === 0),
    comparison, route_comparison: differences(observer.routeMeasurements.open, reference.native_observer.route_open_fraction),
    browser_growth: observer.growth, native_growth: reference.native_observer.growth,
    browser_ecology: observer.ecology, native_ecology: reference.final_sample.ecology,
    browser_replay_exact: true, native_replay_exact: reference.exact_native_replay,
    browser_mean_tick_ms: timings.reduce((a,b) => a+b, 0)/timings.length,
    native_mean_tick_ms: reference.timing_ms.mean, elapsed_s: (performance.now() - began)/1000,
    scope: '200 matched physical ticks; constant diagnostic MOTOR92, actual rays, routes, growth, wing loads; no learned control or cross-platform bit-exactness claim.'};
} finally {world.dispose();}
await writeFile(args.output, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify({output: args.output, match: report.strict_numerical_match, comparison: report.comparison,
  routes: report.route_comparison, browser_mean_tick_ms: report.browser_mean_tick_ms}));
