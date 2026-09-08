// SPDX-License-Identifier: AGPL-3.0-or-later
// No DOM or browser automation. Run after the native portable_resident_probe.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const moduleDir = path.resolve(process.argv[2]);
const fixture = path.resolve(process.argv[3]);
const { ResidentRuntime } = require(path.join(moduleDir, 'resident_runtime.js'));
const config = fs.readFileSync(path.join(fixture, 'config.json'), 'utf8');
const floats = name => {
  const b = fs.readFileSync(path.join(fixture, `${name}.f32`));
  return Float32Array.from({length:b.length/4}, (_,i)=>b.readFloatLE(i*4));
};
const weights = ['core','predictor','sequence'].map(floats);
const make = () => new ResidentRuntime(config,...weights);
const steps = JSON.parse(fs.readFileSync(path.join(fixture,'steps.json'),'utf8'));
const resident = make(); let maxContextError = 0, maxStateError = 0;
const compare = (a,b) => Math.max(0,...a.map((v,i)=>Math.abs(v-b[i])));
const tickArray = step => new BigUint64Array(2).fill(BigInt(step.tick));
const run = (runtime, step) => runtime.stepFlat(Float32Array.from(step.z),Float32Array.from(step.previous),tickArray(step),Uint8Array.from(step.reset));
let nativeRestored = make();
nativeRestored.loadBytes(fs.readFileSync(path.join(fixture,'native-checkpoint.json')));
for (const step of steps) {
  const result = run(resident,step);
  const context = result.proposedContext;
  maxContextError = Math.max(maxContextError, compare(Array.from(context),step.context));
  const diagnostics = JSON.parse(result.diagnosticsJson);
  maxStateError = Math.max(maxStateError,compare(diagnostics.cns_recurrent_state,step.diagnostics.cns_recurrent_state));
  assert.deepEqual(diagnostics.selected_candidate,step.diagnostics.selected_candidate);
  assert.deepEqual(diagnostics.goal_origin_slot,step.diagnostics.goal_origin_slot);
  assert.deepEqual(diagnostics.memory_count,step.diagnostics.memory_count);
  const pending = resident.saveBytes();
  assert.throws(()=>run(resident,step), /unacknowledged/);
  assert.deepEqual(resident.saveBytes(),pending);
  assert.throws(()=>resident.acknowledge(new BigUint64Array(2).fill(999n),context), /boundary/);
  assert.deepEqual(resident.saveBytes(),pending);
  resident.acknowledge(tickArray(step),context);
  if (step.tick > 6) {
    const n = run(nativeRestored,step);
    assert.ok(compare(Array.from(n.proposedContext),step.context)<2e-5);
    nativeRestored.acknowledge(tickArray(step),n.proposedContext); n.free();
  }
  if (step.tick === 6) {
    const snapshot = resident.saveBytes();
    const restored = make(); restored.loadBytes(snapshot);
    assert.deepEqual(restored.saveBytes(),snapshot);
    const next = steps[7]; const a = run(resident,next), b = run(restored,next);
    assert.deepEqual(a.proposedContext,b.proposedContext);
    assert.deepEqual(resident.saveBytes(),restored.saveBytes());
    a.free(); b.free(); restored.free(); resident.loadBytes(snapshot);
    const expanded = resident.expanded(1,19n,23n);
    assert.equal(expanded.batch,3);
    const oldPrivate = JSON.parse(JSON.parse(Buffer.from(snapshot).toString()).private);
    const newPrivate = JSON.parse(JSON.parse(Buffer.from(expanded.saveBytes()).toString()).private);
    assert.deepEqual(newPrivate.state.slice(0,512),oldPrivate.state);
    const bad = JSON.parse(Buffer.from(snapshot).toString());bad.format='wrong';
    assert.throws(()=>resident.loadBytes(Buffer.from(JSON.stringify(bad))),/identity/);
    assert.deepEqual(resident.saveBytes(),snapshot);
    expanded.free();
  }
  // Getter copies survive later engine calls, including an independent restore.
  assert.deepEqual(result.proposedContext,context);
  result.free();
}
assert.ok(maxContextError<2e-5,`context error ${maxContextError}`);
assert.ok(maxStateError<2e-5,`state error ${maxStateError}`);
const wasmPrivate=JSON.parse(Buffer.from(resident.saveBytes()).toString());
const nativePrivate=JSON.parse(fs.readFileSync(path.join(fixture,'native-final.json'),'utf8'));
assert.deepEqual(JSON.parse(wasmPrivate.sequence_control).rng,JSON.parse(nativePrivate.sequence_control).rng);
assert.deepEqual(JSON.parse(wasmPrivate.goal_memory).rng,JSON.parse(nativePrivate.goal_memory).rng);
console.log(JSON.stringify({passed:true,residents:2,ticks:steps.length,maxContextError,maxStateError,
  checks:['sampled native/Wasm decisions','private RNG','native snapshot to Wasm continuation','pending receipt rejection','exact Wasm replay','cohort expansion','atomic invalid restore','stable copied outputs']}));
nativeRestored.free();resident.free();
