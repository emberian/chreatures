import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createBrowserWorld } from "./runtime.mjs";
const fixture = JSON.parse(
  await readFile(new URL("./fixtures/garden.json", import.meta.url), "utf8"),
);
const xml = await readFile(
  new URL("./fixtures/garden.xml", import.meta.url),
  "utf8",
);
const coreWasm = await readFile(
  new URL("./pkg/chreatures_browser_world_bg.wasm", import.meta.url),
);
const world = await createBrowserWorld({ fixture, xml, coreWasm });
try {
  let start = performance.now();
  const black = world.sample();
  assert.equal(black.optic.length, fixture.bodies.length * 5313);
  assert.equal(black.body.length, fixture.bodies.length * 110);
  world.setScreenFrame(new Float32Array(12).fill(1), 2, 2);
  const white = world.sample();
  const affected = white.optic.reduce(
    (n, x, i) => n + Number(Math.abs(x - black.optic[i]) > 0.01),
    0,
  );
  assert(affected > 0, "Physical screen must affect retinal rays");
  const before = world.observe();
  const command = new Float64Array(world.residents * 34);
  // CNS motor-head values recruit synthetic antagonists directly; no supplied gait.
  command[0] = 0.8;
  command[4] = 0.7;
  command[8] = 0.8;
  command[12 + 1] = 0.7;
  command[16 + 1] = 0.8;
  command[20 + 1] = 0.7;
  for (let i = 0; i < 20; i++) world.advance(command);
  const sensed = world.sample();
  assert([...sensed.optic, ...sensed.body].every(Number.isFinite));
  assert(
    Array.from(sensed.body.subarray(98, 110)).some((v) => v > 0),
    "Local synthetic actuator fatigue must feed back through BODY110",
  );
  assert(
    Array.from(sensed.body.subarray(56, 92)).some((v) => Math.abs(v) > 1e-5),
    "Joint position, velocity and measured load must feed back through BODY110",
  );
  assert(Math.abs(world.time - 1) < 1e-9);
  const saved = world.snapshot();
  world.advance(command);
  const future = world.snapshot();
  world.restore(saved);
  world.advance(command);
  assert.deepEqual(
    world.snapshot(),
    future,
    "Full integration + private ecology/RNG replay",
  );
  assert.notDeepEqual(
    world.observe().positions,
    before.positions,
    "Articulated contact physics must move",
  );
  world.setMemoryCheckpoint("opaque-CNS-derived-private-memory");
  const coherent = world.snapshot();
  await assert.rejects(
    world.insertObject({ position: [6, 4, -0.1] }),
    /penetrate/,
  );
  assert.deepEqual(
    world.snapshot(),
    coherent,
    "Rejected topology leaves last coherent life untouched",
  );
  const inserted = await world.insertObject({
    position: [1.2, 2.15, 0.6],
    size: [0.02, 0.5, 0.5],
  });
  assert.equal(world.observe().geometry.length, 113);
  assert.equal(
    JSON.parse(world.snapshot().core).memory,
    "opaque-CNS-derived-private-memory",
  );
  world.setScreenFrame(new Float32Array(12), 2, 2);
  const obscuredBlack = world.sample();
  world.setScreenFrame(new Float32Array(12).fill(1), 2, 2);
  const obscuredWhite = world.sample();
  const blockedDelta = obscuredWhite.optic
    .subarray(0, 5313)
    .reduce(
      (n, x, i) => n + Number(Math.abs(x - obscuredBlack.optic[i]) > 0.01),
      0,
    );
  assert(
    blockedDelta < affected * 0.1,
    "Actual inserted MuJoCo geometry must substantially occlude screen",
  );
  world.queueVisitorForce(inserted.id, [0.2, 0, 0]);
  world.visitorSound([0.8, 2.15, 0.1], 720, 0.5, 0.2);
  world.advance(command);
  assert(
    Array.from(world.sample().body.subarray(27, 43)).some((v) => v > 0),
    "Propagated tone must excite the fixed log-frequency filterbank",
  );
  const grownSnapshot = world.snapshot();
  const isolated = await createBrowserWorld({
    fixture: grownSnapshot.fixture,
    xml: grownSnapshot.xml,
    coreWasm,
    seed: 42,
  });
  try {
    isolated.restore(grownSnapshot);
    assert.deepEqual(isolated.sample(), world.sample());
    world.advance(command);
    isolated.advance(command);
    assert.deepEqual(
      isolated.snapshot(),
      world.snapshot(),
      "Recompiled topology, delayed fields, visitor queue and private memory resume exactly",
    );
    isolated.advance(command);
    assert.notEqual(
      isolated.time,
      world.time,
      "Independent MuJoCo data and Rust states",
    );
  } finally {
    isolated.dispose();
  }
  console.log(
    JSON.stringify(
      {
        engine: world.engine,
        residents: world.residents,
        geoms: before.geometry.length,
        optic: sensed.optic.length,
        body: sensed.body.length,
        screenAffectedScalars: affected,
        insertedOcclusionDelta: blockedDelta,
        topologyGeoms: world.observe().geometry.length,
        elapsedMs: Math.round(performance.now() - start),
        seconds: world.time,
        replay: "exact",
      },
      null,
      2,
    ),
  );
} finally {
  world.dispose();
}
