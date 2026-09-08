import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createBrowserWorld, ENGINE } from "./runtime.mjs";

const base = new URL("./fixtures/fly-ecology/", import.meta.url);
const fixture = JSON.parse(await readFile(new URL("world.json", base), "utf8"));
const xml = await readFile(new URL("scene.xml", base), "utf8");
const coreWasm = await readFile(new URL("./pkg/chreatures_browser_world_bg.wasm", import.meta.url));
const assets = new Map(await Promise.all(fixture.mesh_assets.map(async ({ path }) =>
  [path, new Uint8Array(await readFile(new URL(path, base)))],
)));

const world = await createBrowserWorld({ fixture, xml, assets, coreWasm, seed: 20260908 });
try {
  const started = performance.now();
  let advanceMs = 0, advances = 0, sampleMs = 0, samples = 0;
  const initial = world.observe();
  assert.equal(world.engine, ENGINE);
  assert.equal(world.residents, 2);
  assert.deepEqual(initial.worldSizeMm, [50, 40, 16]);
  assert.equal(initial.geometry.length, 168);
  assert.equal(initial.meshes.length, 69);
  assert.equal(initial.residents[0].head, fixture.bodies[0].head);
  assert(initial.geometry.some((g) => g.mesh_id >= 0 && g.name && g.resident_id === fixture.bodies[0].id));

  let measuredAt = performance.now();
  const sensed = world.sample();
  sampleMs += performance.now() - measuredAt; samples++;
  assert.equal(sensed.optic.length, 2 * 5313);
  assert.equal(sensed.body.length, 2 * 807);
  assert([...sensed.optic, ...sensed.body].every(Number.isFinite));
  const research = world.researchObserve();
  assert.equal(research.qpos.length, 322);
  assert.equal(research.qvel.length, 312);
  assert.equal(research.bodyPositions.length, 167 * 3);
  assert.equal(research.bodyQuaternions.length, 167 * 4);
  assert.equal(research.sensordata.length, 2 * 6 * 16);
  assert.equal(research.ctrl.length, 180);
  assert.equal(research.bodyAfferents.length, 2 * 807);
  assert.equal(research.bodyMap.residents[1].contact_sensors6x16[5].data_address, 176);

  world.setRouteMeasurements(
    Float64Array.from(fixture.ecology.routes, (route) => route.base_open_fraction),
    new Float64Array(fixture.ecology.routes.length),
  );
  const visualCheckpoint = world.snapshot();
  const screenFacing = structuredClone(visualCheckpoint);
  // mjSTATE_INTEGRATION starts with time then qpos. Rotate resident00's free
  // root from the fixture's -X heading to +X, toward the compiled screen.
  screenFacing.physical[4] = 1;
  screenFacing.physical[5] = 0;
  screenFacing.physical[6] = 0;
  screenFacing.physical[7] = 0;
  world.restore(screenFacing);
  measuredAt = performance.now();
  const darkScreenSense = world.sample();
  sampleMs += performance.now() - measuredAt; samples++;
  world.setScreenFrame(new Float32Array(12).fill(1), 2, 2);
  measuredAt = performance.now();
  const distantScreenSense = world.sample();
  sampleMs += performance.now() - measuredAt; samples++;
  let screenAffectedScalars = 0;
  for (let i = 0; i < darkScreenSense.optic.length; i++)
    if (Math.abs(darkScreenSense.optic[i] - distantScreenSense.optic[i]) > 1e-7) screenAffectedScalars++;
  assert(screenAffectedScalars > 0,
    `The actual screen at 18 mm must be visible within the 120 mm fixture ray range: ${JSON.stringify(world.observe().retinalTrace)}`);
  world.restore(visualCheckpoint);
  world.setScreenFrame(new Float32Array(12).fill(1), 2, 2);
  const command = new Float64Array(world.residents * 92);
  command[0] = 0.05;
  command[92 + 42] = -0.04;
  let constructionTick = -1;
  for (let tick = 1; tick <= 500; tick++) {
    measuredAt = performance.now();
    await world.advance(command);
    advanceMs += performance.now() - measuredAt; advances++;
    if (world.observe().geometry.length > initial.geometry.length) {
      constructionTick = tick;
      break;
    }
  }
  assert(constructionTick >= 199 && constructionTick <= 500, "Autonomous resource-funded growth must realize within five seconds");
  const constructed = world.observe();
  const realizedGrowth = constructed.geometry.length - initial.geometry.length;
  assert(realizedGrowth > 0);
  assert(constructed.geometry.some((g) => g.name.startsWith("ecology:growth-") && g.name.endsWith(":geom")));
  assert(constructed.growth.clearanceAccepted >= realizedGrowth);
  assert(constructed.growth.clearanceScratchBuilds < constructed.growth.proposed,
    "Rejected growth must reuse the topology-scoped MuJoCo clearance model");
  assert(constructed.illumination.length > 0 && constructed.illumination.every((v) => v.available_energy_per_s > 0));
  assert(constructed.ecology.accounting.captured_photon_energy > initial.ecology.accounting.captured_photon_energy,
    "Measured physical light must fund native photon capture");

  const toy = await world.insertObject({
    position: [18, 12, 8],
    size: [0.35, 0.35, 0.35],
    shape: "box",
    rgba: [0.25, 0.55, 0.85, 1],
  });
  assert.equal(world.observe().geometry.length, initial.geometry.length + realizedGrowth + 1);
  world.queueVisitorForce(toy.id, [0.001, 0, 0]);
  world.visitorSound([18, 12, 8], 720, 0.3, 0.05);

  const checkpoint = world.snapshot();
  measuredAt = performance.now();
  await world.advance(command);
  advanceMs += performance.now() - measuredAt; advances++;
  const future = world.snapshot();
  world.restore(checkpoint);
  measuredAt = performance.now();
  await world.advance(command);
  advanceMs += performance.now() - measuredAt; advances++;
  assert.deepEqual(world.snapshot(), future, "MuJoCo integration, queued force, ecology and private state replay exactly");
  measuredAt = performance.now();
  const finalSense = world.sample();
  sampleMs += performance.now() - measuredAt; samples++;
  assert([...finalSense.optic, ...finalSense.body].every(Number.isFinite));

  console.log(JSON.stringify({
    engine: world.engine,
    source_mjcf_sha256: fixture.source_mjcf_sha256,
    residents: world.residents,
    compiled: fixture.compiled_counts,
    optic_scalars: finalSense.optic.length,
    supported_rays_per_resident: fixture.supported_sites.filter(Boolean).length,
    screen_affected_scalars: screenAffectedScalars,
    body_scalars: finalSense.body.length,
    construction_tick: constructionTick,
    realized_growth: realizedGrowth,
    growth_at_construction: constructed.growth,
    illumination: constructed.illumination,
    captured_photon_energy: world.observe().ecology.accounting.captured_photon_energy,
    topology_geoms: world.observe().geometry.length,
    mesh_vertices: initial.meshes.reduce((n, mesh) => n + mesh.positions.length / 3, 0),
    elapsed_ms: Math.round(performance.now() - started),
    mean_advance_ms: advanceMs / advances,
    mean_sample_ms: sampleMs / samples,
    replay: "exact",
  }, null, 2));
} finally {
  world.dispose();
}
