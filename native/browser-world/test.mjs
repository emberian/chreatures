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
  const initial = world.observe();
  assert.equal(world.engine, ENGINE);
  assert.equal(world.residents, 2);
  assert.deepEqual(initial.worldSizeMm, [50, 40, 16]);
  assert.equal(initial.geometry.length, 168);
  assert.equal(initial.meshes.length, 69);
  assert.equal(initial.residents[0].head, fixture.bodies[0].head);
  assert(initial.geometry.some((g) => g.mesh_id >= 0 && g.name && g.resident_id === fixture.bodies[0].id));

  const sensed = world.sample();
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
  world.setEcologySites({
    construction_sites: [{
      site_id: "joined-clear-fiber-site",
      organism_id: "colony-0",
      region_id: "region-0-0-2",
      host_template_id: "fiber-capsule",
      physics_binding: "joined-clear-fiber",
      position_m: [-0.020, -0.015, 0.009],
      orientation_xyzw: [0, 0, 0, 1],
      route_hint: null,
    }],
  });
  const command = new Float64Array(world.residents * 92);
  command[0] = 0.05;
  command[92 + 42] = -0.04;
  let constructionTick = -1;
  for (let tick = 1; tick <= 205; tick++) {
    await world.advance(command);
    if (world.observe().geometry.length > initial.geometry.length) {
      constructionTick = tick;
      world.setEcologySites();
      break;
    }
  }
  assert(constructionTick >= 199, "Resource-funded construction must occur after its two-second interval");
  const constructed = world.observe();
  assert.equal(constructed.geometry.length, initial.geometry.length + 1);
  assert(constructed.geometry.some((g) => g.name === "ecology:joined-clear-fiber:geom"));

  const toy = await world.insertObject({
    position: [18, 12, 8],
    size: [0.35, 0.35, 0.35],
    shape: "box",
    rgba: [0.25, 0.55, 0.85, 1],
  });
  assert.equal(world.observe().geometry.length, initial.geometry.length + 2);
  world.queueVisitorForce(toy.id, [0.001, 0, 0]);
  world.visitorSound([18, 12, 8], 720, 0.3, 0.05);

  const checkpoint = world.snapshot();
  await world.advance(command);
  const future = world.snapshot();
  world.restore(checkpoint);
  await world.advance(command);
  assert.deepEqual(world.snapshot(), future, "MuJoCo integration, queued force, ecology and private state replay exactly");
  const finalSense = world.sample();
  assert([...finalSense.optic, ...finalSense.body].every(Number.isFinite));

  console.log(JSON.stringify({
    engine: world.engine,
    source_mjcf_sha256: fixture.source_mjcf_sha256,
    residents: world.residents,
    compiled: fixture.compiled_counts,
    optic_scalars: finalSense.optic.length,
    body_scalars: finalSense.body.length,
    construction_tick: constructionTick,
    topology_geoms: world.observe().geometry.length,
    mesh_vertices: initial.meshes.reduce((n, mesh) => n + mesh.positions.length / 3, 0),
    elapsed_ms: Math.round(performance.now() - started),
    replay: "exact",
  }, null, 2));
} finally {
  world.dispose();
}
