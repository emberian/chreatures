import assert from "node:assert/strict";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import loadMujoco from "../browser-world/node_modules/@mujoco/mujoco/mujoco.js";

const modelPath = resolve(
  process.argv[2] || "assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml",
);
const steps = Number(process.argv[3] || 100);
const receiptPath = process.argv[4];
const mj = await loadMujoco();
let model;
let data;
try {
  assert.equal(mj.mj_versionString(), "3.12.0");
  const modelDir = dirname(modelPath);
  const xml = await readFile(modelPath, "utf8");
  try { mj.FS.mkdir("/fly-body"); } catch (_) {}
  mj.FS.writeFile(`/fly-body/${basename(modelPath)}`, xml);
  for (const file of await readdir(modelDir)) {
    if (file.endsWith(".stl")) {
      mj.FS.writeFile(`/fly-body/${file}`, new Uint8Array(await readFile(join(modelDir, file))));
    }
  }
  model = mj.MjModel.from_xml_path(`/fly-body/${basename(modelPath)}`);
  data = new mj.MjData(model);
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  for (let i = 0; i < steps; i++) mj.mj_step(model, data);
  assert.equal(model.njnt, 127);
  assert.equal(model.nu, 90);
  assert.equal(model.nsensor, 6);
  assert(Array.from(data.qpos).every(Number.isFinite));
  assert(Array.from(data.qvel).every(Number.isFinite));
  assert(Array.from(data.sensordata).every(Number.isFinite));
  const receipt = JSON.stringify({
    engine: `mujoco-${mj.mj_versionString()}-wasm`,
    model: modelPath,
    steps,
    time_s: data.time,
    nq: model.nq,
    nv: model.nv,
    nu: model.nu,
    nbody: model.nbody,
    njnt: model.njnt,
    ngeom: model.ngeom,
    nsite: model.nsite,
    nsensor: model.nsensor,
    nsensordata: model.nsensordata,
    contacts_after_startup: data.ncon,
    finite: true,
  }, null, 2) + "\n";
  if (receiptPath) await writeFile(receiptPath, receipt);
  console.log(receipt);
} finally {
  data?.delete();
  model?.delete();
}
