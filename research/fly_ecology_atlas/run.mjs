#!/usr/bin/env node
// Research orchestration only. All growth, matter and physics run in native cores.
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];
for(const name of ['plan','setting','layout','output'])if(!args[name])throw Error('Missing --'+name);
const sha=b=>createHash('sha256').update(b).digest('hex');
const runnerHash=sha(await readFile(new URL(import.meta.url)));
const planBytes=await readFile(args.plan),plan=JSON.parse(planBytes);
if(plan.format!=='chreatures-fly-ecology-development-atlas-v1'||plan.control_dt!==.01||plan.ticks<800)throw Error('Plan format differs');
if(runnerHash!==plan.runner_sha256)throw Error('Frozen runner checksum differs');
const setting=args.proposal?JSON.parse(await readFile(args.proposal)):plan.settings.find(s=>s.setting_id===args.setting);
const layout=plan.layouts.find(l=>l.layout_id===args.layout);
if(!setting||!layout)throw Error('Unknown setting or layout');
const fixtureBytes=await readFile(layout.fixture);if(sha(fixtureBytes)!==layout.fixture_sha256)throw Error('Layout changed');
const fixture=JSON.parse(fixtureBytes);for(const o of fixture.ecology.organisms)if(o.anchored_region&&o.genotype.development)Object.assign(o.genotype.development,setting.genes);
const xml=await readFile(resolve(dirname(layout.fixture),fixture.scene_xml));
if(sha(xml)!==layout.scene_xml_sha256)throw Error('Layout XML changed');
const runtimeBytes=await readFile(plan.runtime);if(sha(runtimeBytes)!==plan.runtime_sha256)throw Error('Runtime changed');
const coreWasm=await readFile(resolve(dirname(plan.runtime),'pkg/chreatures_browser_world_bg.wasm'));
if(sha(coreWasm)!==plan.core_wasm_sha256)throw Error('Native core changed');
if(sha(await readFile(resolve(dirname(plan.runtime),'pkg/chreatures_browser_world.js')))!==plan.core_js_sha256)throw Error('Native loader changed');
if(process.version!==plan.node_version)throw Error('Node runtime version changed');
for(const [name,expected] of Object.entries(plan.mujoco_sha256))if(sha(await readFile(resolve(dirname(plan.runtime),'node_modules/@mujoco/mujoco',name)))!==expected)throw Error('MuJoCo dependency changed: '+name);
const assets=new Map();for(const a of fixture.mesh_assets)assets.set(a.path,await readFile(resolve(plan.asset_root,a.path)));
const {createBrowserWorld}=await import(pathToFileURL(plan.runtime));
const routePacket=o=>({open:Array.from(o.routeMeasurements.open),flowsM3S:Array.from(o.routeMeasurements.flowsM3S)});
const started=performance.now();let world,completedTicks=0,initial,final,error=null;const trace=[];
try{
 world=await createBrowserWorld({fixture,xml:xml.toString('utf8'),assets,coreWasm,seed:layout.world_seed});
 world.setScreenFrame(new Float32Array([layout.brightness,layout.brightness,layout.brightness]),1,1);
 initial=world.snapshot();const motor=new Float64Array(world.residents*92);
 for(let tick=0;tick<plan.ticks;tick++){
  await world.advance(motor,.01);completedTicks++;
  if(tick===0||tick%100===99||tick===plan.ticks-1){const o=world.observe();trace.push({tick:completedTicks,time:o.time,growth:o.growth,illumination:o.illumination,routeMeasurements:routePacket(o),ecology:o.ecology});console.error(JSON.stringify({tick:completedTicks,time:o.time,growth:o.growth,wall_seconds:(performance.now()-started)/1000}));}
 }
 final=world.snapshot();
}catch(e){error=String(e?.stack??e);}
let o=null;try{o=world?world.observe():null;}catch(e){error=(error??'')+'\nFinal observer unavailable: '+String(e);}
const report={format:'chreatures-fly-ecology-development-run-v1',completed:completedTicks===plan.ticks&&final!==undefined&&error===null,error,setting,layout,
 provenance:{plan_sha256:sha(planBytes),base_fixture_sha256:plan.base_fixture_sha256,derived_gene_fixture_sha256:sha(Buffer.from(JSON.stringify(fixture))),
 source_revision:plan.source_revision,mujoco_sha256:plan.mujoco_sha256,node_version:process.version,runtime_sha256:plan.runtime_sha256,core_wasm_sha256:plan.core_wasm_sha256,core_js_sha256:plan.core_js_sha256,runner_sha256:runnerHash,
 initial_world_sha256:initial?sha(Buffer.from(JSON.stringify(initial))):null,final_world_sha256:final?sha(Buffer.from(JSON.stringify(final))):null},
 diagnostic_motor:plan.neutral_motor,completed_ticks:completedTicks,seconds:completedTicks*.01,wall_seconds:(performance.now()-started)/1000,
 residents:world?.residents??null,trace,
 final: o?{time:o.time,ecology:o.ecology,growth:o.growth,illumination:o.illumination,routeMeasurements:routePacket(o),
 geometry:o.geometry.filter(g=>g.name.startsWith('ecology:growth-')).map(g=>({...g,position:Array.from(g.position),rotation:Array.from(g.rotation)}))}:null};
await writeFile(args.output,JSON.stringify(report)+'\n',{flag:'wx'});
if(final&&args.checkpoint)await writeFile(args.checkpoint,JSON.stringify(final)+'\n',{flag:'wx'});
world?.dispose();console.log(JSON.stringify({output:args.output,completed:report.completed,ticks:completedTicks,wall_seconds:report.wall_seconds,error}));
await new Promise(r=>process.stdout.write('',r));process.exit(report.completed?0:1);
