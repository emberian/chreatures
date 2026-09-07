import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';
import {habitatView} from './habitat-view.js';

const FORMATS = new Set(['chreatures-living-reef-public-recording-v1','chreatures-living-reef-public-recording-v2','chreatures-living-reef-public-recording-v3']);
const GEOMETRY_ENCODING = 'entity-replacement-delta-v1';
const ACTIONS = ['thrust','yaw','gaze_pitch','grip','signal_low','signal_mid','signal_high','posture','oral'];
const ACTOR_ACTIONS = ACTIONS.slice(0,8);
const $ = (selector) => document.querySelector(selector);
const canvas = $('#world-canvas');
const ui = {
  loading: $('#loading'), time: $('#model-time'), status: $('#recording-status'),
  play: $('#play'), scrubber: $('#scrubber'), frame: $('#frame-number'), tick: $('#tick-number'),
  peripheral: $('#peripheral'), foveal: $('#foveal'), readouts: $('#readouts'),
  metabolism: $('#metabolism'), activity: $('#activity'), support: $('#support'),
  goalTime: $('#goal-time'), goalLeft: $('#goal-left'), actions: $('#actions'), moments: $('#moments'),
  decisionGoal: $('#decision-goal'), decisionChoice: $('#decision-choice'), proposal: $('#proposal-values'),
  candidates: $('#candidates'), correction: $('#private-correction'), privateUpdates: $('#private-updates'),
  goalErrorScale: $('#goal-error-scale'),
  proposalContract: $('#proposal-contract'), actionContract: $('#action-contract'),
  physiology: $('#physiology-values'), physiologyContract: $('#physiology-contract'), evidenceLink: $('#recording-evidence-link'),
  resident: $('#resident-select'), residentCoverage: $('#resident-coverage'), events: $('#events'),
  eventFilters: $('#event-filters'), eventContract: $('#event-contract'),
  context: $('#context-diagnostics'), contextUnavailable: $('#context-unavailable'),
  source: $('#recording-source'), hearSignals: $('#hear-signals'),
  suffix: $('#suffix-candidates'), suffixNote: $('#suffix-note'), matterInventory: $('#matter-inventory'),
  matterFlows: $('#matter-flows'), matterNote: $('#matter-note'), matterContract: $('#matter-contract'),
};

let renderer, scene, camera, controls, worldRoot, signalPoints, physicalSun, ambientFill, inspectionLights, pathLine, gazeLine, eventPoints, matterPoints, matterLines;
let recording = null, cursor = 0, playing = false, lastClock = performance.now(), cameraMode = 'orbit';
let activeBody = 0, activeEventKind = 'all';
let audioContext=null,hearingSignals=false,lastAudibleFrame=-1;
const activeVoices=new Set();
let pathFrameIndices=[];
const overlayVisibility={path:true,gaze:true,signals:true,events:true,matter:true};
let pools = new Map();
const dummy = new THREE.Object3D(), tint = new THREE.Color(), qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
const pos = new THREE.Vector3(), posB = new THREE.Vector3(), look = new THREE.Vector3(), direction = new THREE.Vector3();
const upA = new THREE.Vector3(), upB = new THREE.Vector3();
const visualFrames = new WeakMap();

function fail(message, reason) {
  ui.loading.hidden = false;
  ui.loading.textContent = message;
  ui.status.textContent = 'No public recording is loaded.';
  if (reason) console.error(reason);
}

function finiteArray(value, length, label) {
  if (!Array.isArray(value) || value.length !== length || value.some((item) => !Number.isFinite(item))) {
    throw new Error(`${label} is invalid`);
  }
}

function validate(data) {
  if (!data || !FORMATS.has(data.format)) throw new Error('Unsupported public recording format');
  if (data.geometry_encoding !== GEOMETRY_ENCODING) throw new Error(`Expected ${GEOMETRY_ENCODING}`);
  if (!Array.isArray(data.frames) || data.frames.length < 2) throw new Error('Recording has no frame sequence');
  if (!data.geometry || !Array.isArray(data.geometry.bounds)) throw new Error('Recording has no geometry bounds');
  const actionOrder=data.organism_interface?.action_order||ACTIONS,proposalOrder=data.organism_interface?.format==='chreatures-organism-interface-v4'?actionOrder:ACTOR_ACTIONS;
  if(!Array.isArray(actionOrder)||!actionOrder.length||!Array.isArray(proposalOrder))throw new Error('Recording action contract is invalid');
  for (const [index, frame] of data.frames.entries()) {
    if (!Number.isFinite(frame.model_time) || !Number.isInteger(frame.tick)) throw new Error(`Frame ${index} time is invalid`);
    if (!Array.isArray(frame.bodies) || !Array.isArray(frame.entities)) throw new Error(`Frame ${index} is incomplete`);
    const details=Array.isArray(frame.resident_details)?frame.resident_details:(frame.selected?[frame.selected]:[]);
    for(const selected of details){
    if(selected.neural_readouts){finiteArray(selected.neural_readouts.shape,1,`Frame ${index} readout shape`);if(selected.neural_readouts.shape[0]!==384)throw new Error(`Frame ${index} readouts are not 384 values`);}
    const refinement=selected.refinement||selected.consequence_refinement,forecast=selected.forecast||selected.sensory_forecast;
    if(refinement?.status==='unavailable'&&forecast?.status==='unavailable')continue;
    if (!refinement && !forecast && !selected.sampled_proposal) continue;
    if (!refinement || !forecast || !selected.sampled_proposal) throw new Error(`Frame ${index} has a partial decision path`);
    const candidateCount=refinement.candidate_scores?.length;
    for(const [label,value] of [['GAM scores',refinement.candidate_scores],['GAM coverage',refinement.candidate_out_of_domain],['forecast progress',forecast.candidate_progress],['forecast disagreement',forecast.candidate_disagreement],['forecast validity',forecast.candidate_forecast_invalid||forecast.candidate_input_clipped],['forecast tilt',forecast.candidate_logit_tilt]]){
      if(!candidateCount||!Array.isArray(value)||value.length!==candidateCount)throw new Error(`Frame ${index} ${label} candidate count differs`);
    }
    if(!Array.isArray(refinement.selected_private_correction)||refinement.selected_private_correction.length!==3)throw new Error(`Frame ${index} private correction is invalid`);
    if(!proposalOrder.every(name=>Number.isFinite(selected.sampled_proposal[name]))||!actionOrder.every(name=>Number.isFinite(selected.committed_action[name])))throw new Error(`Frame ${index} action contract is invalid`);
    if(!Number.isInteger(refinement.selected_candidate)||refinement.selected_candidate<0||refinement.selected_candidate>=candidateCount)throw new Error(`Frame ${index} selected candidate is invalid`);
    const acquired=selected.acquired_action_candidates;
    if(acquired?.status==='recorded'){
      const value=acquired.value,keys=['available','recalled','slot','generation','length_ticks','support','empirical_score','recall_score','first_action'];
      if(!value||keys.some(key=>!Array.isArray(value[key])||value[key].length!==8)||value.first_action.some(row=>!Array.isArray(row)||row.length!==actionOrder.length))throw new Error(`Frame ${index} acquired-action contract is invalid`);
    }
    }
    if(data.format.endsWith('-v3')){const matter=frame.regional_matter;if(!matter||!['recorded','unavailable'].includes(matter.status))throw new Error(`Frame ${index} regional matter status is invalid`);if(matter.status==='recorded'&&(!Array.isArray(matter.nodes)||!Array.isArray(matter.edges)||!Array.isArray(matter.outlets)||!Array.isArray(matter.last_events)))throw new Error(`Frame ${index} regional matter view is invalid`);}
  }
  return data;
}

function expandEntityDeltas(data) {
  if (!data || data.geometry_encoding !== GEOMETRY_ENCODING || !Array.isArray(data.frames)) return data;
  let entities = new Map();
  for (const [frameIndex,frame] of data.frames.entries()) {
    if (frameIndex === 0) {
      if (!Array.isArray(frame.entities)) throw new Error('First frame must contain complete entity geometry');
      entities = new Map(frame.entities.map(entity=>[entity.entity,entity]));
    } else {
      const delta=frame.entities;
      if (!delta || !Array.isArray(delta.changed) || !Array.isArray(delta.removed)) throw new Error(`Frame ${frameIndex} has an invalid entity delta`);
      entities=new Map(entities);
      for(const key of delta.removed){if(!Number.isInteger(key))throw new Error(`Frame ${frameIndex} removal key is invalid`);entities.delete(key);}
      for(const entity of delta.changed){if(!Number.isInteger(entity?.entity))throw new Error(`Frame ${frameIndex} replacement key is invalid`);entities.set(entity.entity,entity);}
    }
    frame.entities=[...entities.values()].sort((left,right)=>left.entity-right.entity);
  }
  return data;
}

function initThree() {
  renderer = new THREE.WebGLRenderer({canvas, antialias: true, powerPreference: 'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.22;
  renderer.setClearColor('#10251d');
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2('#10251d', .0055);
  camera = new THREE.PerspectiveCamera(40, 1, .015, 180);
  camera.up.set(0, 0, 1);
  camera.position.set(11, -12, 8);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = .065;
  controls.minDistance = .2;
  controls.maxDistance = 90;
  controls.maxPolarAngle = Math.PI * .51;
  controls.target.set(4, 4, .5);
  ambientFill = new THREE.HemisphereLight('#ddd9bf', '#142c23', 0);
  ambientFill.position.set(0, 0, 1);
  scene.add(ambientFill);
  physicalSun = new THREE.DirectionalLight('#fff0c7', 0);
  scene.add(physicalSun);
  inspectionLights = new THREE.Group();
  const sky = new THREE.HemisphereLight('#fff7df', '#779087', 2.25);
  sky.position.set(0, 0, 1);
  const key = new THREE.DirectionalLight('#fff0cf', 3);
  key.position.set(-8, -10, 18);
  const fill = new THREE.DirectionalLight('#b7d5dc', 1.25);
  fill.position.set(10, 12, 10);
  const rim = new THREE.DirectionalLight('#9fc4b1', .75);
  rim.position.set(2, -12, 5);
  inspectionLights.add(sky, key, fill, rim);
  scene.add(inspectionLights);
  worldRoot = new THREE.Group();
  scene.add(worldRoot);
  const resize = () => {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
    if (recording && cameraMode === 'orbit') frameHabitat();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  renderer.setAnimationLoop(animate);
}

const geometries = {
  sphere: () => new THREE.SphereGeometry(1, 18, 12),
  ellipsoid: () => new THREE.SphereGeometry(1, 18, 12),
  box: () => new THREE.BoxGeometry(2, 2, 2),
  cylinder: () => new THREE.CylinderGeometry(1, 1, 2, 16).rotateX(Math.PI / 2),
};

function allShapes(frame) {
  if (visualFrames.has(frame)) return visualFrames.get(frame);
  const shapes = [];
  for (const entity of frame.entities || []) for (const [index,shape] of (entity.shapes || []).entries()) {
    shapes.push({...shape,key:`entity:${entity.entity}:shape:${index}`});
  }
  for (const body of frame.articulations || []) for (const [index,shape] of (body.geoms || []).entries()) {
    shapes.push({...shape,key:`body:${body.body}:geom:${shape.name || index}`});
  }
  const visual = [];
  for (const shape of shapes) {
    if (shape.type !== 'capsule') { visual.push(shape); continue; }
    const quaternion=qFromWxyz(shape.quaternion), offset=new THREE.Vector3(0,0,shape.size[1]).applyQuaternion(quaternion);
    visual.push({...shape,key:`${shape.key}:shaft`,type:'cylinder'});
    visual.push({...shape,key:`${shape.key}:cap-positive`,type:'sphere',size:[shape.size[0]],position:new THREE.Vector3().fromArray(shape.position).add(offset).toArray()});
    visual.push({...shape,key:`${shape.key}:cap-negative`,type:'sphere',size:[shape.size[0]],position:new THREE.Vector3().fromArray(shape.position).sub(offset).toArray()});
  }
  visualFrames.set(frame,visual);
  return visual;
}

function makePools(data) {
  for (const child of [...worldRoot.children]) worldRoot.remove(child);
  pools.clear();
  const keys = {};
  for (const frame of data.frames) {
    for (const shape of allShapes(frame)) (keys[shape.type] ||= new Set()).add(shape.key);
  }
  for (const [kind, keySet] of Object.entries(keys)) {
    if (!geometries[kind]) continue;
    const shapeKeys=[...keySet].sort(), count=shapeKeys.length;
    // setColorAt supplies instance colours. These primitive meshes have no
    // vertex colour attribute; USE_COLOR would multiply them by a missing input.
    const material = new THREE.MeshStandardMaterial({roughness: .72, metalness: .01});
    const mesh = new THREE.InstancedMesh(geometries[kind](), material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    worldRoot.add(mesh);
    pools.set(kind, {mesh, count, keys:shapeKeys});
  }
  const signalGeometry = new THREE.BufferGeometry();
  signalGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 256), 3));
  signalGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(3 * 256), 3));
  signalGeometry.setDrawRange(0, 0);
  signalPoints = new THREE.Points(signalGeometry, new THREE.PointsMaterial({size:.09, vertexColors:true, transparent:true, opacity:.8, depthWrite:false}));
  worldRoot.add(signalPoints);
  pathLine=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#e8bf72',transparent:true,opacity:.86}));
  gazeLine=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#d8ece2',transparent:true,opacity:.9}));
  eventPoints=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:'#e88d61',size:.16,transparent:true,opacity:.9,depthWrite:false}));
  matterPoints=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.18,vertexColors:true,transparent:true,opacity:.9,depthWrite:false}));
  matterLines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.72}));
  worldRoot.add(pathLine,gazeLine,eventPoints,matterLines,matterPoints);
  rebuildResidentOverlays();
  frameHabitat();
}

function frameHabitat() {
  const view = habitatView(recording.geometry.bounds, camera.fov, camera.aspect);
  camera.up.set(0, 0, 1);
  controls.target.fromArray(view.target);
  camera.position.fromArray(view.position);
  camera.far = Math.max(180, view.distance * 5);
  controls.maxDistance = view.distance * 3;
  camera.updateProjectionMatrix();
  controls.update();
}

function qFromWxyz(value, target = new THREE.Quaternion()) {
  return target.set(value[1], value[2], value[3], value[0]).normalize();
}

function interpolatedShape(a, b, alpha, target) {
  const av = a || b, bv = b || a;
  pos.fromArray(av.position).lerp(posB.fromArray(bv.position), alpha);
  qFromWxyz(av.quaternion, qa);
  qFromWxyz(bv.quaternion, qb);
  qa.slerp(qb, alpha);
  const size = av.size.map((value, index) => THREE.MathUtils.lerp(value, bv.size[index] ?? value, alpha));
  target.position.copy(pos);
  target.quaternion.copy(qa);
  if (av.type === 'sphere') target.scale.setScalar(size[0]);
  else if (av.type === 'box' || av.type === 'ellipsoid') target.scale.set(size[0], size[1], size[2]);
  else if (av.type === 'cylinder') target.scale.set(size[0], size[0], size[1]);
  target.updateMatrix();
  return {color: av.color, matrix: target.matrix};
}

function updateGeometry(a, b, alpha) {
  const groupedA = groupShapes(allShapes(a)), groupedB = groupShapes(allShapes(b));
  for (const [kind, pool] of pools) {
    const left = groupedA[kind] || new Map(), right = groupedB[kind] || new Map();
    for (let index = 0; index < pool.count; index++) {
      const key=pool.keys[index], aShape=left.get(key), bShape=right.get(key);
      const visible=(aShape&&bShape) || (aShape&&alpha<1) || (bShape&&alpha>=1);
      if (visible) {
        const part = interpolatedShape(aShape, bShape, alpha, dummy);
        pool.mesh.setMatrixAt(index, part.matrix);
        pool.mesh.setColorAt(index, tint.set(part.color || '#8aa17e'));
      } else {
        dummy.position.set(0,0,-1000); dummy.scale.setScalar(0); dummy.updateMatrix();
        pool.mesh.setMatrixAt(index, dummy.matrix);
      }
    }
    pool.mesh.count = pool.count;
    pool.mesh.instanceMatrix.needsUpdate = true;
    if (pool.mesh.instanceColor) pool.mesh.instanceColor.needsUpdate = true;
  }
  updateSignals(a.signals || []);
  updateLighting(a,b,alpha);
}

function groupShapes(shapes) {
  const result = {};
  for (const shape of shapes) (result[shape.type] ||= new Map()).set(shape.key,shape);
  return result;
}

function updateLighting(a,b,alpha){
  const brightest=(frame)=>{
    const lights=frame.lights||[], directional=lights.filter(light=>light.directional===true);
    return [...(directional.length?directional:lights)].sort((x,y)=>y.intensity-x.intensity)[0];
  };
  const left=brightest(a),right=brightest(b);
  if(!left&&!right){physicalSun.intensity=0;ambientFill.intensity=0;return;}
  const av=left||right,bv=right||left;
  physicalSun.color.setRGB(
    THREE.MathUtils.lerp(av.color[0],bv.color[0],alpha),
    THREE.MathUtils.lerp(av.color[1],bv.color[1],alpha),
    THREE.MathUtils.lerp(av.color[2],bv.color[2],alpha),
  );
  physicalSun.intensity=THREE.MathUtils.lerp(av.intensity,bv.intensity,alpha);
  ambientFill.intensity=THREE.MathUtils.lerp(av.ambient_intensity||0,bv.ambient_intensity||0,alpha);
  direction.fromArray(av.direction).lerp(posB.fromArray(bv.direction),alpha).normalize();
  physicalSun.position.copy(controls.target).addScaledVector(direction,-20);
  physicalSun.target.position.copy(controls.target);
  if(!physicalSun.target.parent)scene.add(physicalSun.target);
}

function updateSignals(signals) {
  if (!signalPoints) return;
  signalPoints.visible=overlayVisibility.signals;
  const positions = signalPoints.geometry.attributes.position.array;
  const colors = signalPoints.geometry.attributes.color.array;
  const palette = ['#d9794d','#b3c686','#72a6a1'];
  const count = Math.min(256, signals.length);
  for (let i=0; i<count; i++) {
    positions.set(signals[i].position, i*3);
    tint.set(palette[Math.max(0, Math.min(2, signals[i].tone))]);
    colors.set([tint.r,tint.g,tint.b], i*3);
  }
  signalPoints.geometry.setDrawRange(0,count);
  signalPoints.geometry.attributes.position.needsUpdate = true;
  signalPoints.geometry.attributes.color.needsUpdate = true;
}

function residentIds(){
  const ids=new Set();
  for(const frame of recording.frames){
    for(const body of frame.bodies||[])ids.add(body.body);
    for(const detail of frame.resident_details||[])ids.add(detail.body);
    if(frame.selected)ids.add(frame.selected.body);
  }
  return [...ids].sort((a,b)=>a-b);
}

function residentDetail(frame,body=activeBody){
  const detail=(frame.resident_details||[]).find(value=>value.body===body)||(frame.selected?.body===body?frame.selected:null);
  const trace=(frame.resident_traces||[]).find(value=>value.body===body);
  if(!detail)return trace||null;
  const bodyState=detail.recorded_body_state||trace?.recorded_body_state,goal=detail.goal?{...detail.goal,commit_remaining_ticks:detail.goal.remaining_ticks??detail.goal.commit_remaining_ticks}:undefined;
  const response=detail.population_response?.status==='recorded'?detail.population_response.value:detail.population_response?.status==='unavailable'?null:detail.population_response;
  const acquired=detail.acquired_action_candidates?.status==='recorded'?detail.acquired_action_candidates.value:null;
  return {...trace,...detail,goal,metabolism:detail.metabolism||trace?.metabolism||(bodyState?{energy:bodyState.energy,gut:bodyState.gut,fatigue:bodyState.fatigue}:undefined),consequence_refinement:detail.refinement||detail.consequence_refinement,sensory_forecast:detail.forecast||detail.sensory_forecast,sequence_memory:detail.memory_summary?.sequence||detail.sequence_memory,context:detail.memory_summary?.contextual||detail.context,population_response:response,acquired_action_candidates:acquired};
}

function bodyPose(frame,body=activeBody){return (frame.bodies||[]).find(value=>value.body===body)||null;}

function rebuildResidentOverlays(){
  if(!recording||!pathLine)return;
  const samples=[];pathFrameIndices=[];recording.frames.forEach((frame,index)=>{const body=bodyPose(frame);if(body){samples.push(body.position);pathFrameIndices.push(index);}});
  pathLine.geometry.dispose();pathLine.geometry=new THREE.BufferGeometry().setFromPoints(samples.map(value=>new THREE.Vector3().fromArray(value)));
  pathLine.geometry.setDrawRange(0,1);
  const eventPositions=[];
  for(const event of (recording.events||[]).filter(value=>activeEventKind==='all'||value.kind===activeEventKind)){
    const frame=recording.frames.reduce((best,item,index)=>Math.abs(item.tick-event.tick)<Math.abs(recording.frames[best].tick-event.tick)?index:best,0);
    const body=event.actors?.bodies?.[0],pose=body===undefined?null:bodyPose(recording.frames[frame],body);
    const entityKey=event.actors?.entities?.[0],entity=entityKey===undefined?null:recording.frames[frame].entities.find(value=>value.entity===entityKey);
    const marker=pose?.position||entity?.shapes?.[0]?.position;
    if(marker)eventPositions.push(...marker);
  }
  eventPoints.geometry.dispose();eventPoints.geometry=new THREE.BufferGeometry();
  eventPoints.geometry.setAttribute('position',new THREE.Float32BufferAttribute(eventPositions,3));
}

function updateResidentOverlays(frameIndex){
  if(!pathLine)return;
  const pathCount=pathFrameIndices.filter(index=>index<=frameIndex).length;pathLine.visible=overlayVisibility.path&&pathCount>0;pathLine.geometry.setDrawRange(0,pathCount);
  const detail=residentDetail(recording.frames[frameIndex]);
  const pose=detail?.retina_pose;
  gazeLine.visible=overlayVisibility.gaze&&Boolean(pose);
  if(pose){
    const start=new THREE.Vector3().fromArray(pose.origin),end=start.clone().addScaledVector(new THREE.Vector3().fromArray(pose.forward).normalize(),1.25);
    gazeLine.geometry.dispose();gazeLine.geometry=new THREE.BufferGeometry().setFromPoints([start,end]);
  }
  eventPoints.visible=overlayVisibility.events;
}

function clearCanvas(canvas,label){
  const context=canvas.getContext('2d');context.fillStyle='#162920';context.fillRect(0,0,canvas.width,canvas.height);
  context.fillStyle='#a9a58f';context.font='10px sans-serif';context.fillText(label,6,Math.max(12,canvas.height/2));
}

function base64Bytes(blob, expected) {
  if (!blob || blob.encoding !== 'base64-u8-linear-0-1') throw new Error('Retina encoding is unsupported');
  const binary = atob(blob.data), bytes = new Uint8Array(binary.length);
  for (let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);
  if (bytes.length !== expected) throw new Error('Retina byte length differs from its shape');
  return bytes;
}

function float32Values(blob) {
  if (!blob || blob.encoding !== 'base64-little-endian-float32') throw new Error('Readout encoding is unsupported');
  const binary = atob(blob.data), bytes = new Uint8Array(binary.length);
  for (let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  return Float32Array.from({length: bytes.length/4}, (_, i) => view.getFloat32(i*4, true));
}

function paintRetina(canvas, blob) {
  const [height,width,channels] = blob.shape;
  if (channels !== 4) throw new Error('Retina must contain RGB plus proximity');
  const values = base64Bytes(blob, width*height*4), context = canvas.getContext('2d');
  canvas.width = width; canvas.height = height + 2;
  const image = context.createImageData(width, height + 2);
  for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
    const source=(y*width+x)*4, dest=(y*width+x)*4;
    image.data.set([values[source],values[source+1],values[source+2],255],dest);
  }
  for (let x=0; x<width; x++) {
    let proximity=0;
    for (let y=0; y<height; y++) proximity=Math.max(proximity,values[(y*width+x)*4+3]);
    for (let y=height; y<height+2; y++) image.data.set([proximity,proximity,proximity,255],(y*width+x)*4);
  }
  context.putImageData(image,0,0);
}

function paintReadouts(blob) {
  const values=float32Values(blob), context=ui.readouts.getContext('2d'), width=ui.readouts.width, height=ui.readouts.height;
  let maximum=1e-8;
  for(const value of values) maximum=Math.max(maximum,Math.abs(value));
  context.fillStyle='#162920'; context.fillRect(0,0,width,height);
  for(let i=0;i<384;i++){
    const normalized=Math.min(1,Math.abs(values[i])/maximum), hue=values[i]>=0 ? [177,196,135] : [211,112,75];
    context.fillStyle=`rgba(${hue.join(',')},${.13+.87*normalized})`;
    context.fillRect(i, height*(1-normalized), 1, height*normalized);
  }
}

function makeActionBars() {
  const names=recording.organism_interface?.action_order||ACTIONS;ui.actionContract.textContent=`${names.length} committed channels`;
  ui.actions.replaceChildren(...names.map((name) => {
    const item=document.createElement('div'); item.className='action'; item.dataset.action=name;
    const label=document.createElement('div'),title=document.createElement('span'),value=document.createElement('span'),bar=document.createElement('div'),fill=document.createElement('i');
    label.className='action-label';title.textContent=name.replaceAll('_',' ');value.textContent='0.00';bar.className='bar';bar.append(fill);label.append(title,value);item.append(label,bar);
    return item;
  }));
}

function makeDecisionDisplay(){
  const names=recording.organism_interface?.format==='chreatures-organism-interface-v4'?recording.organism_interface.action_order:ACTOR_ACTIONS;ui.proposalContract.textContent=`Actor proposal · ${names.length} channels`;
  ui.proposal.replaceChildren(...names.map(name=>{
    const item=document.createElement('div');item.className='proposal-value';item.dataset.action=name;
    item.innerHTML=`<span>${name.replaceAll('_',' ')}</span><span>0.000</span>`;return item;
  }));
  const candidateCount=Math.max(1,...recording.frames.flatMap(frame=>(frame.resident_details||(frame.selected?[frame.selected]:[])).map(detail=>(detail.refinement||detail.consequence_refinement)?.candidate_scores?.length||0)));
  ui.candidates.replaceChildren(...Array.from({length:candidateCount},(_,index)=>{
    const row=document.createElement('div');row.className='candidate-row';row.dataset.candidate=index;
    row.innerHTML=`<span>${index+1}</span><span data-value="score">—</span><span class="coverage" data-value="coverage">—</span><span data-value="progress">—</span><span data-value="disagreement">—</span><span data-value="tilt">—</span>`;
    return row;
  }));
}

function formatNative(value){return Number.isFinite(value)?Number(value).toFixed(3):'—';}

function paintDecision(selected){
  const refinement=selected.consequence_refinement,forecast=selected.sensory_forecast;
  ui.decisionGoal.textContent=selected.goal.valid?`goal at ${selected.goal.recorded_time.toFixed(2)} s`:'no valid goal';
  ui.decisionChoice.textContent=`candidate ${refinement.selected_candidate+1}`;
  for(const item of ui.proposal.children)item.querySelector('span:last-child').textContent=formatNative(selected.sampled_proposal[item.dataset.action]);
  for(const row of ui.candidates.children){
    const index=Number(row.dataset.candidate),ood=refinement.candidate_out_of_domain[index];
    row.classList.toggle('selected',index===refinement.selected_candidate);
    row.querySelector('[data-value="score"]').textContent=formatNative(refinement.candidate_scores[index]);
    const coverage=row.querySelector('[data-value="coverage"]');coverage.textContent=ood?'outside':'within';coverage.classList.toggle('ood',ood);
    const progress=row.querySelector('[data-value="progress"]');progress.textContent=formatNative(forecast.candidate_progress[index]);progress.classList.toggle('clipped',(forecast.candidate_forecast_invalid||forecast.candidate_input_clipped||[])[index]);
    row.querySelector('[data-value="disagreement"]').textContent=formatNative(forecast.candidate_disagreement[index]);
    row.querySelector('[data-value="tilt"]').textContent=formatNative(forecast.candidate_logit_tilt[index]);
  }
  ui.correction.textContent=refinement.selected_private_correction.map(formatNative).join(' · ');
  ui.privateUpdates.textContent=String(refinement.completed_private_updates_before_action);
  ui.goalErrorScale.textContent=formatNative(forecast.empirical_goal_error_scale);
  const response=selected.population_response,coverage=$('#population-response-coverage');
  coverage.hidden=!response;
  if(response)coverage.textContent=`Population GAM: last executed action ${response.executed_transition_in_domain?'within':'outside'} the fitted domain · ${response.in_domain_total.toLocaleString()} covered / ${response.out_of_domain_total.toLocaleString()} outside since birth. Coverage does not establish an effect on the selected action.`;
}

function setDecisionAvailable(available){
  const section=ui.candidates.closest('section');section.classList.toggle('instrument-unavailable',!available);
  if(!available){ui.decisionGoal.textContent='not recorded';ui.decisionChoice.textContent='unavailable';ui.correction.textContent='—';ui.privateUpdates.textContent='—';ui.goalErrorScale.textContent='—';for(const value of ui.proposal.querySelectorAll('span:last-child'))value.textContent='—';for(const row of ui.candidates.children)for(const value of row.querySelectorAll('[data-value]'))value.textContent='—';}
}

function paintContext(selected){
  const streams=[['context',selected?.context],['sequence memory',selected?.sequence_memory],['action prediction',selected?.action_prediction]],rows=[];
  for(const [stream,value] of streams){
    if(value===undefined||value===null||value.status==='unavailable')continue;
    const published=value.status==='recorded'&&Object.hasOwn(value,'value')?value.value:value;
    const entries=typeof published==='object'&&!Array.isArray(published)?Object.entries(published):[['value',published]];
    for(const [name,item] of entries){
      const row=document.createElement('div'),label=document.createElement('span'),output=document.createElement('code');row.className='diagnostic-row';label.textContent=`${stream} · ${name.replaceAll('_',' ')}`;
      output.textContent=Array.isArray(item)?item.map(value=>typeof value==='number'?formatNative(value):String(value)).join(' · '):typeof item==='number'?formatNative(item):String(item);
      row.append(label,output);rows.push(row);
    }
  }
  ui.context.replaceChildren(...rows);ui.contextUnavailable.hidden=rows.length>0;
}

function paintSuffix(selected){
  const summary=selected?.acquired_action_candidates;ui.suffixNote.hidden=Boolean(summary);ui.suffix.replaceChildren();
  if(!summary)return;
  const rows=[];for(let index=0;index<summary.available.length;index++){
    const card=document.createElement('div'),title=document.createElement('strong'),meta=document.createElement('span');card.className='suffix-card';
    const available=summary.available[index],recalled=summary.recalled[index];card.classList.toggle('unavailable',!available);card.classList.toggle('recalled',available&&recalled);card.classList.toggle('selected',index===summary.selected_candidate);
    title.textContent=`${index+1} · ${available?(recalled?'recalled suffix':'local proposal'):'unavailable'}${index===summary.selected_candidate?' · selected':''}`;
    meta.textContent=available?(recalled?`slot ${summary.slot[index]} · generation ${summary.generation[index]} · ${summary.length_ticks[index]} ticks · support ${summary.support[index]} · empirical ${formatNative(summary.empirical_score[index])} · recall ${formatNative(summary.recall_score[index])}`:'current local anatomical candidate'):'no candidate in this slot';
    card.append(title,meta);rows.push(card);
  }
  ui.suffix.replaceChildren(...rows);ui.suffixNote.hidden=false;ui.suffixNote.textContent=`${summary.occupied_slots} recalled slots occupied · ${summary.learned_total} learned total. Support is execution frequency, not confidence; scores and first actions are recorded selection diagnostics, not experienced outcomes.`;
}

function paintMatter(frame){
  const matter=frame.regional_matter,available=matter?.status==='recorded';ui.matterNote.hidden=false;ui.matterInventory.replaceChildren();ui.matterFlows.replaceChildren();
  if(!available){ui.matterNote.textContent=matter?.reason||'Regional matter was unavailable in this recording.';if(matterPoints)matterPoints.visible=false;if(matterLines)matterLines.visible=false;return;}
  ui.matterNote.textContent='Node color reflects recorded total inventory. Edge color reflects recorded movement magnitude and accessibility without implying direction; arrows appear in the list only when a committed event supplies per-pool direction.';
  ui.matterContract.textContent=`step ${matter.step_index} · pools in synthetic chemical amount`;
  const inventories=matter.nodes.map(node=>{const row=document.createElement('div'),name=document.createElement('strong'),values=document.createElement('span');row.className='matter-node';name.textContent=`region ${node.node}`;values.textContent=Object.entries(node.pools).map(([pool,value])=>`${pool} ${formatNative(value)}`).join(' · ')||'empty';row.append(name,values);return row;});ui.matterInventory.replaceChildren(...inventories);
  const flows=[];for(const event of matter.last_events||[]){if(event.kind!=='regional-material-flow')continue;for(const [pool,direction] of Object.entries(event.details?.directions||{})){const quantity=event.quantities?.find(item=>item.name===pool),row=document.createElement('div');row.className='matter-flow';row.textContent=`${pool}: region ${direction.source} → region ${direction.target}${quantity?` · ${formatNative(quantity.value)} ${quantity.unit}`:''} · recorded route event`;flows.push(row);}}
  if(!flows.length){const row=document.createElement('div');row.className='matter-flow';row.textContent='No directional regional flow event was recorded at this frame.';flows.push(row);}ui.matterFlows.replaceChildren(...flows);
  updateMatterGeometry(matter);
}

function updateMatterGeometry(matter){
  if(!matterPoints||!matterLines)return;matterPoints.visible=overlayVisibility.matter;matterLines.visible=overlayVisibility.matter;
  const nodePosition=new Map(matter.nodes.map(node=>[node.node,node.position])),nodeValues=matter.nodes.map(node=>Object.values(node.pools).reduce((sum,value)=>sum+Math.max(0,value),0)),peak=Math.max(1e-12,...nodeValues);
  const positions=new Float32Array(matter.nodes.length*3),colors=new Float32Array(matter.nodes.length*3);matter.nodes.forEach((node,index)=>{positions.set(node.position,index*3);const level=Math.log1p(nodeValues[index])/Math.log1p(peak);tint.set('#cbb866').lerp(new THREE.Color('#e67c4b'),level);colors.set([tint.r,tint.g,tint.b],index*3);});
  const edgePositions=new Float32Array(matter.edges.length*6),edgeColors=new Float32Array(matter.edges.length*6);matter.edges.forEach((edge,index)=>{edgePositions.set(nodePosition.get(edge.source),index*6);edgePositions.set(nodePosition.get(edge.target),index*6+3);const moved=Object.values(edge.last_moved_resources).reduce((sum,value)=>sum+Math.abs(value),0),level=Math.min(1,Math.log1p(moved)*.35),color=new THREE.Color('#547467').lerp(new THREE.Color('#e28955'),level);color.multiplyScalar(.45+.55*edge.accessibility);edgeColors.set([color.r,color.g,color.b,color.r,color.g,color.b],index*6);});
  matterPoints.geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));matterPoints.geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));matterLines.geometry.setAttribute('position',new THREE.BufferAttribute(edgePositions,3));matterLines.geometry.setAttribute('color',new THREE.BufferAttribute(edgeColors,3));
}

function paintActions(values) {
  for(const item of ui.actions.children){
    const value=Math.max(-1,Math.min(1,Number(values[item.dataset.action])||0));
    item.querySelector('.action-label span:last-child').textContent=value.toFixed(2);
    const fill=item.querySelector('i'); fill.style.width=`${Math.abs(value)*50}%`; fill.style.left=value<0?`${50+value*50}%`:'50%';
  }
}

function paintHistory(frameIndex) {
  const context=ui.metabolism.getContext('2d'), width=ui.metabolism.width, height=ui.metabolism.height;
  context.fillStyle='#162920'; context.fillRect(0,0,width,height);
  const names=[['energy','#efb36b'],['gut','#9bb98e'],['fatigue','#d86d55']];
  for(const [name,color] of names){
    const values=recording.frames.map(frame=>residentDetail(frame)?.metabolism?.[name]).filter(Number.isFinite);
    if(!values.length)continue;
    const low=Math.min(...values), high=Math.max(...values), span=Math.max(1e-7,high-low);
    context.beginPath(); context.strokeStyle=color; context.lineWidth=2;
    for(let i=0;i<=frameIndex;i++){
      const value=residentDetail(recording.frames[i])?.metabolism?.[name];if(!Number.isFinite(value))continue;
      const x=i/Math.max(1,recording.frames.length-1)*width, y=height-7-(value-low)/span*(height-14);
      i?context.lineTo(x,y):context.moveTo(x,y);
    }
    context.stroke();
  }
  const x=frameIndex/Math.max(1,recording.frames.length-1)*width;
  context.strokeStyle='#f1eddf66'; context.lineWidth=1; context.beginPath(); context.moveTo(x,0); context.lineTo(x,height); context.stroke();
}

function paintPhysiology(selected){const values=selected.recorded_body_state||selected.metabolism,names=recording.recorded_body_state?.fields?.map(field=>field.name)||Object.keys(values);ui.physiologyContract.textContent=`${names.length} post-physics channels`;ui.physiology.replaceChildren(...names.map(name=>{const row=document.createElement('div'),label=document.createElement('span'),value=document.createElement('b'),unit=recording.recorded_body_state?.fields?.find(field=>field.name===name)?.unit;label.textContent=name.replaceAll('_',' ');value.textContent=`${formatNative(values[name])}${unit?` · ${unit}`:''}`;row.append(label,value);return row}))}

function updateInstruments(index) {
  const frame=recording.frames[index], selected=residentDetail(frame);
  ui.time.textContent=`model time ${frame.model_time.toFixed(3)} s`;
  ui.frame.textContent=`${index+1} / ${recording.frames.length}`;
  ui.tick.textContent=`tick ${frame.tick}`;
  const detailed=Boolean(selected?.retina&&selected?.neural_readouts);
  if(selected?.retina){paintRetina(ui.peripheral,selected.retina.peripheral);paintRetina(ui.foveal,selected.retina.foveal);}else{clearCanvas(ui.peripheral,'not recorded');clearCanvas(ui.foveal,'not recorded');}
  if(selected?.neural_readouts)paintReadouts(selected.neural_readouts);else clearCanvas(ui.readouts,'population readouts not recorded for this resident');
  ui.activity.textContent=Number.isFinite(selected?.neural_summary?.activity)?selected.neural_summary.activity.toFixed(4):'—';
  ui.support.textContent=Number.isFinite(selected?.neural_summary?.support)?selected.neural_summary.support.toFixed(4):'—';
  ui.goalTime.textContent=selected?.goal?.valid?`${selected.goal.recorded_time.toFixed(3)} s`:selected?.goal?'no valid record':'not recorded';
  ui.goalLeft.textContent=selected?.goal?.valid?`${selected.goal.commit_remaining_ticks} ticks`:'—';
  paintActions(selected?.committed_action||{});
  if(selected?.recorded_body_state||selected?.metabolism)paintPhysiology(selected);else ui.physiology.replaceChildren();
  const refinement=selected?.consequence_refinement,forecast=selected?.sensory_forecast;
  const hasDecision=Boolean(refinement&&forecast&&selected?.sampled_proposal&&refinement.status!=='unavailable'&&forecast.status!=='unavailable');setDecisionAvailable(hasDecision);if(hasDecision)paintDecision(selected);
  paintContext(selected);
  paintSuffix(selected);
  paintMatter(frame);
  paintHistory(index);
  updateResidentOverlays(index);
  ui.residentCoverage.textContent=detailed?'Retina, neural readouts and recorded controller detail are available at this frame.':'Physical trace available; detailed sensory and controller streams were not recorded for this resident at this frame.';
  for(const button of ui.moments.querySelectorAll('button')) button.setAttribute('aria-current',String(Number(button.dataset.frame)===index));
}

function populateMoments() {
  ui.moments.replaceChildren(...(recording.phenomena_moments||[]).map((moment)=>{
    const item=document.createElement('li'), button=document.createElement('button');
    button.type='button'; button.dataset.frame=moment.frame;
    const time=document.createElement('time'),label=document.createElement('span');time.textContent=`${moment.model_time.toFixed(2)} s`;label.textContent=moment.phenomena.map(value=>value.replaceAll('-',' ')).join(' · ');button.append(time,label);
    button.addEventListener('click',()=>seek(moment.frame)); item.append(button); return item;
  }));
  if(!ui.moments.children.length){const item=document.createElement('li');item.textContent='No phenomena were indexed in this recording.';ui.moments.append(item);}
}

function populateEvents(){
  const events=recording.events||[],kinds=['all',...new Set(events.map(event=>event.kind))];
  ui.eventFilters.replaceChildren(...kinds.map(kind=>{const button=document.createElement('button');button.type='button';button.textContent=kind.replaceAll('_',' ');button.dataset.kind=kind;button.setAttribute('aria-pressed',String(kind===activeEventKind));button.addEventListener('click',()=>{activeEventKind=kind;populateEvents();rebuildResidentOverlays();});return button;}));
  const shown=events.filter(event=>activeEventKind==='all'||event.kind===activeEventKind);
  ui.events.replaceChildren(...shown.map(event=>{const item=document.createElement('li'),button=document.createElement('button'),time=document.createElement('time'),content=document.createElement('span'),title=document.createElement('span'),quantity=document.createElement('span');time.textContent=`${event.model_time.toFixed(2)} s`;title.textContent=event.kind.replaceAll('_',' ');quantity.className='event-quantity';quantity.textContent=(event.quantities||[]).map(value=>`${value.name.replaceAll('_',' ')} ${formatNative(value.value)} ${value.unit}`).join(' · ')||'recorded event receipt';content.append(title,quantity);button.append(time,content);button.addEventListener('click',()=>seek(nearestFrame(event.tick)));item.append(button);return item;}));
  if(!shown.length){const item=document.createElement('li');item.textContent=events.length?'No events match this filter.':'This recording does not contain the world event stream.';ui.events.append(item);}
  ui.eventContract.textContent=events.length?`${events.length} exact events`:'unavailable in this recording';
}

function nearestFrame(tick){let best=0;for(let index=1;index<recording.frames.length;index++)if(Math.abs(recording.frames[index].tick-tick)<Math.abs(recording.frames[best].tick-tick))best=index;return best;}

function sonifyFrame(index){
  if(!hearingSignals||document.hidden||!audioContext||index===lastAudibleFrame)return;lastAudibleFrame=index;
  const frame=recording.frames[index],ids=new Set(frame.event_ids||[]),events=(recording.events||[]).filter(event=>event.kind==='signal_emission'&&(ids.has(event.event_id)||event.tick===frame.tick)).slice(0,3);
  for(const event of events){
    if(activeVoices.size>=3)break;
    const band=Number(event.details?.signal?.tone),rawStrength=Number(event.details?.signal?.strength);if(!Number.isFinite(band)||!Number.isFinite(rawStrength))continue;
    const strength=Math.max(0,Math.min(1,rawStrength)),oscillator=audioContext.createOscillator(),gain=audioContext.createGain(),start=audioContext.currentTime,voice={oscillator,gain};oscillator.type='sine';oscillator.frequency.value=[220,330,495][Math.max(0,Math.min(2,Math.round(band)))];gain.gain.setValueAtTime(0,start);gain.gain.linearRampToValueAtTime(.06*strength,start+.012);gain.gain.exponentialRampToValueAtTime(.0001,start+.16);oscillator.connect(gain).connect(audioContext.destination);activeVoices.add(voice);oscillator.addEventListener('ended',()=>{oscillator.disconnect();gain.disconnect();activeVoices.delete(voice);},{once:true});oscillator.start(start);oscillator.stop(start+.17);
  }
}

function stopActiveVoices(){
  for(const voice of [...activeVoices]){try{voice.oscillator.stop();}catch{}voice.oscillator.disconnect();voice.gain.disconnect();activeVoices.delete(voice);}
}

function selectedPose(frame) {
  return bodyPose(frame);
}

function updateCamera(a,b,alpha) {
  if(cameraMode==='orbit') return;
  const left=selectedPose(a), right=selectedPose(b);
  if(!left||!right)return;
  pos.fromArray(left.position).lerp(new THREE.Vector3().fromArray(right.position),alpha);
  qFromWxyz(left.quaternion,qa);qFromWxyz(right.quaternion,qb);qa.slerp(qb,alpha);
  if(cameraMode==='follow'){
    direction.set(-1.7,-2.4,1.25).applyQuaternion(qa);
    look.copy(pos).add(new THREE.Vector3(0,0,.12));
    camera.position.lerp(pos.clone().add(direction),.09);controls.target.lerp(look,.11);
  }else{
    const ar=residentDetail(a)?.retina_pose,br=residentDetail(b)?.retina_pose;
    if(!ar||!br)return;
    pos.fromArray(ar.origin).lerp(posB.fromArray(br.origin),alpha);
    direction.fromArray(ar.forward).lerp(posB.fromArray(br.forward),alpha).normalize();
    const up=upA.fromArray(ar.up).lerp(upB.fromArray(br.up),alpha).normalize();
    camera.position.copy(pos);camera.up.copy(up);
    controls.target.copy(look.copy(pos).add(direction));
  }
}

function seek(value) {
  if(!recording)return;
  stopActiveVoices();
  cursor=Math.max(0,Math.min(recording.frames.length-1,Number(value)));
  lastAudibleFrame=Math.round(cursor);
  ui.scrubber.value=cursor;
  updateInstruments(Math.round(cursor));
}

function modelTimeAtCursor(value){
  const lower=Math.floor(value),upper=Math.min(recording.frames.length-1,lower+1),alpha=value-lower;
  return THREE.MathUtils.lerp(recording.frames[lower].model_time,recording.frames[upper].model_time,alpha);
}

function cursorAtModelTime(modelTime){
  const frames=recording.frames;if(modelTime<=frames[0].model_time)return 0;if(modelTime>=frames.at(-1).model_time)return frames.length-1;
  let low=0,high=frames.length-1;while(high-low>1){const middle=(low+high)>>1;if(frames[middle].model_time<=modelTime)low=middle;else high=middle;}
  const span=frames[high].model_time-frames[low].model_time;return low+(modelTime-frames[low].model_time)/span;
}

function animate(now) {
  const delta=Math.min(.1,(now-lastClock)/1000);lastClock=now;
  if(recording){
    if(playing){
      cursor=cursorAtModelTime(modelTimeAtCursor(cursor)+delta);
      if(cursor>=recording.frames.length-1){cursor=recording.frames.length-1;playing=false;stopActiveVoices();ui.play.textContent='▶';ui.play.ariaLabel='Play recording';}
      ui.scrubber.value=cursor;
    }
    const lower=Math.floor(cursor), upper=Math.min(recording.frames.length-1,lower+1), alpha=cursor-lower;
    updateGeometry(recording.frames[lower],recording.frames[upper],alpha);
    updateCamera(recording.frames[lower],recording.frames[upper],alpha);
    const display=Math.round(cursor);
    if(Number(ui.frame.dataset.current)!==display){ui.frame.dataset.current=display;updateInstruments(display);if(playing)sonifyFrame(display);}
  }
  controls.enabled=cameraMode==='orbit';controls.update();renderer.render(scene,camera);
}

export function loadRecording(data) {
  stopActiveVoices();recording=validate(expandEntityDeltas(data));cursor=0;playing=false;
  const ids=residentIds();activeBody=ids.includes(recording.frames[0].selected?.body)?recording.frames[0].selected.body:ids[0];
  ui.resident.replaceChildren(...ids.map((id,index)=>{const option=document.createElement('option');option.value=String(id);option.textContent=`Resident ${index+1} · body ${id}`;return option;}));ui.resident.value=String(activeBody);ui.resident.disabled=false;
  makePools(recording);makeActionBars();makeDecisionDisplay();populateMoments();populateEvents();
  ui.scrubber.max=String(recording.frames.length-1);ui.scrubber.value='0';
  ui.status.textContent=recording.status;
  const provenance=recording.provenance||{},revision=provenance.world_source_revision||provenance.source_revision||'unavailable',content=provenance.world_source_content_sha256||provenance.source_content_sha256;
  ui.source.textContent=`world ${revision}${content?` · ${content.slice(0,12)}…`:''}${provenance.capture_tool?.revision?` · recorder ${provenance.capture_tool.revision}`:''}`;
  const hasSignalEvents=(recording.events||[]).some(event=>event.kind==='signal_emission');ui.hearSignals.disabled=!hasSignalEvents;ui.hearSignals.title=hasSignalEvents?'Sonification of recorded physical signals; not animal sound or mental state':'No recorded signal event stream is available';
  const detailCapability=recording.capabilities?.resident_details,eventCapability=recording.capabilities?.events;
  if(detailCapability?.status==='unavailable')ui.residentCoverage.textContent=detailCapability.reason||'Multi-resident detailed streams are unavailable in this recording.';
  if(eventCapability?.status==='unavailable')ui.eventContract.textContent=eventCapability.reason||'unavailable in this recording';
  for(const name of ['energy','gut','fatigue']){const field=recording.recorded_body_state?.fields?.find(value=>value.name===name),label=document.querySelector(`.legend .${name}`);if(label&&field?.unit)label.textContent=`${name} · ${field.unit}`;}
  const bodyButton=document.querySelector('[data-camera="body"]');
  const hasRetinaPose=recording.frames.some(frame=>residentDetail(frame)?.retina_pose);
  bodyButton.disabled=!hasRetinaPose;
  bodyButton.title=hasRetinaPose?'Recorded retinal viewpoint':'This recording has no retinal pose';
  ui.loading.hidden=true;ui.play.textContent='▶';ui.play.ariaLabel='Play recording';
  updateInstruments(0);return recording;
}

export function consumeRecordingInstrumentsForTest(data){
  recording=validate(expandEntityDeltas(data));makeActionBars();makeDecisionDisplay();populateMoments();populateEvents();
  const bodies=residentIds(),frames=[0,Math.floor(recording.frames.length/2),recording.frames.length-1];
  let updates=0;for(const body of bodies){activeBody=body;for(const frame of frames){updateInstruments(frame);updates++;}}
  return {format:recording.format,residents:bodies.length,frames:recording.frames.length,instrument_updates:updates,events:(recording.events||[]).length};
}

ui.resident.addEventListener('change',()=>{activeBody=Number(ui.resident.value);rebuildResidentOverlays();updateInstruments(Math.round(cursor));const bodyButton=document.querySelector('[data-camera="body"]'),hasPose=recording.frames.some(frame=>residentDetail(frame)?.retina_pose);bodyButton.disabled=!hasPose;bodyButton.title=hasPose?'Recorded retinal viewpoint':'This resident has no recorded retinal pose';});

ui.play.addEventListener('click',()=>{
  if(!recording)return;if(cursor>=recording.frames.length-1)seek(0);
  playing=!playing;if(!playing)stopActiveVoices();ui.play.textContent=playing?'Ⅱ':'▶';ui.play.ariaLabel=playing?'Pause recording':'Play recording';lastClock=performance.now();
});
ui.scrubber.addEventListener('input',()=>{playing=false;ui.play.textContent='▶';seek(ui.scrubber.value);});
for(const button of document.querySelectorAll('[data-camera]'))button.addEventListener('click',()=>{
  cameraMode=button.dataset.camera;
  if(cameraMode!=='body')camera.up.set(0,0,1);
  if(cameraMode==='orbit'&&recording)frameHabitat();
  for(const other of document.querySelectorAll('[data-camera]'))other.setAttribute('aria-pressed',String(other===button));
});
$('#inspection-light').addEventListener('click',event=>{
  inspectionLights.visible=!inspectionLights.visible;
  event.currentTarget.setAttribute('aria-pressed',String(inspectionLights.visible));
});
ui.hearSignals.addEventListener('click',async()=>{if(ui.hearSignals.disabled)return;if(!audioContext)audioContext=new AudioContext();await audioContext.resume();hearingSignals=!hearingSignals;if(!hearingSignals)stopActiveVoices();lastAudibleFrame=Math.round(cursor);ui.hearSignals.setAttribute('aria-pressed',String(hearingSignals));ui.hearSignals.textContent=hearingSignals?'Recorded signals audible':'Hear recorded signals';});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopActiveVoices();if(audioContext?.state==='running')audioContext.suspend();}else if(hearingSignals&&audioContext?.state==='suspended')audioContext.resume();});
for(const button of document.querySelectorAll('[data-overlay]'))button.addEventListener('click',()=>{const key=button.dataset.overlay;overlayVisibility[key]=!overlayVisibility[key];button.setAttribute('aria-pressed',String(overlayVisibility[key]));if(key==='matter'){if(matterPoints)matterPoints.visible=overlayVisibility.matter;if(matterLines)matterLines.visible=overlayVisibility.matter;}});

if(!globalThis.__CHREATURES_LIVING_DOM_STUB__)initThree();
const recordingKey=globalThis.__CHREATURES_LIVING_DOM_STUB__?null:new URLSearchParams(location.search).get('recording');
const recordingAsset=recordingKey==='trained-organs'?'./assets/trained-organs-recording.json':recordingKey==='regional-wave'?'./assets/regional-wave-recording.json':recordingKey==='courtyard'?'./assets/living-reef-recording.json':'./assets/reciprocal-wave-recording.json.gz';
if(recordingKey==='regional-wave'){document.title='Regional world recording — Chreatures';ui.evidenceLink.textContent='Population atlas →';ui.evidenceLink.href='population.html'}
if(recordingKey==='trained-organs'){document.title='Trained organs in a regional world — Chreatures';ui.evidenceLink.textContent='Trained controller receipt →';ui.evidenceLink.href='https://github.com/emberian/chreatures/tree/main/data/training/population-v5-update20'}
if(!recordingKey){document.title='Reciprocal-wave recording — Chreatures';ui.evidenceLink.textContent='Typed recording evidence →';ui.evidenceLink.href='https://github.com/emberian/chreatures/blob/main/integrations/artifacts/reciprocal-v6-research-branch-v1/recording-link.json'}
async function fetchRecording(path){
  const response=await fetch(path);if(!response.ok)throw new Error(`HTTP ${response.status}`);
  if(!path.endsWith('.gz'))return response.json();
  if(typeof DecompressionStream==='undefined')throw new Error('This browser cannot open gzip recordings');
  const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}
if(!globalThis.__CHREATURES_LIVING_DOM_STUB__)fetchRecording(recordingAsset)
  .then(loadRecording)
  .catch(reason=>fail('The recorded reef is being prepared. This observatory will open when its public recording arrives.',reason));
