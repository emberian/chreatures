import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';

const FORMAT = 'chreatures-living-reef-public-recording-v1';
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
};

let renderer, scene, camera, controls, worldRoot, signalPoints, physicalSun, ambientFill;
let recording = null, cursor = 0, playing = false, lastClock = performance.now(), cameraMode = 'orbit';
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
  if (!data || data.format !== FORMAT) throw new Error(`Expected ${FORMAT}`);
  if (data.geometry_encoding !== GEOMETRY_ENCODING) throw new Error(`Expected ${GEOMETRY_ENCODING}`);
  if (!Array.isArray(data.frames) || data.frames.length < 2) throw new Error('Recording has no frame sequence');
  if (!data.geometry || !Array.isArray(data.geometry.bounds)) throw new Error('Recording has no geometry bounds');
  for (const [index, frame] of data.frames.entries()) {
    if (!Number.isFinite(frame.model_time) || !Number.isInteger(frame.tick)) throw new Error(`Frame ${index} time is invalid`);
    if (!Array.isArray(frame.bodies) || !Array.isArray(frame.entities) || !frame.selected) throw new Error(`Frame ${index} is incomplete`);
    finiteArray(frame.selected.neural_readouts?.shape, 1, `Frame ${index} readout shape`);
    if (frame.selected.neural_readouts.shape[0] !== 384) throw new Error(`Frame ${index} readouts are not 384 values`);
    const selected=frame.selected, refinement=selected.consequence_refinement, forecast=selected.sensory_forecast;
    if (!refinement || !forecast || !selected.sampled_proposal) throw new Error(`Frame ${index} has no recorded decision path`);
    for(const [label,value] of [['GAM scores',refinement.candidate_scores],['GAM coverage',refinement.candidate_out_of_domain],['forecast progress',forecast.candidate_progress],['forecast disagreement',forecast.candidate_disagreement],['forecast clipping',forecast.candidate_input_clipped],['forecast tilt',forecast.candidate_logit_tilt]]){
      if(!Array.isArray(value)||value.length!==4)throw new Error(`Frame ${index} ${label} must have four candidates`);
    }
    if(!Array.isArray(refinement.selected_private_correction)||refinement.selected_private_correction.length!==3)throw new Error(`Frame ${index} private correction is invalid`);
    if(!ACTOR_ACTIONS.every(name=>Number.isFinite(selected.sampled_proposal[name])))throw new Error(`Frame ${index} actor proposal is invalid`);
    if(!Number.isInteger(refinement.selected_candidate)||refinement.selected_candidate<0||refinement.selected_candidate>3)throw new Error(`Frame ${index} selected candidate is invalid`);
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
  renderer.toneMappingExposure = 1.08;
  renderer.setClearColor('#10251d');
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2('#10251d', .027);
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
  scene.add(ambientFill);
  physicalSun = new THREE.DirectionalLight('#fff0c7', 0);
  scene.add(physicalSun);
  const fill = new THREE.DirectionalLight('#7ca89c', 1.1);
  fill.position.set(8, 10, 5);
  scene.add(fill);
  worldRoot = new THREE.Group();
  scene.add(worldRoot);
  const resize = () => {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
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
    const material = new THREE.MeshStandardMaterial({roughness: .84, metalness: .02, vertexColors: true});
    const mesh = new THREE.InstancedMesh(geometries[kind](), material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    worldRoot.add(mesh);
    pools.set(kind, {mesh, count, keys:shapeKeys});
  }
  const bounds = data.geometry.bounds;
  const signalGeometry = new THREE.BufferGeometry();
  signalGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 256), 3));
  signalGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(3 * 256), 3));
  signalGeometry.setDrawRange(0, 0);
  signalPoints = new THREE.Points(signalGeometry, new THREE.PointsMaterial({size:.09, vertexColors:true, transparent:true, opacity:.8, depthWrite:false}));
  worldRoot.add(signalPoints);
  controls.target.set(bounds[0] / 2, bounds[1] / 2, Math.max(.4, bounds[2] * .15));
  camera.position.set(bounds[0] * .86, -bounds[1] * .5, Math.max(4, bounds[2] * 1.3));
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
  if(!left&&!right){physicalSun.intensity=0;return;}
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
  ui.actions.replaceChildren(...ACTIONS.map((name) => {
    const item=document.createElement('div'); item.className='action'; item.dataset.action=name;
    item.innerHTML=`<div class="action-label"><span>${name.replaceAll('_',' ')}</span><span>0.00</span></div><div class="bar"><i></i></div>`;
    return item;
  }));
}

function makeDecisionDisplay(){
  ui.proposal.replaceChildren(...ACTOR_ACTIONS.map(name=>{
    const item=document.createElement('div');item.className='proposal-value';item.dataset.action=name;
    item.innerHTML=`<span>${name.replaceAll('_',' ')}</span><span>0.000</span>`;return item;
  }));
  ui.candidates.replaceChildren(...Array.from({length:4},(_,index)=>{
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
    const progress=row.querySelector('[data-value="progress"]');progress.textContent=formatNative(forecast.candidate_progress[index]);progress.classList.toggle('clipped',forecast.candidate_input_clipped[index]);
    row.querySelector('[data-value="disagreement"]').textContent=formatNative(forecast.candidate_disagreement[index]);
    row.querySelector('[data-value="tilt"]').textContent=formatNative(forecast.candidate_logit_tilt[index]);
  }
  ui.correction.textContent=refinement.selected_private_correction.map(formatNative).join(' · ');
  ui.privateUpdates.textContent=String(refinement.completed_private_updates_before_action);
  ui.goalErrorScale.textContent=formatNative(forecast.empirical_goal_error_scale);
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
    const values=recording.frames.map(frame=>frame.selected.metabolism[name]);
    const low=Math.min(...values), high=Math.max(...values), span=Math.max(1e-7,high-low);
    context.beginPath(); context.strokeStyle=color; context.lineWidth=2;
    for(let i=0;i<=frameIndex;i++){
      const x=i/Math.max(1,recording.frames.length-1)*width, y=height-7-(values[i]-low)/span*(height-14);
      i?context.lineTo(x,y):context.moveTo(x,y);
    }
    context.stroke();
  }
  const x=frameIndex/Math.max(1,recording.frames.length-1)*width;
  context.strokeStyle='#f1eddf66'; context.lineWidth=1; context.beginPath(); context.moveTo(x,0); context.lineTo(x,height); context.stroke();
}

function updateInstruments(index) {
  const frame=recording.frames[index], selected=frame.selected;
  ui.time.textContent=`model time ${frame.model_time.toFixed(3)} s`;
  ui.frame.textContent=`${index+1} / ${recording.frames.length}`;
  ui.tick.textContent=`tick ${frame.tick}`;
  paintRetina(ui.peripheral,selected.retina.peripheral);
  paintRetina(ui.foveal,selected.retina.foveal);
  paintReadouts(selected.neural_readouts);
  ui.activity.textContent=selected.neural_summary.activity.toFixed(4);
  ui.support.textContent=selected.neural_summary.support.toFixed(4);
  ui.goalTime.textContent=selected.goal.valid?`${selected.goal.recorded_time.toFixed(3)} s`:'no valid record';
  ui.goalLeft.textContent=selected.goal.valid?`${selected.goal.commit_remaining_ticks} ticks`:'—';
  paintActions(selected.committed_action);
  paintDecision(selected);
  paintHistory(index);
  for(const button of ui.moments.querySelectorAll('button')) button.setAttribute('aria-current',String(Number(button.dataset.frame)===index));
}

function populateMoments() {
  ui.moments.replaceChildren(...(recording.phenomena_moments||[]).map((moment)=>{
    const item=document.createElement('li'), button=document.createElement('button');
    button.type='button'; button.dataset.frame=moment.frame;
    button.innerHTML=`<time>${moment.model_time.toFixed(2)} s</time><span>${moment.phenomena.map(value=>value.replaceAll('-',' ')).join(' · ')}</span>`;
    button.addEventListener('click',()=>seek(moment.frame)); item.append(button); return item;
  }));
  if(!ui.moments.children.length){const item=document.createElement('li');item.textContent='No phenomena were indexed in this recording.';ui.moments.append(item);}
}

function selectedPose(frame) {
  const body=frame.bodies[frame.selected.body];
  return body || frame.bodies[0];
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
    const ar=a.selected.retina_pose,br=b.selected.retina_pose;
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
  cursor=Math.max(0,Math.min(recording.frames.length-1,Number(value)));
  ui.scrubber.value=cursor;
  updateInstruments(Math.round(cursor));
}

function animate(now) {
  const delta=Math.min(.1,(now-lastClock)/1000);lastClock=now;
  if(recording){
    if(playing){
      const interval=recording.sampling.model_interval_seconds || .05;
      cursor+=delta/interval;
      if(cursor>=recording.frames.length-1){cursor=recording.frames.length-1;playing=false;ui.play.textContent='▶';ui.play.ariaLabel='Play recording';}
      ui.scrubber.value=cursor;
    }
    const lower=Math.floor(cursor), upper=Math.min(recording.frames.length-1,lower+1), alpha=cursor-lower;
    updateGeometry(recording.frames[lower],recording.frames[upper],alpha);
    updateCamera(recording.frames[lower],recording.frames[upper],alpha);
    const display=Math.round(cursor);
    if(Number(ui.frame.dataset.current)!==display){ui.frame.dataset.current=display;updateInstruments(display);}
  }
  controls.enabled=cameraMode==='orbit';controls.update();renderer.render(scene,camera);
}

export function loadRecording(data) {
  recording=validate(expandEntityDeltas(data));cursor=0;playing=false;
  makePools(recording);makeActionBars();makeDecisionDisplay();populateMoments();
  ui.scrubber.max=String(recording.frames.length-1);ui.scrubber.value='0';
  ui.status.textContent=recording.status;
  const bodyButton=document.querySelector('[data-camera="body"]');
  const hasRetinaPose=recording.frames.every(frame=>frame.selected.retina_pose);
  bodyButton.disabled=!hasRetinaPose;
  bodyButton.title=hasRetinaPose?'Recorded retinal viewpoint':'This recording has no retinal pose';
  ui.loading.hidden=true;ui.play.textContent='▶';ui.play.ariaLabel='Play recording';
  updateInstruments(0);return recording;
}

ui.play.addEventListener('click',()=>{
  if(!recording)return;if(cursor>=recording.frames.length-1)seek(0);
  playing=!playing;ui.play.textContent=playing?'Ⅱ':'▶';ui.play.ariaLabel=playing?'Pause recording':'Play recording';lastClock=performance.now();
});
ui.scrubber.addEventListener('input',()=>{playing=false;ui.play.textContent='▶';seek(ui.scrubber.value);});
for(const button of document.querySelectorAll('[data-camera]'))button.addEventListener('click',()=>{
  cameraMode=button.dataset.camera;
  if(cameraMode!=='body')camera.up.set(0,0,1);
  for(const other of document.querySelectorAll('[data-camera]'))other.setAttribute('aria-pressed',String(other===button));
});

initThree();
fetch('./assets/living-reef-recording.json')
  .then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();})
  .then(loadRecording)
  .catch(reason=>fail('The recorded reef is being prepared. This observatory will open when its public recording arrives.',reason));
