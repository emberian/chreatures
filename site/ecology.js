import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';

const $ = selector => document.querySelector(selector);
const ui = {
  state: $('#atlas-state'), content: $('#atlas-content'), empty: $('#atlas-empty'), canvas: $('#growth-canvas'),
  genotype: $('#genotype-select'), layout: $('#layout-select'), colony: $('#colony-select'), title: $('#selection-title'),
  worldTitle: $('#world-title'), split: $('#split-badge'), parameters: $('#parameters'), count: $('#committed-count'),
  material: $('#committed-material'), frame: $('#frame-summary'), time: $('#time-output'), slider: $('#time-slider'),
  marks: $('#sample-marks'), playback: $('#playback'), target: $('#target-select'), facts: $('#prediction-facts'),
  plot: $('#comparison-canvas'), plotStatus: $('#plot-status'), plotScale: $('#plot-scale'), identity: $('#record-identity'),
  worldCount: $('#world-count'), evidenceWorldCount: $('#evidence-world-count'), comparisonDescription: $('#comparison-description'),
};

const finite = value => Number.isFinite(value);
const finiteVector = (value, size) => Array.isArray(value) && value.length === size && value.every(finite);
const human = value => String(value).replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
const short = value => String(value || '—').slice(0, 12);
const formatNumber = value => {
  if (!finite(value)) return 'unavailable';
  const magnitude = Math.abs(value);
  if (magnitude && (magnitude < .001 || magnitude >= 10000)) return value.toExponential(3);
  return value.toLocaleString(undefined, {maximumFractionDigits: 5});
};
const isHoldout = split => String(split).toLowerCase().includes('holdout');
const splitKind = split => isHoldout(split) ? 'holdout' : String(split).toLowerCase() === 'fit' ? 'fit' : 'proposal';
const splitLabel = split => splitKind(split) === 'holdout' ? 'whole-genotype holdout' : splitKind(split) === 'fit' ? 'GAM fit world' : human(split);

function validateAtlas(data) {
  if (data?.format !== 'chreatures-native-growth-atlas-v1') throw new Error('Native growth atlas schema differs');
  if (typeof data.scope !== 'string' || !data.scope || !Array.isArray(data.genes) || data.genes.length !== 4)
    throw new Error('Atlas scope or four-gene contract is absent');
  if (!Array.isArray(data.layouts) || !data.layouts.length || !Array.isArray(data.worlds) || !data.worlds.length)
    throw new Error('The completed world atlas is absent');
  const declaredWorldCount = data.manifest?.world_count ?? data.provenance?.world_count ?? data.world_count;
  if (declaredWorldCount !== undefined && (!Number.isSafeInteger(declaredWorldCount) || declaredWorldCount !== data.worlds.length))
    throw new Error('Atlas world count differs from its manifest');
  const geneNames = data.genes.map(gene => gene.name);
  if (new Set(geneNames).size !== 4 || data.genes.some(gene => typeof gene.name !== 'string' || !finite(gene.min) || !finite(gene.max) || gene.min > gene.max))
    throw new Error('Gene bounds are invalid');
  const layouts = new Map();
  for (const layout of data.layouts) {
    if (typeof layout.layout_id !== 'string' || layouts.has(layout.layout_id) || !finiteVector(layout.bounds_mm?.[0], 3) || !finiteVector(layout.bounds_mm?.[1], 3) || !Array.isArray(layout.geometries))
      throw new Error('A physical layout is invalid');
    if (layout.bounds_mm[0].some((low, i) => low >= layout.bounds_mm[1][i])) throw new Error(`Layout ${layout.layout_id} has invalid bounds`);
    for (const geometry of layout.geometries) {
      if (typeof geometry.id !== 'string' || !['box', 'sphere', 'ellipsoid', 'capsule', 'cylinder'].includes(geometry.shape) || typeof geometry.dynamic !== 'boolean')
        throw new Error(`Layout ${layout.layout_id} has invalid geometry`);
      const sizeCount = geometry.shape === 'sphere' ? 1 : geometry.shape === 'box' || geometry.shape === 'ellipsoid' ? 3 : geometry.fromto_mm ? 1 : 2;
      if ((!geometry.fromto_mm && !finiteVector(geometry.position_mm, 3)) || !Array.isArray(geometry.size_mm) || geometry.size_mm.length !== sizeCount || !geometry.size_mm.every(value => finite(value) && value > 0))
        throw new Error(`Geometry ${geometry.id} has invalid dimensions`);
      if (geometry.quaternion_wxyz !== null && !finiteVector(geometry.quaternion_wxyz, 4)) throw new Error(`Geometry ${geometry.id} has invalid quaternion`);
      if (geometry.euler_rad !== null && !finiteVector(geometry.euler_rad, 3)) throw new Error(`Geometry ${geometry.id} has invalid Euler rotation`);
      if (geometry.fromto_mm !== null && !finiteVector(geometry.fromto_mm, 6)) throw new Error(`Geometry ${geometry.id} has invalid endpoints`);
    }
    layouts.set(layout.layout_id, layout);
  }
  const units = new Set(), genotypeContracts = new Map();
  for (const world of data.worlds) {
    if (typeof world.unit_id !== 'string' || units.has(world.unit_id) || typeof world.genotype_id !== 'string' || !layouts.has(world.layout_id))
      throw new Error('A world identity is invalid');
    units.add(world.unit_id);
    if (world.completed !== true || world.error !== null) throw new Error(`World ${world.unit_id} is not a completed committed run`);
    for (const gene of data.genes) {
      const value = world.genes?.[gene.name];
      if (!finite(value) || value < gene.min || value > gene.max) throw new Error(`World ${world.unit_id} has an invalid ${gene.name}`);
    }
    const genotypeContract = JSON.stringify(geneNames.map(name => world.genes[name]));
    if (genotypeContracts.has(world.genotype_id) && genotypeContracts.get(world.genotype_id) !== genotypeContract) throw new Error(`Genotype ${world.genotype_id} changes across layouts`);
    genotypeContracts.set(world.genotype_id, genotypeContract);
    if (!Array.isArray(world.samples) || !world.samples.length) throw new Error(`World ${world.unit_id} has no sparse samples`);
    let previousTick = -1, previousTime = -1;
    for (const [sampleIndex, sample] of world.samples.entries()) {
      if (!Number.isSafeInteger(sample.tick) || sample.tick <= previousTick || !finite(sample.time_s) || sample.time_s <= previousTime || !Number.isSafeInteger(sample.committed_constructions) || sample.committed_constructions < 0 || !finite(sample.construction_material) || !finite(sample.colony_resource) || !Array.isArray(sample.structures))
        throw new Error(`World ${world.unit_id} has an invalid sparse sample`);
      if (sampleIndex === 0 && sample.time_s === 0 ? sample.route_permeability !== null : !finite(sample.route_permeability))
        throw new Error(`World ${world.unit_id} has an invalid route observation`);
      const ids = new Set();
      for (const structure of sample.structures) {
        if (typeof structure.id !== 'string' || ids.has(structure.id) || typeof structure.owner_id !== 'string' || typeof structure.physics_binding !== 'string' || typeof structure.active !== 'boolean' || !finiteVector(structure.position_m, 3) || !finiteVector(structure.orientation_xyzw, 4) || !finite(structure.nominal_length_m) || structure.nominal_length_m <= 0 || !finite(structure.nominal_radius_m) || structure.nominal_radius_m <= 0)
          throw new Error(`World ${world.unit_id} has an invalid committed structure`);
        ids.add(structure.id);
      }
      previousTick = sample.tick; previousTime = sample.time_s;
    }
    const last = world.samples.at(-1);
    if (Math.abs(last.time_s - 32) > 1e-6 || world.metrics?.committed_constructions !== last.committed_constructions || Math.abs(world.metrics?.construction_material - last.construction_material) > 1e-9)
      throw new Error(`World ${world.unit_id} final 32-second metrics differ from its recording`);
    for (const prediction of Object.values(world.predictions || {})) if (prediction && (!finite(prediction.value) || typeof prediction.model !== 'string' || typeof prediction.scope !== 'string'))
      throw new Error(`World ${world.unit_id} has an invalid GAM prediction`);
    for (const baseline of Object.values(world.baselines || {})) if (!baseline || !finite(baseline.value) || typeof baseline.label !== 'string')
      throw new Error(`World ${world.unit_id} has an invalid measured baseline`);
  }
  return data;
}

// These are the exact linear RGBA values authored by the native habitat composer.
const habitatColors = {
  soil: [.18, .10, .045, 1], leaf: [.18, .48, .12, 1], stem: [.28, .22, .08, 1],
  bark: [.31, .16, .07, 1], moist: [.07, .25, .31, .72], grain: [.72, .56, .25, 1], pod: [.58, .24, .16, 1],
};

class GrowthWorldView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = .92;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color().setRGB(.055, .12, .09);
    this.scene.fog = new THREE.Fog(new THREE.Color().setRGB(.055, .12, .09), 75, 155);
    this.camera = new THREE.PerspectiveCamera(38, 1, .02, 350);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .07;
    this.controls.maxPolarAngle = Math.PI * .49;
    this.staticRoot = new THREE.Group(); this.branchRoot = new THREE.Group();
    this.scene.add(this.staticRoot, this.branchRoot);
    const sky = new THREE.HemisphereLight(0xe6eddc, 0x4b3625, 1.25); sky.position.set(0, 0, 1); this.scene.add(sky);
    const sun = new THREE.DirectionalLight(0xffddb0, 1.7); sun.position.set(-36, -25, 58); sun.castShadow = true;
    sun.shadow.mapSize.set(1536, 1536); Object.assign(sun.shadow.camera, {left: -45, right: 45, top: 36, bottom: -36, near: 1, far: 120});
    sun.shadow.camera.updateProjectionMatrix(); sun.shadow.bias = -.0003; sun.shadow.normalBias = .025; this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x9bc7b1, .45); fill.position.set(28, 18, 25); this.scene.add(fill);
    this.materials = new Map(Object.entries(habitatColors).map(([name, rgba]) => [name, new THREE.MeshStandardMaterial({color: new THREE.Color().setRGB(...rgba.slice(0, 3)), roughness: name === 'moist' ? .32 : .84, metalness: 0, transparent: rgba[3] < 1, opacity: rgba[3], depthWrite: rgba[3] === 1})]));
    this.branchMaterial = new THREE.MeshStandardMaterial({color: new THREE.Color().setRGB(...habitatColors.stem.slice(0, 3)), roughness: .7, metalness: 0});
    this.otherBranchMaterial = this.branchMaterial.clone(); this.otherBranchMaterial.transparent = true; this.otherBranchMaterial.opacity = .3; this.otherBranchMaterial.depthWrite = false;
    this.layout = null; this.world = null; this.colony = null; this.mode = 'construction';
    this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(canvas);
    canvas.addEventListener('dblclick', () => { viewMode = 'construction'; updateViewButtons(); this.frameCamera(); });
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }

  resize() {
    const width = Math.max(1, this.canvas.clientWidth), height = Math.max(1, this.canvas.clientHeight);
    this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
  }

  clearGroup(group) {
    for (const child of [...group.children]) { group.remove(child); child.geometry?.dispose(); }
  }

  geometryFor(item) {
    const size = item.size_mm, fromto = item.fromto_mm;
    const segmentLength = fromto ? Math.hypot(fromto[3] - fromto[0], fromto[4] - fromto[1], fromto[5] - fromto[2]) : null;
    if (item.shape === 'box') return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    if (item.shape === 'sphere') return new THREE.SphereGeometry(size[0], 18, 12);
    if (item.shape === 'ellipsoid') { const geometry = new THREE.SphereGeometry(1, 18, 12); geometry.scale(...size.slice(0, 3)); return geometry; }
    if (item.shape === 'cylinder') return new THREE.CylinderGeometry(size[0], size[0], segmentLength ?? size[1] * 2, 16).rotateX(Math.PI / 2);
    if (item.shape === 'capsule') return new THREE.CapsuleGeometry(size[0], segmentLength ?? size[1] * 2, 6, 12).rotateX(Math.PI / 2);
    throw new Error(`Unsupported habitat geometry ${item.shape}`);
  }

  applyTransform(mesh, item) {
    if (item.fromto_mm) {
      const a = new THREE.Vector3(...item.fromto_mm.slice(0, 3)), b = new THREE.Vector3(...item.fromto_mm.slice(3));
      mesh.position.copy(a).add(b).multiplyScalar(.5); mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), b.sub(a).normalize());
    } else {
      mesh.position.set(...item.position_mm);
      if (item.quaternion_wxyz) mesh.quaternion.set(item.quaternion_wxyz[1], item.quaternion_wxyz[2], item.quaternion_wxyz[3], item.quaternion_wxyz[0]).normalize();
      else if (item.euler_rad) mesh.quaternion.setFromEuler(new THREE.Euler(...item.euler_rad, 'XYZ'));
    }
  }

  setWorld(layout, world, colony) {
    const layoutChanged = this.layout?.layout_id !== layout.layout_id;
    this.layout = layout; this.world = world; this.colony = colony;
    if (layoutChanged) {
      this.clearGroup(this.staticRoot);
      for (const item of layout.geometries.filter(geometry => !geometry.dynamic)) {
        const material = this.materials.get(item.material);
        if (!material) throw new Error(`Layout material ${item.material} has no native display value`);
        const mesh = new THREE.Mesh(this.geometryFor(item), material); this.applyTransform(mesh, item);
        mesh.receiveShadow = true; mesh.castShadow = item.category !== 'ground' && !material.transparent; mesh.name = item.id; this.staticRoot.add(mesh);
      }
    }
    this.showFrame(Number(ui.slider.value)); this.frameCamera();
  }

  showFrame(index) {
    this.clearGroup(this.branchRoot);
    const sample = this.world.samples[index];
    for (const structure of sample.structures.filter(item => item.active)) {
      const radius = structure.nominal_radius_m * 1000, length = structure.nominal_length_m * 1000;
      const geometry = new THREE.CapsuleGeometry(radius, length, 6, 12).rotateX(Math.PI / 2);
      const material = structure.owner_id === this.colony ? this.branchMaterial : this.otherBranchMaterial;
      const mesh = new THREE.Mesh(geometry, material); mesh.position.set(...structure.position_m.map(value => value * 1000));
      mesh.quaternion.set(...structure.orientation_xyzw).normalize(); mesh.castShadow = structure.owner_id === this.colony; mesh.receiveShadow = true;
      mesh.name = structure.id; this.branchRoot.add(mesh);
    }
  }

  frameCamera() {
    if (!this.layout || !this.world) return;
    let bounds;
    if (this.mode === 'habitat') bounds = new THREE.Box3(new THREE.Vector3(...this.layout.bounds_mm[0]), new THREE.Vector3(...this.layout.bounds_mm[1]));
    else {
      bounds = new THREE.Box3();
      for (const structure of this.world.samples.at(-1).structures.filter(item => item.active && item.owner_id === this.colony)) {
        const extent = 1000 * (structure.nominal_length_m * .5 + structure.nominal_radius_m), position = new THREE.Vector3(...structure.position_m.map(value => value * 1000));
        bounds.expandByPoint(position.clone().addScalar(extent)); bounds.expandByPoint(position.clone().addScalar(-extent));
      }
      if (bounds.isEmpty()) bounds = new THREE.Box3(new THREE.Vector3(...this.layout.bounds_mm[0]), new THREE.Vector3(...this.layout.bounds_mm[1]));
      else bounds.expandByScalar(2.4);
    }
    const center = bounds.getCenter(new THREE.Vector3()), radius = Math.max(2.2, bounds.getBoundingSphere(new THREE.Sphere()).radius);
    const vertical = this.camera.fov * Math.PI / 360, horizontal = Math.atan(Math.tan(vertical) * Math.max(.5, this.camera.aspect));
    const distance = radius / Math.sin(Math.min(vertical, horizontal)) * 1.08;
    const direction = new THREE.Vector3(.82, -1.15, .86).normalize();
    this.controls.target.copy(center); this.camera.position.copy(center).addScaledVector(direction, distance);
    this.controls.minDistance = Math.max(.35, radius * .28); this.controls.maxDistance = Math.max(14, distance * 3); this.controls.update();
  }
}

let atlas, view, currentWorld, currentLayout, frameIndex = 0, timer = null, viewMode = 'construction';

function availableWorlds() { return atlas.worlds.filter(world => world.genotype_id === ui.genotype.value); }

function setOptions(select, entries, value, label) {
  select.replaceChildren(...entries.map(entry => new Option(label(entry), String(value(entry)))));
}

function stopPlayback() {
  if (timer) clearTimeout(timer); timer = null; ui.playback.setAttribute('aria-pressed', 'false'); ui.playback.innerHTML = '<span aria-hidden="true">▶</span> Play samples';
}

function updateFrame(index) {
  frameIndex = Math.max(0, Math.min(currentWorld.samples.length - 1, index)); ui.slider.value = String(frameIndex);
  const sample = currentWorld.samples[frameIndex]; view.showFrame(frameIndex);
  ui.time.textContent = `${sample.time_s.toFixed(sample.time_s % 1 ? 2 : 0)} model s · tick ${sample.tick.toLocaleString()}`;
  ui.frame.textContent = `sample ${frameIndex + 1} of ${currentWorld.samples.length} · ${sample.structures.filter(item => item.active).length} active committed capsules`;
  ui.count.textContent = sample.committed_constructions.toLocaleString(); ui.material.textContent = formatNumber(sample.construction_material);
}

function playNext() {
  if (!timer) return;
  if (frameIndex >= currentWorld.samples.length - 1) { stopPlayback(); return; }
  updateFrame(frameIndex + 1); timer = setTimeout(playNext, matchMedia('(prefers-reduced-motion: reduce)').matches ? 650 : 280);
}

function updateParameters() {
  ui.parameters.replaceChildren(...atlas.genes.map(gene => {
    const wrapper = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd'), range = document.createElement('small');
    dt.textContent = human(gene.name); dd.textContent = formatNumber(currentWorld.genes[gene.name]); range.textContent = `recorded range ${formatNumber(gene.min)}–${formatNumber(gene.max)}`;
    dd.append(range); wrapper.append(dt, dd); return wrapper;
  }));
}

function predictionTargets() {
  const keys = new Set();
  for (const world of atlas.worlds) for (const key of Object.keys(world.predictions || {})) keys.add(key);
  if (!keys.size && atlas.worlds.some(world => finite(world.metrics?.route_permeability_change))) keys.add('route_permeability_change');
  return [...keys].sort((a, b) => Number(b === 'route_permeability_change') - Number(a === 'route_permeability_change') || a.localeCompare(b));
}

function fact(label, value) {
  const wrapper = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = label; dd.textContent = value; wrapper.append(dt, dd); return wrapper;
}

function renderPrediction() {
  const target = ui.target.value, prediction = currentWorld.predictions?.[target], measured = currentWorld.metrics?.[target];
  const candidates = atlas.worlds.map(world => ({world, measured: world.metrics?.[target], predicted: world.predictions?.[target]?.value, scope: world.predictions?.[target]?.scope})).filter(row => finite(row.measured) && finite(row.predicted));
  const scopeCounts = new Map(); for (const row of candidates) scopeCounts.set(row.scope, (scopeCounts.get(row.scope) || 0) + 1);
  const plotScope = prediction?.scope ?? [...scopeCounts].sort((a, b) => b[1] - a[1])[0]?.[0];
  const rows = candidates.filter(row => row.scope === plotScope);
  const baseline = currentWorld.baselines?.[target];
  ui.facts.replaceChildren(
    fact(baseline?.label || 'measured baseline', baseline ? formatNumber(baseline.value) : 'unavailable'),
    fact('selected measured', formatNumber(measured)), fact('selected predicted', prediction ? formatNumber(prediction.value) : 'unavailable'),
    fact('absolute residual', prediction && finite(measured) ? formatNumber(Math.abs(prediction.value - measured)) : 'unavailable'),
    fact('model', prediction?.model || 'unavailable'), fact('prediction scope', prediction?.scope || 'unavailable'),
    fact('available worlds', `${rows.length} / ${atlas.worlds.length}`),
  );
  const canvas = ui.plot, context = canvas.getContext('2d'), width = canvas.width, height = canvas.height, margin = {left: 94, right: 34, top: 35, bottom: 67};
  context.fillStyle = '#142b22'; context.fillRect(0, 0, width, height);
  if (!rows.length) { ui.plotStatus.hidden = false; ui.plotStatus.textContent = `GAM prediction unavailable for ${human(target)}`; ui.plotScale.textContent = 'No prediction scale'; return; }
  ui.plotStatus.hidden = true;
  const values = rows.flatMap(row => [row.measured, row.predicted]); let low = Math.min(0, ...values), high = Math.max(0, ...values);
  if (high - low < 1e-12) { low -= .5; high += .5; }
  const pad = (high - low) * .07; low -= pad; high += pad; const span = high - low;
  const x = value => margin.left + (value - low) / span * (width - margin.left - margin.right);
  const y = value => height - margin.bottom - (value - low) / span * (height - margin.top - margin.bottom);
  context.lineWidth = 1; context.font = '12px ui-monospace, monospace'; context.fillStyle = '#aebcae';
  for (let i = 0; i <= 4; i++) {
    const value = low + span * i / 4, px = x(value), py = y(value); context.strokeStyle = '#ffffff1b';
    context.beginPath(); context.moveTo(margin.left, py); context.lineTo(width - margin.right, py); context.stroke();
    context.beginPath(); context.moveTo(px, margin.top); context.lineTo(px, height - margin.bottom); context.stroke();
    const label = formatNumber(value); context.fillText(label, margin.left - context.measureText(label).width - 10, py + 4); context.fillText(label, px - context.measureText(label).width / 2, height - margin.bottom + 23);
  }
  context.strokeStyle = '#dce5d644'; context.setLineDash([5, 5]); context.beginPath(); context.moveTo(x(low), y(low)); context.lineTo(x(high), y(high)); context.stroke(); context.setLineDash([]);
  for (const row of rows) {
    const selected = row.world.unit_id === currentWorld.unit_id; context.beginPath(); context.arc(x(row.measured), y(row.predicted), selected ? 8 : 5, 0, Math.PI * 2);
    const kind = splitKind(row.world.split); context.fillStyle = kind === 'holdout' ? '#ef9a67' : kind === 'fit' ? '#78aba0' : '#d8be71'; context.fill();
    if (selected) { context.lineWidth = 3; context.strokeStyle = '#f2efe5'; context.stroke(); }
  }
  context.fillStyle = '#ced8cc'; context.font = '13px system-ui'; context.fillText(`measured ${human(target)}`, width / 2 - 65, height - 16);
  context.save(); context.translate(23, height / 2 + 65); context.rotate(-Math.PI / 2); context.fillText(`GAM predicted ${human(target)}`, 0, 0); context.restore();
  ui.plotScale.textContent = `${formatNumber(low)} to ${formatNumber(high)} · shared axis · ${plotScope}`; 
}

function updateWorld() {
  stopPlayback(); currentWorld = atlas.worlds.find(world => world.genotype_id === ui.genotype.value && world.layout_id === ui.layout.value);
  currentLayout = atlas.layouts.find(layout => layout.layout_id === currentWorld.layout_id);
  const owners = [...new Set(currentWorld.samples.at(-1).structures.filter(item => item.active).map(item => item.owner_id))].sort();
  setOptions(ui.colony, owners, value => value, value => value); ui.colony.disabled = !owners.length; ui.colony.value = owners[0] || '';
  ui.title.textContent = `${currentWorld.genotype_id} · ${currentLayout.description}`; ui.worldTitle.textContent = currentWorld.unit_id;
  ui.split.textContent = splitLabel(currentWorld.split); ui.split.className = `split-badge ${splitKind(currentWorld.split)}`;
  ui.slider.max = String(currentWorld.samples.length - 1); ui.slider.value = String(currentWorld.samples.length - 1);
  ui.marks.replaceChildren(...currentWorld.samples.map(() => document.createElement('i'))); updateParameters();
  view.setWorld(currentLayout, currentWorld, ui.colony.value); updateFrame(currentWorld.samples.length - 1); renderPrediction();
}

function updateLayouts() {
  const worlds = availableWorlds(), ids = new Set(worlds.map(world => world.layout_id));
  const layouts = atlas.layouts.filter(layout => ids.has(layout.layout_id)); setOptions(ui.layout, layouts, item => item.layout_id, item => item.description);
  updateWorld();
}

function updateViewButtons() {
  view.mode = viewMode;
  for (const button of document.querySelectorAll('[data-view]')) button.setAttribute('aria-pressed', String(button.dataset.view === viewMode));
}

function initialize(data) {
  atlas = validateAtlas(data); view = new GrowthWorldView(ui.canvas);
  const genotypes = [...new Set(atlas.worlds.map(world => world.genotype_id))].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
  setOptions(ui.genotype, genotypes, value => value, value => value); ui.target.replaceChildren(...predictionTargets().map(target => new Option(human(target), target)));
  ui.genotype.addEventListener('change', updateLayouts); ui.layout.addEventListener('change', updateWorld);
  ui.colony.addEventListener('change', () => { view.colony = ui.colony.value; view.showFrame(frameIndex); if (viewMode === 'construction') view.frameCamera(); });
  ui.slider.addEventListener('input', () => { stopPlayback(); updateFrame(Number(ui.slider.value)); });
  ui.target.addEventListener('change', renderPrediction);
  ui.playback.addEventListener('click', () => {
    if (timer) { stopPlayback(); return; }
    if (frameIndex >= currentWorld.samples.length - 1) updateFrame(0);
    ui.playback.setAttribute('aria-pressed', 'true'); ui.playback.innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause playback'; timer = setTimeout(playNext, 280);
  });
  for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => { viewMode = button.dataset.view; updateViewButtons(); view.frameCamera(); });
  ui.target.disabled = !ui.target.options.length; updateLayouts(); updateViewButtons();
  if (typeof atlas.gam?.interpretation === 'string' && atlas.gam.interpretation) ui.comparisonDescription.textContent = atlas.gam.interpretation;
  ui.worldCount.textContent = atlas.worlds.length.toLocaleString(); ui.evidenceWorldCount.textContent = atlas.worlds.length.toLocaleString();
  ui.identity.textContent = `scope: ${atlas.scope} · plan SHA-256 ${atlas.provenance.plan_sha256} · native Weave SHA-256 ${atlas.provenance.weave_sha256}`;
  ui.empty.hidden = true; ui.content.hidden = false; ui.state.classList.add('loaded'); ui.state.querySelector('span').textContent = `${atlas.worlds.length} completed native worlds · sparse recorded samples`;
}

try {
  const response = await fetch('./assets/native-growth-atlas.json', {cache: 'no-cache'});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  initialize(await response.json());
} catch (error) {
  ui.state.querySelector('span').textContent = 'Recorded native-growth atlas unavailable';
  ui.empty.querySelector('p').textContent = `The atlas stays empty rather than reconstructing unrecorded ecology. ${error.message}`;
  console.error(error);
}
