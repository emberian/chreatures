import * as THREE from '../vendor/three/three.module.min.js';
import {OrbitControls} from '../vendor/three/OrbitControls.js';
export {NeuronInspector} from './neuron-inspector.js';

const NEURONS = 165122;
const RETINAL_SITES = 1771;
const MUJOCO_SHAPES = new Set([0, 2, 3, 4, 5, 6, 7]);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function worldColor(rgba, target = new THREE.Color()) {
  // MuJoCo exposes model reflectance values. Three stores material colors in
  // linear space; the renderer performs the sole conversion to sRGB output.
  return target.setRGB(rgba[0], rgba[1], rgba[2]);
}

function worldAlpha(item, rgba) {
  return rgba[3] * (/(?:^|\/)boundary-(?:west|east|north|south)$/.test(item.name) ? .055 : 1);
}

function finiteArray(value, length, label) {
  if (!ArrayBuffer.isView(value) || value.length !== length) throw new Error(`${label} has the wrong extent`);
  for (let index = 0; index < value.length; index += 1) {
    if (!Number.isFinite(value[index])) throw new Error(`${label} contains a nonfinite value`);
  }
}

function materialFor(item, rgba) {
  const resident = Boolean(item.resident_id) || /^resident(?:\d+\/|:)/.test(item.name);
  const boundary = /(?:^|\/)boundary-(?:west|east|north|south)$/.test(item.name);
  const wing = /\/[lr]_wing$/.test(item.name);
  const wet = /(?:moist|water)/.test(item.name);
  const color = worldColor(rgba);
  const opacity = worldAlpha(item, rgba);
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: wing ? .34 : wet ? .26 : resident ? .57 : .88,
    metalness: 0,
    clearcoat: wet ? .32 : 0,
    clearcoatRoughness: .38,
    emissive: resident ? color.clone() : new THREE.Color(0x000000),
    emissiveIntensity: resident ? .018 : 0,
    transparent: boundary || opacity < .999,
    opacity,
    depthWrite: !boundary && opacity >= .999,
    side: item.type === 0 || wing ? THREE.DoubleSide : THREE.FrontSide,
  });
}

function geometryFor(item, meshes) {
  const {type, size} = item;
  if (type === 0) return new THREE.PlaneGeometry(2, 2);
  if (type === 7) {
    const source = meshes?.[item.mesh_id];
    if (!source || !(source.positions instanceof Float32Array) || !(source.faces instanceof Uint32Array) || source.positions.length % 3 || source.faces.length % 3) throw new Error(`Compiled anatomical mesh ${item.mesh_id} is absent`);
    finiteArray(source.positions, source.positions.length, 'Compiled anatomical vertices');
    if (source.faces.some(index => index >= source.positions.length / 3)) throw new Error('Anatomical triangle index is outside its mesh');
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(source.positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(source.faces, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }
  if (type === 2) return new THREE.SphereGeometry(1, 20, 13);
  if (type === 3) {
    const geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 6, 12);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  if (type === 4) return new THREE.SphereGeometry(1, 22, 14);
  if (type === 5) {
    const geometry = new THREE.CylinderGeometry(1, 1, 2, 18);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  if (type === 6) return new THREE.BoxGeometry(2, 2, 2);
  throw new Error(`Unsupported MuJoCo geometry type ${type}`);
}

function scaleMesh(mesh, item) {
  const size = item.size;
  if (!Array.isArray(size) || size.length !== 3 || size.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error(`Geometry ${item.id} has invalid size`);
  }
  if (item.type === 0) mesh.scale.set(size[0] || 25, size[1] || 20, 1);
  else if (item.type === 7) mesh.scale.setScalar(1);
  else if (item.type === 2) mesh.scale.setScalar(size[0]);
  else if (item.type === 3) mesh.scale.setScalar(1);
  else if (item.type === 4 || item.type === 6) mesh.scale.set(size[0], size[1], size[2]);
  else if (item.type === 5) mesh.scale.set(size[0], size[0], size[1]);
}

export class LiveView {
  constructor({worldCanvas, brainCanvas, retinaCanvases, onResident, onToy, onPlacement, onNeuron}) {
    this.worldCanvas = worldCanvas;
    this.brainCanvas = brainCanvas;
    if (!Array.isArray(retinaCanvases) || retinaCanvases.length !== 2) throw new Error('Retinal canvases are absent');
    this.retinaCanvases = retinaCanvases;
    this.onResident = onResident;
    this.onToy = onToy;
    this.onPlacement = onPlacement;
    this.onNeuron = onNeuron;
    this.placingToy = false;
    this.pointerDown = null;
    this.meshes = new Map();
    this.geometrySignature = '';
    this.screenGeom = -1;
    this.residents = [];
    this.selectedResident = null;
    this.cameraMode = 'orbit';
    this.lastFrame = null;
    this.habitatBounds = null;
    this.running = true;

    this.worldScene = new THREE.Scene();
    this.worldScene.background = new THREE.Color(0xaebdab);
    this.worldScene.fog = new THREE.Fog(0xaebdab, 48, 118);
    this.worldCamera = new THREE.PerspectiveCamera(39, 1, .035, 180);
    this.worldCamera.position.set(11, -16, 10);
    this.worldCamera.up.set(0, 0, 1);
    this.worldRenderer = new THREE.WebGLRenderer({canvas: worldCanvas, antialias: true, alpha: false});
    this.worldRenderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    this.worldRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.worldRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.worldRenderer.toneMappingExposure = .9;
    this.worldRenderer.shadowMap.enabled = true;
    this.worldRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.worldControls = new OrbitControls(this.worldCamera, worldCanvas);
    this.worldControls.target.set(0, 0, 1);
    this.worldControls.enableDamping = true;
    this.worldControls.dampingFactor = .075;
    this.worldControls.minDistance = 2.2;
    this.worldControls.maxDistance = 72;
    this.worldControls.maxPolarAngle = Math.PI * .47;
    const sky = new THREE.HemisphereLight(0xdde7d4, 0x584535, .92);
    sky.position.set(0, 0, 1);
    this.worldScene.add(sky);
    const sun = new THREE.DirectionalLight(0xffe1bd, 1.55);
    sun.position.set(-28, -18, 42);
    sun.target.position.set(0, 0, 1.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, {left: -31, right: 31, top: 27, bottom: -27, near: 1, far: 90});
    sun.shadow.camera.updateProjectionMatrix();
    sun.shadow.bias = -.00035;
    sun.shadow.normalBias = .025;
    this.worldScene.add(sun, sun.target);
    const fill = new THREE.DirectionalLight(0xa7c9bd, .4);
    fill.position.set(18, 13, 12);
    this.worldScene.add(fill);

    this.brainScene = new THREE.Scene();
    this.brainScene.background = new THREE.Color(0x091713);
    this.brainCamera = new THREE.PerspectiveCamera(34, 1, .01, 20);
    this.brainCamera.position.set(0, 0, 3.1);
    this.brainRenderer = new THREE.WebGLRenderer({canvas: brainCanvas, antialias: true, alpha: false});
    this.brainRenderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    this.brainRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.brainControls = new OrbitControls(this.brainCamera, brainCanvas);
    this.brainControls.enableDamping = true;
    this.brainControls.enablePan = false;
    this.brainControls.minDistance = 1.3;
    this.brainControls.maxDistance = 7;
    this.brainRows = null;
    this.brainColors = null;
    this.brainRates = null;
    this.brainBaseline = null;
    this.brainPoints = null;
    this.selectedNeuron = null;
    this.neuronMarker = new THREE.Mesh(new THREE.SphereGeometry(.018, 12, 8), new THREE.MeshBasicMaterial({color: 0xffffff, wireframe: true, depthTest: false}));
    this.neuronMarker.visible = false;
    this.neuronMarker.renderOrder = 2;
    this.brainScene.add(this.neuronMarker);
    this.neuralScale = .01;
    this.retinalEyes = null;
    this.retinalRGB = null;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    worldCanvas.addEventListener('pointerdown', event => {
      this.pointerDown = {x: event.clientX, y: event.clientY, id: event.pointerId};
    });
    worldCanvas.addEventListener('pointercancel', () => { this.pointerDown = null; });
    worldCanvas.addEventListener('pointerup', event => this.#pick(event));
    brainCanvas.addEventListener('pointerdown', event => {
      this.brainPointerDown = {x: event.clientX, y: event.clientY, id: event.pointerId};
    });
    brainCanvas.addEventListener('pointercancel', () => { this.brainPointerDown = null; });
    brainCanvas.addEventListener('pointerup', event => this.#pickNeuron(event));
    this.boundFrame = this.#loop.bind(this);
    requestAnimationFrame(this.boundFrame);
  }

  setCameraMode(mode) {
    if (!['orbit', 'follow', 'body'].includes(mode)) throw new Error('Unknown camera mode');
    this.cameraMode = mode;
    this.worldControls.enabled = mode !== 'body';
    this.worldCamera.fov = mode === 'body' ? 72 : 39;
    this.worldCamera.near = mode === 'body' ? .012 : .035;
    this.worldCamera.updateProjectionMatrix();
    if (mode !== 'body') this.worldCamera.up.set(0, 0, 1);
    if (mode === 'orbit') this.#frameHabitat();
    if (mode === 'follow') {
      const pose = this.#selectedPose();
      if (pose) {
        this.worldControls.target.copy(pose.position).add(new THREE.Vector3(0, 0, .28));
        this.worldCamera.position.copy(this.worldControls.target)
          .addScaledVector(this.#openingDirection(this.worldControls.target, 10), 10);
      }
    }
  }

  setToyPlacement(active) {
    this.placingToy = Boolean(active);
    this.worldCanvas.style.cursor = this.placingToy ? 'crosshair' : '';
  }

  setResidents(residents) {
    if (!Array.isArray(residents) || !residents.length) throw new Error('Resident observer metadata differs');
    const parsed = residents.map(item => ({id: String(item.id), root: Number(item.root), head: Number(item.head)}));
    if (parsed.some(item => !item.id || !Number.isInteger(item.root) || !Number.isInteger(item.head) || item.root < 0 || item.head < 0)) {
      throw new Error('Resident observer addresses differ');
    }
    this.residents = parsed;
  }

  selectResident(id) {
    if (!this.residents.some(item => item.id === id)) throw new Error('Selected resident is absent');
    const changed = this.selectedResident !== id;
    this.selectedResident = id;
    for (const mesh of this.meshes.values()) {
      if (!mesh.userData.residentId) continue;
      mesh.material.emissiveIntensity = mesh.userData.residentId === id ? .13 : .018;
    }
    if (changed && this.lastFrame && this.cameraMode === 'orbit') this.#frameHabitat();
  }

  setScreenCanvas(canvas) {
    const mesh = this.meshes.get(this.screenGeom);
    if (!mesh) return;
    if (!this.screenTexture) {
      this.screenTexture = new THREE.CanvasTexture(canvas);
      this.screenTexture.colorSpace = THREE.SRGBColorSpace;
      this.screenTexture.minFilter = THREE.LinearFilter;
      mesh.material.map = this.screenTexture;
      mesh.material.emissive = new THREE.Color(0xffffff);
      mesh.material.emissiveMap = this.screenTexture;
      mesh.material.emissiveIntensity = .72;
      mesh.material.needsUpdate = true;
    }
    this.screenTexture.needsUpdate = true;
  }

  initializeBrain(positions, valid, baseline) {
    if (!(positions instanceof Float32Array) || positions.length !== NEURONS * 3) throw new Error('MaleCNS soma-position extent differs');
    if (!(valid instanceof Uint8Array) || valid.length !== NEURONS) throw new Error('MaleCNS soma-valid mask differs');
    if (valid.some(value => value !== 0 && value !== 1)) throw new Error('MaleCNS soma-valid mask is not binary');
    finiteArray(baseline, NEURONS, 'MaleCNS rate baseline');
    if (baseline.some(value => value < 0)) throw new Error('MaleCNS rate baseline contains a negative value');
    const rows = [];
    const center = new THREE.Vector3();
    for (let row = 0; row < NEURONS; row += 1) {
      if (!valid[row]) continue;
      if (![positions[row * 3], positions[row * 3 + 1], positions[row * 3 + 2]].every(Number.isFinite)) throw new Error('A valid MaleCNS soma position is nonfinite');
      rows.push(row);
      center.x += positions[row * 3]; center.y += positions[row * 3 + 1]; center.z += positions[row * 3 + 2];
    }
    if (!rows.length) throw new Error('MaleCNS has no valid soma positions');
    center.multiplyScalar(1 / rows.length);
    let radius = 0;
    for (const row of rows) radius = Math.max(radius, Math.hypot(positions[row * 3] - center.x, positions[row * 3 + 1] - center.y, positions[row * 3 + 2] - center.z));
    if (!(radius > 0)) throw new Error('MaleCNS soma extent is degenerate');
    this.brainRows = Int32Array.from(rows);
    this.brainDisplayIndex = new Int32Array(NEURONS).fill(-1);
    rows.forEach((row, index) => { this.brainDisplayIndex[row] = index; });
    this.brainBaseline = Float32Array.from(baseline);
    const display = new Float32Array(rows.length * 3);
    this.brainColors = new Float32Array(rows.length * 3);
    rows.forEach((row, index) => {
      display[index * 3] = (positions[row * 3] - center.x) / radius;
      display[index * 3 + 1] = (positions[row * 3 + 1] - center.y) / radius;
      display[index * 3 + 2] = (positions[row * 3 + 2] - center.z) / radius;
      this.brainColors.set([.20, .29, .25], index * 3);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(display, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.brainColors, 3));
    const material = new THREE.PointsMaterial({size: .009, vertexColors: true, transparent: true, opacity: .9, sizeAttenuation: true});
    this.brainPoints = new THREE.Points(geometry, material);
    this.brainScene.add(this.brainPoints);
    return rows.length;
  }

  selectNeuron(row) {
    this.selectedNeuron = row;
    const index = this.brainDisplayIndex?.[row] ?? -1;
    this.neuronMarker.visible = index >= 0;
    if (index >= 0) this.neuronMarker.position.fromBufferAttribute(this.brainPoints.geometry.attributes.position, index);
    return index >= 0;
  }

  #pickNeuron(event) {
    const down = this.brainPointerDown;
    this.brainPointerDown = null;
    if (!this.brainPoints || !down || event.button !== 0 || down.id !== event.pointerId ||
        Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return;
    const bounds = this.brainCanvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const pointer = new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = .012;
    this.brainScene.updateMatrixWorld(true); this.brainCamera.updateMatrixWorld(true);
    ray.setFromCamera(pointer, this.brainCamera);
    const hit = ray.intersectObject(this.brainPoints, false)[0];
    if (hit) this.onNeuron?.(this.brainRows[hit.index]);
  }

  initializeRetina(sites, supported) {
    if (!(sites instanceof Int16Array) || sites.length !== RETINAL_SITES * 3) throw new Error('Retinal q/r atlas extent differs');
    if (!(supported instanceof Uint8Array) || supported.length !== RETINAL_SITES) throw new Error('Retinal support-mask extent differs');
    if (supported.some(value => value !== 0 && value !== 1)) throw new Error('Retinal support mask is not binary');
    const eyes = [[], []], seen = new Set();
    for (let row = 0; row < RETINAL_SITES; row += 1) {
      const side = sites[row * 3], q = sites[row * 3 + 1], r = sites[row * 3 + 2];
      if ((side !== 1 && side !== 2) || !Number.isInteger(q) || !Number.isInteger(r)) throw new Error('Retinal site address differs');
      const key = `${side}:${q}:${r}`;
      if (seen.has(key)) throw new Error('Retinal q/r atlas contains a duplicate site');
      seen.add(key);
      eyes[side - 1].push({row, q, r, supported: Boolean(supported[row])});
    }
    if (eyes.some(eye => !eye.length)) throw new Error('A retinal side has no sites');
    this.retinalEyes = eyes;
    this.retinalRGB = null;
    this.renderRetina();
    return {sites: RETINAL_SITES, supported: supported.reduce((sum, value) => sum + value, 0)};
  }

  clearRetina() {
    this.retinalRGB = null;
    this.renderRetina();
  }

  updateRetina(rgb) {
    if (!this.retinalEyes) throw new Error('Retinal atlas is not ready');
    if (!(rgb instanceof Float32Array)) throw new Error('Retinal RGB state type differs');
    finiteArray(rgb, RETINAL_SITES * 3, 'Retinal RGB state');
    if (rgb.some(value => value < 0 || value > 1)) throw new Error('Retinal RGB state is outside [0, 1]');
    this.retinalRGB = rgb;
    this.renderRetina();
  }

  renderRetina() {
    if (!this.retinalEyes) return;
    this.retinalEyes.forEach((eye, index) => this.#drawRetina(this.retinaCanvases[index], eye));
  }

  setNeuralScale(value) {
    if (!Number.isFinite(value) || value <= 0) throw new Error('Neural display scale must be positive');
    this.neuralScale = value;
    if (this.brainRates) this.updateNeural(this.brainRates);
  }

  setNeuralField(field, baseline) {
    this.neuralField = field;
    this.setNeuralBaseline(baseline);
  }

  setNeuralBaseline(baseline) {
    finiteArray(baseline, NEURONS, 'Selected MaleCNS rate baseline');
    this.brainBaseline = Float32Array.from(baseline);
    this.clearNeural();
  }

  clearNeural() {
    this.brainRates = null;
    if (!this.brainColors || !this.brainPoints) return;
    for (let index = 0; index < this.brainColors.length; index += 3) {
      this.brainColors[index] = .20;
      this.brainColors[index + 1] = .29;
      this.brainColors[index + 2] = .25;
    }
    this.brainPoints.geometry.attributes.color.needsUpdate = true;
  }

  updateNeural(rates) {
    if (!this.brainRows || !this.brainColors || !this.brainBaseline) throw new Error('MaleCNS atlas and selected baseline are not ready');
    finiteArray(rates, NEURONS, 'MaleCNS rate state');
    if ((!this.neuralField || this.neuralField === 'rate') && rates.some(value => value < 0)) throw new Error('MaleCNS rate state contains a negative value');
    this.brainRates = rates;
    const colors = this.brainColors;
    for (let index = 0; index < this.brainRows.length; index += 1) {
      const row = this.brainRows[index];
      const delta = clamp((rates[row] - this.brainBaseline[row]) / this.neuralScale, -1, 1);
      const amount = Math.abs(delta);
      const base = index * 3;
      if (delta >= 0) {
        colors[base] = .16 - .08 * amount;
        colors[base + 1] = .25 + .68 * amount;
        colors[base + 2] = .22 + .70 * amount;
      } else {
        colors[base] = .16 + .80 * amount;
        colors[base + 1] = .25 + .18 * amount;
        colors[base + 2] = .22 - .10 * amount;
      }
    }
    this.brainPoints.geometry.attributes.color.needsUpdate = true;
    let squared = 0, peak = 0;
    // Include every modeled row, including those without display coordinates.
    for (let row = 0; row < rates.length; row += 1) {
      const delta = rates[row] - this.brainBaseline[row];
      squared += delta * delta; peak = Math.max(peak, Math.abs(delta));
    }
    return {rms: Math.sqrt(squared / rates.length), peak};
  }

  applyFrame(frame) {
    if (frame.meshes) this.anatomicalMeshes = frame.meshes;
    const geometry = frame.geometry;
    if (!Array.isArray(geometry) || !Array.isArray(frame.residents)) throw new Error('Observer frame metadata differs');
    geometry.forEach((item, index) => {
      if (!item || item.id !== index || !MUJOCO_SHAPES.has(item.type) || typeof item.name !== 'string' ||
          !Array.isArray(item.size) || item.size.length !== 3 || item.size.some(value => !Number.isFinite(value) || value < 0)) {
        throw new Error(`Geometry descriptor ${index} differs`);
      }
    });
    finiteArray(frame.positions, geometry.length * 3, 'Geometry positions');
    finiteArray(frame.rotations, geometry.length * 9, 'Geometry rotations');
    finiteArray(frame.colors, geometry.length * 4, 'Geometry colors');
    if (!ArrayBuffer.isView(frame.bodyPositions) || frame.bodyPositions.length % 3) throw new Error('Body positions differ');
    finiteArray(frame.bodyPositions, frame.bodyPositions.length, 'Body positions');
    this.setResidents(frame.residents);
    if (this.residents.some(item => item.root * 3 + 2 >= frame.bodyPositions.length)) throw new Error('Resident root-body address differs');
    this.screenGeom = Number(frame.screenGeom);
    const signature = geometry.map(item => `${item.id}:${item.type}:${item.name}:${item.size.join(',')}`).join('|');
    if (signature !== this.geometrySignature) this.#rebuildGeometry(geometry, frame.colors, signature, this.anatomicalMeshes);
    if (this.residents.some(item => item.head * 3 + 2 >= frame.bodyPositions.length)) throw new Error('Resident head-body address differs');
    const matrix = new THREE.Matrix4();
    for (const item of geometry) {
      const mesh = this.meshes.get(item.id);
      if (!mesh) throw new Error(`Geometry ${item.id} is absent from view`);
      const p = item.id * 3, r = item.id * 9, c = item.id * 4;
      mesh.position.set(frame.positions[p], frame.positions[p + 1], frame.positions[p + 2]);
      matrix.set(frame.rotations[r], frame.rotations[r + 1], frame.rotations[r + 2], 0, frame.rotations[r + 3], frame.rotations[r + 4], frame.rotations[r + 5], 0, frame.rotations[r + 6], frame.rotations[r + 7], frame.rotations[r + 8], 0, 0, 0, 0, 1);
      mesh.quaternion.setFromRotationMatrix(matrix);
      scaleMesh(mesh, item);
      const rgba = [frame.colors[c], frame.colors[c + 1], frame.colors[c + 2], frame.colors[c + 3]];
      worldColor(rgba, mesh.material.color);
      if (mesh.userData.residentId) mesh.material.emissive.copy(mesh.material.color);
      mesh.material.opacity = worldAlpha(item, rgba);
      mesh.visible = rgba[3] > 0;
    }
    if (this.selectedResident && this.residents.some(item => item.id === this.selectedResident)) this.selectResident(this.selectedResident);
    else if (this.residents.length) this.selectResident(this.residents[0].id);
    this.lastFrame = frame;
    if (!this.habitatBounds) {
      this.worldScene.updateMatrixWorld(true);
      this.habitatBounds = new THREE.Box3();
      for (const mesh of this.meshes.values()) if (mesh.visible) this.habitatBounds.expandByObject(mesh);
      this.#frameHabitat();
    }
  }

  stop() { this.running = false; }

  #rebuildGeometry(items, colors, signature, anatomicalMeshes) {
    for (const mesh of this.meshes.values()) {
      this.worldScene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();
    }
    this.meshes.clear(); this.screenTexture?.dispose(); this.screenTexture = null;
    for (const item of items) {
      if (!Number.isInteger(item.id) || !MUJOCO_SHAPES.has(item.type) || typeof item.name !== 'string') throw new Error('Observer geometry descriptor differs');
      if (!Array.isArray(item.size) || item.size.length !== 3 || item.size.some(value => !Number.isFinite(value) || value < 0)) throw new Error(`Geometry ${item.id} has invalid size`);
      const c = item.id * 4;
      const rgba = [colors[c], colors[c + 1], colors[c + 2], colors[c + 3]];
      const mesh = new THREE.Mesh(geometryFor(item, anatomicalMeshes), materialFor(item, rgba));
      const resident = /^resident:([^:]+):/.exec(item.name) || /^(resident\d+)\//.exec(item.name);
      const visitor = /^entity:(visitor-[^:]+):/.exec(item.name);
      mesh.userData = {residentId: item.resident_id ?? resident?.[1] ?? null, toyId: visitor?.[1] ?? null, body: item.body,
        boundary: /(?:^|\/)boundary-(?:west|east|north|south)$/.test(item.name)};
      const translucent = mesh.material.opacity < .999;
      mesh.castShadow = !mesh.userData.boundary && !translucent && item.type !== 0;
      mesh.receiveShadow = !mesh.userData.boundary && !translucent;
      scaleMesh(mesh, item);
      this.meshes.set(item.id, mesh); this.worldScene.add(mesh);
    }
    this.geometrySignature = signature;
  }

  #selectedPose() {
    if (!this.lastFrame || !this.selectedResident) return null;
    const resident = this.residents.find(item => item.id === this.selectedResident);
    if (!resident) return null;
    const body = resident.root * 3;
    const position = new THREE.Vector3(this.lastFrame.bodyPositions[body], this.lastFrame.bodyPositions[body + 1], this.lastFrame.bodyPositions[body + 2]);
    const h = resident.head * 3;
    const head = {position: new THREE.Vector3(this.lastFrame.bodyPositions[h], this.lastFrame.bodyPositions[h + 1], this.lastFrame.bodyPositions[h + 2]), quaternion: new THREE.Quaternion()};
    const rotations = this.lastFrame.bodyRotations;
    if (rotations) {
      const k = resident.head * 9;
      const matrix = new THREE.Matrix4().set(rotations[k], rotations[k+1], rotations[k+2], 0, rotations[k+3], rotations[k+4], rotations[k+5], 0, rotations[k+6], rotations[k+7], rotations[k+8], 0, 0, 0, 0, 1);
      head.quaternion.setFromRotationMatrix(matrix);
    }
    return {position, head};
  }

  #frameHabitat() {
    if (!this.habitatBounds || this.habitatBounds.isEmpty()) return;
    const center = this.habitatBounds.getCenter(new THREE.Vector3());
    const radius = this.habitatBounds.getSize(new THREE.Vector3()).length() * .5;
    const pose = this.#selectedPose();
    const focus = pose ? pose.position.clone().add(new THREE.Vector3(0, 0, .32)) : center;
    const vertical = this.worldCamera.fov * Math.PI / 360;
    const horizontal = Math.atan(Math.tan(vertical) * Math.max(.5, this.worldCamera.aspect));
    const distance = pose
      ? clamp(68 / Math.max(.94, Math.sqrt(this.worldCamera.aspect)), 52, 68)
      : radius / Math.sin(Math.min(vertical, horizontal)) * 1.06;
    this.worldControls.target.copy(focus);
    this.worldCamera.position.copy(focus).addScaledVector(this.#openingDirection(focus, distance), distance);
    this.worldCamera.far = Math.max(180, distance + radius * 3);
    this.worldControls.maxDistance = Math.max(72, radius * 2.2);
    this.worldCamera.updateProjectionMatrix();
    this.worldCamera.lookAt(focus);
  }

  #openingDirection(focus, distance) {
    const center = this.habitatBounds?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3();
    const radial = focus.clone().sub(center); radial.z = 0;
    if (radial.lengthSq() < .01) radial.set(1, -1, 0);
    radial.normalize();
    const tangent = new THREE.Vector3(-radial.y, radial.x, 0);
    const candidates = [
      radial.clone().multiplyScalar(.17).add(new THREE.Vector3(0, 0, 1.3)).normalize(),
      tangent.clone().multiplyScalar(.4).addScaledVector(radial, .18).add(new THREE.Vector3(0, 0, 1.16)).normalize(),
      tangent.clone().multiplyScalar(-.4).addScaledVector(radial, .18).add(new THREE.Vector3(0, 0, 1.16)).normalize(),
      new THREE.Vector3(0, 0, 1),
    ];
    const obstacles = [...this.meshes.values()].filter(mesh => mesh.visible && !mesh.userData.residentId && !mesh.userData.boundary && mesh.geometry.type !== 'PlaneGeometry');
    const ray = new THREE.Raycaster();
    ray.near = .75; ray.far = Math.max(.75, distance - 1);
    let choice = candidates[0], clearance = -1;
    for (const direction of candidates) {
      ray.set(focus, direction);
      const hit = ray.intersectObjects(obstacles, false)[0];
      const available = hit?.distance ?? Infinity;
      if (available > clearance) { choice = direction; clearance = available; }
    }
    return choice;
  }

  #drawRetina(canvas, eye) {
    const width = Math.floor(canvas.clientWidth), height = Math.floor(canvas.clientHeight);
    if (width < 1 || height < 1) return;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
      canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio);
    }
    const context = canvas.getContext('2d', {alpha: false});
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = '#07100e'; context.fillRect(0, 0, width, height);
    const centers = eye.map(site => ({...site, x: site.q + site.r * .5, y: site.r * Math.sqrt(3) * .5}));
    const xs = centers.map(site => site.x), ys = centers.map(site => site.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys), padding = 7;
    const scale = Math.min((width - padding * 2) / Math.max(1, maxX - minX + 1), (height - padding * 2) / Math.max(1, maxY - minY + 1));
    const radius = Math.max(.65, scale * .54);
    for (const site of centers) {
      const x = padding + (site.x - minX + .5) * scale;
      const y = padding + (site.y - minY + .5) * scale;
      context.beginPath();
      for (let corner = 0; corner < 6; corner += 1) {
        const angle = Math.PI / 3 * corner;
        const px = x + Math.cos(angle) * radius, py = y + Math.sin(angle) * radius;
        if (!corner) context.moveTo(px, py); else context.lineTo(px, py);
      }
      context.closePath();
      if (site.supported) {
        const offset = site.row * 3;
        if (this.retinalRGB) {
          const red = Math.round(this.retinalRGB[offset] * 255), green = Math.round(this.retinalRGB[offset + 1] * 255), blue = Math.round(this.retinalRGB[offset + 2] * 255);
          context.fillStyle = `rgb(${red} ${green} ${blue})`;
        } else context.fillStyle = '#172a24';
        context.fill();
      } else {
        context.fillStyle = '#0a1512'; context.fill();
        context.strokeStyle = '#5d746a'; context.lineWidth = .55; context.stroke();
      }
    }
  }

  #updateCamera() {
    const pose = this.#selectedPose();
    if (!pose || this.cameraMode === 'orbit') return;
    if (this.cameraMode === 'follow') {
      const oldTarget = this.worldControls.target.clone();
      const desired = pose.position.clone().add(new THREE.Vector3(0, 0, .28));
      this.worldControls.target.lerp(desired, .16);
      this.worldCamera.position.add(this.worldControls.target.clone().sub(oldTarget));
      return;
    }
    if (!pose.head) return;
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.head.quaternion);
    const up = new THREE.Vector3(0, 0, 1).applyQuaternion(pose.head.quaternion);
    this.worldCamera.position.copy(pose.head.position).addScaledVector(forward, .35).addScaledVector(up, .18);
    this.worldCamera.up.copy(up);
    this.worldCamera.lookAt(pose.head.position.clone().addScaledVector(forward, 1));
  }

  #pick(event) {
    const down = this.pointerDown;
    this.pointerDown = null;
    if (!down || down.id !== event.pointerId || event.button !== 0 ||
        Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return;
    const bounds = this.worldCanvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    this.pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
    this.worldScene.updateMatrixWorld(true);
    this.raycaster.setFromCamera(this.pointer, this.worldCamera);
    const hit = this.raycaster.intersectObjects([...this.meshes.values()].filter(mesh => mesh.visible && !mesh.userData.boundary), false)[0];
    if (this.placingToy) {
      if (!hit?.face || hit.object.userData.residentId) return;
      const normal = hit.face.normal.clone().applyMatrix3(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)).normalize();
      if (normal.dot(this.raycaster.ray.direction) > 0) normal.negate();
      // Visitor-chosen world geometry is an external intervention. The physical
      // host still rejects penetration against the current world at commit.
      const position = hit.point.clone().addScaledVector(normal, .37);
      this.onPlacement?.(position.toArray());
      return;
    }
    if (hit?.object.userData.residentId) this.onResident(hit.object.userData.residentId);
    if (hit?.object.userData.toyId) this.onToy(hit.object.userData.toyId);
  }

  #resize(renderer, camera, canvas) {
    const width = Math.max(1, Math.floor(canvas.clientWidth));
    const height = Math.max(1, Math.floor(canvas.clientHeight));
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) renderer.setSize(width, height, false);
    camera.aspect = width / height; camera.updateProjectionMatrix();
  }

  #loop() {
    if (!this.running) return;
    this.#resize(this.worldRenderer, this.worldCamera, this.worldCanvas);
    this.#resize(this.brainRenderer, this.brainCamera, this.brainCanvas);
    this.#updateCamera();
    if (this.cameraMode !== 'body') this.worldControls.update();
    this.brainControls.update();
    this.worldRenderer.render(this.worldScene, this.worldCamera);
    this.brainRenderer.render(this.brainScene, this.brainCamera);
    requestAnimationFrame(this.boundFrame);
  }
}
