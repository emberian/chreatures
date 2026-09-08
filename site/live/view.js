import * as THREE from '../vendor/three/three.module.min.js';
import {OrbitControls} from '../vendor/three/OrbitControls.js';

const NEURONS = 165122;
const RETINAL_SITES = 1771;
const MUJOCO_SHAPES = new Set([2, 3, 4, 5, 6]);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function finiteArray(value, length, label) {
  if (!ArrayBuffer.isView(value) || value.length !== length) throw new Error(`${label} has the wrong extent`);
  for (let index = 0; index < value.length; index += 1) {
    if (!Number.isFinite(value[index])) throw new Error(`${label} contains a nonfinite value`);
  }
}

function materialFor(item, rgba) {
  const resident = item.name.startsWith('resident:');
  const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: resident ? .28 : .72,
    metalness: resident ? .58 : .04,
    emissive: resident ? color.clone().multiplyScalar(.12) : new THREE.Color(0x000000),
    emissiveIntensity: resident ? .55 : 0,
    transparent: rgba[3] < .999,
    opacity: rgba[3],
    depthWrite: rgba[3] >= .999,
  });
}

function geometryFor(item) {
  const {type, size} = item;
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
  if (item.type === 2) mesh.scale.setScalar(size[0]);
  else if (item.type === 3) mesh.scale.setScalar(1);
  else if (item.type === 4 || item.type === 6) mesh.scale.set(size[0], size[1], size[2]);
  else if (item.type === 5) mesh.scale.set(size[0], size[0], size[1]);
}

export class LiveView {
  constructor({worldCanvas, brainCanvas, retinaCanvases, onResident, onToy}) {
    this.worldCanvas = worldCanvas;
    this.brainCanvas = brainCanvas;
    if (!Array.isArray(retinaCanvases) || retinaCanvases.length !== 2) throw new Error('Retinal canvases are absent');
    this.retinaCanvases = retinaCanvases;
    this.onResident = onResident;
    this.onToy = onToy;
    this.meshes = new Map();
    this.geometrySignature = '';
    this.screenGeom = -1;
    this.residents = [];
    this.selectedResident = null;
    this.cameraMode = 'orbit';
    this.lastFrame = null;
    this.running = true;

    this.worldScene = new THREE.Scene();
    this.worldScene.background = new THREE.Color(0xf3eee0);
    this.worldScene.fog = new THREE.FogExp2(0xe8f0df, .018);
    this.worldCamera = new THREE.PerspectiveCamera(42, 1, .015, 120);
    this.worldCamera.position.set(7.6, -8.8, 5.6);
    this.worldCamera.up.set(0, 0, 1);
    this.worldRenderer = new THREE.WebGLRenderer({canvas: worldCanvas, antialias: true, alpha: false});
    this.worldRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.worldRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.worldRenderer.toneMappingExposure = 1.12;
    this.worldControls = new OrbitControls(this.worldCamera, worldCanvas);
    this.worldControls.target.set(2.5, 2.5, .45);
    this.worldControls.enableDamping = true;
    this.worldControls.minDistance = .18;
    this.worldControls.maxDistance = 42;
    this.worldControls.maxPolarAngle = Math.PI * .495;
    const sky = new THREE.HemisphereLight(0xfff6dc, 0x6e9f86, 2.15);
    sky.position.set(0, 0, 1);
    this.worldScene.add(sky);
    const sun = new THREE.DirectionalLight(0xffd5ae, 3.1);
    sun.position.set(-6, -4, 10);
    this.worldScene.add(sun);
    const mint = new THREE.DirectionalLight(0xa8f2cf, 1.35);
    mint.position.set(8, 4, 4);
    this.worldScene.add(mint);

    this.brainScene = new THREE.Scene();
    this.brainScene.background = new THREE.Color(0x091713);
    this.brainCamera = new THREE.PerspectiveCamera(34, 1, .01, 20);
    this.brainCamera.position.set(0, 0, 3.1);
    this.brainRenderer = new THREE.WebGLRenderer({canvas: brainCanvas, antialias: true, alpha: false});
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
    this.neuralScale = .01;
    this.retinalEyes = null;
    this.retinalRGB = null;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    worldCanvas.addEventListener('pointerup', event => this.#pick(event));
    this.boundFrame = this.#loop.bind(this);
    requestAnimationFrame(this.boundFrame);
  }

  setCameraMode(mode) {
    if (!['orbit', 'follow', 'body'].includes(mode)) throw new Error('Unknown camera mode');
    this.cameraMode = mode;
    this.worldControls.enabled = mode !== 'body';
    if (mode !== 'body') this.worldCamera.up.set(0, 0, 1);
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
    this.selectedResident = id;
    for (const mesh of this.meshes.values()) {
      if (!mesh.userData.residentId) continue;
      mesh.material.emissiveIntensity = mesh.userData.residentId === id ? 1.35 : .55;
    }
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
    if (signature !== this.geometrySignature) this.#rebuildGeometry(geometry, frame.colors, signature);
    if (this.residents.some(item => !this.meshes.has(item.head))) throw new Error('Resident head-geometry address differs');
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
      mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      mesh.material.opacity = rgba[3];
    }
    if (this.selectedResident && this.residents.some(item => item.id === this.selectedResident)) this.selectResident(this.selectedResident);
    else if (this.residents.length) this.selectResident(this.residents[0].id);
    this.lastFrame = frame;
  }

  stop() { this.running = false; }

  #rebuildGeometry(items, colors, signature) {
    for (const mesh of this.meshes.values()) {
      this.worldScene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();
    }
    this.meshes.clear(); this.screenTexture?.dispose(); this.screenTexture = null;
    for (const item of items) {
      if (!Number.isInteger(item.id) || !MUJOCO_SHAPES.has(item.type) || typeof item.name !== 'string') throw new Error('Observer geometry descriptor differs');
      if (!Array.isArray(item.size) || item.size.length !== 3 || item.size.some(value => !Number.isFinite(value) || value < 0)) throw new Error(`Geometry ${item.id} has invalid size`);
      const c = item.id * 4;
      const rgba = [colors[c], colors[c + 1], colors[c + 2], colors[c + 3]];
      const mesh = new THREE.Mesh(geometryFor(item), materialFor(item, rgba));
      const resident = /^resident:([^:]+):/.exec(item.name);
      const visitor = /^entity:(visitor-[^:]+):/.exec(item.name);
      mesh.userData = {residentId: resident?.[1] ?? null, toyId: visitor?.[1] ?? null, body: item.body};
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
    const head = this.meshes.get(resident.head);
    return {position, head};
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
      this.worldControls.target.lerp(pose.position, .12);
      return;
    }
    if (!pose.head) return;
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.head.quaternion);
    const up = new THREE.Vector3(0, 0, 1).applyQuaternion(pose.head.quaternion);
    this.worldCamera.position.copy(pose.head.position).addScaledVector(forward, .045).addScaledVector(up, .018);
    this.worldCamera.up.copy(up);
    this.worldCamera.lookAt(pose.head.position.clone().addScaledVector(forward, 1));
  }

  #pick(event) {
    const bounds = this.worldCanvas.getBoundingClientRect();
    this.pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.worldCamera);
    const hit = this.raycaster.intersectObjects([...this.meshes.values()], false)[0]?.object;
    if (hit?.userData.residentId) this.onResident(hit.userData.residentId);
    if (hit?.userData.toyId) this.onToy(hit.userData.toyId);
  }

  #resize(renderer, camera, canvas) {
    const width = Math.max(1, Math.floor(canvas.clientWidth));
    const height = Math.max(1, Math.floor(canvas.clientHeight));
    const ratio = Math.min(devicePixelRatio || 1, 2);
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
