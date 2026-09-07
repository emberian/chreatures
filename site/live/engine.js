// One authority for a coupled life. Rust owns bodies/cognition; WebGPU owns CNS.
// This host only orders transfers and commits complete physical/neural ticks.
import { loadBlob, loadJSON, digest } from './assets.js';
import { createBrowserWorld } from './world-runtime.mjs';
import initResident, { ResidentRuntime } from './pkg/resident_runtime.js';
import initWorld, * as worldModule from './pkg/chreatures_browser_world.js';
import mujocoFactory from './vendor/mujoco/mujoco.js';
import { MaleCNSWebGPU } from './cns-webgpu.js';

const encoder = new TextEncoder(), decoder = new TextDecoder();
const MAGIC = encoder.encode('CHLIVE1\0');
export class LiveEngine {
  static async create({baseURL, device, progress = () => {}, modules = {}}) {
    const engine = new LiveEngine();
    engine.baseURL = new URL(baseURL);
    engine.progress = progress;
    engine.modelURL = new URL('model/', engine.baseURL);
    const manifests = await Promise.all(['cns', 'resident', 'observer'].map(name => loadJSON(new URL(`${name}-manifest.json`, engine.modelURL))));
    const [cns, resident, observer] = manifests;
    if (resident.cnsServiceArtifactSha256 !== cns.serviceArtifactSha256 || observer.graph !== cns.identity.graph) throw new Error('Coupled model identities differ');
    const runtime = await loadJSON(new URL('runtime-manifest.json', engine.baseURL));
    if (runtime.format !== 'chreatures-live-runtime-v1') throw new Error('Unknown live runtime identity');
    engine.identity = await digest(encoder.encode(JSON.stringify({manifests, runtimeFiles: runtime.files})));
    const runtimeBytes = async name => loadBlob(runtime.files[name], engine.baseURL);
    const [residentBinary, worldBinary, physicsBinary] = await Promise.all(['pkg/resident_runtime_bg.wasm', 'pkg/chreatures_browser_world_bg.wasm', 'vendor/mujoco/mujoco.wasm'].map(runtimeBytes));
    const fixture = await loadJSON(new URL('fixtures/garden.json', engine.baseURL));
    const xmlResponse = await fetch(new URL('fixtures/garden.xml', engine.baseURL));
    if (!xmlResponse.ok) throw new Error('Physical world download failed');
    engine.factoryOptions = {mujocoFactory: modules.mujocoFactory ?? mujocoFactory, coreModule: modules.worldModule ?? worldModule, coreWasm: modules.worldWasm ?? worldBinary};
    engine.physicsBinary = new Uint8Array(physicsBinary);
    engine.world = await createBrowserWorld({...engine.factoryOptions, fixture, xml: await xmlResponse.text(), seed: 20260907, mujocoOptions: {wasmBinary: engine.physicsBinary}});
    engine.batch = engine.world.residents;
    if (engine.batch !== resident.config.batch) throw new Error('Physical and cognitive cohort differ');
    await (modules.initResident ?? initResident)({module_or_path: residentBinary});
    const Resident = modules.ResidentRuntime ?? ResidentRuntime;
    let loaded = 0;
    const add = bytes => {loaded += bytes; progress({stage: 'download', label: 'Loading your local nervous systems', loaded});};
    const packed = await Promise.all(['core', 'predictor', 'sequence'].map(name => loadBlob(resident.buffers[name], engine.modelURL, add)));
    engine.resident = new Resident(JSON.stringify(resident.config), ...packed.map(bytes => new Float32Array(bytes)));
    engine.brain = await MaleCNSWebGPU.load({device, manifest: cns, baseURL: engine.modelURL, capacity: engine.batch, onProgress: ({bytes}) => add(bytes), shaderBaseURL: new URL("cns-webgpu.js", engine.baseURL)});
    await engine.brain.reset(Array.from({length: engine.batch}, (_, i) => i), engine.world.observe().residents.map(r => ({id: r.id, physicalEpoch: engine.world.engine, seed: 20260907})));
    const [soma, valid] = await Promise.all(['positions', 'valid'].map(name => loadBlob(observer.buffers[name], engine.modelURL, add)));
    engine.brainPositions = new Float32Array(soma); engine.brainValid = new Uint8Array(valid);
    engine.neuralBaseline = engine.brain.baselineRates;
    engine.modelStatus = cns.trainingStatus;
    engine.controllerStatus = resident.trainingStatus;
    engine.previous = new Float32Array(engine.batch * 12);
    engine.tick = 0; engine.selected = 0; engine.failed = false; engine.journal = []; engine.externalEvents = [];
    engine.loadedBytes = loaded;
    return engine;
  }
  describe() {
    return {engine: 'Full MaleCNS · WebGPU + Rust/Wasm + MuJoCo', neurons: 165122,
      validSoma: this.brainValid.reduce((sum, value) => sum + value, 0), residents: this.world.observe().residents,
      modelStatus: this.modelStatus, controllerStatus: this.controllerStatus,
      brainPositions: this.brainPositions, brainValid: this.brainValid, neuralBaseline: this.neuralBaseline,
      identity: this.identity};
  }
  async advance(capture = false) {
    if (this.failed) throw new Error('An incomplete tick paused this life; restore a coherent checkpoint');
    const started = performance.now();
    try {
      while (this.externalEvents.length && this.externalEvents[0].tick <= this.tick) {
        const event = this.externalEvents.shift();
        this.world.visitorSound(event.position, event.amplitude);
      }
      const {optic, body} = this.world.sample();
      const neural = await this.brain.step({dt: .05, activeMask: (1 << this.batch) - 1, opticRGB: optic, body,
        ...(capture ? {selectedResident: this.selected} : {})});
      const ticks = new BigUint64Array(this.batch).fill(BigInt(this.tick));
      const resets = new Uint8Array(this.batch).fill(this.tick === 0 ? 1 : 0);
      const decision = this.resident.stepFlat(neural.latent, this.previous, ticks, resets);
      const command = decision.proposedCommand;
      const diagnostics = decision.diagnosticsJson;
      decision.free();
      this.world.advance(command, .05);
      const receipt = this.resident.acknowledge(ticks, command);
      if (!receipt.every(Boolean)) throw new Error('Delivered motor receipt rejected');
      this.previous = command; this.tick++;
      return {...this.world.observe(), neuralRates: neural.selectedRates,
        selectedResidentId: this.world.observe().residents[this.selected].id,
        diagnostics: JSON.parse(diagnostics), tick: this.tick, wallMilliseconds: performance.now() - started};
    } catch (error) {this.failed = true; throw error;}
  }
  observe() { return {...this.world.observe(), tick: this.tick}; }
  record(kind, details) {
    this.journal.push({tick: this.tick, time: this.world.time, kind, details});
    // The journal is observer history, never a memory input to the creature.
    if (this.journal.length > 4096) this.journal.splice(0, 512);
  }
  screen(rgb, width, height, filmTime) {
    this.world.setScreenFrame(rgb, width, height);
    this.filmTime = filmTime ?? null;
  }
  greet(notes = [0, 1, 2]) {
    if (notes.length > 8 || notes.some(n => !Number.isInteger(n) || n < 0 || n > 2)) throw new Error('Choose up to eight notes in three bands');
    // A sound source exists in the world. No visitor ID goes to the controller.
    notes.forEach((note, index) => {
      const amplitude = [0, 0, 0]; amplitude[note] = .65;
      this.externalEvents.push({tick: this.tick + index * 5, position: [1.1, 1.8, .4], amplitude});
    });
    this.externalEvents.sort((a, b) => a.tick - b.tick);
    this.record('visitor-sound', {notes});
  }
  async insertToy() {
    const result = await this.world.insertObject({position: [1.2, 2.15, .6], size: [.07, .07, .07], shape: 'sphere', rgba: [.83, .37, .16, 1]});
    this.lastToy = result.id ?? result; this.record('insert-object', {id: this.lastToy}); return result;
  }
  shove(id, force = [1.2, .4, 0]) {
    this.world.queueVisitorForce(id ?? this.lastToy, force); this.record('visitor-force', {id: id ?? this.lastToy, force});
  }
  destroy() { this.world.dispose(); this.resident.free(); this.brain.destroy(); }
  async save() {
    if (this.failed) throw new Error('Cannot save a partially advanced life');
    const cns = new Uint8Array(await this.brain.snapshot()), resident = this.resident.saveBytes();
    const header = encoder.encode(JSON.stringify({format: 'chreatures-live-life-v1', identity: this.identity,
      tick: this.tick, selected: this.selected, previous: Array.from(this.previous), world: this.world.snapshot(),
      journal: this.journal, externalEvents: this.externalEvents, filmTime: this.filmTime ?? null, lastToy: this.lastToy ?? null,
      cnsBytes: cns.length, residentBytes: resident.length, cnsSHA: await digest(cns), residentSHA: await digest(resident)}));
    const result = new Uint8Array(12 + header.length + cns.length + resident.length);
    result.set(MAGIC); new DataView(result.buffer).setUint32(8, header.length, true);
    result.set(header, 12); result.set(cns, 12 + header.length); result.set(resident, 12 + header.length + cns.length);
    return result.buffer;
  }
  async load(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 12 || buffer.byteLength > 128 * 1024**2) throw new Error('Invalid life file');
    const bytes = new Uint8Array(buffer);
    if (!MAGIC.every((b, i) => b === bytes[i])) throw new Error('Different life format');
    const length = new DataView(buffer).getUint32(8, true);
    if (length > 8 * 1024**2 || length + 12 > bytes.length) throw new Error('Invalid life envelope');
    const h = JSON.parse(decoder.decode(bytes.subarray(12, 12 + length)));
    if (h.format !== 'chreatures-live-life-v1' || h.identity !== this.identity ||
        !Number.isSafeInteger(h.tick) || h.tick < 0 || !Number.isInteger(h.selected) || h.selected < 0 || h.selected >= this.batch ||
        h.previous.length !== this.batch * 12 || !h.previous.every(Number.isFinite) ||
        !Number.isSafeInteger(h.cnsBytes) || h.cnsBytes < 0 || !Number.isSafeInteger(h.residentBytes) || h.residentBytes < 0 ||
        12 + length + h.cnsBytes + h.residentBytes !== bytes.length || Math.abs(h.world.physical[0] - h.tick * .05) > 1e-7) throw new Error('Life identity, length or clock differs');
    if (!Array.isArray(h.externalEvents) || h.externalEvents.length > 4096 || h.externalEvents.some(e => !Number.isSafeInteger(e.tick) || e.tick < h.tick || e.position?.length !== 3 || e.amplitude?.length !== 3 || !e.position.every(Number.isFinite) || !e.amplitude.every(x => Number.isFinite(x) && x >= 0 && x <= 1))) throw new Error('Invalid pending physical signals');
    const cns = bytes.slice(12 + length, 12 + length + h.cnsBytes);
    const resident = bytes.slice(12 + length + h.cnsBytes);
    if (await digest(cns) !== h.cnsSHA || await digest(resident) !== h.residentSHA) throw new Error('Life state checksum differs');
    // Construct the possibly grown world separately before any current state changes.
    const world = await createBrowserWorld({...this.factoryOptions, fixture: h.world.fixture, xml: h.world.xml, mujocoOptions: {wasmBinary: this.physicsBinary}});
    try { world.restore(h.world); } catch (error) {world.dispose(); throw error;}
    const oldCns = await this.brain.snapshot(), oldResident = this.resident.saveBytes();
    try {await this.brain.restore(cns.buffer); this.resident.loadBytes(resident);}
    catch (error) {
      world.dispose();
      try {await this.brain.restore(oldCns); this.resident.loadBytes(oldResident);} catch {this.failed = true;}
      throw error;
    }
    this.world.dispose(); this.world = world; this.tick = h.tick; this.selected = h.selected;
    this.previous = Float32Array.from(h.previous); this.journal = h.journal; this.externalEvents = h.externalEvents; this.filmTime = h.filmTime; this.lastToy = h.lastToy; this.failed = false;
  }
}
