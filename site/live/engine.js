// One authority for a coupled life. Rust owns bodies/cognition; WebGPU owns CNS.
// This host only orders transfers and commits complete physical/neural ticks.
import { loadBlob, loadJSON, digest } from './assets.js';
import { createPhysicalWorld } from './physical-world.js';
import initResident, { ResidentRuntime } from './pkg/resident_runtime.js';
import flyWorldFactory from './pkg/chreatures-fly-world.mjs';
import { MaleCNSWebGPU } from './cns-webgpu.js';

const encoder = new TextEncoder(), decoder = new TextDecoder();
const MAGIC = encoder.encode('CHLIVE5\0');
const FIELDS = ['rate', 'adaptation', 'support', 'release', 'dopamine', 'octopamine', 'serotonin'];
const sameTime = (a, b) => Number.isFinite(a) && Math.abs(a - b) <= 1e-7 + Math.abs(b) * 1e-9;
export class LiveEngine {
  static async create({baseURL, device, progress = () => {}, modules = {}}) {
    const engine = new LiveEngine();
    try {
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
      const [residentBinary, worldBinary] = await Promise.all(['pkg/resident_runtime_bg.wasm', 'pkg/chreatures-fly-world.wasm'].map(runtimeBytes));
      const [fixtureBytes, xmlBytes] = await Promise.all(['fixtures/fly-ecology/world.json', 'fixtures/fly-ecology/scene.xml'].map(runtimeBytes));
      const fixture = JSON.parse(decoder.decode(fixtureBytes));
      const motorSchemaBytes = await runtimeBytes('fixtures/fly-ecology/motor92.json');
      if (await digest(motorSchemaBytes) !== fixture.actuator_schema_sha256) throw new Error('Motor display schema differs from the physical/CNS interface');
      engine.motorSchema = JSON.parse(decoder.decode(motorSchemaBytes));
      // CNS morphology names the semantic body schema. The separately hashed
      // mesh asset set is bound by the runtime fixture and its file manifest.
      for (const [key, identityKey] of [['body_schema_sha256','morphology'],['sensory_schema_sha256','sensorySchema'],['actuator_schema_sha256','actuatorSchema']]) {
        if (fixture[key] !== cns.identity[identityKey]) throw new Error(`Physical/CNS ${key} differs`);
      }
      const assets = Object.fromEntries(await Promise.all(fixture.mesh_assets.map(async item => [item.path, new Uint8Array(await runtimeBytes(`fixtures/fly-ecology/${item.path}`))])));
      engine.retinalSites = Int16Array.from(fixture.anatomical_sites.flat());
      engine.retinalSupported = Uint8Array.from(fixture.supported_sites, Number);
      engine.factoryOptions = {assets, moduleFactory: modules.flyWorldFactory ?? flyWorldFactory,
        wasmBinary: new Uint8Array(worldBinary), fixtureBytes: new Uint8Array(fixtureBytes),
        xmlBytes: new Uint8Array(xmlBytes), seed: 20260908};
      engine.world = await createPhysicalWorld(engine.factoryOptions);
      engine.batch = engine.world.residents;
      if (engine.batch !== resident.config.batch) throw new Error('Physical and cognitive cohort differ');
      await (modules.initResident ?? initResident)({module_or_path: residentBinary});
      const Resident = modules.ResidentRuntime ?? ResidentRuntime;
      let loaded = 0;
      const add = bytes => {loaded += bytes; progress({stage: 'download', label: 'Loading your local nervous systems', loaded});};
      const packed = await Promise.all(['core', 'predictor', 'sequence'].map(name => loadBlob(resident.buffers[name], engine.modelURL, add)));
      engine.resident = new Resident(JSON.stringify(resident.config), ...packed.map(bytes => new Float32Array(bytes)));
      engine.brain = await MaleCNSWebGPU.load({device, manifest: cns, baseURL: engine.modelURL, capacity: engine.batch, onProgress: ({bytes}) => add(bytes), shaderBaseURL: new URL("cns-webgpu.js", engine.baseURL)});
      await engine.brain.reset(Array.from({length: engine.batch}, (_, i) => i), engine.world.residentDescriptors.map(r => ({id: r.id, physicalEpoch: engine.world.engine, seed: 20260908})));
      const [soma, valid] = await Promise.all(['positions', 'valid'].map(name => loadBlob(observer.buffers[name], engine.modelURL, add)));
      engine.brainPositions = new Float32Array(soma); engine.brainValid = new Uint8Array(valid);
      engine.neuralBaseline = engine.brain.baselineRates;
      engine.modelStatus = cns.trainingStatus;
      engine.controllerStatus = resident.trainingStatus;
      engine.deliveredContext = new Float32Array(engine.batch * 12);
      engine.pendingContext = new Float32Array(engine.batch * 12);
      engine.pendingTick = null;
      engine.lastMotor = new Float32Array(engine.batch * 92);
      engine.tick = 0; engine.selected = 0; engine.neuralField = 'rate'; engine.failed = false; engine.journal = []; engine.externalEvents = [];
      engine.loadedBytes = loaded;
      return engine;
    } catch (error) {
      engine.destroy();
      throw error;
    }
  }
  describe() {
    return {engine: 'MaleCNS V5 · private synaptic learning · NeuroMechFly body', neurons: 165122,
      validSoma: this.brainValid.reduce((sum, value) => sum + value, 0), residents: this.world.residentDescriptors,
      modelStatus: this.modelStatus, controllerStatus: this.controllerStatus,
      brainPositions: this.brainPositions, brainValid: this.brainValid, neuralBaseline: this.neuralBaseline,
      retinalSites: this.retinalSites, retinalSupported: this.retinalSupported, motorSchema: this.motorSchema,
      identity: this.identity};
  }
  async advance(capture = false) {
    if (this.failed) throw new Error('An incomplete tick paused this life; restore a coherent checkpoint');
    const started = performance.now();
    try {
      while (this.externalEvents.length && this.externalEvents[0].tick <= this.tick) {
        const event = this.externalEvents.shift();
        this.world.visitorSound(event.position, event.frequency, event.amplitude, event.duration);
      }
      const {optic, body} = this.world.sample();
      const neural = await this.brain.step({dt: .01, activeMask: (1 << this.batch) - 1,
        opticRGB: optic, body, context: this.pendingContext,
        ...(capture ? {selectedResident: this.selected, selectedField: this.neuralField} : {})});
      if (!(neural.latent instanceof Float32Array) || neural.latent.length !== this.batch * 512 ||
          !neural.latent.every(Number.isFinite)) throw new Error('Invalid CNS cognitive readout');
      // Context is an internal neural intervention, not a physical command. A
      // previous decision is acknowledged only after the CNS actually used it.
      if (this.pendingTick !== null) {
        const pendingTicks = new BigUint64Array(this.batch).fill(BigInt(this.pendingTick));
        const accepted = this.resident.acknowledge(pendingTicks, this.pendingContext);
        if (!accepted.every(Boolean)) throw new Error('Delivered CNS context receipt rejected');
      }
      this.deliveredContext = this.pendingContext.slice();
      if (!(neural.motor instanceof Float32Array) || neural.motor.length !== this.batch * 92 ||
          !neural.motor.every((v, i) => Number.isFinite(v) && v <= 1 && v >= (i % 92 < 84 ? -1 : 0)))
        throw new Error('Invalid anatomical motor output');
      await this.world.advance(neural.motor, .01);
      this.lastMotor = neural.motor.slice();
      const ticks = new BigUint64Array(this.batch).fill(BigInt(this.tick));
      const resets = new Uint8Array(this.batch).fill(this.tick === 0 ? 1 : 0);
      const decision = this.resident.stepFlat(neural.latent, this.deliveredContext, ticks, resets);
      let diagnostics;
      try {
        this.pendingContext = decision.proposedContext;
        diagnostics = decision.diagnosticsJson;
      } finally { decision.free(); }
      if (this.pendingContext.length !== this.batch * 12 || !this.pendingContext.every(v => Number.isFinite(v) && Math.abs(v) <= 1))
        throw new Error('Invalid cognitive context proposal');
      this.pendingTick = this.tick;
      this.tick++;
      const physical = this.world.observe();
      return {...physical, neuralRates: neural.selectedRates,
        neuralSignal: neural.selectedSignal, neuralField: neural.selectedField,
        retinalRGB: capture ? optic.slice(this.selected * 5313, (this.selected + 1) * 5313) : undefined,
        // Read-only observation of the exact input supplied above. It is never
        // passed to the private resident, which receives CNS latent512 only.
        bodySense: capture ? body.slice(this.selected * 807, (this.selected + 1) * 807) : undefined,
        bodySenseTime: (this.tick - 1) * .01,
        selectedResidentId: this.world.residentDescriptors[this.selected].id,
        motorActivation: this.lastMotor.slice(this.selected * 92, (this.selected + 1) * 92),
        deliveredContext: this.deliveredContext.slice(this.selected * 12, (this.selected + 1) * 12),
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
  tone(frequency, duration = .6, amplitude = .65) {
    if (!Number.isFinite(frequency) || frequency < 40 || frequency > 1600 ||
        !Number.isFinite(duration) || duration <= 0 || duration > 5 ||
        !Number.isFinite(amplitude) || amplitude < 0 || amplitude > 1)
      throw new Error('Tone outside the physical source limits');
    this.externalEvents.push({tick: this.tick, position: [0, 0, 2], frequency, amplitude, duration});
    this.externalEvents.sort((a, b) => a.tick - b.tick);
    this.record('visitor-tone', {frequency, duration, amplitude});
  }
  greet(notes = [0, 1, 2]) {
    if (notes.length > 8 || notes.some(n => !Number.isInteger(n) || n < 0 || n > 2)) throw new Error('Choose up to eight notes');
    const frequencies = [100, 250, 630];
    notes.forEach((note, index) => this.externalEvents.push({tick: this.tick + index * 30,
      position: [0, 0, 2], frequency: frequencies[note], amplitude: .65, duration: .2}));
    this.externalEvents.sort((a, b) => a.tick - b.tick);
    this.record('visitor-sound', {notes, frequencies: notes.map(n => frequencies[n])});
  }
  async insertToy(position) {
    const result = await this.world.insertObject({position, size: [.35, .35, .35], shape: 'sphere', rgba: [.83, .37, .16, 1]});
    this.lastToy = result.id ?? result; this.record('insert-object', {id: this.lastToy, position: Array.from(position)}); return result;
  }
  shove(id, force = [.03, .01, 0]) {
    this.world.queueVisitorForce(id ?? this.lastToy, force); this.record('visitor-force', {id: id ?? this.lastToy, force});
  }
  destroy() {
    try { this.world?.dispose(); }
    finally {
      try { this.resident?.free(); }
      finally { this.brain?.destroy(); }
    }
    this.world = this.resident = this.brain = null;
  }
  async save() {
    if (this.failed) throw new Error('Cannot save a partially advanced life');
    if (!sameTime(this.world.time, this.tick * .01) ||
        !this.brain.times.every(time => sameTime(time, this.tick * .01))) throw new Error('Coupled life clocks differ');
    const world = this.world.snapshot();
    const cns = new Uint8Array(await this.brain.snapshot()), resident = this.resident.saveBytes();
    const [worldSHA, cnsSHA, residentSHA] = await Promise.all([world, cns, resident].map(digest));
    const header = encoder.encode(JSON.stringify({format: 'chreatures-live-life-v5', identity: this.identity,
      tick: this.tick, selected: this.selected, deliveredContext: Array.from(this.deliveredContext),
      pendingContext: Array.from(this.pendingContext), pendingTick: this.pendingTick, lastMotor: Array.from(this.lastMotor),
      journal: this.journal, externalEvents: this.externalEvents, filmTime: this.filmTime ?? null, lastToy: this.lastToy ?? null,
      neuralField: this.neuralField, worldTime: this.world.time,
      worldBytes: world.length, cnsBytes: cns.length, residentBytes: resident.length, worldSHA, cnsSHA, residentSHA}));
    const result = new Uint8Array(12 + header.length + world.length + cns.length + resident.length);
    result.set(MAGIC); new DataView(result.buffer).setUint32(8, header.length, true);
    result.set(header, 12);
    let offset = 12 + header.length;
    for (const segment of [world, cns, resident]) {result.set(segment, offset); offset += segment.length;}
    return result.buffer;
  }
  async load(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 12 || buffer.byteLength > 512 * 1024**2) throw new Error('Invalid life file');
    const bytes = new Uint8Array(buffer);
    if (!MAGIC.every((b, i) => b === bytes[i])) throw new Error('Different life format');
    const length = new DataView(buffer).getUint32(8, true);
    if (length > 8 * 1024**2 || length + 12 > bytes.length) throw new Error('Invalid life envelope');
    const h = JSON.parse(decoder.decode(bytes.subarray(12, 12 + length)));
    if (h.format !== 'chreatures-live-life-v5' || h.identity !== this.identity ||
        !Number.isSafeInteger(h.tick) || h.tick < 0 || !Number.isInteger(h.selected) || h.selected < 0 || h.selected >= this.batch ||
        !Array.isArray(h.deliveredContext) || h.deliveredContext.length !== this.batch * 12 ||
        !h.deliveredContext.every(v => Number.isFinite(v) && Math.abs(v) <= 1) ||
        !Array.isArray(h.pendingContext) || h.pendingContext.length !== this.batch * 12 ||
        !h.pendingContext.every(v => Number.isFinite(v) && Math.abs(v) <= 1) ||
        h.pendingTick !== (h.tick === 0 ? null : h.tick - 1) ||
        !Array.isArray(h.lastMotor) || h.lastMotor.length !== this.batch * 92 ||
        !h.lastMotor.every((v,i) => Number.isFinite(v) && v <= 1 && v >= (i % 92 < 84 ? -1 : 0)) ||
        ![h.worldBytes, h.cnsBytes, h.residentBytes].every(size => Number.isSafeInteger(size) && size > 0) ||
        12 + length + h.worldBytes + h.cnsBytes + h.residentBytes !== bytes.length ||
        !sameTime(h.worldTime, h.tick * .01) || !FIELDS.includes(h.neuralField)) throw new Error('Life identity, length or clock differs');
    if (!Array.isArray(h.externalEvents) || h.externalEvents.length > 4096 || h.externalEvents.some(e => !Number.isSafeInteger(e.tick) || e.tick < h.tick || e.position?.length !== 3 || !e.position.every(Number.isFinite) || !Number.isFinite(e.frequency) || e.frequency < 40 || e.frequency > 1600 || !Number.isFinite(e.amplitude) || e.amplitude < 0 || e.amplitude > 1 || !Number.isFinite(e.duration) || e.duration <= 0 || e.duration > 5)) throw new Error('Invalid pending physical signals');
    if (h.externalEvents.some((event, index) => index > 0 && event.tick < h.externalEvents[index - 1].tick) ||
        !Array.isArray(h.journal) || h.journal.length > 4096 ||
        h.journal.some(event => !Number.isSafeInteger(event.tick) || event.tick < 0 || event.tick > h.tick ||
          !Number.isFinite(event.time) || event.time < 0 || typeof event.kind !== 'string') ||
        !(h.filmTime === null || Number.isFinite(h.filmTime) && h.filmTime >= 0) ||
        !(h.lastToy === null || typeof h.lastToy === 'string' && h.lastToy.length <= 256)) throw new Error('Invalid observer history');
    const worldOffset = 12 + length, cnsOffset = worldOffset + h.worldBytes;
    const residentOffset = cnsOffset + h.cnsBytes;
    const worldBytes = bytes.slice(worldOffset, cnsOffset);
    const cns = bytes.slice(cnsOffset, residentOffset), resident = bytes.slice(residentOffset);
    const hashes = await Promise.all([worldBytes, cns, resident].map(digest));
    if (hashes[0] !== h.worldSHA || hashes[1] !== h.cnsSHA || hashes[2] !== h.residentSHA) throw new Error('Life state checksum differs');
    // Native restore reconstructs appended geometry, delayed forces and ecology
    // inside a fresh world; the existing life remains untouched until commit.
    const world = await createPhysicalWorld(this.factoryOptions);
    let oldCns, oldResident;
    try {
      world.restore(worldBytes);
      if (!sameTime(world.time, h.worldTime) || world.residents !== this.batch) throw new Error('Restored physical clock or cohort differs');
      oldCns = await this.brain.snapshot();
      oldResident = this.resident.saveBytes();
    } catch (error) {world.dispose(); throw error;}
    try {
      await this.brain.restore(cns.buffer);
      if (!this.brain.times.every(time => sameTime(time, h.worldTime))) throw new Error('Restored CNS clock differs from the body');
      this.resident.loadBytes(resident);
    }
    catch (error) {
      world.dispose();
      try {await this.brain.restore(oldCns); this.resident.loadBytes(oldResident);} catch {this.failed = true;}
      throw error;
    }
    this.world.dispose(); this.world = world; this.tick = h.tick; this.selected = h.selected;
    this.deliveredContext = Float32Array.from(h.deliveredContext);
    this.pendingContext = Float32Array.from(h.pendingContext); this.pendingTick = h.pendingTick;
    this.lastMotor = Float32Array.from(h.lastMotor); this.journal = h.journal; this.externalEvents = h.externalEvents;
    this.filmTime = h.filmTime; this.lastToy = h.lastToy; this.neuralField = h.neuralField; this.failed = false;
  }
}
