import { LiveEngine } from './engine.js';
let engine, running = false, queue = Promise.resolve(), scheduled = false;
const post = (type, detail = {}, transfer = []) => self.postMessage({type, ...detail}, transfer);
function exclusive(operation, requestId) {
  queue = queue.then(operation).catch(error => {
    const recoverable = Boolean(engine && !engine.failed);
    if (!recoverable) running = false;
    post('error', {message: error.message ?? String(error), requestId, paused: !running, recoverable});
  });
}
function frame(value) {
  const transferable = [];
  for (const field of ['positions', 'rotations', 'colors', 'bodyPositions', 'food', 'neuralRates', 'retinalRGB']) {
    if (value[field]?.buffer) transferable.push(value[field].buffer);
  }
  post('frame', {...value, paused: !running}, transferable);
}
function pump() {
  if (!running || scheduled) return;
  scheduled = true;
  setTimeout(() => {
    scheduled = false;
    exclusive(async () => {
      if (!running) return;
      const observation = await engine.advance(true);
      frame(observation);
    });
    queue.then(pump);
  }, 0);
}
self.onmessage = ({data: message}) => {
  exclusive(async () => {
    const {type, requestId} = message;
    if (type === 'start') {
      if (!engine) {
        if (!navigator.gpu) throw new Error('This live habitat needs WebGPU. The recorded experiment remains available on the Bad Apple page.');
        post('progress', {stage: 'gpu', label: 'Opening the full-connectome GPU engine'});
        const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
        if (!adapter) throw new Error('No WebGPU adapter is available');
        const device = await adapter.requestDevice();
        device.lost.then(info => {running = false; if (engine) engine.failed = true; post('error', {message: `GPU stopped: ${info.message}. Restore your last saved life to continue.`, paused: true});});
        engine = await LiveEngine.create({baseURL: new URL('./', import.meta.url), device, progress: value => post('progress', value)});
        post('ready', engine.describe()); frame(engine.observe());
      }
      running = true; pump(); return;
    }
    if (!engine) throw new Error('Start the habitat first');
    switch (type) {
      case 'pause': running = false; frame(engine.observe()); break;
      case 'resume': if (engine.failed) throw new Error('Restore a coherent life before resuming'); running = true; pump(); break;
      case 'screen-frame': engine.screen(message.rgb, message.width, message.height, message.filmTime); break;
      case 'stimulus': engine.record('screen-mode', {kind: message.kind}); break;
      case 'greet': engine.greet(message.notes); break;
      case 'insert-toy': post('inserted', {requestId, object: await engine.insertToy()}); frame(engine.observe()); break;
      case 'shove': engine.shove(message.id, message.force); break;
      case 'select': {
        const index = engine.world.observe().residents.findIndex(r => r.id === message.residentId);
        if (index < 0) throw new Error('Unknown resident'); engine.selected = index; break;
      }
      case 'save': {running = false; const snapshot = await engine.save(); post('saved', {requestId, snapshot}, [snapshot]); frame(engine.observe()); break;}
      case 'load': running = false; await engine.load(message.snapshot); post('loaded', {requestId}); frame(engine.observe()); break;
      default: throw new Error(`Unknown habitat operation: ${type}`);
    }
  }, message.requestId);
};
