import {LiveView, NeuronInspector} from './live/view.js';

const $ = selector => document.querySelector(selector);
const stateLabel = $('#state-label');
const stateLight = stateLabel.parentElement;
const startLayer = $('#start-layer');
const startButton = $('#start');
const progress = $('#load-progress');
const loadDetail = $('#load-detail');
const modelTime = $('#model-time');
const filmTime = $('#film-time');
const residentList = $('#resident-list');
const selectedName = $('#selected-name');
const pauseButton = $('#pause');
const saveButton = $('#save');
const loadInput = $('#load');
const greetButton = $('#greet');
const motorDisplay = $('#motor-output');
let motorBars = [];
const addToyButton = $('#add-toy');
const shoveToyButton = $('#shove-toy');
const stimulusVideo = $('#stimulus-video');
const stimulusCanvas = $('#stimulus-canvas');
const stimulusContext = stimulusCanvas.getContext('2d', {willReadFrequently: true});
const interactive = [...document.querySelectorAll('.instrument-panel button, .instrument-panel input, .instrument-panel select, .camera-bar button')];

let worker = null;
let view = null;
let inspector = null;
let ready = false;
let paused = false;
let selectedResident = null;
let neuralField = 'rate', loadedRateBaseline = null;
let selectedToy = null;
let placingToy = false, pendingToyRequest = null;
let requestCounter = 0;
const requestedStimulus = new URLSearchParams(location.search).get('stimulus');
const initialStimulus = ['bad-apple', 'grating'].includes(requestedStimulus) ? requestedStimulus : 'blank';
let stimulusMode = 'blank';
let stimulusTimer = null;
let stimulusStarted = performance.now();
let noticeTimer = null;
const acousticBars = Array.from({length: 16}, (_, index) => {
  const frequency = 40 * 40 ** (index / 15);
  const column = document.createElement('span'), fill = document.createElement('i');
  column.title = `${frequency.toFixed(1)} Hz · no input yet`;
  column.append(fill); $('#acoustic-bands').append(column);
  return {column, fill, frequency};
});
const footReadouts = ['Left front', 'Left middle', 'Left hind', 'Right front', 'Right middle', 'Right hind'].map(name => {
  const row = document.createElement('div'), label = document.createElement('span'), value = document.createElement('output');
  label.textContent = name; value.textContent = '—'; row.append(label, value); $('#foot-senses').append(row);
  return value;
});

function updateBodySenses(body, time) {
  if (!(body instanceof Float32Array) || body.length !== 807 || !body.every(Number.isFinite))
    throw new Error('Body sensory observer extent differs');
  $('#sense-time').textContent = `${time.toFixed(2)} s`;
  for (let i = 0; i < 16; i++) {
    const {column, fill, frequency} = acousticBars[i], value = body[46 + i];
    fill.style.height = `${100 * Math.min(1, Math.log1p(Math.max(0, value) * 100) / Math.log(101))}%`;
    column.title = `${frequency.toFixed(1)} Hz · ${value.toExponential(3)}`;
  }
  const magnitude = offset => Math.hypot(body[offset], body[offset + 1], body[offset + 2]);
  for (let i = 0; i < 6; i++) {
    const offset = 459 + 6 * i;
    footReadouts[i].textContent = `${magnitude(offset).toPrecision(3)} / ${magnitude(offset + 3).toPrecision(3)}`;
  }
  $('#mouth-contact').textContent = `${(100 * body[711]).toFixed(1)}% of sampled interval`;
  $('#antenna-flow').textContent = `${magnitude(40).toPrecision(3)} / ${magnitude(43).toPrecision(3)} mm/s`;
  let square = 0;
  for (let i = 207; i < 333; i++) square += body[i] ** 2;
  $('#joint-speed').textContent = `${Math.sqrt(square / 126).toPrecision(3)} rad/s`;
  $('#body-reserves').textContent = `${body[69].toFixed(3)} / ${body[73].toFixed(3)}`;
}

function initializeMotorDisplay(schema) {
  const channels = schema?.channels;
  if (!Array.isArray(channels) || channels.length !== 92 || channels.some((c, i) => c.index !== i)) {
    throw new Error('Anatomical motor schema is missing');
  }
  const names = {walking: 'Six legs', head: 'Head', pedicels: 'Antennae', proboscis: 'Proboscis',
    abdomen: 'Abdomen', wings: 'Wings', halteres: 'Halteres', adhesion: 'Foot adhesion',
    pharyngeal_pump: 'Pharyngeal pump', salivary_drive: 'Salivary drive'};
  const groups = new Map();
  motorBars = Array(92);
  motorDisplay.replaceChildren();
  for (const channel of channels) {
    let group = groups.get(channel.group);
    if (!group) {
      group = document.createElement('details');
      group.open = ['walking', 'head', 'proboscis', 'adhesion', 'pharyngeal_pump'].includes(channel.group);
      const summary = document.createElement('summary'); summary.textContent = names[channel.group] || channel.group;
      group.append(summary); groups.set(channel.group, group); motorDisplay.append(group);
    }
    const row = document.createElement('div'); row.className = 'motor-channel';
    const parts = channel.target.split('-');
    const label = document.createElement('span');
    label.textContent = parts.length > 1 ? `${parts.at(-2).replaceAll('_', ' ')} ${parts.at(-1)}` : channel.target.replaceAll('_', ' ');
    const track = document.createElement('span'); track.className = 'motor-track';
    const signed = channel.normalized_range[0] < 0;
    track.dataset.signed = String(signed);
    const fill = document.createElement('i'); track.append(fill);
    const value = document.createElement('output'); value.textContent = '—';
    row.title = channel.id; row.append(label, track, value); group.append(row);
    motorBars[channel.index] = {fill, value, signed};
  }
}

function updateMotorDisplay(values) {
  if (values.length !== 92 || motorBars.length !== 92) throw new Error('Motor observer extent differs');
  for (let i = 0; i < 92; i++) {
    const bar = motorBars[i], x = values[i];
    if (!Number.isFinite(x) || x < (bar.signed ? -1 : 0) || x > 1) throw new Error('Invalid motor observation');
    bar.fill.style.left = `${bar.signed ? 50 + Math.min(0, x) * 50 : 0}%`;
    bar.fill.style.width = `${Math.abs(x) * (bar.signed ? 50 : 100)}%`;
    bar.fill.dataset.negative = String(x < 0);
    bar.value.textContent = x.toFixed(2);
  }
}

function createView() {
  return new LiveView({
    worldCanvas: $('#world-canvas'),
    brainCanvas: $('#brain-canvas'),
    retinaCanvases: [$('#retina-side-1'), $('#retina-side-2')],
    onResident: selectResident,
    onNeuron(row) { inspector?.select(row); },
    onToy(id) {
      selectedToy = id;
      shoveToyButton.disabled = !ready;
      setNotice(`Selected physical object ${id}.`);
    },
    onPlacement(position) {
      if (!ready || !placingToy || pendingToyRequest) return;
      setToyPlacement(false);
      addToyButton.disabled = true;
      pendingToyRequest = request('insert-toy', {position});
      setNotice('Checking space for the toy…');
    },
  });
}

function setToyPlacement(active) {
  placingToy = Boolean(active);
  view?.setToyPlacement(placingToy);
  addToyButton.setAttribute('aria-pressed', String(placingToy));
  addToyButton.textContent = placingToy ? 'Cancel toy placement' : 'Place a movable toy';
}

function setInteractive(enabled) {
  for (const control of interactive) control.disabled = !enabled;
  shoveToyButton.disabled = !enabled || !selectedToy;
}

function setNotice(text, kind = ready ? 'ready' : '') {
  stateLabel.textContent = text;
  stateLight.className = `live-state ${kind}`.trim();
}

function fatal(reason) {
  if (noticeTimer) clearTimeout(noticeTimer);
  ready = false;
  setInteractive(false);
  stopStimulus();
  view?.stop();
  startLayer.hidden = false;
  startButton.hidden = true;
  progress.style.width = '0%';
  loadDetail.textContent = reason;
  setNotice('stopped', 'failed');
}

function recoverableError(message) {
  const reason = message.reason || message.message || 'The requested physical operation was rejected.';
  if (typeof message.paused === 'boolean') paused = message.paused;
  pauseButton.textContent = paused ? 'Resume' : 'Pause';
  setNotice(reason, 'notice');
  if (noticeTimer) clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    if (ready) setNotice(paused ? 'paused' : 'running locally', paused ? 'paused' : 'ready');
    noticeTimer = null;
  }, 5200);
}

function post(type, fields = {}, transfer = []) {
  if (!worker) throw new Error('Live Worker is not initialized');
  worker.postMessage({type, ...fields}, transfer);
}

function request(type, fields = {}) {
  const requestId = `ui-${++requestCounter}`;
  post(type, {requestId, ...fields});
  return requestId;
}

function residentButtons(residents) {
  residentList.replaceChildren(...residents.map(resident => {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = resident.id;
    button.dataset.resident = resident.id;
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => selectResident(resident.id));
    return button;
  }));
}

function selectResident(id) {
  if (!view || !view.residents.some(item => item.id === id)) return;
  selectedResident = id;
  selectedName.textContent = id;
  view.selectResident(id);
  view.clearNeural();
  view.clearRetina();
  resetInspector();
  $('#neural-rms').textContent = '—'; $('#neural-peak').textContent = '—';
  $('#sense-time').textContent = '—';
  for (const {fill} of acousticBars) fill.style.height = '0%';
  for (const value of footReadouts) value.textContent = '—';
  for (const id of ['mouth-contact', 'antenna-flow', 'joint-speed', 'body-reserves']) $(`#${id}`).textContent = '—';
  for (const button of residentList.querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.resident === id));
  if (worker) post('select', {residentId: id});
}

function resetInspector() {
  if (view) inspector?.reset({resident: selectedResident, field: neuralField, reference: view.brainBaseline, scale: view.neuralScale});
}

function updateProgress(message) {
  const loaded = Number(message.loaded ?? message.value ?? 0);
  const total = Number(message.total);
  const track = progress.parentElement;
  if (Number.isFinite(total) && total > 0) {
    track.classList.remove('indeterminate');
    progress.style.width = `${Math.round(Math.max(0, Math.min(1, loaded / total)) * 100)}%`;
  } else {
    track.classList.add('indeterminate');
    progress.style.width = '38%';
  }
  const bytes = Number.isFinite(loaded) && loaded > 0 ? ` · ${(loaded / 1048576).toFixed(1)} MB received` : '';
  loadDetail.textContent = `${message.label || message.stage || 'Loading the local runtime…'}${bytes}`;
}

function handleReady(message) {
  initializeMotorDisplay(message.motorSchema);
  const residents = Array.isArray(message.residents) ? message.residents : [];
  if (!residents.length) throw new Error('Worker reported no residents');
  if (!(message.brainPositions instanceof Float32Array) || !(message.brainValid instanceof Uint8Array) || !(message.neuralBaseline instanceof Float32Array)) {
    throw new Error('Worker did not provide the pinned MaleCNS atlas and baseline');
  }
  const visible = view.initializeBrain(message.brainPositions, message.brainValid, message.neuralBaseline);
  loadedRateBaseline = message.neuralBaseline.slice();
  if (message.neurons !== 165122 || message.validSoma !== visible || visible !== 140024) throw new Error('Worker MaleCNS extent differs from the pinned observer atlas');
  const retina = view.initializeRetina(message.retinalSites, message.retinalSupported);
  if (retina.sites !== 1771 || retina.supported !== 1486) throw new Error('Worker retinal atlas extent differs from the pinned observer atlas');
  $('#brain-count').textContent = `${visible.toLocaleString()} positioned / 165,122 rows`;
  $('#model-status').textContent = message.modelStatus || 'status absent';
  $('#controller-status').textContent = message.controllerStatus || 'status absent';
  $('#engine-status').textContent = message.engine || 'engine identity absent';
  view.setResidents(residents);
  residentButtons(residents);
  ready = true; paused = false;
  progress.parentElement.classList.remove('indeterminate'); progress.style.width = '100%';
  startLayer.hidden = true;
  setInteractive(true);
  selectResident(residents[0].id);
  void inspector.load();
  setNotice('running locally', 'ready');
  startStimulusClock();
  if (initialStimulus !== 'blank') {
    const button = document.querySelector(`[data-stimulus="${initialStimulus}"]`);
    void chooseStimulus(initialStimulus, button);
  }
}

function updateFrame(message) {
  view.applyFrame(message);
  modelTime.textContent = `${Number(message.time).toFixed(2)} s`;
  paused = Boolean(message.paused);
  pauseButton.textContent = paused ? 'Resume' : 'Pause';
  if (!noticeTimer) setNotice(paused ? 'paused' : 'running locally', paused ? 'paused' : 'ready');
  if (message.selectedResidentId === selectedResident) {
    if (message.neuralSignal && message.neuralField === neuralField) {
      const activity = view.updateNeural(message.neuralSignal);
      $('#neural-rms').textContent = activity.rms.toExponential(2);
      $('#neural-peak').textContent = activity.peak.toExponential(2);
      inspector?.update(Number(message.time), message.neuralSignal);
    }
    if (message.retinalRGB) view.updateRetina(message.retinalRGB);
    if (message.motorActivation) updateMotorDisplay(message.motorActivation);
    if (message.bodySense) updateBodySenses(message.bodySense, message.bodySenseTime);
  }
  const visitorIds = new Set();
  for (const item of message.geometry) {
    const match = /^entity:(visitor-[^:]+):/.exec(item.name);
    if (match) visitorIds.add(match[1]);
  }
  if (selectedToy && !visitorIds.has(selectedToy)) selectedToy = null;
  if (!selectedToy && visitorIds.size) selectedToy = [...visitorIds].at(-1);
  shoveToyButton.disabled = !selectedToy;
}

function downloadSnapshot(message) {
  if (!(message.snapshot instanceof ArrayBuffer)) throw new Error('Saved checkpoint payload differs');
  const url = URL.createObjectURL(new Blob([message.snapshot], {type: 'application/octet-stream'}));
  const link = document.createElement('a');
  link.href = url; link.download = `chreature-${Date.now()}.chreature`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setNotice('checkpoint saved', paused ? 'paused' : 'ready');
}

function workerMessage(event) {
  const message = event.data;
  if (!message || typeof message.type !== 'string') return fatal('The Worker returned an invalid message. The world is stopped.');
  try {
    if (message.type === 'progress') updateProgress(message);
    else if (message.type === 'ready') handleReady(message);
    else if (message.type === 'frame') updateFrame(message);
    else if (message.type === 'saved') downloadSnapshot(message);
    else if (message.type === 'loaded') {
      resetInspector();
      if (selectedResident) post('select', {residentId: selectedResident});
      setNotice('checkpoint loaded', message.paused ? 'paused' : 'ready');
    }
    else if (message.type === 'inserted') {
      pendingToyRequest = null;
      addToyButton.disabled = !ready;
      selectedToy = message.object?.id || null;
      shoveToyButton.disabled = !selectedToy;
      setNotice(selectedToy ? `placed ${selectedToy}` : 'physical object placed', paused ? 'paused' : 'ready');
    } else if (message.type === 'error') {
      if (pendingToyRequest && message.requestId === pendingToyRequest) {
        pendingToyRequest = null;
        addToyButton.disabled = !ready;
        if (message.recoverable && ready) setToyPlacement(true);
      }
      if (message.recoverable && ready) recoverableError(message);
      else fatal(message.reason || message.message || 'The compute Worker stopped without a reason.');
    }
  } catch (error) {
    fatal(`Observer boundary rejected the live state: ${error.message}`);
  }
}

function blankFrame() {
  stimulusContext.fillStyle = '#050807'; stimulusContext.fillRect(0, 0, stimulusCanvas.width, stimulusCanvas.height);
}

function gratingFrame(seconds) {
  const width = stimulusCanvas.width, height = stimulusCanvas.height;
  const image = stimulusContext.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const light = Math.sin(x * .22 + seconds * 4.1) > 0 ? 235 : 18;
      const offset = (y * width + x) * 4;
      image.data[offset] = light; image.data[offset + 1] = light; image.data[offset + 2] = light; image.data[offset + 3] = 255;
    }
  }
  stimulusContext.putImageData(image, 0, 0);
}

function sendStimulusFrame() {
  if (!ready) return;
  if (stimulusMode === 'bad-apple') {
    if (stimulusVideo.readyState < 2) return;
    stimulusContext.drawImage(stimulusVideo, 0, 0, stimulusCanvas.width, stimulusCanvas.height);
  } else if (stimulusMode === 'grating') gratingFrame((performance.now() - stimulusStarted) / 1000);
  else blankFrame();
  const bytes = stimulusContext.getImageData(0, 0, stimulusCanvas.width, stimulusCanvas.height).data;
  const rgb = new Float32Array(stimulusCanvas.width * stimulusCanvas.height * 3);
  for (let source = 0, target = 0; source < bytes.length; source += 4) {
    rgb[target++] = bytes[source] / 255; rgb[target++] = bytes[source + 1] / 255; rgb[target++] = bytes[source + 2] / 255;
  }
  view.setScreenCanvas(stimulusCanvas);
  post('screen-frame', {rgb, width: stimulusCanvas.width, height: stimulusCanvas.height, filmTime: stimulusMode === 'bad-apple' ? stimulusVideo.currentTime : (performance.now() - stimulusStarted) / 1000}, [rgb.buffer]);
  filmTime.textContent = stimulusMode === 'bad-apple' ? `${stimulusVideo.currentTime.toFixed(2)} s · wall clock` : `${stimulusMode} · wall clock`;
}

function startStimulusClock() {
  if (stimulusTimer) return;
  blankFrame(); sendStimulusFrame();
  stimulusTimer = setInterval(sendStimulusFrame, 50);
}

function stopStimulus() {
  if (stimulusTimer) clearInterval(stimulusTimer);
  stimulusTimer = null; stimulusVideo.pause();
}

async function chooseStimulus(kind, button) {
  if (!ready || !['bad-apple', 'grating', 'blank'].includes(kind)) return;
  if (kind === 'bad-apple') {
    try { await stimulusVideo.play(); }
    catch (error) { setNotice(`film could not start: ${error.message}`, 'paused'); return; }
  } else stimulusVideo.pause();
  stimulusMode = kind; stimulusStarted = performance.now();
  for (const item of document.querySelectorAll('[data-stimulus]')) item.setAttribute('aria-pressed', String(item === button));
  post('stimulus', {kind});
  sendStimulusFrame();
}

startButton.addEventListener('click', () => {
  if (!navigator.gpu) return fatal('WebGPU is unavailable in this browser. No life was started. Try a current browser with WebGPU enabled, or watch the archived recording.');
  startButton.disabled = true;
  try {
    view = createView();
    inspector = new NeuronInspector($('#neuron-inspector'), row => view.selectNeuron(row));
    loadDetail.textContent = 'Opening the isolated compute Worker…';
    worker = new Worker('./live/worker.js', {type: 'module'});
    worker.addEventListener('message', workerMessage);
    worker.addEventListener('error', event => fatal(`The compute Worker failed to load: ${event.message || 'unknown module error'}`));
    post('start');
  } catch (error) {
    fatal(`The 3D observer could not start: ${error.message}`);
  }
});

for (const button of document.querySelectorAll('[data-stimulus]')) button.addEventListener('click', () => chooseStimulus(button.dataset.stimulus, button));
for (const button of document.querySelectorAll('[data-camera]')) button.addEventListener('click', () => {
  if (!view || !ready) return;
  view.setCameraMode(button.dataset.camera);
  for (const item of document.querySelectorAll('[data-camera]')) item.setAttribute('aria-pressed', String(item === button));
});
for (const button of document.querySelectorAll('[data-neural-panel]')) button.addEventListener('click', () => {
  const panel = button.dataset.neuralPanel;
  for (const item of document.querySelectorAll('[data-neural-panel]')) item.setAttribute('aria-selected', String(item === button));
  $('#brain-panel').hidden = panel !== 'brain';
  $('#eyes-panel').hidden = panel !== 'eyes';
  $('#body-panel').hidden = panel !== 'body';
  if (panel === 'eyes') requestAnimationFrame(() => view?.renderRetina());
});

$('#neural-field').addEventListener('change', event => {
  if (!view || !ready) return;
  neuralField = event.target.value;
  $('#brain-canvas').setAttribute('aria-label', `Actual MaleCNS soma point cloud: ${event.target.selectedOptions[0].textContent}`);
  const reservoir = neuralField === 'support' || neuralField === 'release';
  const baseline = neuralField === 'rate' ? loadedRateBaseline : new Float32Array(165122).fill(reservoir ? 1 : 0);
  view.setNeuralField(neuralField, baseline);
  resetInspector();
  $('#neural-reference').textContent = neuralField === 'rate' ? 'change from loaded rate baseline' : reservoir ? 'change from full resource (1)' : 'signed deviation from zero';
  const descriptions = {
    rate: 'Model activity relative to its loaded baseline, not measured firing rate or Hz.',
    adaptation: 'Slow, activity-dependent neural adaptation. Signed model state; not a measurement of fatigue or feeling.',
    support: 'Local activity-dependent support relative to its fully replenished value. An engineered regulatory mechanism.',
    release: 'Neuron-shared effective release resource relative to full recovery. This approximates resource use; individual boutons are not modeled.',
    dopamine: 'Target-local effective modulation carried by dopamine-annotated connections. A signed deviation from tonic state, not a chemical concentration or reward score.',
    octopamine: 'Target-local effective modulation carried by octopamine-annotated connections. A signed deviation from tonic state, not a chemical concentration.',
    serotonin: 'Target-local effective modulation carried by serotonin-annotated connections. A signed deviation from tonic state, not a chemical concentration.'
  };
  $('#neural-description').textContent = descriptions[neuralField] + ' Color limits stay fixed until you change them.';
  $('#neural-rms').textContent = '—'; $('#neural-peak').textContent = '—';
  post('neural-field', {field: neuralField});
});

$('#neural-scale').addEventListener('input', event => {
  if (!view || !ready) return;
  const value = 10 ** Number(event.target.value);
  view.setNeuralScale(value); $('#scale-output').textContent = `±${value.toPrecision(2)}`;
  inspector?.setScale(value);
});
pauseButton.addEventListener('click', () => post(paused ? 'resume' : 'pause'));
saveButton.addEventListener('click', () => request('save'));
loadInput.addEventListener('change', async () => {
  const file = loadInput.files?.[0]; if (!file || !ready) return;
  const snapshot = await file.arrayBuffer();
  const requestId = `ui-${++requestCounter}`;
  worker.postMessage({type: 'load', requestId, snapshot}, [snapshot]);
  loadInput.value = '';
});
greetButton.addEventListener('click', () => post('greet', {notes: [0, 1, 2]}));
for (const button of document.querySelectorAll('[data-tone]')) button.addEventListener('click', () => {
  post('tone', {frequency: Number(button.dataset.tone), duration: .8, amplitude: .65});
  setNotice(`Sent ${button.dataset.tone} Hz into the garden`, 'ready');
});
addToyButton.addEventListener('click', () => {
  if (!ready || pendingToyRequest) return;
  setToyPlacement(!placingToy);
  setNotice(placingToy ? 'Click a surface to place the toy. Drag to orbit; Esc cancels.' : 'Toy placement cancelled.');
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && placingToy) {
    setToyPlacement(false);
    setNotice('Toy placement cancelled.');
  }
});
shoveToyButton.addEventListener('click', () => {
  if (selectedToy) post('shove', {id: selectedToy, force: [0, 4, 1]});
});
stimulusVideo.addEventListener('error', () => {
  if (stimulusMode === 'bad-apple') {
    stimulusMode = 'blank'; blankFrame(); sendStimulusFrame();
    setNotice('official film asset unavailable; screen returned to blank', paused ? 'paused' : 'ready');
  }
});

setInteractive(false);

if (initialStimulus === 'bad-apple') startButton.textContent = 'Start a life with Bad Apple';
