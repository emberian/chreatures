import {LiveView} from './live/view.js';

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
const addToyButton = $('#add-toy');
const shoveToyButton = $('#shove-toy');
const stimulusVideo = $('#stimulus-video');
const stimulusCanvas = $('#stimulus-canvas');
const stimulusContext = stimulusCanvas.getContext('2d', {willReadFrequently: true});
const interactive = [...document.querySelectorAll('.instrument-panel button, .instrument-panel input, .camera-bar button')];

let worker = null;
let view = null;
let ready = false;
let paused = false;
let selectedResident = null;
let selectedToy = null;
let requestCounter = 0;
let stimulusMode = 'blank';
let stimulusTimer = null;
let stimulusStarted = performance.now();
let noticeTimer = null;

function createView() {
  return new LiveView({
    worldCanvas: $('#world-canvas'),
    brainCanvas: $('#brain-canvas'),
    onResident: selectResident,
    onToy(id) {
      selectedToy = id;
      shoveToyButton.disabled = !ready;
      setNotice(`Selected physical object ${id}.`);
    },
  });
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
  for (const button of residentList.querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.resident === id));
  if (worker) post('select', {residentId: id});
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
  const residents = Array.isArray(message.residents) ? message.residents : [];
  if (!residents.length) throw new Error('Worker reported no residents');
  if (!(message.brainPositions instanceof Float32Array) || !(message.brainValid instanceof Uint8Array) || !(message.neuralBaseline instanceof Float32Array)) {
    throw new Error('Worker did not provide the pinned MaleCNS atlas and baseline');
  }
  const visible = view.initializeBrain(message.brainPositions, message.brainValid, message.neuralBaseline);
  if (message.neurons !== 165122 || message.validSoma !== visible || visible !== 140024) throw new Error('Worker MaleCNS extent differs from the pinned observer atlas');
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
  setNotice('running locally', 'ready');
  startStimulusClock();
}

function updateFrame(message) {
  view.applyFrame(message);
  modelTime.textContent = `${Number(message.time).toFixed(2)} s`;
  paused = Boolean(message.paused);
  pauseButton.textContent = paused ? 'Resume' : 'Pause';
  setNotice(paused ? 'paused' : 'running locally', paused ? 'paused' : 'ready');
  if (message.neuralRates && message.selectedResidentId === selectedResident) view.updateNeural(message.neuralRates);
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
      if (selectedResident) post('select', {residentId: selectedResident});
      setNotice('checkpoint loaded', message.paused ? 'paused' : 'ready');
    }
    else if (message.type === 'inserted') {
      selectedToy = message.object?.id || null;
      shoveToyButton.disabled = !selectedToy;
      setNotice(selectedToy ? `placed ${selectedToy}` : 'physical object placed', paused ? 'paused' : 'ready');
    } else if (message.type === 'error') {
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

$('#neural-scale').addEventListener('input', event => {
  if (!view || !ready) return;
  const value = Number(event.target.value);
  view.setNeuralScale(value); $('#scale-output').textContent = `±${value.toFixed(2)}`;
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
addToyButton.addEventListener('click', () => request('insert-toy'));
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
