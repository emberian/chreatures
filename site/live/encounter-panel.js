// SPDX-License-Identifier: AGPL-3.0-or-later

const TICKS_PER_SECOND = 100;
const finite = (value, label, low, high) => {
  const number = Number(value);
  if (!Number.isFinite(number) || number < low || number > high) throw new Error(`${label} must be ${low} to ${high}`);
  return number;
};
const rgb = value => {
  if (!/^#[0-9a-f]{6}$/i.test(value)) throw new Error('Screen color is invalid');
  return [1, 3, 5].map(offset => parseInt(value.slice(offset, offset + 2), 16) / 255);
};
const field = (label, input) => {
  const wrapper = document.createElement('label');
  wrapper.textContent = label; wrapper.append(input); return wrapper;
};
const input = (type, value, min, max, step) => {
  const element = document.createElement('input');
  element.type = type; element.value = value;
  if (min !== undefined) element.min = min;
  if (max !== undefined) element.max = max;
  if (step !== undefined) element.step = step;
  return element;
};
const exactKeys = (value, keys, label) => {
  const actual = Object.keys(value).sort(), expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) throw new Error(`${label} fields differ from the native schema`);
};

function validateEvent(step, index) {
  if (!step || typeof step !== 'object' || !Number.isSafeInteger(step.offset_ticks) || step.offset_ticks < 0 ||
      !step.event || typeof step.event !== 'object') throw new Error(`Step ${index + 1} has an invalid model-time event`);
  exactKeys(step, ['offset_ticks', 'event'], `Step ${index + 1}`);
  const event = step.event;
  if (event.kind === 'tone') {
    exactKeys(event, ['kind','position_mm','frequency_hz','amplitude','duration_s'], `Step ${index + 1} tone`);
    if (!Array.isArray(event.position_mm) || event.position_mm.length !== 3) throw new Error(`Step ${index + 1} tone position differs`);
    event.position_mm.forEach((value, axis) => finite(value, `Tone position ${'XYZ'[axis]}`, -10000, 10000));
    finite(event.frequency_hz, 'Tone frequency', 40, 1600); finite(event.amplitude, 'Tone amplitude', 0, 1); finite(event.duration_s, 'Tone duration', .01, 5);
  } else if (event.kind === 'toy_force') {
    exactKeys(event, ['kind','entity_id','force'], `Step ${index + 1} toy force`);
    if (typeof event.entity_id !== 'string' || !event.entity_id || event.entity_id.length > 256 || /[\u0000-\u001f\u007f]/.test(event.entity_id) ||
        !Array.isArray(event.force) || event.force.length !== 3) throw new Error(`Step ${index + 1} toy force differs`);
    const force = event.force.map((value, axis) => finite(value, `Push ${'XYZ'[axis]}`, -60, 60));
    if (Math.hypot(...force) > 60) throw new Error(`Step ${index + 1} push magnitude exceeds 60`);
  } else if (event.kind === 'pattern') {
    exactKeys(event, ['kind','pattern'], `Step ${index + 1} pattern event`);
    const pattern = event.pattern;
    if (!pattern || !['blank','uniform','grating'].includes(pattern.mode)) throw new Error(`Step ${index + 1} screen pattern differs`);
    exactKeys(pattern, pattern.mode === 'blank' ? ['mode'] : pattern.mode === 'uniform' ? ['mode','rgb'] :
      ['mode','low_rgb','high_rgb','phase_radians','orientation_radians','spatial_cycles'], `Step ${index + 1} screen pattern`);
    const colors = pattern.mode === 'uniform' ? [pattern.rgb] : pattern.mode === 'grating' ? [pattern.low_rgb, pattern.high_rgb] : [];
    for (const color of colors) {
      if (!Array.isArray(color) || color.length !== 3) throw new Error(`Step ${index + 1} screen color differs`);
      color.forEach(value => finite(value, 'Screen color channel', 0, 1));
    }
    if (pattern.mode === 'grating') {
      finite(pattern.phase_radians, 'Grating phase', -2 * Math.PI, 2 * Math.PI);
      finite(pattern.orientation_radians, 'Grating orientation', -Math.PI, Math.PI);
      finite(pattern.spatial_cycles, 'Grating cycles', .1, 64);
    }
  } else throw new Error(`Step ${index + 1} has an unknown event kind`);
}

export class EncounterPanel {
  constructor(root, {submit, onScreenProgram}) {
    this.root = root;
    this.submit = submit;
    this.onScreenProgram = onScreenProgram;
    this.rows = root.querySelector('[data-encounter-steps]');
    this.addKind = root.querySelector('[data-encounter-kind]');
    this.addButton = root.querySelector('[data-encounter-add]');
    this.sendButton = root.querySelector('[data-encounter-send]');
    this.status = root.querySelector('[data-encounter-status]');
    this.nativeStatus = root.querySelector('[data-encounter-native-status]');
    if (!this.rows || !this.addKind || !this.addButton || !this.sendButton || !this.status || !this.nativeStatus) {
      throw new Error('Encounter composer markup differs');
    }
    this.steps = [];
    this.nextId = 0;
    this.ready = false;
    this.pendingRequest = null;
    this.selectedToy = null;
    this.addButton.addEventListener('click', () => this.add(this.addKind.value));
    this.sendButton.addEventListener('click', () => this.send());
    this.add('tone');
  }

  setReady(ready) {
    this.ready = Boolean(ready);
    this.#buttons();
  }

  setSelectedToy(id) {
    const selected = id || null;
    if (selected === this.selectedToy) return;
    this.selectedToy = selected;
    this.#render();
  }

  add(kind) {
    if (!['tone', 'screen', 'push'].includes(kind) || this.steps.length >= 12) return;
    const step = {id: ++this.nextId, kind, delay: .2};
    if (kind === 'tone') Object.assign(step, {frequency: 250, amplitude: .65, duration: .2});
    if (kind === 'screen') Object.assign(step, {mode: 'blank', uniform: '#d9e5cf', low: '#101614', high: '#eef1df', phase: 0, orientation: 0, cycles: 8});
    if (kind === 'push') Object.assign(step, {force: [0, 4, 1]});
    if (this.steps.length === 0) step.delay = 0;
    this.steps.push(step); this.#render();
  }

  clear() {
    this.pendingRequest = null;
    this.status.textContent = 'Compose external events; native model time schedules delivery.';
    this.#buttons();
  }

  sendProgram(events) {
    if (!this.ready || this.pendingRequest) return null;
    if (!Array.isArray(events) || !events.length) throw new Error('Encounter program is empty');
    if (events.length > 12) throw new Error('Encounter programs contain at most twelve events');
    events.forEach(validateEvent);
    const containsPattern = events.some(step => step.event.kind === 'pattern');
    if (containsPattern) this.onScreenProgram();
    this.pendingRequest = this.submit({events});
    this.status.textContent = 'Validating and scheduling in the native world…';
    this.#buttons();
    return this.pendingRequest;
  }

  send() {
    try {
      this.sendProgram(this.#program());
    } catch (error) {
      this.status.textContent = error.message;
    }
  }

  scheduled(message) {
    if (message.requestId !== this.pendingRequest) return false;
    this.pendingRequest = null;
    const receipt = message.receipt;
    if (!receipt || !Number.isInteger(receipt.event_count) || !Number.isInteger(receipt.scheduled_at_tick)) throw new Error('Native encounter receipt differs');
    this.status.textContent = `${receipt.event_count} event${receipt.event_count === 1 ? '' : 's'} accepted at ${(receipt.scheduled_at_tick / TICKS_PER_SECOND).toFixed(2)} s model time.`;
    this.updateStatus(message.status);
    this.#buttons();
    return true;
  }

  reject(requestId) {
    if (requestId !== this.pendingRequest) return;
    this.pendingRequest = null;
    this.status.textContent = 'The native world rejected this program. Edit it and send again.';
    this.#buttons();
  }

  updateStatus(value) {
    if (!value) return;
    const current = Number(value.current_tick), pending = Number(value.pending_events), completed = Number(value.completed_events);
    if (![current, pending, completed].every(Number.isInteger) || [current, pending, completed].some(number => number < 0)) {
      throw new Error('Native encounter status differs');
    }
    const next = value.next_event_tick === null || value.next_event_tick === undefined ? 'none' : `${(Number(value.next_event_tick) / TICKS_PER_SECOND).toFixed(2)} s`;
    this.nativeStatus.textContent = `native ${(current / TICKS_PER_SECOND).toFixed(2)} s · ${pending} pending · ${completed} delivered · next ${next}`;
  }

  #program() {
    let offset = 0;
    return this.steps.map((step, index) => {
      const delay = finite(step.delay, `Step ${index + 1} delay`, 0, 300);
      const ticks = Math.round(delay * TICKS_PER_SECOND);
      if (Math.abs(delay * TICKS_PER_SECOND - ticks) > 1e-6) throw new Error(`Step ${index + 1} delay must use 0.01-second increments`);
      offset += ticks;
      if (step.kind === 'tone') return {offset_ticks: offset, event: {kind: 'tone', position_mm: [0, 0, 2],
        frequency_hz: finite(step.frequency, 'Tone frequency', 40, 1600), amplitude: finite(step.amplitude, 'Tone amplitude', 0, 1),
        duration_s: finite(step.duration, 'Tone duration', .01, 5)}};
      if (step.kind === 'push') {
        if (!this.selectedToy) throw new Error('Select an existing toy before scheduling a push');
        return {offset_ticks: offset, event: {kind: 'toy_force', entity_id: this.selectedToy,
          force: step.force.map((value, axis) => finite(value, `Push ${'XYZ'[axis]}`, -60, 60))}};
      }
      const pattern = {mode: step.mode};
      if (step.mode === 'uniform') pattern.rgb = rgb(step.uniform);
      else if (step.mode === 'grating') Object.assign(pattern, {low_rgb: rgb(step.low), high_rgb: rgb(step.high),
        phase_radians: finite(step.phase, 'Grating phase', -2 * Math.PI, 2 * Math.PI),
        orientation_radians: finite(step.orientation, 'Grating orientation', -Math.PI, Math.PI),
        spatial_cycles: finite(step.cycles, 'Grating cycles', .1, 64)});
      else if (step.mode !== 'blank') throw new Error('Unknown screen pattern');
      return {offset_ticks: offset, event: {kind: 'pattern', pattern}};
    });
  }

  #render() {
    this.rows.replaceChildren(...this.steps.map((step, index) => {
      const row = document.createElement('fieldset'); row.className = 'encounter-step';
      const legend = document.createElement('legend'); legend.textContent = `${index + 1} · ${step.kind === 'screen' ? 'static screen' : step.kind}`;
      const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove'; remove.disabled = this.steps.length === 1;
      remove.addEventListener('click', () => {this.steps = this.steps.filter(item => item.id !== step.id); this.#render();});
      legend.append(remove); row.append(legend);
      const delay = input('number', step.delay, 0, 300, .01); delay.addEventListener('input', () => {step.delay = delay.value;});
      row.append(field(index ? 'Wait after prior step (model s)' : 'Start after (model s)', delay));
      if (step.kind === 'tone') {
        for (const [label, key, low, high, increment] of [['Frequency (Hz)','frequency',40,1600,1],['Amplitude','amplitude',0,1,.01],['Duration (s)','duration',.01,5,.01]]) {
          const control = input('number', step[key], low, high, increment); control.addEventListener('input', () => {step[key] = control.value;}); row.append(field(label, control));
        }
      } else if (step.kind === 'push') {
        const target = document.createElement('p'); target.className = 'encounter-target'; target.textContent = this.selectedToy ? `Selected toy: ${this.selectedToy}` : 'Select a toy in the world first'; row.append(target);
        step.force.forEach((value, axis) => {const control = input('number', value, -60, 60, .1); control.addEventListener('input', () => {step.force[axis] = control.value;}); row.append(field(`Force ${'XYZ'[axis]}`, control));});
      } else {
        const mode = document.createElement('select');
        for (const value of ['blank','uniform','grating']) {const option = document.createElement('option'); option.value = value; option.textContent = value; option.selected = value === step.mode; mode.append(option);}
        mode.addEventListener('change', () => {step.mode = mode.value; this.#render();}); row.append(field('Pattern', mode));
        const colorControl = (label, key) => {const control = input('color', step[key]); control.addEventListener('input', () => {step[key] = control.value;}); row.append(field(label, control));};
        if (step.mode === 'uniform') colorControl('Color', 'uniform');
        if (step.mode === 'grating') {
          colorControl('Low color', 'low'); colorControl('High color', 'high');
          for (const [label,key,low,high,increment] of [['Phase (rad)','phase',-6.283,6.283,.01],['Orientation (rad)','orientation',-3.141,3.141,.01],['Spatial cycles','cycles',.1,64,.1]]) {
            const control = input('number', step[key], low, high, increment); control.addEventListener('input', () => {step[key] = control.value;}); row.append(field(label, control));
          }
        }
      }
      return row;
    }));
    this.#buttons();
  }

  #buttons() {
    this.addButton.disabled = !this.ready || this.steps.length >= 12 || Boolean(this.pendingRequest);
    this.sendButton.disabled = !this.ready || !this.steps.length || Boolean(this.pendingRequest);
  }
}
