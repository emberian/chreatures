const MAX_SAMPLES = 600;
const SAMPLE_INTERVAL_MS = 100;
const READOUT_COLUMNS = 16;

const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const label = value => String(value).replaceAll('_', ' ').replace(/\b\w/g, match => match.toUpperCase());
const number = value => !finite(value) ? '—' : Math.abs(value) > 0 && Math.abs(value) < .001 ? value.toExponential(2) : value.toFixed(4);
const node = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};

function validRetina(value, rows, columns) {
  return Array.isArray(value) && value.length === rows && value.every(
    row => Array.isArray(row) && row.length === columns && row.every(
      pixel => Array.isArray(pixel) && pixel.length >= 4 && pixel.slice(0, 4).every(finite)
    )
  );
}

function readoutGroup(name) {
  const match = /^readout\/([^/]+)\//.exec(name);
  return match ? match[1] : 'other';
}

function traceValue(state, resident) {
  const body = state.bodies?.find(item => item.id === resident);
  const outcome = state.outcomes?.[resident];
  if (!body) return null;
  return {
    time: state.time,
    speed: finite(body.speed) ? body.speed : null,
    effort: finite(outcome?.effort) ? outcome.effort : null,
    energy: finite(body.energy) ? body.energy : null,
  };
}

export class LiveInspection {
  constructor(root, fetcher = (...args) => globalThis.fetch(...args)) {
    this.root = root;
    this.fetcher = fetcher;
    this.buffers = new Map();
    this.lastSampleAt = 0;
    this.lastSampleTime = new Map();
    this.lastRenderAt = 0;
    this.worldId = null;
    this.selected = null;
    this.state = null;
    this.metadata = null;
    this.metadataError = null;
    this.metadataRequested = false;
    this.populationCanvases = new Map();
    this.build();
  }

  build() {
    this.root.innerHTML = `
      <details class="inspection-shell">
        <summary><span>Live evidence inspector</span><small data-inspection-summary>state samples</small></summary>
        <div class="inspection-intro">
          <strong data-inspection-resident>Selected resident</strong>
          <span>Live values come from the world state. Traces remain only in this browser.</span>
        </div>
        <div class="inspection-grid">
          <section class="inspection-card retina-inspection">
            <header><h3>Body-bound vision</h3><span>1,024 physical rays · RGB + proximity</span></header>
            <div class="retina-pair" data-retina-rasters></div>
            <p class="inspection-note">Peripheral: 200° × 100°. Central: 60° × 44°. Each raster is shown at its own angular scale. Both come from collision rays at the physical head.</p>
            <output class="inspection-hover" data-retina-hover>Hover a ray for its four delivered values.</output>
          </section>
          <section class="inspection-card sensor-inspection">
            <header><h3>Delivered sensor channels</h3><span data-sensor-count>—</span></header>
            <div data-sensors></div>
            <output class="inspection-hover" data-sensor-hover>Hover a channel for its name and value.</output>
          </section>
          <section class="inspection-card population-inspection">
            <header><h3>Current population readouts</h3><span data-population-scale>metadata loading</span></header>
            <p class="inspection-note">These are current simulated population readouts named by measured connectome ports. They are not individual-neuron recordings.</p>
            <div data-populations></div>
            <output class="inspection-hover" data-population-hover>Hover a cell for its real readout name and current value.</output>
          </section>
          <section class="inspection-card trace-inspection">
            <header><h3>Recent physical state</h3><span data-trace-window>browser buffer</span></header>
            <canvas data-traces width="640" height="210"></canvas>
            <output class="inspection-hover" data-trace-hover>Hover the trace for model time, speed, measured effort, and energy.</output>
          </section>
          <section class="inspection-card action-inspection">
            <header><h3>Currently executed motor action</h3><span data-held-ticks>—</span></header>
            <div data-actions></div>
          </section>
          <section class="inspection-card memory-inspection">
            <header><h3>An encounter remembered</h3><span>private achieved-history goals</span></header>
            <dl data-memory></dl>
            <p class="inspection-note">A goal refers to an experienced four-frame sensory window. Remembering it does not establish that it is reachable now.</p>
          </section>
        </div>
      </details>`;
    const physiology = this.root.parentElement?.querySelector('.physiology');
    if (physiology) this.root.parentElement.insertBefore(this.root, physiology);
    this.details = this.root.querySelector('details');
    this.details.addEventListener('toggle', () => {
      document.body.classList.toggle('inspection-open', this.details.open);
      if (this.details.open && this.state) this.render(true);
    });
    this.retinaCanvases = [];
    const rasterHost = this.root.querySelector('[data-retina-rasters]');
    for (const [raster, rows, description] of [['peripheral', 8, 'Peripheral'], ['foveal', 24, 'Central']]) {
      for (const component of ['color', 'depth']) {
        const figure = node('figure');
        const canvas = node('canvas');
        canvas.width = 320;
        canvas.height = rows * 10;
        canvas.dataset.raster = raster;
        canvas.dataset.component = component;
        canvas.dataset.rows = rows;
        figure.append(canvas, node('figcaption', '', `${description} ${rows} × 32 · ${component === 'color' ? 'RGB after illumination' : 'proximity: 0 far, 1 near'}`));
        rasterHost.append(figure);
        this.retinaCanvases.push(canvas);
      }
    }
    this.traceCanvas = this.root.querySelector('[data-traces]');
    for (const canvas of this.retinaCanvases) {
      canvas.addEventListener('pointermove', event => this.retinaHover(event));
      canvas.addEventListener('pointerleave', () => {
        this.root.querySelector('[data-retina-hover]').textContent = 'Hover a ray for its four delivered values.';
      });
    }
    this.traceCanvas.addEventListener('pointermove', event => this.traceHover(event));
    this.traceCanvas.addEventListener('pointerleave', () => {
      this.root.querySelector('[data-trace-hover]').textContent = 'Hover the trace for model time, speed, measured effort, and energy.';
    });
  }

  requestMetadata() {
    if (this.metadataRequested) return;
    this.metadataRequested = true;
    this.fetcher('/api/connectome', {headers: {Accept: 'application/json'}})
      .then(response => {
        if (!response.ok) throw Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then(value => {
        if (!Array.isArray(value.readouts) || !value.readouts.every(name => typeof name === 'string')) {
          throw Error('connectome readout names are unavailable');
        }
        this.metadata = value;
        if (this.details.open) this.render(true);
      })
      .catch(error => {
        this.metadataError = error.message;
        if (this.details.open) this.render(true);
      });
  }

  update(state, resident) {
    const worldChanged = this.worldId !== null && state.id !== this.worldId;
    if (worldChanged) {
      this.buffers.clear();
      this.lastSampleTime.clear();
      this.lastSampleAt = 0;
    }
    this.worldId = state.id;
    this.state = state;
    const selectionChanged = resident !== this.selected;
    this.selected = resident;
    this.requestMetadata();
    const now = performance.now();
    const sample = traceValue(state, resident);
    const previousTime = this.lastSampleTime.get(resident);
    if (sample && finite(sample.time) && sample.time !== previousTime && (worldChanged || selectionChanged || now - this.lastSampleAt >= SAMPLE_INTERVAL_MS)) {
      const buffer = this.buffers.get(resident) || [];
      buffer.push(sample);
      if (buffer.length > MAX_SAMPLES) buffer.splice(0, buffer.length - MAX_SAMPLES);
      this.buffers.set(resident, buffer);
      this.lastSampleTime.set(resident, sample.time);
      this.lastSampleAt = now;
    }
    const body = state.bodies?.find(item => item.id === resident);
    const features = state.neural?.[resident]?.features;
    this.root.querySelector('[data-inspection-summary]').textContent = `${Array.isArray(features) ? features.length : 0} population readouts · 10 Hz view`;
    this.root.querySelector('[data-inspection-resident]').textContent = body?.name || resident || 'Selected resident';
    if (this.details.open && (selectionChanged || now - this.lastRenderAt >= SAMPLE_INTERVAL_MS)) this.render(selectionChanged);
  }

  render(force = false) {
    if (!this.state || !this.selected) return;
    this.lastRenderAt = performance.now();
    const senses = this.state.senses?.[this.selected] || {};
    const cognition = this.state.cognition?.[this.selected] || {};
    this.drawRetina(senses.rich_retina);
    this.drawSensors(senses);
    this.drawPopulations(this.state.neural?.[this.selected]?.features, force);
    this.drawTraces();
    this.drawActions(cognition);
    this.drawMemory(cognition);
  }

  drawRetina(retina) {
    this.currentRetina = retina;
    for (const canvas of this.retinaCanvases) {
      const rows = Number(canvas.dataset.rows), raster = retina?.[canvas.dataset.raster];
      const available = validRetina(raster, rows, 32);
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, canvas.width, canvas.height);
      canvas.classList.toggle('unavailable', !available);
      if (!available) continue;
      raster.forEach((row, elevation) => row.forEach((pixel, azimuth) => {
        const channels = canvas.dataset.component === 'color' ? pixel.slice(0, 3) : [pixel[3], pixel[3], pixel[3]];
        context.fillStyle = `rgb(${channels.map(value => Math.round(clamp(value, 0, 1) * 255)).join(',')})`;
        context.fillRect(azimuth * 10, (rows - 1 - elevation) * 10, 10, 10);
      }));
    }
  }

  retinaHover(event) {
    if (!this.currentRetina) return;
    const canvas = event.currentTarget, rect = canvas.getBoundingClientRect();
    const rows = Number(canvas.dataset.rows), raster = this.currentRetina[canvas.dataset.raster];
    if (!validRetina(raster, rows, 32)) return;
    const column = clamp(Math.floor((event.clientX - rect.left) / rect.width * 32), 0, 31);
    const visualRow = clamp(Math.floor((event.clientY - rect.top) / rect.height * rows), 0, rows - 1);
    const elevation = rows - 1 - visualRow, pixel = raster[elevation][column];
    this.root.querySelector('[data-retina-hover]').textContent = `${canvas.dataset.raster} · elevation ${elevation + 1}, ray ${column + 1} · R ${number(pixel[0])} · G ${number(pixel[1])} · B ${number(pixel[2])} · proximity ${number(pixel[3])}`;
  }

  drawSensors(senses) {
    const host = this.root.querySelector('[data-sensors]');
    const hover = this.root.querySelector('[data-sensor-hover]');
    const channels = [];
    const ecologyNames = Array.isArray(this.state.ecology?.channels) ? this.state.ecology.channels : [];
    const add = (group, name, value) => { if (finite(value)) channels.push({group, name, value}); };
    if (Array.isArray(senses.odor)) senses.odor.forEach((antenna, side) => {
      if (Array.isArray(antenna)) antenna.forEach((value, index) => add('Odor', `${side === 0 ? 'left' : 'right'} antenna · ${ecologyNames[index] || `channel ${index + 1}`}`, value));
    });
    if (Array.isArray(senses.touch)) senses.touch.forEach((value, index) => add('Touch', `touch channel ${index + 1}`, value));
    if (Array.isArray(senses.tarsal_contact)) senses.tarsal_contact.forEach((value, index) => add('Tarsal contact', `tarsal contact ${index + 1}`, value));
    if (Array.isArray(senses.joint_position)) senses.joint_position.forEach((value, index) => add('Joint position', `joint ${index + 1} position`, value));
    if (Array.isArray(senses.joint_velocity)) senses.joint_velocity.forEach((value, index) => add('Joint velocity', `joint ${index + 1} velocity`, value));
    host.replaceChildren();
    for (const groupName of [...new Set(channels.map(item => item.group))]) {
      const items = channels.filter(item => item.group === groupName);
      const row = node('div', 'sensor-row');
      row.append(node('span', 'sensor-label', `${groupName} · ${items.length}`));
      const strip = node('div', 'sensor-strip');
      const peak = Math.max(.000001, ...items.map(item => Math.abs(item.value)));
      for (const item of items) {
        const cell = node('i', 'sensor-cell');
        const magnitude = clamp(Math.abs(item.value) / peak, 0, 1);
        cell.style.background = item.value < 0 ? `rgba(231,154,111,${.14 + magnitude * .82})` : `rgba(131,189,165,${.14 + magnitude * .82})`;
        cell.setAttribute('aria-label', `${item.name}: ${number(item.value)}`);
        cell.addEventListener('pointerenter', () => { hover.textContent = `${item.name} · ${number(item.value)}`; });
        strip.append(cell);
      }
      row.append(strip);
      host.append(row);
    }
    this.root.querySelector('[data-sensor-count]').textContent = `${channels.length} shown`;
  }

  metadataMatches(features) {
    return this.metadata && this.metadata.graph?.sha256 === this.state.anatomy?.sha256 && this.metadata.readouts.length === features.length;
  }

  drawPopulations(features, force) {
    const host = this.root.querySelector('[data-populations]');
    const scale = this.root.querySelector('[data-population-scale]');
    if (!Array.isArray(features) || !features.every(finite)) {
      host.textContent = 'Population values are unavailable in this state.';
      scale.textContent = 'unavailable';
      return;
    }
    if (!this.metadataMatches(features)) {
      host.textContent = this.metadataError ? `Readout metadata unavailable: ${this.metadataError}` : this.metadata ? 'Readout metadata does not match this world graph.' : 'Loading readout names…';
      scale.textContent = `${features.length} unnamed values`;
      return;
    }
    const peak = Math.max(.000001, ...features.map(value => Math.abs(value)));
    scale.textContent = `${features.length} readouts · current |peak| ${number(peak)}`;
    const groups = new Map();
    this.metadata.readouts.forEach((name, index) => {
      const group = readoutGroup(name);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push({name, value: features[index], index});
    });
    const signature = [...groups].map(([name, items]) => `${name}:${items.length}`).join('|');
    if (force || host.dataset.signature !== signature) {
      host.dataset.signature = signature;
      host.replaceChildren();
      this.populationCanvases.clear();
      for (const [group, items] of groups) {
        const block = node('div', 'population-group');
        const heading = node('div', 'population-heading');
        heading.append(node('strong', '', label(group)), node('span', '', `${items.length} readouts`));
        const canvas = node('canvas', 'population-canvas');
        canvas.width = 512;
        canvas.height = Math.ceil(items.length / READOUT_COLUMNS) * 12;
        canvas.dataset.group = group;
        canvas.addEventListener('pointermove', event => this.populationHover(event));
        canvas.addEventListener('pointerleave', () => {
          this.root.querySelector('[data-population-hover]').textContent = 'Hover a cell for its real readout name and current value.';
        });
        block.append(heading, canvas);
        host.append(block);
        this.populationCanvases.set(group, canvas);
      }
    }
    for (const [group, items] of groups) {
      const canvas = this.populationCanvases.get(group);
      canvas._readouts = items;
      const context = canvas.getContext('2d');
      const cellWidth = canvas.width / READOUT_COLUMNS;
      context.clearRect(0, 0, canvas.width, canvas.height);
      items.forEach((item, index) => {
        const magnitude = clamp(Math.abs(item.value) / peak, 0, 1);
        context.fillStyle = item.value < 0 ? `rgb(${Math.round(74 + 181 * magnitude)},${Math.round(49 + 91 * magnitude)},${Math.round(47 + 54 * magnitude)})` : `rgb(${Math.round(20 + 111 * magnitude)},${Math.round(34 + 155 * magnitude)},${Math.round(40 + 125 * magnitude)})`;
        context.fillRect((index % READOUT_COLUMNS) * cellWidth + 1, Math.floor(index / READOUT_COLUMNS) * 12 + 1, cellWidth - 2, 10);
      });
    }
  }

  populationHover(event) {
    const canvas = event.currentTarget, rect = canvas.getBoundingClientRect();
    const column = clamp(Math.floor((event.clientX - rect.left) / rect.width * READOUT_COLUMNS), 0, READOUT_COLUMNS - 1);
    const row = Math.floor((event.clientY - rect.top) / rect.height * (canvas.height / 12));
    const item = canvas._readouts?.[row * READOUT_COLUMNS + column];
    if (item) this.root.querySelector('[data-population-hover]').textContent = `${item.name} · ${number(item.value)} · index ${item.index}`;
  }

  drawTraces() {
    const samples = this.buffers.get(this.selected) || [];
    const canvas = this.traceCanvas, context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#0b171b';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const lanes = [
      {key: 'speed', name: 'body speed', color: '#7fb9ce', fixed: false},
      {key: 'effort', name: 'measured effort', color: '#e4ad78', fixed: true},
      {key: 'energy', name: 'body energy', color: '#a9c878', fixed: true},
    ];
    const left = 112, right = 12, laneHeight = canvas.height / lanes.length;
    lanes.forEach((lane, laneIndex) => {
      const values = samples.map(sample => sample[lane.key]).filter(finite);
      const maximum = lane.fixed ? 1 : Math.max(.000001, ...values);
      const top = laneIndex * laneHeight;
      context.strokeStyle = '#ffffff12';
      context.beginPath(); context.moveTo(left, top + laneHeight - 12); context.lineTo(canvas.width - right, top + laneHeight - 12); context.stroke();
      context.fillStyle = '#8ba09b'; context.font = '18px system-ui'; context.fillText(lane.name, 10, top + 26);
      const current = values.at(-1);
      context.fillStyle = lane.color; context.font = '16px ui-monospace, monospace'; context.fillText(number(current), 10, top + 49);
      if (samples.length < 2) return;
      context.strokeStyle = lane.color; context.lineWidth = 2; context.beginPath();
      let started = false;
      samples.forEach((sample, index) => {
        if (!finite(sample[lane.key])) { started = false; return; }
        const x = left + index / Math.max(1, samples.length - 1) * (canvas.width - left - right);
        const y = top + laneHeight - 12 - clamp(sample[lane.key] / maximum, 0, 1) * (laneHeight - 24);
        if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
      });
      context.stroke();
    });
    this.traceSamples = samples;
    const duration = samples.length > 1 ? samples.at(-1).time - samples[0].time : 0;
    this.root.querySelector('[data-trace-window]').textContent = `${samples.length} samples · ${Math.max(0, duration).toFixed(1)} model s`;
  }

  traceHover(event) {
    if (!this.traceSamples?.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const index = clamp(Math.round((event.clientX - rect.left) / rect.width * (this.traceSamples.length - 1)), 0, this.traceSamples.length - 1);
    const sample = this.traceSamples[index];
    this.root.querySelector('[data-trace-hover]').textContent = `model ${sample.time.toFixed(2)} s · speed ${number(sample.speed)} · effort ${number(sample.effort)} · energy ${number(sample.energy)}`;
  }

  drawActions(cognition) {
    const host = this.root.querySelector('[data-actions]');
    const action = cognition.executed_action;
    host.replaceChildren();
    if (!action || typeof action !== 'object') {
      host.textContent = 'Executed action is unavailable for this controller.';
      this.root.querySelector('[data-held-ticks]').textContent = 'unavailable';
      return;
    }
    for (const [name, value] of Object.entries(action)) {
      if (!finite(value)) continue;
      const row = node('div', 'action-row');
      const nameNode = node('span', '', label(name));
      const track = node('i', 'action-track');
      const bar = node('b', value < 0 ? 'negative' : 'positive');
      bar.style.width = `${clamp(Math.abs(value), 0, 1) * 50}%`;
      bar.style.left = value < 0 ? `${50 - clamp(Math.abs(value), 0, 1) * 50}%` : '50%';
      track.append(bar);
      row.append(nameNode, track, node('output', '', number(value)));
      host.append(row);
    }
    this.root.querySelector('[data-held-ticks]').textContent = '20 Hz · last completed physical tick';
  }

  drawMemory(cognition) {
    const host = this.root.querySelector('[data-memory]');
    host.replaceChildren();
    const goal = cognition.goal || {};
    const refinement = cognition.consequence_refinement || {};
    const rows = [
      ['Stored sensory encounters', cognition.memory_count],
      ['Goal available', goal.valid === undefined ? undefined : goal.valid ? 'yes' : 'warming up'],
      ['Selected memory slot', goal.valid ? goal.slot : undefined],
      ['Encounter recorded at model seconds', goal.valid ? goal.recorded_time : undefined],
      ['Encounter tick', goal.valid ? goal.recorded_tick : undefined],
      ['Attempt time remaining (seconds)', goal.valid ? goal.remaining_ticks * .05 : undefined],
      ['Selection changed this tick', goal.changed === undefined ? undefined : goal.changed ? 'yes' : 'no'],
      ['Private consequence updates before action', refinement.completed_private_updates_before_action],
      ['Selected motor proposal', finite(refinement.selected_candidate) ? refinement.selected_candidate + 1 : undefined],
      ['Proposals outside fitted domain', Array.isArray(refinement.candidate_out_of_domain) ? `${refinement.candidate_out_of_domain.filter(Boolean).length} / ${refinement.candidate_out_of_domain.length}` : undefined],
    ];
    for (const [name, value] of rows) {
      if (value === undefined || value === null || (typeof value === 'number' && !finite(value))) continue;
      host.append(node('dt', '', name), node('dd', '', typeof value === 'number' ? Number.isInteger(value) ? value.toLocaleString() : number(value) : String(value)));
    }
  }
}

export function mountInspection(root, options = {}) {
  if (!root) return null;
  return new LiveInspection(root, options.fetcher);
}
