import {loadNeuronAnnotations, loadRetinalNeuronRows} from './neuron-annotations.js';

const text = (tag, value, className) => {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
};
const number = value => Number.isFinite(value) ? value.toExponential(3) : '—';

/** Observer-only metadata and sampled trace. Nothing is sent to the worker. */
export class NeuronInspector {
  constructor(root, selectPoint) {
    this.root = root;
    this.selectPoint = selectPoint;
    this.row = null;
    this.records = null;
    this.history = [];
    this.reference = null;
    this.scale = .01;
    this.resident = null;
    this.field = 'rate';
    this.query = root.querySelector('input');
    this.status = root.querySelector('[data-cell-status]');
    this.results = root.querySelector('[data-cell-results]');
    this.details = root.querySelector('[data-cell-details]');
    this.readout = root.querySelector('[data-cell-value]');
    this.caption = root.querySelector('[data-cell-trace-caption]');
    this.canvas = root.querySelector('canvas');
    this.context = this.canvas.getContext('2d');
    root.querySelector('form').addEventListener('submit', event => {
      event.preventDefault();
      void this.search();
    });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  async load() {
    if (!this.loading) this.loading = (async () => {
      this.status.textContent = 'Loading anatomical annotations…';
      const response = await fetch(new URL('./model/cns-manifest.json', import.meta.url));
      if (!response.ok) throw new Error(`model identity HTTP ${response.status}`);
      const model = await response.json();
      [this.records, this.retinal] = await Promise.all([
        loadNeuronAnnotations({
          baseURL: new URL('./assets/observer/', import.meta.url),
          graphSha256: model.identity?.graph,
          atlasSha256: model.identity?.anatomy,
        }),
        loadRetinalNeuronRows({modelManifest: model, baseURL: new URL('./model/', import.meta.url)}),
      ]);
      this.status.textContent = 'Click a CNS point, or search a body ID, cell type or class.';
      if (this.row !== null) this.describe();
    })().catch(error => {
      this.status.textContent = `Annotations unavailable: ${error.message}. The life continues.`;
    });
    await this.loading;
  }

  async search() {
    await this.load();
    if (!this.records) return;
    const query = this.query.value.trim();
    this.results.replaceChildren();
    if (!query) return;
    const found = this.records.find(query, 16);
    this.status.textContent = found.length ? `${found.length} matching rows shown. Select one to inspect it.` : 'No matching annotated row.';
    for (const record of found) {
      const button = text('button', `${record.annotations.type || record.annotations.class || 'Unlabeled cell'} · ${record.bodyId}`);
      button.type = 'button';
      button.addEventListener('click', () => this.select(record.row));
      this.results.append(button);
    }
  }

  select(row) {
    if (!Number.isInteger(row) || row < 0 || row >= 165122) return;
    this.row = row;
    this.history = [];
    this.readout.textContent = 'Waiting for the next sampled state';
    this.positioned = this.selectPoint(row);
    this.root.open = true;
    this.describe();
    this.draw();
    void this.load();
  }

  describe() {
    this.details.replaceChildren();
    if (this.row === null) return;
    const record = this.records?.get(this.row);
    this.details.append(text('h3', record?.annotations.type || `Canonical row ${this.row}`));
    if (!record) return;
    const list = document.createElement('dl');
    const a = record.annotations, nt = record.neurotransmitter, routes = record.routes;
    const fields = [
      ['Body ID / row', `${record.bodyId} / ${record.row}`],
      ['Class', [a.superclass, a.class, a.subclass].filter(Boolean).join(' / ') || 'Unannotated'],
      ['Type / MANC type', [a.type, a.mancType].filter(Boolean).join(' / ') || 'Unannotated'],
      ['Side', a.side || 'Unannotated'],
      ['Entry / exit nerve', `${a.entryNerve || 'Unannotated'} / ${a.exitNerve || 'Unannotated'}`],
      ['Predicted transmitter', `${nt.predicted || 'Unannotated'}${nt.predictionConfidence === null ? '' : ` · confidence ${nt.predictionConfidence.toFixed(3)}`}`],
      ['Model transmitter', `${nt.effective || 'Unresolved'} · ${nt.basis || 'basis unspecified'}`],
      ['Current routing', [this.retinal?.supported.has(record.row) && 'Retinal afferent', this.retinal?.rows.has(record.row) && !this.retinal?.supported.has(record.row) && 'Retinal receptor: no supported input sites', routes.body807 && 'BODY807 afferent', routes.descendingContext && 'descending context', routes.motor92 && 'MOTOR92 readout', routes.sensoryAtlas && !routes.body807 && 'sensory atlas: no supported BODY807 input', routes.motorAnnotation && !routes.motor92 && 'motor neuron: no supported MOTOR92 output'].filter(Boolean).join('; ') || 'No direct sensory/context/motor port'],
      ['Soma position', this.positioned ? 'Supplied anatomical coordinate' : 'No supplied soma; trace remains available'],
    ];
    for (const [label, value] of fields) list.append(text('dt', label), text('dd', value));
    this.details.append(list);
    for (const [name, members] of [['Sensory port membership', record.sensoryMemberships], ['Motor output membership', record.motorMemberships]]) {
      if (!members.length) continue;
      const group = document.createElement('details');
      group.append(text('summary', `${name} (${members.length})`));
      const list = document.createElement('ul');
      for (const member of members) list.append(text('li', `${member.name} · ${member.evidence}`));
      group.append(list); this.details.append(group);
    }
  }

  reset({resident, field, reference, scale}) {
    this.resident = resident;
    this.field = field;
    this.reference = reference;
    this.scale = scale;
    this.history = [];
    this.readout.textContent = 'Waiting for the next sampled state';
    this.draw();
  }

  setScale(scale) { this.scale = scale; this.draw(); }

  update(time, signal) {
    if (this.row === null || !Number.isFinite(time)) return;
    const value = signal[this.row], baseline = this.reference?.[this.row];
    if (!Number.isFinite(value) || !Number.isFinite(baseline)) return;
    // A restore or repeated paused frame starts no invented continuous history.
    if (this.history.length && time < this.history.at(-1).time) this.history = [];
    const point = {time, value, delta: value - baseline};
    if (this.history.at(-1)?.time === time) this.history[this.history.length - 1] = point;
    else this.history.push(point);
    if (this.history.length > 256) this.history.shift();
    this.readout.textContent = `${this.field}: ${number(value)} · reference ${number(baseline)} · Δ ${number(point.delta)}`;
    this.draw();
  }

  draw() {
    const width = Math.max(1, this.canvas.clientWidth), height = 100;
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(width * ratio); this.canvas.height = height * ratio;
    const c = this.context;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.fillStyle = '#0b1a16'; c.fillRect(0, 0, width, height);
    c.strokeStyle = '#52695c'; c.setLineDash([3, 4]);
    c.beginPath(); c.moveTo(0, 50); c.lineTo(width, 50); c.stroke(); c.setLineDash([]);
    c.fillStyle = '#c6d4c6'; c.font = '10px monospace';
    c.fillText(`+${number(this.scale)}`, 4, 12); c.fillText(`−${number(this.scale)}`, 4, 96);
    const points = this.history;
    if (!points.length) { this.caption.textContent = 'Selected field Δ · fixed display scale · awaiting samples'; return; }
    const first = points[0].time, last = points.at(-1).time, span = Math.max(.01, last - first);
    c.strokeStyle = '#edaa76'; c.lineWidth = 1.5; c.beginPath();
    points.forEach((p, index) => {
      const x = 6 + (width - 12) * (p.time - first) / span;
      const y = 50 - 36 * Math.max(-1, Math.min(1, p.delta / this.scale));
      if (index) c.lineTo(x, y); else c.moveTo(x, y);
    });
    c.stroke();
    const clipped = points.some(p => Math.abs(p.delta) > this.scale);
    this.caption.textContent = `${this.resident || 'Selected resident'} · ${first.toFixed(2)}–${last.toFixed(2)} s model time${clipped ? ' · trace clipped at fixed color scale' : ''}`;
  }
}
