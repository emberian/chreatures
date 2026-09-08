// SPDX-License-Identifier: AGPL-3.0-or-later

const EDGES = 4184;
const WIDTH = 256;
const HEIGHT = Math.ceil(EDGES / WIDTH);
const TARGET_EDGES = [2048, 2136];

function finiteField(value, label, low, high) {
  if (!(value instanceof Float32Array) || value.length !== EDGES) throw new Error(`${label} must be Float32Array[${EDGES}]`);
  for (let index = 0; index < value.length; index += 1) {
    if (!Number.isFinite(value[index]) || value[index] < low - 1e-6 || value[index] > high + 1e-6) {
      throw new Error(`${label} is outside its fixed model range`);
    }
  }
}

function heatmap(canvas, values, color) {
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext('2d');
  const image = context.createImageData(WIDTH, HEIGHT);
  for (let index = 0; index < values.length; index += 1) {
    const rgba = color(values[index]);
    image.data.set(rgba, index * 4);
  }
  context.putImageData(image, 0, 0);
}

function identityText(identity) {
  if (typeof identity === 'string' && identity) return identity;
  if (!identity || typeof identity !== 'object') return 'plasticity identity absent';
  return String(identity.format ?? identity.rule ?? identity.identity ?? 'private plasticity rule');
}

export class PlasticityPanel {
  constructor(root, requestInspection) {
    if (!(root instanceof HTMLElement)) throw new Error('Plasticity panel root is absent');
    this.root = root;
    this.requestInspection = requestInspection;
    this.button = root.querySelector('[data-plasticity-refresh]');
    this.status = root.querySelector('[data-plasticity-status]');
    this.time = root.querySelector('[data-plasticity-time]');
    this.identity = root.querySelector('[data-plasticity-identity]');
    this.efficacyCanvas = root.querySelector('[data-plasticity-efficacy]');
    this.eligibilityCanvas = root.querySelector('[data-plasticity-eligibility]');
    this.targets = [...root.querySelectorAll('[data-plasticity-target]')];
    if (!this.button || !this.status || !this.time || !this.identity || !this.efficacyCanvas ||
        !this.eligibilityCanvas || this.targets.length !== 2) throw new Error('Plasticity panel markup differs');
    this.resident = null;
    this.ready = false;
    this.pending = null;
    this.button.addEventListener('click', () => {
      if (!this.ready || !this.resident || this.pending) return;
      this.pending = this.requestInspection(this.resident);
      this.button.disabled = true;
      this.status.textContent = 'reading private GPU state…';
    });
    this.clear(null);
  }

  setReady(ready) {
    this.ready = Boolean(ready);
    this.button.disabled = !this.ready || !this.resident || Boolean(this.pending);
  }

  reject(requestId) {
    if (requestId !== this.pending) return;
    this.pending = null;
    this.button.disabled = !this.ready || !this.resident;
    this.status.textContent = 'readback failed · try again';
  }

  clear(resident) {
    this.resident = resident;
    this.pending = null;
    this.button.disabled = !this.ready || !resident;
    this.status.textContent = resident ? 'not read · refresh manually' : 'select a resident';
    this.time.textContent = '—';
    this.identity.textContent = 'Rule identity appears after a read.';
    for (const canvas of [this.efficacyCanvas, this.eligibilityCanvas]) {
      canvas.width = WIDTH; canvas.height = HEIGHT;
      canvas.getContext('2d').clearRect(0, 0, WIDTH, HEIGHT);
    }
    for (const target of this.targets) target.querySelector('output').textContent = '—';
  }

  accept(message, selectedResident) {
    if (message.requestId !== this.pending) return false;
    this.pending = null;
    this.button.disabled = !this.ready || !this.resident;
    if (!this.resident || message.residentId !== this.resident || message.residentId !== selectedResident) {
      this.status.textContent = 'selection changed · result ignored';
      return false;
    }
    const plasticity = message.plasticity;
    // The GPU field is a private batch slot; residentId above is the public ID.
    if (!plasticity || !Number.isInteger(plasticity.resident) || plasticity.resident < 0 || plasticity.resident >= 4) throw new Error('Plasticity resident slot differs');
    finiteField(plasticity.efficacy, 'efficacy deviation', -.8, 0);
    finiteField(plasticity.eligibility, 'eligibility', 0, 1);
    if (!Array.isArray(plasticity.targets) || plasticity.targets.length !== 2) throw new Error('Plasticity target summaries differ');
    plasticity.targets.forEach((target, index) => {
      if (Number(target.edgeCount) !== TARGET_EDGES[index]) throw new Error('Plasticity target edge partition differs');
      const numbers = [target.changedEdges, target.meanEfficacyDeviation, target.meanEligibility].map(Number);
      if (numbers.some(value => !Number.isFinite(value))) throw new Error('Plasticity target summary is nonfinite');
      if (numbers[0] < 0 || numbers[0] > TARGET_EDGES[index] || numbers[1] < -.8 - 1e-6 || numbers[1] > 1e-6 ||
          numbers[2] < -1e-6 || numbers[2] > 1 + 1e-6) throw new Error('Plasticity target summary is outside its fixed model range');
      this.targets[index].querySelector('output').textContent =
        `${numbers[0].toLocaleString()} changed · mean d ${numbers[1].toFixed(4)} · mean e ${numbers[2].toFixed(4)}`;
    });
    heatmap(this.efficacyCanvas, plasticity.efficacy, value => {
      const x = Math.max(0, Math.min(1, -value / .8));
      return [Math.round(235 - 32 * x), Math.round(239 - 141 * x), Math.round(225 - 151 * x), 255];
    });
    heatmap(this.eligibilityCanvas, plasticity.eligibility, value => {
      const x = Math.max(0, Math.min(1, value));
      return [Math.round(18 + 78 * x), Math.round(48 + 167 * x), Math.round(40 + 133 * x), 255];
    });
    this.time.textContent = `${Number(message.time).toFixed(2)} s`;
    this.identity.textContent = identityText(message.identity);
    this.status.textContent = `${EDGES.toLocaleString()} measured edges · fixed scales`;
    return true;
  }
}
