// Observer-only MaleCNS annotations. This module has no controller or DOM imports.

const MANIFEST_NAME = 'male-cns-observer-annotations-v1.manifest.json';
const MANIFEST_FORMAT = 'chreatures-male-cns-observer-annotations-v1-manifest';
const PAYLOAD_FORMAT = 'chreatures-male-cns-observer-annotations-v1';
const ROWS = 165122;
const SHA256 = /^[0-9a-f]{64}$/;
const COLUMN_KEYS = Object.freeze([
  'superclasses', 'classes', 'subclasses', 'types', 'manc_types', 'sides',
  'entry_nerves', 'exit_nerves', 'predicted_nt', 'effective_nt', 'nt_basis',
]);
const SEARCH_COLUMN_KEYS = Object.freeze([
  'types', 'manc_types', 'classes', 'subclasses', 'superclasses',
]);
const FLAG_BITS = Object.freeze({
  sensory_annotation: 0,
  sensory_atlas_row: 1,
  descending_context_row: 2,
  motor_annotation: 3,
  motor92_routed: 4,
  body807_routed: 5,
});

function fail(message) {
  throw new Error(`Neuron annotations: ${message}`);
}

function requireSha256(value, label) {
  if (typeof value !== 'string' || !SHA256.test(value)) fail(`${label} must be a lowercase SHA-256`);
  return value;
}

async function sha256(bytes) {
  if (!globalThis.crypto?.subtle) fail('Web Crypto SHA-256 is unavailable');
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, value => value.toString(16).padStart(2, '0')).join('');
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) fail(`${label} download failed with HTTP ${response.status}`);
  return response.arrayBuffer();
}

async function fetchJSON(url, label) {
  const bytes = await fetchBytes(url, label);
  try {
    return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  } catch (error) {
    fail(`${label} is not valid UTF-8 JSON (${error.message})`);
  }
}

async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'function') fail('gzip DecompressionStream is unavailable');
  const body = new Response(bytes).body;
  if (body === null) fail('gzip response stream is unavailable');
  return new Response(body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

function base64Bytes(value, label) {
  if (typeof value !== 'string' || value.length % 4 !== 0 || typeof atob !== 'function') {
    fail(`${label} is not base64`);
  }
  let binary;
  try {
    binary = atob(value);
  } catch {
    fail(`${label} is not base64`);
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function decodeUnsigned(value, dtype, count, label) {
  const width = dtype === 'u1' ? 1 : dtype === 'u2' ? 2 : dtype === 'u4' ? 4 : 0;
  if (width === 0) fail(`${label} has unsupported dtype ${dtype}`);
  const bytes = base64Bytes(value, label);
  if (bytes.byteLength !== count * width) fail(`${label} byte count differs`);
  if (width === 1) return bytes;
  const output = width === 2 ? new Uint16Array(count) : new Uint32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) {
    output[index] = width === 2
      ? view.getUint16(index * width, true)
      : view.getUint32(index * width, true);
  }
  return output;
}

function decodeFloat32(value, count, label) {
  const bytes = base64Bytes(value, label);
  if (bytes.byteLength !== count * 4) fail(`${label} byte count differs`);
  const output = new Float32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) output[index] = view.getFloat32(index * 4, true);
  return output;
}

function decodeInt64(value, count, label) {
  const bytes = base64Bytes(value, label);
  if (bytes.byteLength !== count * 8) fail(`${label} byte count differs`);
  const output = new BigInt64Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) output[index] = view.getBigInt64(index * 8, true);
  return output;
}

function decodeColumn(specification, rows, label) {
  if (!specification || !Array.isArray(specification.dictionary)) fail(`${label} dictionary is missing`);
  if (!specification.dictionary.every(value => typeof value === 'string')) fail(`${label} dictionary differs`);
  const codes = decodeUnsigned(specification.codes, specification.codes_dtype, rows, `${label} codes`);
  for (const code of codes) {
    if (code >= specification.dictionary.length) fail(`${label} code is outside its dictionary`);
  }
  const missing = specification.missing_code;
  if (missing !== null && (!Number.isInteger(missing) || specification.dictionary[missing] !== '')) {
    fail(`${label} missing code differs`);
  }
  return Object.freeze({dictionary: Object.freeze(specification.dictionary.slice()), codes, missing});
}

function decodeCSR(specification, rows, manifestCount, label, descriptors) {
  if (!specification || !descriptors.every(key => Array.isArray(specification[key]))) {
    fail(`${label} descriptors are missing`);
  }
  const descriptorCount = specification[descriptors[0]].length;
  if (!descriptors.every(key => specification[key].length === descriptorCount)) {
    fail(`${label} descriptor lengths differ`);
  }
  const rowPtr = decodeUnsigned(specification.row_ptr, 'u4', rows + 1, `${label} row_ptr`);
  if (rowPtr[0] !== 0) fail(`${label} row_ptr must begin at zero`);
  for (let index = 1; index < rowPtr.length; index += 1) {
    if (rowPtr[index] < rowPtr[index - 1]) fail(`${label} row_ptr is not monotonic`);
  }
  const membershipCount = rowPtr[rows];
  if (membershipCount !== manifestCount) fail(`${label} membership count differs`);
  const indices = decodeUnsigned(specification.indices, 'u2', membershipCount, `${label} indices`);
  for (const index of indices) {
    if (index >= descriptorCount) fail(`${label} index is outside its descriptor table`);
  }
  const tables = {};
  for (const key of descriptors) tables[key] = Object.freeze(specification[key].slice());
  return Object.freeze({rowPtr, indices, ...tables});
}

function checkedRow(row) {
  if (!Number.isInteger(row) || row < 0 || row >= ROWS) {
    throw new RangeError(`Neuron annotation row must be an integer in 0..${ROWS - 1}`);
  }
  return row;
}

function binarySearchBodyId(bodyIds, target) {
  let low = 0;
  let high = bodyIds.length;
  while (low < high) {
    const middle = low + ((high - low) >> 1);
    if (bodyIds[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low < bodyIds.length && bodyIds[low] === target ? low : -1;
}

function frozenMemberships(row, csr, kind) {
  const result = [];
  for (let position = csr.rowPtr[row]; position < csr.rowPtr[row + 1]; position += 1) {
    const index = csr.indices[position];
    result.push(Object.freeze(kind === 'sensory'
      ? {index, name: csr.names[index], modality: csr.modalities[index], evidence: csr.evidence_grades[index]}
      : {index, name: csr.names[index], target: csr.targets[index], group: csr.groups[index], evidence: csr.evidence_grades[index]}));
  }
  return Object.freeze(result);
}

function buildSearchLists(columns) {
  const lists = [];
  for (const key of SEARCH_COLUMN_KEYS) {
    const column = columns[key];
    const rowsByCode = Array.from({length: column.dictionary.length}, () => []);
    for (let row = 0; row < ROWS; row += 1) rowsByCode[column.codes[row]].push(row);
    for (let code = 0; code < column.dictionary.length; code += 1) {
      const value = column.dictionary[code];
      if (value !== '') lists.push(Object.freeze({term: value.toLocaleLowerCase('en-US'), rows: rowsByCode[code]}));
    }
  }
  return Object.freeze(lists);
}

export async function loadRetinalNeuronRows({modelManifest, baseURL} = {}) {
  if (
    !modelManifest
    || modelManifest.format !== 'chreatures-cns-webgpu-v5'
    || modelManifest.version !== 5
  ) fail('selected CNS model manifest differs');
  if (baseURL === undefined || baseURL === null) fail('retinal row baseURL is required');
  const neuronCount = modelManifest.counts?.neurons;
  const receptorCount = modelManifest.counts?.receptors;
  const siteEdgeCount = modelManifest.counts?.receptorSiteEdges;
  if (
    neuronCount !== ROWS
    || receptorCount !== 4107
    || siteEdgeCount !== 4669
  ) fail('selected CNS neuron, receptor, or receptor-site count differs');
  const root = new URL(String(baseURL), import.meta.url);
  if (!root.pathname.endsWith('/')) root.pathname += '/';

  async function loadBuffer(key, url, dtype, count, label) {
    const entry = modelManifest.buffers?.[key];
    if (
      !entry
      || entry.url !== url
      || entry.dtype !== dtype
      || entry.encoding !== 'gzip'
      || !Array.isArray(entry.shape)
      || entry.shape.length !== 1
      || entry.shape[0] !== count
      || entry.byteLength !== count * 4
      || !Number.isInteger(entry.transportByteLength)
      || entry.transportByteLength <= 0
    ) fail(`${label} buffer contract differs`);
    requireSha256(entry.sha256, `${label} content identity`);
    requireSha256(entry.transportSha256, `${label} transport identity`);
    const transport = await fetchBytes(new URL(entry.url, root), label);
    if (transport.byteLength !== entry.transportByteLength) fail(`${label} transport byte count differs`);
    if (await sha256(transport) !== entry.transportSha256) fail(`${label} transport SHA-256 differs`);
    const unpacked = await gunzip(transport);
    if (unpacked.byteLength !== entry.byteLength) fail(`${label} byte count differs`);
    if (await sha256(unpacked) !== entry.sha256) fail(`${label} content SHA-256 differs`);
    return unpacked;
  }

  const [rowBytes, pointerBytes, weightBytes] = await Promise.all([
    loadBuffer('atlas.receptor_rows', 'atlas-receptor_rows.bin.gz', 'u32', receptorCount, 'retinal receptor rows'),
    loadBuffer('atlas.receptor_ptr', 'atlas-receptor_ptr.bin.gz', 'u32', receptorCount + 1, 'retinal receptor pointers'),
    loadBuffer('atlas.site_weight', 'atlas-site_weight.bin.gz', 'f32', siteEdgeCount, 'retinal site weights'),
  ]);
  const rowView = new DataView(rowBytes);
  const pointerView = new DataView(pointerBytes);
  const weightView = new DataView(weightBytes);
  const pointers = new Uint32Array(receptorCount + 1);
  for (let index = 0; index <= receptorCount; index += 1) {
    pointers[index] = pointerView.getUint32(index * 4, true);
    if (index > 0 && pointers[index] < pointers[index - 1]) fail('retinal receptor pointers are not monotonic');
  }
  if (pointers[0] !== 0 || pointers[receptorCount] !== siteEdgeCount) {
    fail('retinal receptor pointer boundary differs');
  }
  const rows = new Set();
  const supported = new Set();
  for (let index = 0; index < receptorCount; index += 1) {
    const row = rowView.getUint32(index * 4, true);
    if (row >= neuronCount) fail('retinal receptor row is outside the canonical graph');
    rows.add(row);
    let support = 0;
    for (let edge = pointers[index]; edge < pointers[index + 1]; edge += 1) {
      const weight = weightView.getFloat32(edge * 4, true);
      if (!Number.isFinite(weight) || weight < 0) fail('retinal site weight is not finite and nonnegative');
      support += weight;
    }
    if (support > 0) supported.add(row);
  }
  if (rows.size !== receptorCount) fail('retinal receptor rows are not unique');
  if (supported.size !== 3936) fail('supported retinal receptor count differs');
  return Object.freeze({rows, supported});
}

export async function loadNeuronAnnotations({baseURL, graphSha256, atlasSha256} = {}) {
  requireSha256(graphSha256, 'graph identity');
  requireSha256(atlasSha256, 'atlas identity');
  if (baseURL === undefined || baseURL === null) fail('baseURL is required');
  const root = new URL(String(baseURL), import.meta.url);
  if (!root.pathname.endsWith('/')) root.pathname += '/';
  const manifest = await fetchJSON(new URL(MANIFEST_NAME, root), 'manifest');
  if (manifest.format !== MANIFEST_FORMAT || manifest.version !== 1 || manifest.rows !== ROWS) {
    fail('manifest format, version, or row count differs');
  }
  if (manifest.graph_sha256 !== graphSha256) fail('manifest graph identity differs');
  if (manifest.sources?.fly_body_neural_atlas?.sha256 !== atlasSha256) fail('manifest atlas identity differs');
  const artifact = manifest.artifact;
  if (
    !artifact
    || artifact.filename !== 'male-cns-observer-annotations-v1.json.gz'
    || artifact.encoding !== 'gzip JSON; deterministic mtime=0'
    || !Number.isInteger(artifact.transport_bytes)
    || !Number.isInteger(artifact.uncompressed_bytes)
  ) fail('manifest artifact contract differs');
  requireSha256(artifact.transport_sha256, 'payload transport identity');
  const transport = await fetchBytes(new URL(artifact.filename, root), 'payload');
  if (transport.byteLength !== artifact.transport_bytes) fail('payload transport byte count differs');
  if (await sha256(transport) !== artifact.transport_sha256) fail('payload transport SHA-256 differs');
  const unpacked = await gunzip(transport);
  if (unpacked.byteLength !== artifact.uncompressed_bytes) fail('payload uncompressed byte count differs');
  let payload;
  try {
    payload = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(unpacked));
  } catch (error) {
    fail(`payload is not valid UTF-8 JSON (${error.message})`);
  }
  if (
    payload.format !== PAYLOAD_FORMAT
    || payload.version !== 1
    || payload.row_count !== ROWS
    || payload.graph_sha256 !== graphSha256
  ) fail('payload format, version, row count, or graph identity differs');
  if (payload.scope !== manifest.semantics?.information_boundary) fail('payload observer boundary differs');
  if (payload.body_ids?.dtype !== 'i8' || !String(payload.body_ids?.encoding).includes('little-endian')) {
    fail('body ID encoding differs');
  }
  const bodyIds = decodeInt64(payload.body_ids.data, ROWS, 'body IDs');
  for (let row = 1; row < ROWS; row += 1) {
    if (bodyIds[row] <= bodyIds[row - 1]) fail('body IDs are not strictly ascending');
  }
  const payloadColumnKeys = Object.keys(payload.columns || {}).sort();
  if (payloadColumnKeys.join('\0') !== COLUMN_KEYS.slice().sort().join('\0')) fail('annotation columns differ');
  const columns = {};
  for (const key of COLUMN_KEYS) columns[key] = decodeColumn(payload.columns[key], ROWS, key);
  if (payload.nt_confidence?.dtype !== 'f4') fail('NT confidence dtype differs');
  const confidence = decodeFloat32(payload.nt_confidence.data, ROWS, 'NT confidence');
  for (const value of confidence) {
    if (!Number.isFinite(value) || value < 0 || value > 1) fail('NT confidence is not finite in [0,1]');
  }
  if (payload.route_flags?.dtype !== 'u1') fail('route flag dtype differs');
  const payloadFlagBits = payload.route_flags.bits || {};
  if (
    Object.keys(payloadFlagBits).length !== Object.keys(FLAG_BITS).length
    || Object.entries(FLAG_BITS).some(([name, bit]) => payloadFlagBits[name] !== bit)
  ) fail('route flag definitions differ');
  const flags = decodeUnsigned(payload.route_flags.data, 'u1', ROWS, 'route flags');
  const counts = manifest.counts || {};
  const countFlags = Object.fromEntries(Object.keys(FLAG_BITS).map(name => [name, 0]));
  for (const value of flags) {
    if (value & ~0x3f) fail('route flags contain unknown bits');
    for (const [name, bit] of Object.entries(FLAG_BITS)) countFlags[name] += (value >> bit) & 1;
  }
  const expectedFlagCounts = {
    sensory_annotation: counts.sensory_annotation,
    sensory_atlas_row: counts.sensory_atlas_rows,
    descending_context_row: counts.descending_context_rows,
    motor_annotation: counts.motor_annotation,
    motor92_routed: counts.motor92_routed_rows,
    body807_routed: counts.body807_routed_rows,
  };
  for (const name of Object.keys(FLAG_BITS)) {
    if (countFlags[name] !== expectedFlagCounts[name]) fail(`${name} count differs`);
  }
  if (
    counts.body807_unsupported_sensory_rows !== counts.sensory_atlas_rows - counts.body807_routed_rows
    || counts.motor92_unsupported_rows !== counts.motor_annotation - counts.motor92_routed_rows
  ) fail('unsupported routing counts differ');
  const sensory = decodeCSR(
    payload.sensory_ports, ROWS, counts.sensory_port_memberships, 'sensory ports',
    ['names', 'modalities', 'evidence_grades'],
  );
  const motor = decodeCSR(
    payload.motor_outputs, ROWS, counts.motor_output_memberships, 'motor outputs',
    ['names', 'targets', 'groups', 'evidence_grades'],
  );
  const recordCache = new Map();
  let searchLists = null;

  function annotation(column, row) {
    const code = column.codes[row];
    return code === column.missing ? null : column.dictionary[code];
  }

  function get(row) {
    row = checkedRow(row);
    const cached = recordCache.get(row);
    if (cached !== undefined) return cached;
    const predicted = annotation(columns.predicted_nt, row);
    const route = flags[row];
    const record = Object.freeze({
      row,
      bodyId: bodyIds[row].toString(),
      annotations: Object.freeze({
        superclass: annotation(columns.superclasses, row),
        class: annotation(columns.classes, row),
        subclass: annotation(columns.subclasses, row),
        type: annotation(columns.types, row),
        mancType: annotation(columns.manc_types, row),
        side: annotation(columns.sides, row),
        entryNerve: annotation(columns.entry_nerves, row),
        exitNerve: annotation(columns.exit_nerves, row),
      }),
      neurotransmitter: Object.freeze({
        predicted,
        predictionConfidence: predicted === null ? null : confidence[row],
        effective: annotation(columns.effective_nt, row),
        basis: annotation(columns.nt_basis, row),
      }),
      routes: Object.freeze({
        sensoryAnnotation: Boolean(route & (1 << FLAG_BITS.sensory_annotation)),
        sensoryAtlas: Boolean(route & (1 << FLAG_BITS.sensory_atlas_row)),
        descendingContext: Boolean(route & (1 << FLAG_BITS.descending_context_row)),
        motorAnnotation: Boolean(route & (1 << FLAG_BITS.motor_annotation)),
        motor92: Boolean(route & (1 << FLAG_BITS.motor92_routed)),
        body807: Boolean(route & (1 << FLAG_BITS.body807_routed)),
      }),
      sensoryMemberships: frozenMemberships(row, sensory, 'sensory'),
      motorMemberships: frozenMemberships(row, motor, 'motor'),
    });
    recordCache.set(row, record);
    return record;
  }

  function find(query, limit = 30) {
    if (!Number.isInteger(limit) || limit < 1 || limit > 1000) {
      throw new RangeError('Neuron annotation find limit must be an integer in 1..1000');
    }
    const text = String(query ?? '').trim();
    if (text === '') return Object.freeze([]);
    const explicitRow = /^row\s*:?\s*(\d+)$/i.exec(text);
    if (explicitRow) {
      const row = Number(explicitRow[1]);
      return Object.freeze(row < ROWS ? [get(row)] : []);
    }
    if (/^-?\d+$/.test(text)) {
      const bodyId = BigInt(text);
      const row = binarySearchBodyId(bodyIds, bodyId);
      if (row >= 0) return Object.freeze([get(row)]);
      const fallbackRow = Number(text);
      if (Number.isSafeInteger(fallbackRow) && fallbackRow >= 0 && fallbackRow < ROWS) {
        return Object.freeze([get(fallbackRow)]);
      }
      return Object.freeze([]);
    }
    if (searchLists === null) searchLists = buildSearchLists(columns);
    const needle = text.toLocaleLowerCase('en-US');
    const lists = searchLists.filter(entry => entry.term.includes(needle)).map(entry => entry.rows);
    const positions = new Uint32Array(lists.length);
    const rows = [];
    while (rows.length < limit) {
      let next = ROWS;
      for (let index = 0; index < lists.length; index += 1) {
        const candidate = lists[index][positions[index]];
        if (candidate !== undefined && candidate < next) next = candidate;
      }
      if (next === ROWS) break;
      rows.push(get(next));
      for (let index = 0; index < lists.length; index += 1) {
        while (lists[index][positions[index]] === next) positions[index] += 1;
      }
    }
    return Object.freeze(rows);
  }

  return Object.freeze({rows: ROWS, get, find});
}
