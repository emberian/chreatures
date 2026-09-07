const PORTABLE_URL = 'assets/live-cns-evidence.json';
const MAX_JSON_BYTES = 1024 * 1024;
const $ = selector => document.querySelector(selector);

const ui = {
  loading: $('#loading'), shell: $('#evidence-shell'), footer: $('#graph-footer'), failure: $('#failure'),
  failureReason: $('#failure-reason'), description: $('#archive-description'), nodes: $('#node-count'),
  edges: $('#edge-count'), joins: $('#join-count'), list: $('#record-list'), filter: $('#record-filter'),
  filterEmpty: $('#filter-empty'), stage: $('#detail-stage'), title: $('#detail-title'), status: $('#detail-status'),
  text: $('#detail-text'), parents: $('#parent-list'), children: $('#child-list'), fields: $('#field-data'),
  blobs: $('#blob-list'), artifactSha: $('#artifact-sha'), verification: $('#verification-state'), library: $('#library-state'),
};

let graph = null;
let records = [];
let bySource = new Map();
let incoming = new Map();
let outgoing = new Map();
let selected = null;

function fail(message) {
  ui.loading.hidden = true; ui.shell.hidden = true; ui.footer.hidden = true; ui.failure.hidden = false;
  ui.failureReason.textContent = message;
}

function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} is absent`);
  return value;
}

function text(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} is absent`);
  return value;
}

function hex(value, label, length = 64) {
  if (typeof value !== 'string' || !new RegExp(`^[0-9a-f]{${length}}$`).test(value)) throw new Error(`${label} is not a lowercase hexadecimal identity`);
  return value;
}

function displayType(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
}

function stageOf(record) {
  const time = object(record.time, 'record time');
  if (time.domain !== 'live_cns_evidence_stage' || !Number.isSafeInteger(time.value) || time.value < 0) throw new Error('record stage differs');
  return time.value;
}

function statusOf(record) {
  const fields = object(record.source.fields, 'record fields');
  if (record.record_type.includes('failure')) return {label: 'recorded failure', kind: 'failed'};
  if (record.record_type === 'comparison_identity_diagnostic') return {label: 'refined diagnostic', kind: 'mixed'};
  if (fields.outcome === 'mixed') return {label: 'mixed result', kind: 'mixed'};
  if (fields.passed === true || record.record_type.includes('confirmation')) return {label: 'executed', kind: ''};
  if (record.record_type.includes('prediction')) return {label: 'prediction', kind: 'mixed'};
  if (record.record_type.includes('fit') || record.record_type.includes('training')) return {label: 'fitted', kind: ''};
  return {label: 'recorded', kind: ''};
}

function validate(data) {
  object(data, 'portable graph');
  if (data.format !== 'chreatures-portable-weave-biography-v1' || data.integration !== 'native-universal-weave-dag') throw new Error('portable Weave format differs');
  if (!Array.isArray(data.records) || !Array.isArray(data.edges) || !Array.isArray(data.topological_order)) throw new Error('portable Weave arrays are absent');
  if (!Number.isSafeInteger(data.node_count) || data.node_count !== data.records.length || data.node_count < 1 || data.node_count > 256) throw new Error('portable node count differs');
  if (!Number.isSafeInteger(data.edge_count) || data.edge_count !== data.edges.length || data.edge_count > 1024) throw new Error('portable edge count differs');
  const nodeIds = new Set(), sourceIds = new Set();
  for (const record of data.records) {
    object(record, 'record');
    hex(record.node_id, 'native node id', 32); text(record.source_id, 'source id'); text(record.record_type, 'record type'); text(record.text, 'record narrative');
    object(record.source, 'source record'); object(record.source.fields, 'source fields'); stageOf(record);
    if (record.source.id !== record.source_id || record.source.record_type !== record.record_type || record.source.text !== record.text) throw new Error('portable and source record identity differs');
    if (nodeIds.has(record.node_id) || sourceIds.has(record.source_id)) throw new Error('portable graph repeats an identity');
    if (!Array.isArray(record.parent_source_ids) || new Set(record.parent_source_ids).size !== record.parent_source_ids.length ||
        !Array.isArray(record.source.parent_ids) || JSON.stringify(record.source.parent_ids) !== JSON.stringify(record.parent_source_ids) ||
        !Array.isArray(record.blob_refs) || record.blob_refs.length > 16) throw new Error('record references differ');
    if (JSON.stringify(record.source.fields).length > 40000) throw new Error('recorded fields exceed the public bound');
    for (const item of record.blob_refs) { object(item, 'blob reference'); text(item.role, 'blob role'); hex(item.sha256, 'blob SHA-256'); if (item.uri !== `urn:sha256:${item.sha256}`) throw new Error('blob URI differs from its digest'); }
    nodeIds.add(record.node_id); sourceIds.add(record.source_id);
  }
  if (data.topological_order.length !== data.node_count || new Set(data.topological_order).size !== data.node_count || data.topological_order.some(id => !nodeIds.has(id))) throw new Error('native topological order differs');
  const order = new Map(data.topological_order.map((id, index) => [id, index]));
  const byNode = new Map(data.records.map(record => [record.node_id, record]));
  const edgeKeys = new Set();
  for (const edge of data.edges) {
    object(edge, 'edge'); text(edge.source, 'edge source'); text(edge.target, 'edge target'); text(edge.role, 'edge role');
    if (!sourceIds.has(edge.source) || !sourceIds.has(edge.target)) throw new Error('edge endpoint is absent');
    const key = `${edge.source}\0${edge.target}`; if (edgeKeys.has(key)) throw new Error('portable graph repeats an edge'); edgeKeys.add(key);
    const parent = data.records.find(record => record.source_id === edge.source), child = data.records.find(record => record.source_id === edge.target);
    if (order.get(parent.node_id) >= order.get(child.node_id)) throw new Error('edge violates native topological order');
  }
  for (const record of data.records) {
    const edgeParents = data.edges.filter(edge => edge.target === record.source_id).map(edge => edge.source);
    if (JSON.stringify([...record.parent_source_ids].sort()) !== JSON.stringify(edgeParents.sort())) throw new Error('record parents and edge table differ');
    if (record.parent_source_ids.some(id => !sourceIds.has(id))) throw new Error('record parent is absent');
  }
  hex(data.artifact_sha256, 'native serialization SHA-256');
  if (data.artifact !== 'live-cns-evidence.weave.json' || !Number.isSafeInteger(data.bytes) || data.bytes < 1 || data.bytes > MAX_JSON_BYTES) throw new Error('native serialization reference differs');
  const library = object(data.library, 'library identity'); text(library.name, 'library name'); text(library.version, 'library version'); hex(library.source_commit, 'library source revision', 40);
  text(object(data.archive, 'archive metadata').description, 'archive description');
  return data.topological_order.map(id => byNode.get(id));
}

async function fetchJSON() {
  const response = await fetch(PORTABLE_URL, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`portable graph request returned HTTP ${response.status}`);
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > MAX_JSON_BYTES) throw new Error('portable graph exceeds the public size bound');
  const source = await response.text();
  if (new TextEncoder().encode(source).length > MAX_JSON_BYTES) throw new Error('portable graph exceeds the public size bound');
  return JSON.parse(source);
}

function relationButton(edge, direction) {
  const id = direction === 'parent' ? edge.source : edge.target;
  const record = bySource.get(id);
  const button = document.createElement('button'); button.type = 'button';
  const role = document.createElement('span'), title = document.createElement('strong');
  role.textContent = displayType(edge.role); title.textContent = displayType(record.record_type);
  button.append(role, title); button.addEventListener('click', () => choose(id, true));
  return button;
}

function fillRelations(container, edges, direction) {
  if (!edges.length) { const empty = document.createElement('p'); empty.className = 'none'; empty.textContent = direction === 'parent' ? 'Origin record' : 'No later record in this graph'; container.replaceChildren(empty); return; }
  container.replaceChildren(...edges.map(edge => relationButton(edge, direction)));
}

function fillBlobs(record) {
  const blobs = [...record.blob_refs];
  if (!blobs.length) { const empty = document.createElement('p'); empty.className = 'none'; empty.textContent = 'No external blob on this node.'; ui.blobs.replaceChildren(empty); return; }
  ui.blobs.replaceChildren(...blobs.map(item => {
    const card = document.createElement('div'); card.className = 'blob';
    const role = document.createElement('strong'), hash = document.createElement('code'), detail = document.createElement('span');
    role.textContent = displayType(item.role); hash.textContent = item.sha256;
    const size = Number.isSafeInteger(item.bytes) ? `${item.bytes.toLocaleString()} bytes · ` : '';
    detail.textContent = `${size}${item.media_type || 'media type unspecified'} · ${item.verification || 'verification unspecified'}`;
    card.append(role, hash, detail); return card;
  }));
}

function choose(sourceId, updateHash = false) {
  const record = bySource.get(sourceId); if (!record) return;
  selected = sourceId;
  for (const button of ui.list.querySelectorAll('button')) button.setAttribute('aria-current', String(button.dataset.source === sourceId));
  const status = statusOf(record);
  ui.stage.textContent = `stage ${stageOf(record)} · ${record.record_type.replaceAll('_', ' ')}`;
  ui.title.textContent = displayType(record.record_type); ui.status.textContent = status.label; ui.status.className = `status ${status.kind}`.trim();
  ui.text.textContent = record.text;
  const fields = {...record.source.fields}; delete fields.parent_roles;
  const figure = $('#record-figure');
  const plot = fields.plot_url;
  const showPlot = typeof plot === 'string' && /^assets\/[A-Za-z0-9_.-]+\.svg$/.test(plot);
  figure.hidden = !showPlot;
  if (showPlot) { $('#figure-image').src = plot; $('#figure-link').href = plot; }
  else { $('#figure-image').removeAttribute('src'); $('#figure-link').removeAttribute('href'); }
  ui.fields.textContent = JSON.stringify(fields, null, 2);
  fillRelations(ui.parents, incoming.get(sourceId) || [], 'parent');
  fillRelations(ui.children, outgoing.get(sourceId) || [], 'child');
  fillBlobs(record);
  if (updateHash) history.replaceState(null, '', `#${encodeURIComponent(sourceId)}`);
}

function recordClass(record) {
  if (record.record_type.includes('failure')) return 'failure-record';
  if (record.record_type.includes('prediction') || record.record_type.includes('diagnostic')) return 'prediction-record';
  if (record.record_type.includes('confirmation') || record.record_type.includes('rollout') || record.record_type.includes('export')) return 'result-record';
  return '';
}

function buildList() {
  ui.list.replaceChildren(...records.map(record => {
    const item = document.createElement('li'); item.className = recordClass(record); item.dataset.search = `${record.record_type} ${record.text}`.toLowerCase();
    const button = document.createElement('button'); button.type = 'button'; button.dataset.source = record.source_id; button.setAttribute('aria-current', 'false');
    const stage = document.createElement('small'), title = document.createElement('strong'), excerpt = document.createElement('span');
    stage.textContent = `stage ${stageOf(record)}`; title.textContent = displayType(record.record_type); excerpt.textContent = record.text;
    button.append(stage, title, excerpt); button.addEventListener('click', () => choose(record.source_id, true)); item.append(button); return item;
  }));
}

function applyFilter() {
  const query = ui.filter.value.trim().toLowerCase(); let visible = 0;
  for (const item of ui.list.children) { item.hidden = query && !item.dataset.search.includes(query); if (!item.hidden) visible += 1; }
  ui.filterEmpty.hidden = visible !== 0;
}

async function verifyArtifact() {
  try {
    const response = await fetch(new URL(graph.artifact, new URL(PORTABLE_URL, location.href)), {cache: 'no-cache'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== graph.bytes) throw new Error('byte count differs');
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(value => value.toString(16).padStart(2, '0')).join('');
    if (digest !== graph.artifact_sha256) throw new Error('SHA-256 differs');
    ui.verification.textContent = 'Native serialization SHA-256 checked in this tab.'; ui.verification.className = 'verified';
  } catch (error) {
    ui.verification.textContent = `Native serialization could not be checked: ${error.message}`;
  }
}

async function start() {
  try {
    graph = await fetchJSON(); records = validate(graph); bySource = new Map(records.map(record => [record.source_id, record]));
    incoming = new Map(records.map(record => [record.source_id, []])); outgoing = new Map(records.map(record => [record.source_id, []]));
    for (const edge of graph.edges) { incoming.get(edge.target).push(edge); outgoing.get(edge.source).push(edge); }
    ui.description.textContent = graph.archive.description; ui.nodes.textContent = graph.node_count.toLocaleString();
    ui.edges.textContent = graph.edge_count.toLocaleString(); ui.joins.textContent = graph.multi_parent_nodes.toLocaleString();
    ui.artifactSha.textContent = graph.artifact_sha256;
    ui.library.textContent = `${graph.library.name} ${graph.library.version} · source ${graph.library.source_commit}`;
    buildList(); ui.filter.addEventListener('input', applyFilter);
    ui.loading.hidden = true; ui.shell.hidden = false; ui.footer.hidden = false;
    const requested = location.hash ? decodeURIComponent(location.hash.slice(1)) : '';
    choose(bySource.has(requested) ? requested : records[0].source_id);
    verifyArtifact();
  } catch (error) { fail(`The published evidence failed its reader boundary: ${error.message}`); }
}

start();
