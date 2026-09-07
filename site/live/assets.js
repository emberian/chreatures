// Authenticated transport only; simulation arithmetic belongs to Rust/WGSL.
export async function digest(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
}
export async function loadBlob(entry, baseURL, progress = () => {}) {
  const response = await fetch(new URL(entry.url, baseURL));
  if (!response.ok) throw new Error(`Model download ${response.status}: ${entry.url}`);
  const packed = await response.arrayBuffer();
  if (packed.byteLength !== (entry.transportByteLength ?? entry.byteLength) ||
      await digest(packed) !== (entry.transportSha256 ?? entry.sha256)) throw new Error(`Transport hash differs: ${entry.url}`);
  const raw = entry.encoding === 'gzip'
    ? await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
    : packed;
  if (raw.byteLength !== entry.byteLength || await digest(raw) !== entry.sha256) throw new Error(`Model hash differs: ${entry.url}`);
  progress(entry.transportByteLength ?? entry.byteLength);
  return raw;
}
export async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Asset download ${response.status}: ${url}`);
  return response.json();
}
