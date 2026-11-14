// Read exact recorded JSON bytes, with or without a gzip transport wrapper.
export async function fetchRecordedJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  // HTTP Content-Encoding may already have been decoded by fetch.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return new Response(bytes).json();
  if (typeof DecompressionStream !== 'function') throw new Error('This browser cannot open gzip recordings');
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}
