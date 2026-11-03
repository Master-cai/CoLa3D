/* Session-local LRU of compressed GLB bytes, shared by both galleries. */
(function () {
  const entries = new Map();
  const limit = 256 * 1024 * 1024;
  let bytes = 0;

  async function load(loader, url, signal) {
    const key = new URL(url, document.baseURI).href;
    signal.throwIfAborted();
    let buffer = entries.get(key);
    if (buffer) {
      entries.delete(key);
      entries.set(key, buffer);
    } else {
      const response = await fetch(key, { signal });
      if (!response.ok) throw new Error(`Model HTTP ${response.status}: ${url}`);
      buffer = await response.arrayBuffer();
      signal.throwIfAborted();
      if (buffer.byteLength <= limit) {
        // A second consumer may have completed the same URL while we fetched.
        if (entries.has(key)) bytes -= entries.get(key).byteLength;
        entries.delete(key);
        entries.set(key, buffer);
        bytes += buffer.byteLength;
        while (bytes > limit) {
          const oldest = entries.keys().next().value;
          bytes -= entries.get(oldest).byteLength;
          entries.delete(oldest);
        }
      }
    }
    signal.throwIfAborted();
    // Decoder workers may transfer ArrayBuffers: never give them cache storage.
    // Parsing cannot be cancelled; callers dispose stale decoded scenes.
    return new Promise((resolve, reject) => {
      loader.parse(buffer.slice(0), new URL('.', key).href, resolve, reject);
    });
  }

  window.ModelAssetCache = Object.freeze({ load });
})();
