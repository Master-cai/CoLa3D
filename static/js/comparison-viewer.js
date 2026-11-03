/* One selected case, four synchronized views, viewport-gated cached loading. */
(async function () {
  const root = document.getElementById('compare-grid');
  if (!root) return;
  const methods = [['gt', 'Ground Truth'], ['midi', 'MIDI-3D'], ['sam', 'SAM 3D'], ['ours', 'Ours']];
  let manifest;
  try {
    const response = await fetch('static/scenes/comparison/manifest.json');
    if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
    manifest = await response.json();
  } catch (error) {
    root.textContent = 'Unable to load comparison scenes. Please reload the page.';
    console.error(error);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.className = 'comparison-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (error) {
    root.textContent = '3D previews are unavailable. Enable WebGL in your browser and reload.';
    console.error(error);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.autoClear = false;
  root.replaceChildren(canvas);

  const decoder = new THREE.DRACOLoader();
  decoder.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
  decoder.setWorkerLimit(2);
  const loader = new THREE.GLTFLoader().setDRACOLoader(decoder);
  const header = document.createElement('div');
  header.className = 'cmp-row cmp-header';
  ['Input Image', ...methods.map(([, label]) => label)].forEach(label => {
    const cell = document.createElement('div');
    cell.className = 'cmp-head' + (label === 'Ours' ? ' ours' : '');
    cell.textContent = label;
    header.appendChild(cell);
  });
  root.appendChild(header);

  const rows = [];
  let selectedIndex = 0;
  const navigation = document.createElement('div');
  navigation.className = 'case-carousel cmp-case-carousel';
  const previous = document.createElement('button');
  previous.id = 'compare-prev';
  previous.type = 'button';
  previous.className = 'case-nav';
  previous.textContent = '‹';
  previous.setAttribute('aria-label', 'Previous comparison');
  const track = document.createElement('div');
  track.id = 'compare-track';
  track.className = 'case-track';
  track.setAttribute('aria-label', 'Choose a comparison image');
  const tiles = manifest.cases.map((data, index) => {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'case-tile';
    tile.setAttribute('aria-label', `Show comparison ${index + 1}`);
    const thumbnail = document.createElement('div');
    thumbnail.className = 'case-thumb';
    const image = document.createElement('img');
    image.src = data.input;
    image.alt = '';
    image.loading = 'lazy';
    thumbnail.appendChild(image);
    tile.appendChild(thumbnail);
    tile.addEventListener('click', () => selectCase(index));
    track.appendChild(tile);
    return tile;
  });
  const next = document.createElement('button');
  next.id = 'compare-next';
  next.type = 'button';
  next.className = 'case-nav';
  next.textContent = '›';
  next.setAttribute('aria-label', 'Next comparison');
  navigation.append(previous, track, next);
  function selectCase(index) {
    const selected = (index + rows.length) % rows.length;
    if (selected !== selectedIndex) {
      releaseRow(rows[selectedIndex]);
      rows[selectedIndex].active = false;
    }
    selectedIndex = selected;
    tiles.forEach((tile, i) => {
      tile.classList.toggle('active', i === selected);
      tile.setAttribute('aria-pressed', String(i === selected));
    });
    const tile = tiles[selected];
    // Scroll only the thumbnail strip, keeping the 3D views in place.
    if (tile.offsetLeft < track.scrollLeft || tile.offsetLeft + tile.offsetWidth > track.scrollLeft + track.clientWidth) {
      track.scrollTo({ left: tile.offsetLeft - (track.clientWidth - tile.offsetWidth) / 2, behavior: 'smooth' });
    }
    rows.forEach((row, i) => {
      row.element.hidden = i !== selected;
    });
    if (!rows[selected].image.src) rows[selected].image.src = rows[selected].data.input;
    invalidate();
  }
  previous.addEventListener('click', () => selectCase(selectedIndex - 1));
  next.addEventListener('click', () => selectCase(selectedIndex + 1));
  let needsRender = true;
  const invalidate = () => { needsRender = true; };
  window.addEventListener('scroll', invalidate, { passive: true, capture: true });
  window.addEventListener('resize', invalidate);
  document.addEventListener('visibilitychange', invalidate);
  let inflight = 0;
  let stopped = false;
  const MAX_REQUESTS = 2;


  function disposeModel(model) {
    const geometries = new Set(), materials = new Set(), textures = new Set();
    model.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) {
        (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => {
          materials.add(material);
          Object.values(material).forEach(value => { if (value && value.isTexture) textures.add(value); });
        });
      }
    });
    geometries.forEach(geometry => geometry.dispose());
    materials.forEach(material => material.dispose());
    textures.forEach(texture => {
      texture.dispose();
      if (texture.image && typeof texture.image.close === 'function') texture.image.close();
    });
  }

  function resetCamera(row) {
    invalidate();
    const center = new THREE.Vector3(...row.data.center);
    const distance = row.data.radius / Math.sin(THREE.MathUtils.degToRad(17.5)) * 1.12;
    row.camera.position.copy(center).add(new THREE.Vector3(1, 0.8, 1.5).normalize().multiplyScalar(distance));
    row.camera.lookAt(center);
    row.controls.forEach(control => {
      control.target.copy(center);
      control.minDistance = distance * 0.25;
      control.maxDistance = distance * 4;
      control.update();
    });
  }

  for (const data of manifest.cases) {
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'cmp-reset';
    reset.title = 'Reset all views';
    reset.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 11a9 9 0 1 1 2.4 7"/><path d="M3 4v7h7"/></svg>';
    ['pointerdown', 'pointermove', 'pointerup', 'keydown'].forEach(type => {
      reset.addEventListener(type, event => event.stopPropagation());
    });
    reset.setAttribute('aria-label', `Reset all four views for ${data.label}`);
    const element = document.createElement('div');
    element.className = 'cmp-row';
    const input = document.createElement('div');
    input.className = 'cmp-cell cmp-input';
    const image = document.createElement('img');

    image.alt = `${data.label} input image`;
    image.loading = 'lazy';
    input.appendChild(image);
    element.appendChild(input);
    const row = { data, element, image, views: [], controls: [], active: false, visible: false,
      camera: new THREE.PerspectiveCamera(35, 1, Math.max(data.radius / 1000, 0.001), data.radius * 100) };
    for (const [key, name] of methods) {
      const cell = document.createElement('div');
      cell.className = 'cmp-cell cmp-model' + (key === 'ours' ? ' ours' : '');
      cell.tabIndex = 0;
      cell.setAttribute('role', 'region');
      cell.setAttribute('aria-label', `${data.label}, ${name}. Drag to rotate all four views; scroll to zoom. Arrow keys rotate, plus and minus zoom.`);
      const status = document.createElement('span');
      status.className = 'cmp-status';
      status.textContent = '3D · loads when visible';
      cell.appendChild(status);
      if (key === 'ours') cell.appendChild(reset);
      element.appendChild(cell);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color('#fafbfc');
      scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 1.1));
      const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
      keyLight.position.set(3, 7, 5);
      scene.add(keyLight);
      const fill = new THREE.DirectionalLight(0xe4ecff, 0.5);
      fill.position.set(-4, 2, -3);
      scene.add(fill);
      const control = new THREE.OrbitControls(row.camera, cell);
      control.enableDamping = false;
      ['pointerdown', 'pointermove', 'pointerup', 'pointercancel', 'wheel'].forEach(type => {
        cell.addEventListener(type, invalidate, { passive: true });
      });
      row.controls.push(control);
      cell.addEventListener('keydown', event => {
        const offset = row.camera.position.clone().sub(control.target);
        const spherical = new THREE.Spherical().setFromVector3(offset);
        if (event.key === 'ArrowLeft') spherical.theta -= 0.12;
        else if (event.key === 'ArrowRight') spherical.theta += 0.12;
        else if (event.key === 'ArrowUp') spherical.phi -= 0.12;
        else if (event.key === 'ArrowDown') spherical.phi += 0.12;
        else if (event.key === '+' || event.key === '=') spherical.radius *= 0.9;
        else if (event.key === '-') spherical.radius *= 1.1;
        else return;
        event.preventDefault();
        invalidate();
        spherical.makeSafe();
        spherical.radius = Math.max(control.minDistance, Math.min(control.maxDistance, spherical.radius));
        row.camera.position.copy(control.target).add(offset.setFromSpherical(spherical));
        row.camera.lookAt(control.target);
      });
      row.views.push({ cell, status, scene, presentation: data.display?.[key], method: key, url: data.models[key], model: null, state: 'idle', generation: 0, controller: null });
    }
    reset.addEventListener('click', event => { event.stopPropagation(); resetCamera(row); });
    resetCamera(row);
    root.append(element);
    rows.push(row);
  }

  root.appendChild(navigation);
  selectCase(0);

  function releaseRow(row) {
    for (const view of row.views) {
      if (view.controller) { view.controller.abort(); view.controller = null; }
      // Already-started decoding finishes; stale results are disposed.
      if (view.state === 'loading' || view.model) view.generation++;
      if (view.model) {
        view.scene.remove(view.model);
        disposeModel(view.model);
        view.model = null;
      }
      if (view.state !== 'error') view.state = 'idle';
      view.status.hidden = false;
      view.status.textContent = view.state === 'error' ? 'Load failed · click to retry' : '3D · loads when visible';
    }
    renderer.renderLists.dispose();
  }

  function loadView(view) {
    inflight++;
    view.state = 'loading';
    view.status.hidden = false;
    view.status.textContent = 'Loading 3D…';
    const generation = ++view.generation;
    const controller = new AbortController();
    view.controller = controller;
    ModelAssetCache.load(loader, view.url, controller.signal).then(gltf => {
      if (generation !== view.generation || stopped) { disposeModel(gltf.scene); return; }
      // Apply one rigid transform to the complete method scene, never its parts.
      view.model = new THREE.Group();
      view.model.add(gltf.scene);
      if (view.presentation) {
        view.model.matrix.set(...view.presentation.matrix.flat());
        view.model.matrixAutoUpdate = false;
      }
      const sourceMaterials = new Set();
      view.model.traverse(object => {
        if (object.isMesh && object.material) {
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          const color = view.presentation?.colors[object.name];
          const replacement = materials.map(material => {
            if (!color) { material.side = THREE.DoubleSide; return material; }
            sourceMaterials.add(material);
            const copy = material.clone();
            copy.vertexColors = false;
            copy.color.setRGB(color[0] / 255, color[1] / 255, color[2] / 255);
            copy.side = THREE.DoubleSide;
            return copy;
          });
          object.material = Array.isArray(object.material) ? replacement : replacement[0];
        }
      });
      sourceMaterials.forEach(material => material.dispose());
      view.scene.add(view.model);
      view.state = 'ready';
      view.status.hidden = true;
      invalidate();
    }).catch(error => {
      if (generation !== view.generation || stopped) return;
      if (error.name === 'AbortError') { view.state = 'idle'; return; }
      view.state = 'error';
      view.status.textContent = 'Load failed · click to retry';
      console.error('Comparison GLB failed', view.url, error);
    }).finally(() => {
      inflight--;
      if (view.controller === controller) view.controller = null;
    });
    view.cell.onclick = () => { if (view.state === 'error') view.state = 'idle'; };
  }

  let width = 0, height = 0;
  function animate() {
    if (stopped) return;
    requestAnimationFrame(animate);
    if (document.hidden) return;
    if (width !== window.innerWidth || height !== window.innerHeight) {
      width = window.innerWidth; height = window.innerHeight;
      renderer.setSize(width, height);
      invalidate();
    }
    const clip = root.getBoundingClientRect();
    const visible = rows.filter((row, index) => {
      if (index !== selectedIndex) return false;
      const rect = row.element.getBoundingClientRect();
      row.visible = rect.bottom > 0 && rect.top < height && rect.right > 0 && rect.left < width;
      return row.visible;
    });
    const active = new Set(visible);
    for (const row of rows) {
      if (row.active !== active.has(row)) invalidate();
      if (row.active && !active.has(row)) releaseRow(row);
      row.active = active.has(row);
      if (row.active) {
        row.controls.forEach(control => control.update());
        for (const view of row.views) {
          if (view.state === 'idle' && inflight < MAX_REQUESTS) loadView(view);
        }
      }
    }
    if (!needsRender) return;
    needsRender = false;
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.setScissorTest(true);
    for (const row of visible) {
      if (!row.active) continue;
      for (const view of row.views) {
        if (!view.model) continue;
        const rect = view.cell.getBoundingClientRect();
        const left = Math.max(0, clip.left, rect.left + 2);
        const right = Math.min(width, clip.right, rect.right - 2);
        const top = Math.max(0, rect.top + 2);
        const bottom = Math.min(height, rect.bottom - 2);
        if (right <= left || bottom <= top) continue;
        row.camera.aspect = rect.width / rect.height;
        row.camera.updateProjectionMatrix();
        renderer.setViewport(rect.left, height - rect.bottom, rect.width, rect.height);
        renderer.setScissor(left, height - bottom, right - left, bottom - top);
        renderer.render(view.scene, row.camera);
      }
    }
  }
  animate();
  window.addEventListener('pagehide', event => {
    if (event.persisted) return;
    stopped = true;
    rows.forEach(releaseRow);
    decoder.dispose();
    renderer.dispose();
  });
})();
