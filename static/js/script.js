/* Real-image viewer, application videos, and citation controls. */

// DATA
// CASES is populated at boot from static/scenes/manifest.json.
// Each entry: { id, label, url }  where url points at a .glb file.
// currentCaseObjects is rebuilt from the loaded GLB's top-level children.
let CASES = [];
let currentCaseObjects = [];   // [{ name, color }] for Inst control rendering

// THREE.js scene
let scene, camera, renderer, controls, viewerRoom;
let objectGroup;       // holds current case meshes (reparented from GLB)
let autoSpin = true;
let currentCaseIdx = 0;
let gltfLoader = null;
let caseLoadVersion = 0;
let caseRequest = null;
let realViewerVisible = false;
let realCaseState = 'idle';
let defaultCameraPos = new THREE.Vector3(4.5, 3.2, 4.5);
let defaultCameraTarget = new THREE.Vector3(0, 0.4, 0);

function initThree() {
  const canvas = document.getElementById("viewer3d");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe8e7e2);

  camera = new THREE.PerspectiveCamera(35, 16/9, 0.1, 100);
  camera.position.set(4.5, 3.2, 4.5);
  camera.lookAt(0, 0.4, 0);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  // Proper colorspace + tone-mapping so GLB PBR textures don't look washed out
  if (THREE.sRGBEncoding) renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  resizeRenderer();

  // ---- IBL: bake a neutral RoomEnvironment into a PMREM so PBR materials
  // (metalness / roughness / baseColor textures) sample correct reflections
  // and indirect lighting instead of looking flat-gray.
  if (THREE.RoomEnvironment) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envScene = new THREE.RoomEnvironment();
    scene.environment = pmrem.fromScene(envScene, 0.04).texture;
    pmrem.dispose();
  } else {
    console.warn("THREE.RoomEnvironment not found — PBR materials will look flat. Check that the RoomEnvironment.js script tag loaded.");
  }

  // ---- Three-point key/fill/rim, neutral white so albedo isn't tinted.
  // Hemisphere stays small — IBL is doing most of the ambient work now.
  const hemi = new THREE.HemisphereLight(0xffffff, 0x404449, 0.25);
  scene.add(hemi);

  // Key light: warm-neutral, casts the main highlight direction.
  const key = new THREE.DirectionalLight(0xfff4e6, 1.1);
  key.position.set(5, 8, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.bias = -0.0002;
  key.shadow.radius = 3;
  scene.add(key, key.target);

  // Fill light from the opposite side, softer + cool-neutral to round shadows.
  const fill = new THREE.DirectionalLight(0xdfe6f2, 0.45);
  fill.position.set(-5, 4, -2);
  scene.add(fill);

  // Subtle rim from behind/above to separate objects from the dark stage.
  const rim = new THREE.DirectionalLight(0xc8d4ff, 0.35);
  rim.position.set(0, 6, -6);
  scene.add(rim);

  viewerRoom = createViewerRoom(scene, key, renderer);

  controls = new THREE.OrbitControls(camera, canvas);
  controls.target.set(0, 0.4, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2.5;
  controls.maxDistance = 12;
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.autoRotate = autoSpin;
  controls.autoRotateSpeed = 0.03;

  gltfLoader = new THREE.GLTFLoader();
  // Draco-compressed GLBs (KHR_draco_mesh_compression) need a decoder.
  // Decoder is loaded on demand from Google's CDN; plain GLBs are unaffected.
  if (THREE.DRACOLoader) {
    const draco = new THREE.DRACOLoader();
    draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.6/");
    gltfLoader.setDRACOLoader(draco);
  }

  window.addEventListener("resize", resizeRenderer);
  animate();
}

function resizeRenderer() {
  const canvas = renderer.domElement;
  const stage = canvas.parentElement;
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  const bounds = renderer.domElement.getBoundingClientRect();
  if (document.hidden || bounds.bottom <= 0 || bounds.top >= window.innerHeight) return;
  controls.update();
  viewerRoom.update(camera);
  renderer.render(scene, camera);
}

function disposeObjectTree(root) {
  root.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      materials.forEach(material => {
        Object.values(material).forEach(value => {
          if (value && value.isTexture) {
            value.dispose();
            if (value.image && typeof value.image.close === "function") value.image.close();
          }
        });
        material.dispose();
      });
    }
  });
}

function clearObjectGroup() {
  if (objectGroup) {
    disposeObjectTree(objectGroup);
    scene.remove(objectGroup);
    objectGroup = null;
    if (renderer) renderer.shadowMap.needsUpdate = true;
  }
}

// Pick a representative color for an object node — used as the swatch on its
// Inst button. We walk into descendants and take the first material.color.
function extractRepColor(node) {
  let hex = null;
  node.traverse(o => {
    if (hex !== null) return;
    if (o.isMesh && o.material) {
      const m = Array.isArray(o.material) ? o.material[0] : o.material;
      if (m && m.color) hex = m.color.getHex();
    }
  });
  return hex ?? 0x9aa2ad;
}

// Auto-frame the camera on a freshly loaded scene.
function fitCameraToGroup(group, fillRatio = 1.45) {
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const fov = camera.fov * Math.PI / 180;
  const distance = (maxDim / 2) / Math.tan(fov / 2) * fillRatio;

  const dir = new THREE.Vector3(1, 0.7, 1).normalize();
  defaultCameraPos = center.clone().add(dir.multiplyScalar(distance));
  defaultCameraTarget = center.clone();

  camera.position.copy(defaultCameraPos);
  camera.lookAt(defaultCameraTarget);
  controls.target.copy(defaultCameraTarget);
  controls.minDistance = distance * 0.35;
  controls.maxDistance = distance * 3.0;
  controls.update();
}

function loadCase(idx) {
  if (!CASES.length) return;
  if (idx === currentCaseIdx && (realCaseState === 'ready' || realCaseState === 'loading')) return;
  if (caseRequest) caseRequest.abort();
  caseRequest = null;
  const loadVersion = ++caseLoadVersion;
  currentCaseIdx = idx;
  const c = CASES[idx];
  clearObjectGroup();
  realCaseState = 'idle';
  currentCaseObjects = [];
  renderInstControls();
  document.getElementById("viewer-count").textContent = "Loading…";
  document.getElementById("viewer-case-label").textContent = "Loading scene…";
  renderCaseTrack();

  if (!realViewerVisible || !gltfLoader) {
    document.getElementById("viewer-case-label").textContent = "Interactive scene";
    document.getElementById("viewer-count").textContent = "Ready to load";
    return;
  }
  const controller = new AbortController();
  caseRequest = controller;
  realCaseState = 'loading';
  ModelAssetCache.load(gltfLoader, c.url, controller.signal).then((gltf) => {
      // A slower previous request must not replace the user's latest selection.
      if (loadVersion !== caseLoadVersion) {
        disposeObjectTree(gltf.scene);
        return;
      }
      const root = gltf.scene;
      // Each child of the exported world group represents one object.
      const world = root.getObjectByName("world") || root;
      const children = [...world.children];
      if (children.length === 0) children.push(...root.children);

      objectGroup = new THREE.Group();
      currentCaseObjects = [];
      children.forEach((child, i) => {
        child.userData.instIdx = i;
        // Keep open surfaces visible from either side; this does not fill holes.
        child.traverse(o => {
          if (o.isMesh && o.material) {
            o.castShadow = true;
            const mats = Array.isArray(o.material) ? o.material : [o.material];
            mats.forEach(m => { m.side = THREE.DoubleSide; });
          }
        });
        objectGroup.add(child);  // reparent
        currentCaseObjects.push({
          name: child.name || `inst_${i}`,
          color: extractRepColor(child)
        });
      });
      scene.add(objectGroup);

      // Convert positive Z-up sources to Y-up, preserving all relative object poses.
      const upAxis = (c.upAxis || "z").toLowerCase();
      if (upAxis === "z") objectGroup.rotation.x = -Math.PI / 2;
      objectGroup.updateMatrixWorld(true);

      // Ground the complete scene without moving individual objects.
      const gbox = new THREE.Box3().setFromObject(objectGroup);
      if (!gbox.isEmpty()) {
        objectGroup.position.y -= gbox.min.y;
        objectGroup.updateMatrixWorld(true);
      }

      viewerRoom.fit(objectGroup);
      fitCameraToGroup(objectGroup);
      viewerRoom.update(camera);

      // HUD
      document.getElementById("viewer-case-label").textContent = "Interactive scene";
      document.getElementById("viewer-count").textContent = `${currentCaseObjects.length} OBJECTS`;

      renderInstControls();
      renderCaseTrack();
      setActiveInstance("all");
      realCaseState = 'ready';
    }).catch((err) => {
      if (loadVersion !== caseLoadVersion) return;
      realCaseState = err.name === 'AbortError' ? 'idle' : 'error';
      if (err.name === 'AbortError') return;
      console.error("Failed to load GLB", c.url, err);
      document.getElementById("viewer-case-label").textContent = "Unable to load scene";
    }).finally(() => {
      if (caseRequest === controller) caseRequest = null;
    });
}

// Instance toggle
function renderInstControls() {
  const root = document.getElementById("inst-controls");
  root.innerHTML = "";
  currentCaseObjects.forEach((o, i) => {
    const btn = document.createElement("button");
    btn.className = "inst-btn";
    btn.dataset.inst = String(i);
    const hex = "#" + o.color.toString(16).padStart(6, "0");
    btn.innerHTML = `<span class="swatch" style="background:${hex}"></span> Inst ${i+1}`;
    btn.addEventListener("click", () => setActiveInstance(i));
    root.appendChild(btn);
  });
  const all = document.createElement("button");
  all.className = "inst-btn show-all active";
  all.dataset.inst = "all";
  all.innerHTML = `Show All`;
  all.addEventListener("click", () => setActiveInstance("all"));
  root.appendChild(all);
}

function setActiveInstance(which) {
  // toggle button states
  document.querySelectorAll(".inst-btn").forEach(b => {
    b.classList.toggle("active", String(b.dataset.inst) === String(which));
  });
  if (!objectGroup) return;
  // Toggle each object together with its child meshes.
  objectGroup.children.forEach((node, i) => {
    node.visible = (which === "all") || (i === which);
  });
  renderer.shadowMap.needsUpdate = true;
}

// Case carousel
function renderCaseTrack() {
  const track = document.getElementById("case-track");
  track.innerHTML = "";
  CASES.forEach((c, i) => {
    const tile = document.createElement("button");
    tile.className = "case-tile" + (i === currentCaseIdx ? " active" : "");
    tile.setAttribute("aria-label", `Show scene ${i + 1}`);
    tile.setAttribute("aria-pressed", String(i === currentCaseIdx));
    // Cover image: set "thumb" on the manifest entry (e.g. a square PNG/JPG).
    // Falls back to the ▦ glyph if absent or if the image fails to load.
    const thumb = c.thumb
      ? `<img loading="lazy" src="${c.thumb}" alt="${c.label || c.id}"
              onerror="this.parentElement.textContent='▦'">`
      : "▦";
    tile.innerHTML = `
      <div class="case-thumb">${thumb}</div>
    `;
    tile.addEventListener("click", () => loadCase(i));
    track.appendChild(tile);
  });
}

document.getElementById("case-prev").addEventListener("click", () => {
  if (CASES.length) loadCase((currentCaseIdx - 1 + CASES.length) % CASES.length);
});
document.getElementById("case-next").addEventListener("click", () => {
  if (CASES.length) loadCase((currentCaseIdx + 1) % CASES.length);
});

// HUD buttons
document.getElementById("viewer-reset").addEventListener("click", () => {
  if (camera && controls && defaultCameraPos) {
    camera.position.copy(defaultCameraPos);
    controls.target.copy(defaultCameraTarget);
    controls.update();
  }
});
document.getElementById("viewer-spin").addEventListener("click", (e) => {
  autoSpin = !autoSpin;
  if (controls) controls.autoRotate = autoSpin;
  const btn = e.currentTarget;
  btn.classList.toggle("active", autoSpin);
  const lbl = document.getElementById("spin-label");
  if (lbl) lbl.textContent = autoSpin ? "Spin: On" : "Spin: Off";
});

// Videos are listed explicitly under static/videos/downstream/<application>/.
const APPLICATIONS = [
  {
    id: "sim",
    tab: "Simulator Ready",
    cases: [
      { file: "case01.mp4" },
      { file: "case02.mp4" },
      { file: "case03.mp4" },
      { file: "case04.mp4" },
      { file: "case05.mp4" },
      { file: "case06.mp4" },
      { file: "case07.mp4" },
      { file: "case08.mp4" },
      { file: "case09.mp4" }
    ]
  },
  {
    id: "decompose",
    tab: "Holistic Mesh → Decompose",
    cases: [
      { file: "Text_to_Mesh.mp4" },
      { file: "Image_to_Mesh.mp4" },
      { file: "Mesh_to_Mesh.mp4" }
    ]
  }
];

let appIdx = 0;
let galleryIdx = 0;

function activeCases() { return APPLICATIONS[appIdx].cases; }

function renderTabs() {
  const root = document.getElementById("app-tabs");
  if (!root) return;
  root.innerHTML = "";
  APPLICATIONS.forEach((app, i) => {
    const btn = document.createElement("button");
    btn.className = "app-tab" + (i === appIdx ? " active" : "");
    btn.type = "button";
    btn.textContent = app.tab;
    btn.addEventListener("click", () => setApplication(i));
    root.appendChild(btn);
  });
}

function setApplication(i) {
  appIdx = (i + APPLICATIONS.length) % APPLICATIONS.length;
  document.querySelectorAll(".app-tab").forEach((b, idx) => {
    b.classList.toggle("active", idx === appIdx);
  });
  galleryIdx = 0;
  renderGallery();
}

// Downstream carousel: play ONLY the centered (front) slide, and only while
// the gallery is on screen. Side peeks and hidden slides stay paused.
let galleryVisible = false;
function syncGalleryPlayback() {
  document.querySelectorAll("#gallery-stage .gallery-slide").forEach(el => {
    const v = el.querySelector("video");
    if (!v) return;
    if (galleryVisible && el.classList.contains("center")) {
      const p = v.play(); if (p) p.catch(() => {});
    } else if (!v.paused) {
      v.pause();
    }
  });
}
const galleryVisObserver = new IntersectionObserver((entries) => {
  galleryVisible = entries.some(e => e.isIntersecting);
  syncGalleryPlayback();
}, { threshold: 0.15 });

function renderGallery() {
  const cases = activeCases();
  const baseDir = `static/videos/downstream/${APPLICATIONS[appIdx].id}/`;
  const stage = document.getElementById("gallery-stage");
  stage.innerHTML = "";
  cases.forEach((item, i) => {
    const slide = document.createElement("div");
    slide.className = "gallery-slide";
    slide.dataset.idx = String(i);
    // Use a text fallback when a video cannot be loaded.
    const src = baseDir + item.file;
    const ph = "Video unavailable";
    const media = src
      ? `<video class="slide-video" src="${src}" muted loop playsinline preload="metadata"
                onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'slide-bg',textContent:'${ph}'}))"></video>`
      : `<div class="slide-bg">${ph}</div>`;
    slide.innerHTML = media;
    slide.addEventListener("click", () => {
      if (i !== galleryIdx) setGallery(i);
    });
    stage.appendChild(slide);
  });

  const dots = document.getElementById("gallery-dots");
  dots.innerHTML = "";
  cases.forEach((_, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `Show video ${i + 1}`);
    b.addEventListener("click", () => setGallery(i));
    dots.appendChild(b);
  });

  setGallery(galleryIdx);
}

function setGallery(i) {
  const cases = activeCases();
  const n = cases.length;
  galleryIdx = (i + n) % n;
  const slides = document.querySelectorAll(".gallery-slide");
  slides.forEach((el, idx) => {
    el.classList.remove("center", "left", "right", "hidden");
    if (idx === galleryIdx) el.classList.add("center");
    else if (idx === (galleryIdx - 1 + n) % n) el.classList.add("left");
    else if (idx === (galleryIdx + 1) % n) el.classList.add("right");
    else el.classList.add("hidden");
  });

  document.querySelectorAll(".gallery-dots button").forEach((d, idx) => {
    d.classList.toggle("active", idx === galleryIdx);
    d.setAttribute("aria-pressed", String(idx === galleryIdx));
  });

  syncGalleryPlayback();
}

document.getElementById("gal-prev").addEventListener("click", () => setGallery(galleryIdx - 1));
document.getElementById("gal-next").addEventListener("click", () => setGallery(galleryIdx + 1));

renderTabs();
renderGallery();
galleryVisObserver.observe(document.getElementById("gallery-stage"));

// Baseline comparisons are initialized by comparison-viewer.js.

// BibTeX copy
document.getElementById("copy-bib").addEventListener("click", async (e) => {
  const code = document.querySelector(".code-block code").textContent;
  try {
    await navigator.clipboard.writeText(code);
    e.target.textContent = "Copied ✓";
    setTimeout(() => { e.target.textContent = "Copy"; }, 1600);
  } catch (err) {
    e.target.textContent = "Error";
  }
});

// Boot
function showViewerError(msg) {
  document.getElementById("viewer3d").parentElement.innerHTML =
    `<div style="color:#888;display:flex;align-items:center;justify-content:center;height:100%;font-family:monospace;padding:20px;text-align:center">${msg}</div>`;
}

async function bootViewer() {
  if (typeof THREE === "undefined" || !THREE.GLTFLoader) {
    showViewerError("The 3D viewer could not start. Please reload the page.");
    return;
  }
  let manifest;
  try {
    const resp = await fetch("static/scenes/manifest.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`manifest HTTP ${resp.status}`);
    manifest = await resp.json();
  } catch (err) {
    console.error(err);
    showViewerError("Scenes are unavailable. Please reload the page.");
    return;
  }
  CASES = Array.isArray(manifest.cases) ? manifest.cases : [];
  if (CASES.length === 0) {
    showViewerError("No scenes are available yet.");
    return;
  }
  renderCaseTrack();
  document.getElementById("viewer-case-label").textContent = "Interactive scene";
  const observer = new IntersectionObserver(entries => {
    realViewerVisible = entries.some(entry => entry.isIntersecting);
    if (realViewerVisible) {
      if (!gltfLoader) initThree();
      loadCase(currentCaseIdx);
    } else if (realCaseState === 'loading') {
      ++caseLoadVersion;
      if (caseRequest) caseRequest.abort();
      caseRequest = null;
      realCaseState = 'idle';
    }
  }, { threshold: 0 });
  observer.observe(document.getElementById("viewer3d"));
}

bootViewer();
