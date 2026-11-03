/* Lightweight orbit controls:
   - left-drag to rotate
   - scroll to zoom (dolly)
   - touch drag to rotate
   - autoRotate
   - damping
   Exposes THREE.OrbitControls(camera, domElement) with .update(), .target,
   .enableDamping, .dampingFactor, .autoRotate, .autoRotateSpeed,
   .minDistance, .maxDistance, .maxPolarAngle.
*/
(function(){
  function OrbitControls(camera, dom){
    this.camera = camera;
    this.dom = dom;
    this.target = new THREE.Vector3(0,0,0);
    this.enableDamping = false;
    this.dampingFactor = 0.08;
    this.autoRotate = false;
    this.autoRotateSpeed = 0.5;     // rad/sec scaled
    this.minDistance = 0.1;
    this.maxDistance = Infinity;
    this.maxPolarAngle = Math.PI;
    this.minPolarAngle = 0;

    // spherical state (relative to target)
    const offset = new THREE.Vector3();
    const spherical = new THREE.Spherical();
    const sphericalDelta = new THREE.Spherical();
    let scale = 1;

    let isDragging = false;
    let lastX = 0, lastY = 0;

    function onDown(e){
      isDragging = true;
      const p = pt(e);
      lastX = p.x; lastY = p.y;
      dom.setPointerCapture && e.pointerId !== undefined && dom.setPointerCapture(e.pointerId);
    }
    function onMove(e){
      if (!isDragging) return;
      const p = pt(e);
      const dx = p.x - lastX;
      const dy = p.y - lastY;
      lastX = p.x; lastY = p.y;
      const w = dom.clientWidth || 1;
      const h = dom.clientHeight || 1;
      // rotate
      sphericalDelta.theta -= 2 * Math.PI * dx / w;
      sphericalDelta.phi   -= 2 * Math.PI * dy / h;
    }
    function onUp(e){ isDragging = false; }
    function onWheel(e){
      e.preventDefault();
      const factor = Math.pow(0.95, -e.deltaY / 100);
      scale *= factor;
    }
    function pt(e){
      if (e.touches && e.touches[0]) return { x: e.touches[0].clientX, y: e.touches[0].clientY };
      return { x: e.clientX, y: e.clientY };
    }

    dom.addEventListener('pointerdown', onDown);
    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('pointerup', onUp);
    dom.addEventListener('pointercancel', onUp);
    dom.addEventListener('pointerleave', onUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    let lastTime = performance.now();

    this.update = function(){
      const now = performance.now();
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      // auto rotate
      if (this.autoRotate && !isDragging){
        sphericalDelta.theta -= this.autoRotateSpeed * dt;
      }

      // current spherical
      offset.copy(this.camera.position).sub(this.target);
      spherical.setFromVector3(offset);

      // apply deltas
      spherical.theta += sphericalDelta.theta;
      spherical.phi   += sphericalDelta.phi;
      // clamp phi
      spherical.phi = Math.max(this.minPolarAngle + 0.001, Math.min(this.maxPolarAngle, spherical.phi));
      spherical.makeSafe();
      // dolly
      spherical.radius *= scale;
      spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, spherical.radius));

      offset.setFromSpherical(spherical);
      this.camera.position.copy(this.target).add(offset);
      this.camera.lookAt(this.target);

      // damping decay
      if (this.enableDamping){
        sphericalDelta.theta *= (1 - this.dampingFactor);
        sphericalDelta.phi   *= (1 - this.dampingFactor);
        scale = 1 + (scale - 1) * (1 - this.dampingFactor);
      } else {
        sphericalDelta.set(0,0,0);
        scale = 1;
      }
    };
  }
  THREE.OrbitControls = OrbitControls;
})();
