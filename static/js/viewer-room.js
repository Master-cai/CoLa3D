/* Reusable architectural backdrop; furniture stays in its original coordinates. */
(function () {
  window.createViewerRoom = function (scene, keyLight, renderer) {
    const root = new THREE.Group();
    root.name = 'shared_display_room';
    root.visible = false;
    scene.add(root);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 512;
    const context = canvas.getContext('2d');
    const colors = ['#d5c3a5', '#dccbae', '#d8c5a7', '#d1bfa0'];
    for (let row = 0; row < 8; row++) {
      context.fillStyle = colors[row % colors.length];
      context.fillRect(0, row * 64, 512, 64);
      context.fillStyle = 'rgba(100,76,48,0.10)';
      context.fillRect(0, row * 64, 512, 1);
      const joint = row % 2 ? 330 : 150;
      context.fillRect(joint, row * 64, 1, 64);
      for (let grain = 0; grain < 10; grain++) {
        context.strokeStyle = 'rgba(120,92,60,0.035)';
        context.beginPath();
        context.moveTo(0, row * 64 + grain * 6 + 3);
        context.bezierCurveTo(180, row * 64 + grain * 6, 320, row * 64 + grain * 6 + 7, 512, row * 64 + grain * 6 + 3);
        context.stroke();
      }
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.encoding = THREE.sRGBEncoding;
    texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.92, metalness: 0 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    root.add(floor);
    const wallMaterials = [0xeeeae2, 0xe3e4de, 0xeeeae2, 0xe3e4de].map(color => new THREE.MeshStandardMaterial({ color, roughness: 1 }));
    const trimMaterial = new THREE.MeshStandardMaterial({ color: 0xfaf7ee, roughness: 0.85 });
    const walls = wallMaterials.map((material, index) => {
      const group = new THREE.Group();
      const wall = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
      wall.receiveShadow = true;
      const trim = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), trimMaterial);
      group.add(wall, trim);
      root.add(group);
      return { group, wall, trim, index };
    });
    // A high frosted window provides an architectural cue without adding furniture.
    const windowFrame = new THREE.Group();
    const frame = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), trimMaterial);
    const glass = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xdce8e9 }));
    glass.position.z = 0.002;
    windowFrame.add(frame, glass);
    const mullion = new THREE.Mesh(new THREE.PlaneGeometry(0.025, 1), trimMaterial);
    mullion.position.z = 0.004;
    windowFrame.add(mullion);
    walls[0].group.add(windowFrame);
    const center = new THREE.Vector3();
    let span = 1;

    function fit(group) {
      const bounds = new THREE.Box3().setFromObject(group);
      if (bounds.isEmpty()) return;
      root.visible = true;
      const size = bounds.getSize(new THREE.Vector3());
      bounds.getCenter(center);
      span = Math.max(size.x, size.z, size.y * 0.75, 0.1);
      const width = Math.max(size.x + span * 0.7, span * 1.4);
      const depth = Math.max(size.z + span * 0.7, span * 1.4);
      const height = Math.max(size.y + span * 0.35, span * 0.9);
      root.position.set(center.x, -span * 0.004, center.z);
      floor.scale.set(width, depth, 1);
      texture.repeat.set(width / span * 2, depth / span * 2);
      walls.forEach(({ group: wallGroup, wall, trim, index }) => {
        const side = index % 2;
        const length = side ? depth : width;
        wallGroup.position.set(index === 1 ? -width / 2 : index === 3 ? width / 2 : 0, 0, index === 0 ? -depth / 2 : index === 2 ? depth / 2 : 0);
        wallGroup.rotation.y = [0, Math.PI / 2, Math.PI, -Math.PI / 2][index];
        wall.position.set(0, height / 2, 0);
        wall.scale.set(length, height, 1);
        trim.position.set(0, span * 0.025, span * 0.006);
        trim.scale.set(length, span * 0.05, span * 0.012);
      });
      windowFrame.position.set(width * 0.22, height * 0.75, span * 0.003);
      windowFrame.scale.set(width * 0.28, height * 0.28, 1);
      glass.scale.set(0.92, 0.89, 1);
      keyLight.position.set(center.x - span * 1.4, height * 2, center.z + span * 1.6);
      keyLight.target.position.set(center.x, 0, center.z);
      const shadow = keyLight.shadow.camera;
      shadow.left = shadow.bottom = -span * 1.8;
      shadow.right = shadow.top = span * 1.8;
      shadow.near = span * 0.01;
      shadow.far = span * 8;
      shadow.updateProjectionMatrix();
      keyLight.shadow.normalBias = span * 0.008;
      renderer.shadowMap.needsUpdate = true;
    }

    function update(camera) {
      // Hide walls between the camera and the furniture during a full orbit.
      const visible = [camera.position.z >= center.z, camera.position.x >= center.x,
        camera.position.z < center.z, camera.position.x < center.x];
      walls.forEach(({ group }, index) => { group.visible = visible[index]; });
    }
    return { fit, update, root, floor };
  };
})();
