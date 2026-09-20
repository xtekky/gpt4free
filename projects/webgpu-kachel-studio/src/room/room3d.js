/* ==========================================================================
   FILL THE ROOM - 3D INTERIOR SCENE RENDERER (THREE.JS)
   ========================================================================== */

import * as THREE from 'three';

export class Room3DManager {
  constructor(containerEl, onSurfaceClickCallback, onSurfaceDropCallback) {
    this.container = containerEl;
    this.onSurfaceClick = onSurfaceClickCallback;
    this.onSurfaceDropCallback = onSurfaceDropCallback;

    this.scene = null;
    this.camera = null;
    this.renderer = null;

    this.floorMesh = null;
    this.backWallMesh = null;
    this.sideWallMesh = null;
    this.backsplashMesh = null;
    this.furnitureGroup = null;

    this.lightsGroup = null;
    this.activeSurfaceTarget = 'floor'; // 'floor' | 'wall' | 'backsplash' | 'all'
    this.tileRepeatScale = 6;

    this.currentCanvasTexture = null;
    this.currentNormalTexture = null;

    // Orbit Controls manual state
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };
    this.cameraRotation = { theta: Math.PI / 4, phi: Math.PI / 6, radius: 10 };

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
  }

  init() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x06080d);
    this.scene.fog = new THREE.FogExp2(0x06080d, 0.04);

    // Camera setup
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.updateCameraPosition();

    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    // Remove existing canvases
    while (this.container.firstChild) {
      this.container.removeChild(this.container.firstChild);
    }
    this.container.appendChild(this.renderer.domElement);

    // Lighting setup
    this.setupLighting('daylight');

    // Build Room Geometry
    this.buildRoomGeometry();

    // Event listeners for camera orbit & surface click
    this.attachEventListeners();

    // Start Animation Loop
    this.animate();

    // Handle Window Resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  updateCameraPosition() {
    const r = this.cameraRotation.radius;
    const theta = this.cameraRotation.theta;
    const phi = this.cameraRotation.phi;

    this.camera.position.x = r * Math.sin(theta) * Math.cos(phi);
    this.camera.position.y = Math.max(1.0, r * Math.sin(phi));
    this.camera.position.z = r * Math.cos(theta) * Math.cos(phi);
    this.camera.lookAt(0, 1.2, 0);
  }

  setupLighting(preset = 'daylight') {
    if (this.lightsGroup) {
      this.scene.remove(this.lightsGroup);
    }
    this.lightsGroup = new THREE.Group();

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.lightsGroup.add(ambientLight);

    const mainSun = new THREE.DirectionalLight(0xffffff, 1.2);
    mainSun.position.set(6, 12, 8);
    mainSun.castShadow = true;
    mainSun.shadow.mapSize.width = 2048;
    mainSun.shadow.mapSize.height = 2048;
    mainSun.shadow.bias = -0.0001;
    this.lightsGroup.add(mainSun);

    const fillLight = new THREE.PointLight(0x00f2fe, 0.6, 15);
    fillLight.position.set(-4, 3, 2);
    this.lightsGroup.add(fillLight);

    if (preset === 'warm') {
      mainSun.color.setHex(0xffaa55);
      mainSun.intensity = 1.0;
      fillLight.color.setHex(0xff7700);
      ambientLight.color.setHex(0xffeecc);
    } else if (preset === 'nordic') {
      mainSun.color.setHex(0xe0f2fe);
      mainSun.intensity = 0.9;
      fillLight.color.setHex(0x38bdf8);
      ambientLight.color.setHex(0xf1f5f9);
    } else if (preset === 'cyber') {
      mainSun.color.setHex(0xec4899);
      mainSun.intensity = 1.5;
      fillLight.color.setHex(0x00f2fe);
      fillLight.intensity = 2.0;
      ambientLight.color.setHex(0x3b0764);
    }

    this.scene.add(this.lightsGroup);
  }

  buildRoomGeometry() {
    // Default Material
    const defaultMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.4,
      metalness: 0.1
    });

    // 1. FLOOR (8m x 8m)
    const floorGeo = new THREE.PlaneGeometry(8, 8);
    this.floorMesh = new THREE.Mesh(floorGeo, defaultMat.clone());
    this.floorMesh.rotation.x = -Math.PI / 2;
    this.floorMesh.receiveShadow = true;
    this.floorMesh.name = 'floor';
    this.scene.add(this.floorMesh);

    // 2. BACK WALL (8m x 4m)
    const backWallGeo = new THREE.PlaneGeometry(8, 4);
    this.backWallMesh = new THREE.Mesh(backWallGeo, defaultMat.clone());
    this.backWallMesh.position.set(0, 2, -4);
    this.backWallMesh.receiveShadow = true;
    this.backWallMesh.name = 'wall';
    this.scene.add(this.backWallMesh);

    // 3. SIDE ACCENT WALL (8m x 4m)
    const sideWallGeo = new THREE.PlaneGeometry(8, 4);
    this.sideWallMesh = new THREE.Mesh(sideWallGeo, defaultMat.clone());
    this.sideWallMesh.position.set(-4, 2, 0);
    this.sideWallMesh.rotation.y = Math.PI / 2;
    this.sideWallMesh.receiveShadow = true;
    this.sideWallMesh.name = 'wall';
    this.scene.add(this.sideWallMesh);

    // 4. KITCHEN SPLASHBACK PANEL (4m x 1.2m)
    const splashGeo = new THREE.PlaneGeometry(4, 1.2);
    this.backsplashMesh = new THREE.Mesh(splashGeo, defaultMat.clone());
    this.backsplashMesh.position.set(0, 1.8, -3.95);
    this.backsplashMesh.receiveShadow = true;
    this.backsplashMesh.name = 'backsplash';
    this.scene.add(this.backsplashMesh);

    // 5. MODERN FURNITURE DECORATION
    this.buildFurniture();
  }

  buildFurniture() {
    if (this.furnitureGroup) {
      this.scene.remove(this.furnitureGroup);
    }
    this.furnitureGroup = new THREE.Group();

    const sofaMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.8 });
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.6 });
    const metalMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.8, roughness: 0.2 });

    // Sofa Base
    const sofa = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.6, 1.1), sofaMat);
    sofa.position.set(0, 0.3, 1);
    sofa.castShadow = true;
    sofa.receiveShadow = true;
    this.furnitureGroup.add(sofa);

    // Coffee Table
    const tableTop = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.1, 32), woodMat);
    tableTop.position.set(0, 0.45, -0.8);
    tableTop.castShadow = true;
    this.furnitureGroup.add(tableTop);

    // Table Legs
    for (let i = 0; i < 3; i++) {
      const angle = (i * Math.PI * 2) / 3;
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.45), metalMat);
      leg.position.set(Math.cos(angle) * 0.6, 0.225, -0.8 + Math.sin(angle) * 0.6);
      leg.castShadow = true;
      this.furnitureGroup.add(leg);
    }

    // Modern Plant Pot
    const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.25, 0.7, 16), new THREE.MeshStandardMaterial({ color: 0xf8fafc }));
    pot.position.set(-2.8, 0.35, -2.5);
    pot.castShadow = true;
    this.furnitureGroup.add(pot);

    this.scene.add(this.furnitureGroup);
  }

  // Update room textures with WebGPU tile canvas & generated normal map
  applyTileTextures(tileCanvas, normalCanvas, config = {}) {
    const repeatScale = config.repeatScale || this.tileRepeatScale;
    const targetSurface = config.targetSurface || this.activeSurfaceTarget;
    const roughness = config.roughness !== undefined ? config.roughness : 0.3;

    // Create Three.js Canvas Textures
    const colorTex = new THREE.CanvasTexture(tileCanvas);
    colorTex.wrapS = THREE.RepeatWrapping;
    colorTex.wrapT = THREE.RepeatWrapping;
    colorTex.repeat.set(repeatScale, repeatScale);
    colorTex.colorSpace = THREE.SRGBColorSpace;
    colorTex.needsUpdate = true;

    let normalTex = null;
    if (normalCanvas) {
      normalTex = new THREE.CanvasTexture(normalCanvas);
      normalTex.wrapS = THREE.RepeatWrapping;
      normalTex.wrapT = THREE.RepeatWrapping;
      normalTex.repeat.set(repeatScale, repeatScale);
      normalTex.needsUpdate = true;
    }

    const pbrMaterial = new THREE.MeshStandardMaterial({
      map: colorTex,
      normalMap: normalTex,
      normalScale: new THREE.Vector2(0.8, 0.8),
      roughness: roughness,
      metalness: 0.15
    });

    // Apply to requested surface
    if (targetSurface === 'floor' || targetSurface === 'all') {
      this.floorMesh.material = pbrMaterial;
    }
    if (targetSurface === 'wall' || targetSurface === 'all') {
      this.backWallMesh.material = pbrMaterial.clone();
      this.sideWallMesh.material = pbrMaterial.clone();
    }
    if (targetSurface === 'backsplash' || targetSurface === 'all') {
      this.backsplashMesh.material = pbrMaterial.clone();
    }

    this.currentCanvasTexture = colorTex;
    this.currentNormalTexture = normalTex;
  }

  attachEventListeners() {
    const dom = this.renderer.domElement;

    // --- MOUSE DRAG ORBIT ---
    dom.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    dom.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const deltaX = e.clientX - this.previousMousePosition.x;
      const deltaY = e.clientY - this.previousMousePosition.y;

      this.cameraRotation.theta -= deltaX * 0.008;
      this.cameraRotation.phi = Math.max(0.1, Math.min(Math.PI / 2.2, this.cameraRotation.phi + deltaY * 0.008));

      this.previousMousePosition = { x: e.clientX, y: e.clientY };
      this.updateCameraPosition();
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // --- TOUCH GESTURES (MOBILE / TOUCHSCREEN / TABSTART) ---
    let initialPinchDist = 0;

    dom.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        // Pinch zoom
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        initialPinchDist = Math.sqrt(dx * dx + dy * dy);
      }
    }, { passive: true });

    dom.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && this.isDragging) {
        const deltaX = e.touches[0].clientX - this.previousMousePosition.x;
        const deltaY = e.touches[0].clientY - this.previousMousePosition.y;

        this.cameraRotation.theta -= deltaX * 0.008;
        this.cameraRotation.phi = Math.max(0.1, Math.min(Math.PI / 2.2, this.cameraRotation.phi + deltaY * 0.008));

        this.previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.updateCameraPosition();
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const delta = initialPinchDist - dist;

        this.cameraRotation.radius = Math.max(4, Math.min(15, this.cameraRotation.radius + delta * 0.02));
        this.updateCameraPosition();
        initialPinchDist = dist;
      }
    }, { passive: true });

    dom.addEventListener('touchend', () => {
      this.isDragging = false;
    });

    // --- WHEEL ZOOM ---
    dom.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.cameraRotation.radius = Math.max(4, Math.min(15, this.cameraRotation.radius + e.deltaY * 0.005));
      this.updateCameraPosition();
    }, { passive: false });

    // --- HTML5 DRAG & DROP ONTO 3D SURFACES ---
    dom.addEventListener('dragover', (e) => {
      e.preventDefault();
      dom.style.cursor = 'copy';
    });

    dom.addEventListener('dragleave', () => {
      dom.style.cursor = 'default';
    });

    dom.addEventListener('drop', (e) => {
      e.preventDefault();
      dom.style.cursor = 'default';

      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects([this.floorMesh, this.backWallMesh, this.sideWallMesh, this.backsplashMesh]);

      if (intersects.length > 0) {
        const hitMesh = intersects[0].object;
        const rawData = e.dataTransfer.getData('application/json');
        if (rawData && this.onSurfaceDropCallback) {
          try {
            const tileConfig = JSON.parse(rawData);
            this.onSurfaceDropCallback(hitMesh.name, tileConfig, intersects[0].uv);
          } catch (err) {
            console.warn('Invalid drop data', err);
          }
        }
      }
    });

    // Click raycast for surface selection & mutation popover
    dom.addEventListener('click', (e) => {
      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects([this.floorMesh, this.backWallMesh, this.sideWallMesh, this.backsplashMesh]);

      if (intersects.length > 0 && this.onSurfaceClick) {
        const hitMesh = intersects[0].object;
        const uv = intersects[0].uv || { x: 0.5, y: 0.5 };
        this.onSurfaceClick(hitMesh.name, intersects[0].point, uv);
      }
    });
  }

  onWindowResize() {
    if (!this.container || !this.renderer) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  resetView() {
    this.cameraRotation = { theta: Math.PI / 4, phi: Math.PI / 6, radius: 10 };
    this.updateCameraPosition();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }
}
