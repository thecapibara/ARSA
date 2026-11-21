import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { createBoneMesh, ShapeType } from './mannequin.ts';

interface CachedBody {
  object: THREE.Object3D;
  signature: string;
  isProp: boolean;
}

interface ImpulseRing {
  mesh: THREE.Mesh;
  age: number;
  duration: number;
}

/**
 * Manages the Three.js Aperture/Euphoria test-chamber scene, lighting,
 * procedural meter-grid floor, static obstacles, dynamic bodies, and grab tether.
 */
export class LabScene {
  public readonly scene: THREE.Scene;
  public readonly camera: THREE.PerspectiveCamera;
  public readonly renderer: THREE.WebGLRenderer;
  public readonly controls: OrbitControls;

  private readonly staticGroup = new THREE.Group();
  private readonly dynamicGroup = new THREE.Group();
  private readonly bodyPool: CachedBody[] = [];

  private readonly propBoxMat: THREE.MeshStandardMaterial;
  private readonly propSphereMat: THREE.MeshStandardMaterial;
  private readonly propAccentMat: THREE.MeshStandardMaterial;
  private readonly propEdgeMat: THREE.LineBasicMaterial;
  private readonly staticBoxMat: THREE.MeshStandardMaterial;
  private readonly staticEdgeMat: THREE.LineBasicMaterial;

  private readonly tetherLine: THREE.Line;
  private readonly tetherStartDot: THREE.Mesh;
  private readonly tetherEndDot: THREE.Mesh;
  private readonly tetherPositions = new Float32Array(6);

  private readonly impulseRings: ImpulseRing[] = [];
  private readonly impulseRingGeo: THREE.RingGeometry;

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14181f);
    this.scene.fog = new THREE.FogExp2(0x14181f, 0.02);

    this.camera = new THREE.PerspectiveCamera(
      48,
      window.innerWidth / window.innerHeight,
      0.08,
      140
    );
    this.camera.position.set(3.2, 2.35, 4.6);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight, false);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.06;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.95, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 26;
    this.controls.maxPolarAngle = Math.PI * 0.492;
    // Leave Left Mouse Button free for physics interaction tools
    this.controls.mouseButtons = {
      LEFT: null as unknown as THREE.MOUSE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.ROTATE,
    };
    this.controls.update();

    this.propBoxMat = new THREE.MeshStandardMaterial({
      color: 0x364050,
      roughness: 0.46,
      metalness: 0.25,
    });
    this.propSphereMat = new THREE.MeshStandardMaterial({
      color: 0x566275,
      roughness: 0.26,
      metalness: 0.82,
    });
    this.propAccentMat = new THREE.MeshStandardMaterial({
      color: 0xe8942c,
      roughness: 0.35,
      metalness: 0.2,
    });
    this.propEdgeMat = new THREE.LineBasicMaterial({
      color: 0xe8942c,
      transparent: true,
      opacity: 0.45,
    });

    this.staticBoxMat = new THREE.MeshStandardMaterial({
      color: 0x262d38,
      roughness: 0.65,
      metalness: 0.14,
    });
    this.staticEdgeMat = new THREE.LineBasicMaterial({
      color: 0x526075,
      transparent: true,
      opacity: 0.55,
    });

    this.setupLighting();
    this.setupFloor();

    this.scene.add(this.staticGroup);
    this.scene.add(this.dynamicGroup);

    // Grab tether line & endpoint markers
    const tetherGeo = new THREE.BufferGeometry();
    tetherGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.tetherPositions, 3)
    );
    const tetherMat = new THREE.LineBasicMaterial({
      color: 0xe8942c,
      transparent: true,
      opacity: 0.92,
    });
    this.tetherLine = new THREE.Line(tetherGeo, tetherMat);
    this.tetherLine.frustumCulled = false;
    this.tetherLine.visible = false;
    this.scene.add(this.tetherLine);

    const dotGeo = new THREE.SphereGeometry(0.032, 12, 10);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xf5a623 });
    this.tetherStartDot = new THREE.Mesh(dotGeo, dotMat);
    this.tetherEndDot = new THREE.Mesh(dotGeo, dotMat);
    this.tetherStartDot.visible = false;
    this.tetherEndDot.visible = false;
    this.scene.add(this.tetherStartDot);
    this.scene.add(this.tetherEndDot);

    this.impulseRingGeo = new THREE.RingGeometry(0.08, 0.125, 28);
  }

  private setupLighting(): void {
    const hemi = new THREE.HemisphereLight(0xcbe0fa, 0x1a212c, 0.58);
    this.scene.add(hemi);

    // Primary studio key light (warm neutral)
    const keyLight = new THREE.DirectionalLight(0xfff4e6, 2.25);
    keyLight.position.set(7.5, 13.5, 8.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 42;
    const frustum = 11;
    keyLight.shadow.camera.left = -frustum;
    keyLight.shadow.camera.right = frustum;
    keyLight.shadow.camera.top = frustum;
    keyLight.shadow.camera.bottom = -frustum;
    keyLight.shadow.bias = -0.0004;
    keyLight.shadow.normalBias = 0.022;
    this.scene.add(keyLight);

    // Cool rim light for crisp mannequin edge separation
    const rimLight = new THREE.DirectionalLight(0x7ca9e8, 0.9);
    rimLight.position.set(-8.5, 6.5, -7.5);
    this.scene.add(rimLight);

    // Low front fill light
    const fillLight = new THREE.DirectionalLight(0xa3b7cf, 0.38);
    fillLight.position.set(0, 2.8, 10);
    this.scene.add(fillLight);
  }

  private setupFloor(): void {
    const texCanvas = document.createElement('canvas');
    texCanvas.width = 512;
    texCanvas.height = 512;
    const ctx = texCanvas.getContext('2d');
    if (ctx) {
      // Base architectural slate
      ctx.fillStyle = '#1b2029';
      ctx.fillRect(0, 0, 512, 512);

      // Subtle quarter-meter subgrid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.032)';
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const p = (i * 512) / 4;
        ctx.beginPath();
        ctx.moveTo(p, 0);
        ctx.lineTo(p, 512);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, p);
        ctx.lineTo(512, p);
        ctx.stroke();
      }

      // Crisp 1-meter major lines
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, 512, 512);

      // Corner precision calibration ticks
      ctx.strokeStyle = 'rgba(232, 148, 44, 0.35)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 14);
      ctx.lineTo(0, 0);
      ctx.lineTo(14, 0);
      ctx.stroke();
    }

    const gridTex = new THREE.CanvasTexture(texCanvas);
    gridTex.wrapS = THREE.RepeatWrapping;
    gridTex.wrapT = THREE.RepeatWrapping;
    gridTex.repeat.set(64, 64);
    gridTex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    gridTex.colorSpace = THREE.SRGBColorSpace;

    const floorGeo = new THREE.PlaneGeometry(64, 64);
    const floorMat = new THREE.MeshStandardMaterial({
      map: gridTex,
      roughness: 0.56,
      metalness: 0.2,
    });

    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  /**
   * Rebuilds static lab obstacles (stairs, drop platforms, walls) from `get_static_boxes()`.
   * Expects 6 floats per box: `[cx, cy, cz, hx, hy, hz]`.
   */
  public syncStaticBoxes(boxes: Float32Array): void {
    while (this.staticGroup.children.length > 0) {
      const child = this.staticGroup.children.pop()!;
      child.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
        }
      });
    }

    const count = Math.floor(boxes.length / 6);
    for (let i = 0; i < count; i++) {
      const o = i * 6;
      const cx = boxes[o];
      const cy = boxes[o + 1];
      const cz = boxes[o + 2];
      const hx = boxes[o + 3];
      const hy = boxes[o + 4];
      const hz = boxes[o + 5];

      // Skip huge ground-plane slabs since our procedural grid floor sits at y = 0
      if (hx >= 14 && hz >= 14 && cy + hy <= 0.05) {
        continue;
      }

      const geo = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
      const mesh = new THREE.Mesh(geo, this.staticBoxMat);
      mesh.position.set(cx, cy, cz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        this.staticEdgeMat
      );
      mesh.add(edges);

      this.staticGroup.add(mesh);
    }
  }

  /**
   * Synchronizes all dynamic rigid bodies (mannequin bones & props).
   */
  public syncBodies(
    transforms: Float32Array,
    descriptors: Float32Array,
    count: number
  ): void {
    // Shrink pool if simulation was reset
    while (this.bodyPool.length > count) {
      const removed = this.bodyPool.pop()!;
      this.dynamicGroup.remove(removed.object);
      this.disposeObjectGeometries(removed.object);
    }

    for (let i = 0; i < count; i++) {
      const dOff = i * 6;
      const shapeType = Math.round(descriptors[dOff]);
      const dim0 = descriptors[dOff + 1];
      const dim1 = descriptors[dOff + 2];
      const dim2 = descriptors[dOff + 3];
      const ragdollId = Math.round(descriptors[dOff + 4]);
      const boneId = Math.round(descriptors[dOff + 5]);

      const sig = `${shapeType}|${dim0.toFixed(4)}|${dim1.toFixed(4)}|${dim2.toFixed(4)}|${ragdollId}|${boneId}`;
      let entry = this.bodyPool[i];

      if (!entry || entry.signature !== sig) {
        if (entry) {
          this.dynamicGroup.remove(entry.object);
          this.disposeObjectGeometries(entry.object);
        }

        const isProp = ragdollId < 0 || boneId < 0;
        const object = isProp
          ? this.createPropMesh(shapeType, dim0, dim1, dim2)
          : createBoneMesh(boneId, shapeType, dim0, dim1, dim2);

        this.dynamicGroup.add(object);
        entry = { object, signature: sig, isProp };
        this.bodyPool[i] = entry;
      }

      const tOff = i * 7;
      entry.object.position.set(
        transforms[tOff],
        transforms[tOff + 1],
        transforms[tOff + 2]
      );
      entry.object.quaternion.set(
        transforms[tOff + 3],
        transforms[tOff + 4],
        transforms[tOff + 5],
        transforms[tOff + 6]
      );
    }
  }

  private createPropMesh(
    shapeType: number,
    dim0: number,
    dim1: number,
    dim2: number
  ): THREE.Object3D {
    const group = new THREE.Group();

    if (shapeType === ShapeType.Box) {
      const geo = new THREE.BoxGeometry(dim0 * 2, dim1 * 2, dim2 * 2);
      const mesh = new THREE.Mesh(geo, this.propBoxMat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), this.propEdgeMat));
      group.add(mesh);
    } else if (shapeType === ShapeType.Sphere) {
      const r = Math.max(0.04, dim0);
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(r, 24, 18),
        this.propSphereMat
      );
      sphere.castShadow = true;
      sphere.receiveShadow = true;
      group.add(sphere);

      // Equatorial amber band so sphere rotation reads clearly
      const band = new THREE.Mesh(
        new THREE.TorusGeometry(r * 0.995, r * 0.045, 8, 28),
        this.propAccentMat
      );
      band.rotation.x = Math.PI / 2;
      group.add(band);
    } else {
      const r = Math.max(0.03, dim0);
      const len = Math.max(0.04, dim1 * 2);
      const cap = new THREE.Mesh(
        new THREE.CapsuleGeometry(r, len, 8, 16),
        this.propBoxMat
      );
      cap.castShadow = true;
      cap.receiveShadow = true;
      group.add(cap);
    }

    return group;
  }

  private disposeObjectGeometries(root: THREE.Object3D): void {
    root.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
      }
    });
  }

  /**
   * Updates or hides the interactive spring grab tether line.
   */
  public setGrabTether(
    active: boolean,
    bodyPos?: THREE.Vector3,
    targetPos?: THREE.Vector3
  ): void {
    if (!active || !bodyPos || !targetPos) {
      this.tetherLine.visible = false;
      this.tetherStartDot.visible = false;
      this.tetherEndDot.visible = false;
      return;
    }

    this.tetherPositions[0] = bodyPos.x;
    this.tetherPositions[1] = bodyPos.y;
    this.tetherPositions[2] = bodyPos.z;
    this.tetherPositions[3] = targetPos.x;
    this.tetherPositions[4] = targetPos.y;
    this.tetherPositions[5] = targetPos.z;

    const attr = this.tetherLine.geometry.getAttribute(
      'position'
    ) as THREE.BufferAttribute;
    attr.needsUpdate = true;

    this.tetherStartDot.position.copy(bodyPos);
    this.tetherEndDot.position.copy(targetPos);

    this.tetherLine.visible = true;
    this.tetherStartDot.visible = true;
    this.tetherEndDot.visible = true;
  }

  /**
   * Spawns a brief expanding shockwave ring at an impulse impact point.
   */
  public spawnImpulseEffect(pos: THREE.Vector3): void {
    const mat = new THREE.MeshBasicMaterial({
      color: 0xe8942c,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(this.impulseRingGeo, mat);
    ring.position.copy(pos);
    ring.quaternion.copy(this.camera.quaternion);
    this.scene.add(ring);
    this.impulseRings.push({ mesh: ring, age: 0, duration: 0.28 });
  }

  public updateEffects(dt: number): void {
    for (let i = this.impulseRings.length - 1; i >= 0; i--) {
      const fx = this.impulseRings[i];
      fx.age += dt;
      const t = Math.min(1, fx.age / fx.duration);
      const scale = 1 + t * 3.4;
      fx.mesh.scale.setScalar(scale);
      fx.mesh.quaternion.copy(this.camera.quaternion);
      (fx.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.85;

      if (t >= 1) {
        this.scene.remove(fx.mesh);
        (fx.mesh.material as THREE.Material).dispose();
        this.impulseRings.splice(i, 1);
      }
    }
  }

  public resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  public render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
