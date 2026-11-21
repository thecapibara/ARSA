import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { createBoneMesh, ShapeType } from './mannequin.ts';

export type CameraMode = 'orbit' | 'fly';
export type CameraPresetName = 'front' | 'back' | 'left' | 'right' | 'top';
export type ImpulseEffectVariant = 'shove' | 'punch';

interface CameraPreset {
  position: [number, number, number];
  target: [number, number, number];
}

const CAMERA_PRESETS: Record<CameraPresetName, CameraPreset> = {
  front: { position: [0, 1.35, 3.6], target: [0, 1.0, 0] },
  back: { position: [0, 1.35, -3.6], target: [0, 1.0, 0] },
  left: { position: [3.6, 1.35, 0], target: [0, 1.0, 0] },
  right: { position: [-3.6, 1.35, 0], target: [0, 1.0, 0] },
  top: { position: [0, 5.2, 0.25], target: [0, 0.5, 0] },
};

interface PresetTransition {
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  startTarget: THREE.Vector3;
  endTarget: THREE.Vector3;
  elapsed: number;
  duration: number;
}

interface CachedBody {
  object: THREE.Object3D;
  signature: string;
  isProp: boolean;
}

interface ImpulseFx {
  group: THREE.Group;
  ring: THREE.Mesh;
  flashCore?: THREE.Mesh;
  age: number;
  duration: number;
  maxScale: number;
}

interface TelemetryVisual {
  group: THREE.Group;
  comMarker: THREE.Mesh;
  plumbLine: THREE.Line;
  plumbPositions: Float32Array;
  cpRing: THREE.Mesh;
  cpTether: THREE.Line;
  cpTetherPositions: Float32Array;
}

const TELEMETRY_STRIDE = 11;

/**
 * Manages the Three.js Aperture/Euphoria test-chamber scene, lighting,
 * procedural meter-grid floor, static obstacles, dynamic bodies, camera modes/presets,
 * place-box preview, impact FX, and debug telemetry overlay.
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

  private readonly placePreview: THREE.Group;

  private readonly impulseEffects: ImpulseFx[] = [];
  private readonly impulseRingGeo: THREE.RingGeometry;
  private readonly punchCoreGeo: THREE.SphereGeometry;

  // Debug Telemetry Overlay (CoM + plumb line + Capture Point ring)
  private readonly telemetryGroup = new THREE.Group();
  private readonly telemetryPool: TelemetryVisual[] = [];
  private readonly comGeo: THREE.SphereGeometry;
  private readonly comMat: THREE.MeshBasicMaterial;
  private readonly plumbMat: THREE.LineDashedMaterial;
  private readonly cpRingGeo: THREE.RingGeometry;
  private readonly cpRingMat: THREE.MeshBasicMaterial;
  private readonly cpTetherMat: THREE.LineBasicMaterial;

  // Camera Mode & Preset Transition state
  private cameraMode: CameraMode = 'orbit';
  private presetTransition: PresetTransition | null = null;
  private flyYaw = 0;
  private flyPitch = 0;
  private flySpeed = 4.0;
  private readonly pressedFlyKeys = new Set<string>();
  private readonly tempVecA = new THREE.Vector3();
  private readonly tempVecB = new THREE.Vector3();
  private readonly tempVecC = new THREE.Vector3();
  private readonly tempEuler = new THREE.Euler(0, 0, 0, 'YXZ');

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
    this.controls.minDistance = 1.0;
    this.controls.maxDistance = 26;
    this.controls.maxPolarAngle = Math.PI * 0.492;
    // Leave Left Mouse Button free for physics interaction tools
    this.controls.mouseButtons = {
      LEFT: null as unknown as THREE.MOUSE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.ROTATE,
    };
    this.controls.update();
    this.syncFlyAnglesFromCamera();

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

    // Place-Box ghost preview (matches hx=0.35, hy=0.14, hz=0.18)
    this.placePreview = new THREE.Group();
    const previewGeo = new THREE.BoxGeometry(0.7, 0.28, 0.36);
    const previewFill = new THREE.Mesh(
      previewGeo,
      new THREE.MeshBasicMaterial({
        color: 0xe8942c,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      })
    );
    const previewWire = new THREE.LineSegments(
      new THREE.EdgesGeometry(previewGeo),
      new THREE.LineBasicMaterial({
        color: 0xf5a623,
        transparent: true,
        opacity: 0.85,
      })
    );
    this.placePreview.add(previewFill, previewWire);
    this.placePreview.visible = false;
    this.scene.add(this.placePreview);

    this.impulseRingGeo = new THREE.RingGeometry(0.08, 0.13, 28);
    this.punchCoreGeo = new THREE.SphereGeometry(0.075, 14, 12);

    // Debug Telemetry shared resources
    this.comGeo = new THREE.SphereGeometry(0.044, 14, 12);
    this.comMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.92,
      depthTest: false,
    });
    this.plumbMat = new THREE.LineDashedMaterial({
      color: 0x22d3ee,
      dashSize: 0.06,
      gapSize: 0.04,
      transparent: true,
      opacity: 0.78,
      depthTest: false,
    });
    this.cpRingGeo = new THREE.RingGeometry(0.085, 0.125, 28);
    this.cpRingMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.cpTetherMat = new THREE.LineBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    });
    this.scene.add(this.telemetryGroup);
  }

  private setupLighting(): void {
    const hemi = new THREE.HemisphereLight(0xcbe0fa, 0x1a212c, 0.58);
    this.scene.add(hemi);

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

    const rimLight = new THREE.DirectionalLight(0x7ca9e8, 0.9);
    rimLight.position.set(-8.5, 6.5, -7.5);
    this.scene.add(rimLight);

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
      ctx.fillStyle = '#1b2029';
      ctx.fillRect(0, 0, 512, 512);

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

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, 512, 512);

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

  public getCameraMode(): CameraMode {
    return this.cameraMode;
  }

  public getFlySpeed(): number {
    return this.flySpeed;
  }

  public setCameraMode(mode: CameraMode): void {
    if (this.cameraMode === mode) return;
    this.cameraMode = mode;
    this.clearFlyKeys();

    if (mode === 'fly') {
      this.controls.enabled = false;
      this.syncFlyAnglesFromCamera();
    } else {
      // Place orbit target cleanly in front of current camera view
      const forward = this.tempVecA
        .set(0, 0, -1)
        .applyQuaternion(this.camera.quaternion)
        .normalize();
      let targetDist = 3.0;
      if (forward.y < -0.05) {
        const planeDist = (0.9 - this.camera.position.y) / forward.y;
        if (planeDist > 0.8 && planeDist < 8.0) {
          targetDist = planeDist;
        }
      }
      this.controls.target
        .copy(this.camera.position)
        .addScaledVector(forward, targetDist);
      this.controls.enabled = true;
      this.controls.update();
    }
  }

  public applyCameraPreset(name: CameraPresetName): void {
    const preset = CAMERA_PRESETS[name];
    let startTarget: THREE.Vector3;

    if (this.cameraMode === 'orbit') {
      startTarget = this.controls.target.clone();
    } else {
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(
        this.camera.quaternion
      );
      startTarget = this.camera.position.clone().addScaledVector(forward, 3.0);
    }

    this.presetTransition = {
      startPos: this.camera.position.clone(),
      endPos: new THREE.Vector3(...preset.position),
      startTarget,
      endTarget: new THREE.Vector3(...preset.target),
      elapsed: 0,
      duration: 0.35,
    };
  }

  public rotateFlyCamera(dx: number, dy: number): void {
    if (this.cameraMode !== 'fly') return;
    this.presetTransition = null;
    const sensitivity = 0.0032;
    this.flyYaw -= dx * sensitivity;
    this.flyPitch = THREE.MathUtils.clamp(
      this.flyPitch - dy * sensitivity,
      -Math.PI * 0.485,
      Math.PI * 0.485
    );
    this.tempEuler.set(this.flyPitch, this.flyYaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(this.tempEuler);
  }

  public adjustFlySpeed(wheelDeltaY: number): number {
    const factor = wheelDeltaY < 0 ? 1.12 : 0.89;
    this.flySpeed = THREE.MathUtils.clamp(this.flySpeed * factor, 0.8, 18.0);
    return this.flySpeed;
  }

  public setFlyKey(code: string, pressed: boolean): void {
    if (pressed) {
      this.pressedFlyKeys.add(code);
      if (this.cameraMode === 'fly') {
        this.presetTransition = null;
      }
    } else {
      this.pressedFlyKeys.delete(code);
    }
  }

  public clearFlyKeys(): void {
    this.pressedFlyKeys.clear();
  }

  private syncFlyAnglesFromCamera(): void {
    this.tempEuler.setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.flyPitch = this.tempEuler.x;
    this.flyYaw = this.tempEuler.y;
  }

  public updateCamera(dt: number): void {
    if (this.presetTransition) {
      const anim = this.presetTransition;
      anim.elapsed += dt;
      const rawT = Math.min(1, anim.elapsed / anim.duration);
      const ease = 1 - Math.pow(1 - rawT, 3);

      this.camera.position.lerpVectors(anim.startPos, anim.endPos, ease);
      this.tempVecA.lerpVectors(anim.startTarget, anim.endTarget, ease);
      this.controls.target.copy(this.tempVecA);
      this.camera.lookAt(this.tempVecA);
      this.syncFlyAnglesFromCamera();

      if (rawT >= 1) {
        this.presetTransition = null;
        if (this.cameraMode === 'orbit') {
          this.controls.update();
        }
      }
      return;
    }

    if (this.cameraMode === 'fly') {
      const forward = this.tempVecA
        .set(0, 0, -1)
        .applyQuaternion(this.camera.quaternion)
        .normalize();
      const right = this.tempVecB
        .set(1, 0, 0)
        .applyQuaternion(this.camera.quaternion)
        .normalize();
      const move = this.tempVecC.set(0, 0, 0);

      if (this.pressedFlyKeys.has('KeyW') || this.pressedFlyKeys.has('ArrowUp')) {
        move.add(forward);
      }
      if (this.pressedFlyKeys.has('KeyS') || this.pressedFlyKeys.has('ArrowDown')) {
        move.sub(forward);
      }
      if (this.pressedFlyKeys.has('KeyD') || this.pressedFlyKeys.has('ArrowRight')) {
        move.add(right);
      }
      if (this.pressedFlyKeys.has('KeyA') || this.pressedFlyKeys.has('ArrowLeft')) {
        move.sub(right);
      }
      if (this.pressedFlyKeys.has('KeyE') || this.pressedFlyKeys.has('Space')) {
        move.y += 1;
      }
      if (
        this.pressedFlyKeys.has('KeyQ') ||
        this.pressedFlyKeys.has('ShiftLeft') ||
        this.pressedFlyKeys.has('ShiftRight')
      ) {
        move.y -= 1;
      }

      if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(this.flySpeed * dt);
        this.camera.position.add(move);
        this.camera.position.y = Math.max(0.06, this.camera.position.y);
      }
    } else {
      this.controls.update();
    }
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

  public setDebugTelemetryVisible(visible: boolean): void {
    this.telemetryGroup.visible = visible;
  }

  public isDebugTelemetryVisible(): boolean {
    return this.telemetryGroup.visible;
  }

  /**
   * Updates 3D debug markers (cyan CoM + dashed plumb line + amber Capture Point ring)
   * from `get_ragdoll_telemetry()`.
   */
  public syncRagdollTelemetry(telemetry: Float32Array): void {
    const count = Math.floor(telemetry.length / TELEMETRY_STRIDE);

    while (this.telemetryPool.length < count) {
      this.telemetryPool.push(this.createTelemetryVisual());
    }

    for (let i = 0; i < this.telemetryPool.length; i++) {
      const entry = this.telemetryPool[i];
      if (i >= count) {
        entry.group.visible = false;
        continue;
      }

      const o = i * TELEMETRY_STRIDE;
      const comX = telemetry[o];
      const comY = telemetry[o + 1];
      const comZ = telemetry[o + 2];
      const cpX = telemetry[o + 3];
      const cpZ = telemetry[o + 4];
      const floorY = 0.015;

      entry.group.visible = true;
      entry.comMarker.position.set(comX, comY, comZ);

      entry.plumbPositions[0] = comX;
      entry.plumbPositions[1] = comY;
      entry.plumbPositions[2] = comZ;
      entry.plumbPositions[3] = comX;
      entry.plumbPositions[4] = floorY;
      entry.plumbPositions[5] = comZ;
      const plumbAttr = entry.plumbLine.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      plumbAttr.needsUpdate = true;
      entry.plumbLine.computeLineDistances();

      entry.cpRing.position.set(cpX, floorY, cpZ);

      entry.cpTetherPositions[0] = comX;
      entry.cpTetherPositions[1] = floorY;
      entry.cpTetherPositions[2] = comZ;
      entry.cpTetherPositions[3] = cpX;
      entry.cpTetherPositions[4] = floorY;
      entry.cpTetherPositions[5] = cpZ;
      const tetherAttr = entry.cpTether.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      tetherAttr.needsUpdate = true;
    }
  }

  private createTelemetryVisual(): TelemetryVisual {
    const group = new THREE.Group();

    const comMarker = new THREE.Mesh(this.comGeo, this.comMat);
    comMarker.renderOrder = 10;
    group.add(comMarker);

    const plumbPositions = new Float32Array(6);
    const plumbGeo = new THREE.BufferGeometry();
    plumbGeo.setAttribute('position', new THREE.BufferAttribute(plumbPositions, 3));
    const plumbLine = new THREE.Line(plumbGeo, this.plumbMat);
    plumbLine.frustumCulled = false;
    plumbLine.renderOrder = 9;
    group.add(plumbLine);

    const cpRing = new THREE.Mesh(this.cpRingGeo, this.cpRingMat);
    cpRing.rotation.x = -Math.PI / 2;
    cpRing.renderOrder = 5;
    group.add(cpRing);

    const cpTetherPositions = new Float32Array(6);
    const cpTetherGeo = new THREE.BufferGeometry();
    cpTetherGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(cpTetherPositions, 3)
    );
    const cpTether = new THREE.Line(cpTetherGeo, this.cpTetherMat);
    cpTether.frustumCulled = false;
    cpTether.renderOrder = 4;
    group.add(cpTether);

    this.telemetryGroup.add(group);
    return {
      group,
      comMarker,
      plumbLine,
      plumbPositions,
      cpRing,
      cpTether,
      cpTetherPositions,
    };
  }

  /**
   * Updates or hides the translucent Place Box wireframe preview on the floor.
   */
  public setPlaceBoxPreview(visible: boolean, x = 0, z = 0): void {
    this.placePreview.visible = visible;
    if (visible) {
      this.placePreview.position.set(x, 0.14, z);
    }
  }

  /**
   * Raycasts against dynamic meshes to locate exact surface hit coordinates for FX.
   */
  public raycastDynamicPoint(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    const hits = raycaster.intersectObjects(this.dynamicGroup.children, true);
    return hits.length > 0 ? hits[0].point.clone() : null;
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
   * Spawns a brief expanding shockwave FX at an impact point.
   */
  public spawnImpulseEffect(
    pos: THREE.Vector3,
    variant: ImpulseEffectVariant = 'shove'
  ): void {
    const isPunch = variant === 'punch';
    const group = new THREE.Group();
    group.position.copy(pos);
    group.quaternion.copy(this.camera.quaternion);

    const ringMat = new THREE.MeshBasicMaterial({
      color: isPunch ? 0xff4518 : 0xe8942c,
      transparent: true,
      opacity: isPunch ? 0.95 : 0.85,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(this.impulseRingGeo, ringMat);
    group.add(ring);

    let flashCore: THREE.Mesh | undefined;
    if (isPunch) {
      const coreMat = new THREE.MeshBasicMaterial({
        color: 0xff8a3d,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      });
      flashCore = new THREE.Mesh(this.punchCoreGeo, coreMat);
      group.add(flashCore);
    }

    this.scene.add(group);
    this.impulseEffects.push({
      group,
      ring,
      flashCore,
      age: 0,
      duration: isPunch ? 0.18 : 0.34,
      maxScale: isPunch ? 2.8 : 4.2,
    });
  }

  public updateEffects(dt: number): void {
    for (let i = this.impulseEffects.length - 1; i >= 0; i--) {
      const fx = this.impulseEffects[i];
      fx.age += dt;
      const t = Math.min(1, fx.age / fx.duration);
      const scale = 1 + t * fx.maxScale;

      fx.ring.scale.setScalar(scale);
      fx.group.quaternion.copy(this.camera.quaternion);
      (fx.ring.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.88;

      if (fx.flashCore) {
        fx.flashCore.scale.setScalar(1 + t * 0.9);
        (fx.flashCore.material as THREE.MeshBasicMaterial).opacity =
          Math.max(0, 1 - t * 1.45);
      }

      if (t >= 1) {
        this.scene.remove(fx.group);
        (fx.ring.material as THREE.Material).dispose();
        if (fx.flashCore) {
          (fx.flashCore.material as THREE.Material).dispose();
        }
        this.impulseEffects.splice(i, 1);
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
    this.renderer.render(this.scene, this.camera);
  }
}
