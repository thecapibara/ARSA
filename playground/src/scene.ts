import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  LabCameraController,
  type CameraMode,
  type CameraPresetName,
} from './camera.ts';
import {
  LabEffectsManager,
  type DynamiteFuseTip,
  type ImpulseEffectVariant,
} from './effects.ts';
import { createBoneMesh, ShapeType } from './mannequin.ts';

export type { CameraMode, CameraPresetName, ImpulseEffectVariant };
export type PlacerMode = 'ragdoll' | 'curb' | 'cube' | 'sphere' | 'dynamite';

export const DYNAMITE_BONE_ID = 250;

interface CachedBody {
  object: THREE.Object3D;
  signature: string;
  isProp: boolean;
}

interface CableVisual {
  group: THREE.Group;
  cable: THREE.Mesh;
  shackle: THREE.Group;
}

const UP_AXIS = new THREE.Vector3(0, 1, 0);

interface TelemetryVisual {
  group: THREE.Group;
  comMarker: THREE.Mesh;
  plumbLine: THREE.Line;
  plumbPositions: Float32Array;
  cpRing: THREE.Mesh;
  cpTether: THREE.Line;
  cpTetherPositions: Float32Array;
  woundGroup: THREE.Group;
  woundRing: THREE.Mesh;
}

/**
 * Computes the per-ragdoll float stride for `get_ragdoll_telemetry()`,
 * supporting both Phase 2 (11 floats) and Phase 3 (16 floats).
 */
export function getTelemetryStride(telemetry: Float32Array): number {
  return telemetry.length >= 16 && telemetry.length % 16 === 0 ? 16 : 11;
}

/**
 * Manages the Three.js Aperture/Euphoria test-chamber scene, lighting,
 * procedural meter-grid floor, static obstacles, dynamic bodies, tether cables,
 * placement ghost previews, and debug telemetry overlay, composing
 * `LabCameraController` and `LabEffectsManager`.
 */
export class LabScene {
  public readonly scene: THREE.Scene;
  public readonly camera: THREE.PerspectiveCamera;
  public readonly renderer: THREE.WebGLRenderer;
  public readonly controls: OrbitControls;

  private readonly cameraController: LabCameraController;
  private readonly effects: LabEffectsManager;

  private readonly staticGroup = new THREE.Group();
  private readonly dynamicGroup = new THREE.Group();
  private readonly bodyPool: CachedBody[] = [];

  private readonly propBoxMat: THREE.MeshStandardMaterial;
  private readonly propSphereMat: THREE.MeshStandardMaterial;
  private readonly propAccentMat: THREE.MeshStandardMaterial;
  private readonly propEdgeMat: THREE.LineBasicMaterial;
  private readonly dynamiteRedMat: THREE.MeshStandardMaterial;
  private readonly dynamiteTapeMat: THREE.MeshStandardMaterial;
  private readonly dynamiteCordMat: THREE.MeshStandardMaterial;
  private readonly staticBoxMat: THREE.MeshStandardMaterial;
  private readonly staticEdgeMat: THREE.LineBasicMaterial;
  private readonly terrainGroup = new THREE.Group();
  private readonly terrainMat: THREE.MeshStandardMaterial;
  private readonly terrainWireMat: THREE.MeshBasicMaterial;
  private terrainSurfaceMesh: THREE.Mesh | null = null;

  private readonly placePreviewGroup: THREE.Group;
  private readonly placePreviewRagdoll: THREE.Group;
  private readonly placePreviewCurb: THREE.Group;
  private readonly placePreviewCube: THREE.Group;
  private readonly placePreviewSphere: THREE.Group;
  private readonly placePreviewDynamite: THREE.Group;

  private readonly dynamiteFuseTips = new Set<DynamiteFuseTip>();

  // Steel Cable / Tether Renderer (Wrecking Ball)
  private readonly tetherCableGroup = new THREE.Group();
  private readonly cablePool: CableVisual[] = [];
  private readonly cableGeo: THREE.CylinderGeometry;
  private readonly cableMat: THREE.MeshStandardMaterial;
  private readonly shacklePlateGeo: THREE.CylinderGeometry;
  private readonly shackleRingGeo: THREE.TorusGeometry;
  private readonly shackleMat: THREE.MeshStandardMaterial;

  // Debug Telemetry Overlay (CoM + plumb line + Capture Point + Wound markers)
  private readonly telemetryGroup = new THREE.Group();
  private readonly telemetryPool: TelemetryVisual[] = [];
  private readonly comGeo: THREE.SphereGeometry;
  private readonly comMat: THREE.MeshBasicMaterial;
  private readonly plumbMat: THREE.LineDashedMaterial;
  private readonly cpRingGeo: THREE.RingGeometry;
  private readonly cpRingMat: THREE.MeshBasicMaterial;
  private readonly cpTetherMat: THREE.LineBasicMaterial;
  private readonly woundRingGeo: THREE.RingGeometry;
  private readonly woundRingMat: THREE.MeshBasicMaterial;
  private readonly woundCoreGeo: THREE.SphereGeometry;
  private readonly woundCoreMat: THREE.MeshBasicMaterial;

  private readonly tempVecA = new THREE.Vector3();
  private readonly tempVecB = new THREE.Vector3();
  private readonly tempVecC = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14181f);
    this.scene.fog = new THREE.FogExp2(0x14181f, 0.02);

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

    this.cameraController = new LabCameraController(canvas);
    this.camera = this.cameraController.camera;
    this.controls = this.cameraController.controls;
    this.effects = new LabEffectsManager(this.scene, this.camera);

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
    this.dynamiteRedMat = new THREE.MeshStandardMaterial({
      color: 0xc92a2a,
      roughness: 0.68,
      metalness: 0.06,
    });
    this.dynamiteTapeMat = new THREE.MeshStandardMaterial({
      color: 0x181b22,
      roughness: 0.55,
      metalness: 0.16,
    });
    this.dynamiteCordMat = new THREE.MeshStandardMaterial({
      color: 0xd6cfc2,
      roughness: 0.85,
      metalness: 0.0,
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
    this.terrainMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.78,
      metalness: 0.12,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    this.terrainWireMat = new THREE.MeshBasicMaterial({
      color: 0x64748b,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });

    this.setupLighting();
    this.setupFloor();

    this.scene.add(this.staticGroup);
    this.scene.add(this.terrainGroup);
    this.scene.add(this.dynamicGroup);

    // Multi-mode 3D Placement Ghost Preview
    this.placePreviewGroup = new THREE.Group();
    this.placePreviewRagdoll = this.buildRagdollGhostPreview();
    this.placePreviewCurb = this.buildBoxGhostPreview(0.7, 0.28, 0.36, 0xe8942c, 0xf5a623);
    this.placePreviewCube = this.buildBoxGhostPreview(0.48, 0.48, 0.48, 0xe8942c, 0xf5a623);
    this.placePreviewSphere = this.buildSphereGhostPreview(0.28, 0xe8942c, 0xf5a623);
    this.placePreviewDynamite = this.buildDynamiteGhostPreview();
    this.placePreviewGroup.add(
      this.placePreviewRagdoll,
      this.placePreviewCurb,
      this.placePreviewCube,
      this.placePreviewSphere,
      this.placePreviewDynamite
    );
    this.placePreviewGroup.visible = false;
    this.scene.add(this.placePreviewGroup);

    // Steel cable & gunmetal anchor shackle shared assets
    this.cableGeo = new THREE.CylinderGeometry(0.015, 0.015, 1, 10);
    this.cableMat = new THREE.MeshStandardMaterial({
      color: 0x8b95a5,
      roughness: 0.3,
      metalness: 0.88,
    });
    this.shacklePlateGeo = new THREE.CylinderGeometry(0.075, 0.09, 0.045, 16);
    this.shackleRingGeo = new THREE.TorusGeometry(0.042, 0.014, 10, 20);
    this.shackleMat = new THREE.MeshStandardMaterial({
      color: 0x353d4a,
      roughness: 0.42,
      metalness: 0.85,
    });
    this.scene.add(this.tetherCableGroup);

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
    this.woundRingGeo = new THREE.RingGeometry(0.052, 0.082, 24);
    this.woundRingMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    this.woundCoreGeo = new THREE.SphereGeometry(0.02, 10, 8);
    this.woundCoreMat = new THREE.MeshBasicMaterial({
      color: 0xff2e4c,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
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
    return this.cameraController.getCameraMode();
  }

  public getFlySpeed(): number {
    return this.cameraController.getFlySpeed();
  }

  public setCameraMode(mode: CameraMode): void {
    this.cameraController.setCameraMode(mode);
  }

  public applyCameraPreset(name: CameraPresetName): void {
    this.cameraController.applyCameraPreset(name);
  }

  /**
   * Smoothly transitions the camera position and focus target.
   */
  public frameCamera(
    position: [number, number, number],
    target: [number, number, number],
    duration = 0.35
  ): void {
    this.cameraController.frameCamera(position, target, duration);
  }

  public rotateFlyCamera(dx: number, dy: number): void {
    this.cameraController.rotateFlyLook(dx, dy);
  }

  public adjustFlySpeed(wheelDeltaY: number): number {
    return this.cameraController.adjustFlySpeed(wheelDeltaY);
  }

  public setFlyKey(code: string, pressed: boolean): void {
    this.cameraController.setFlyKey(code, pressed);
  }

  public clearFlyKeys(): void {
    this.cameraController.clearFlyKeys();
  }

  public updateCamera(dt: number): void {
    this.cameraController.update(dt);
  }

  private clearStaticBoxes(): void {
    while (this.staticGroup.children.length > 0) {
      const child = this.staticGroup.children.pop()!;
      child.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
        }
      });
    }
  }

  private addStaticBoxMesh(
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
    qx = 0,
    qy = 0,
    qz = 0,
    qw = 1
  ): void {
    // Skip huge ground-plane slabs since our procedural grid floor sits at y = 0
    if (hx >= 14 && hz >= 14 && cy + hy <= 0.05) {
      return;
    }

    const geo = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
    const mesh = new THREE.Mesh(geo, this.staticBoxMat);
    mesh.position.set(cx, cy, cz);
    mesh.quaternion.set(qx, qy, qz, qw);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      this.staticEdgeMat
    );
    mesh.add(edges);

    this.staticGroup.add(mesh);
  }

  /**
   * Rebuilds static lab obstacles (stairs, drop platforms, walls) from `get_static_boxes()`.
   * Expects 6 floats per box: `[cx, cy, cz, hx, hy, hz]`.
   */
  public syncStaticBoxes(boxes: Float32Array | number[]): void {
    this.clearStaticBoxes();
    const count = Math.floor(boxes.length / 6);
    for (let i = 0; i < count; i++) {
      const o = i * 6;
      this.addStaticBoxMesh(
        boxes[o],
        boxes[o + 1],
        boxes[o + 2],
        boxes[o + 3],
        boxes[o + 4],
        boxes[o + 5]
      );
    }
  }

  /**
   * Rebuilds static lab obstacles with full 3D orientation from `get_static_boxes_oriented()`.
   * Expects 10 floats per box: `[cx, cy, cz, hx, hy, hz, qx, qy, qz, qw]`.
   */
  public syncStaticBoxesOriented(flat10: Float32Array | number[]): void {
    this.clearStaticBoxes();
    const count = Math.floor(flat10.length / 10);
    for (let i = 0; i < count; i++) {
      const o = i * 10;
      this.addStaticBoxMesh(
        flat10[o],
        flat10[o + 1],
        flat10[o + 2],
        flat10[o + 3],
        flat10[o + 4],
        flat10[o + 5],
        flat10[o + 6],
        flat10[o + 7],
        flat10[o + 8],
        flat10[o + 9]
      );
    }
  }

  /**
   * Builds or clears the heightfield terrain mesh from `get_heightfield_meta()`
   * and `get_heightfield_heights()`.
   */
  public syncHeightfield(
    meta: Float32Array | number[],
    heights: Float32Array | number[]
  ): void {
    while (this.terrainGroup.children.length > 0) {
      const child = this.terrainGroup.children.pop()!;
      if (child instanceof THREE.Mesh && child === this.terrainSurfaceMesh) {
        child.geometry.dispose();
      }
    }
    this.terrainSurfaceMesh = null;

    if (meta.length < 5 || heights.length === 0) {
      return;
    }

    const originX = meta[0];
    const originZ = meta[1];
    const cellSize = meta[2];
    const cols = Math.round(meta[3]);
    const rows = Math.round(meta[4]);
    if (cols < 2 || rows < 2 || heights.length < cols * rows) {
      return;
    }

    const vertexCount = cols * rows;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);

    let minH = Infinity;
    let maxH = -Infinity;
    for (let i = 0; i < vertexCount; i++) {
      const h = heights[i];
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;
    }
    const span = Math.max(0.05, maxH - minH);

    // Subtle elevation tint: deep architectural slate in valleys -> cool slate-blue on peaks
    const lowR = 0x22 / 255;
    const lowG = 0x2c / 255;
    const lowB = 0x3a / 255;
    const highR = 0x3b / 255;
    const highG = 0x4f / 255;
    const highB = 0x6b / 255;

    for (let r = 0; r < rows; r++) {
      const z = originZ + r * cellSize;
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const h = heights[idx];
        const pOff = idx * 3;
        positions[pOff] = originX + c * cellSize;
        positions[pOff + 1] = h;
        positions[pOff + 2] = z;

        const t = (h - minH) / span;
        colors[pOff] = lowR + (highR - lowR) * t;
        colors[pOff + 1] = lowG + (highG - lowG) * t;
        colors[pOff + 2] = lowB + (highB - lowB) * t;
      }
    }

    const quadCount = (cols - 1) * (rows - 1);
    const indices = new Uint32Array(quadCount * 6);
    let ptr = 0;
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const i00 = r * cols + c;
        const i10 = i00 + 1;
        const i01 = i00 + cols;
        const i11 = i01 + 1;

        indices[ptr++] = i00;
        indices[ptr++] = i01;
        indices[ptr++] = i10;

        indices[ptr++] = i10;
        indices[ptr++] = i01;
        indices[ptr++] = i11;
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();

    const surface = new THREE.Mesh(geo, this.terrainMat);
    surface.receiveShadow = true;
    surface.castShadow = true;

    const wire = new THREE.Mesh(geo, this.terrainWireMat);
    wire.receiveShadow = false;
    wire.castShadow = false;

    this.terrainSurfaceMesh = surface;
    this.terrainGroup.add(surface, wire);
  }

  /**
   * Renders metallic steel cables and gunmetal anchor shackles from `get_tethers()`.
   * Expects 6 floats per tether: `[ax, ay, az, bx, by, bz]`.
   */
  public syncTethers(tethers: Float32Array): void {
    const count = Math.floor(tethers.length / 6);

    if (count <= 0) {
      while (this.cablePool.length > 0) {
        const removed = this.cablePool.pop()!;
        this.tetherCableGroup.remove(removed.group);
      }
      return;
    }

    while (this.cablePool.length > count) {
      const removed = this.cablePool.pop()!;
      this.tetherCableGroup.remove(removed.group);
    }

    while (this.cablePool.length < count) {
      this.cablePool.push(this.createCableVisual());
    }

    for (let i = 0; i < count; i++) {
      const o = i * 6;
      const entry = this.cablePool[i];

      this.tempVecA.set(tethers[o], tethers[o + 1], tethers[o + 2]);
      this.tempVecB.set(tethers[o + 3], tethers[o + 4], tethers[o + 5]);
      this.tempVecC.subVectors(this.tempVecB, this.tempVecA);
      const len = this.tempVecC.length();

      if (len < 1e-4) {
        entry.group.visible = false;
        continue;
      }

      entry.group.visible = true;
      entry.shackle.position.copy(this.tempVecA);

      entry.cable.position
        .addVectors(this.tempVecA, this.tempVecB)
        .multiplyScalar(0.5);
      this.tempVecC.divideScalar(len);
      entry.cable.quaternion.setFromUnitVectors(UP_AXIS, this.tempVecC);
      entry.cable.scale.set(1, len, 1);
    }
  }

  private createCableVisual(): CableVisual {
    const group = new THREE.Group();

    const cable = new THREE.Mesh(this.cableGeo, this.cableMat);
    cable.castShadow = true;
    cable.receiveShadow = true;
    group.add(cable);

    const shackle = new THREE.Group();
    const plate = new THREE.Mesh(this.shacklePlateGeo, this.shackleMat);
    plate.castShadow = true;
    plate.receiveShadow = true;
    const ring = new THREE.Mesh(this.shackleRingGeo, this.shackleMat);
    ring.position.y = -0.032;
    ring.castShadow = true;
    shackle.add(plate, ring);
    group.add(shackle);

    this.tetherCableGroup.add(group);
    return { group, cable, shackle };
  }

  /**
   * Synchronizes all dynamic rigid bodies (mannequin bones & props).
   */
  public syncBodies(
    transforms: Float32Array,
    descriptors: Float32Array,
    count: number,
    forceRebuild = false
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

      if (forceRebuild || !entry || entry.signature !== sig) {
        if (entry) {
          this.dynamicGroup.remove(entry.object);
          this.disposeObjectGeometries(entry.object);
        }

        const isProp = ragdollId < 0 || boneId < 0 || boneId === DYNAMITE_BONE_ID;
        const object = isProp
          ? this.createPropMesh(shapeType, dim0, dim1, dim2, boneId)
          : createBoneMesh(boneId, shapeType, dim0, dim1, dim2);

        this.dynamicGroup.add(object);
        entry = { object, signature: sig, isProp };
        this.bodyPool[i] = entry;
      }

      const tOff = i * 7;
      const py = transforms[tOff + 1];
      const fuseTip = entry.object.userData.fuseTip as DynamiteFuseTip | undefined;
      if (py < -20.0) {
        entry.object.visible = false;
        if (fuseTip) {
          this.dynamiteFuseTips.delete(fuseTip);
        }
        continue;
      }

      entry.object.visible = true;
      if (fuseTip) {
        this.dynamiteFuseTips.add(fuseTip);
      }
      entry.object.position.set(
        transforms[tOff],
        py,
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
    dim2: number,
    boneId = -1
  ): THREE.Object3D {
    if (boneId === DYNAMITE_BONE_ID) {
      return this.createDynamiteMesh(shapeType, dim0, dim1, dim2);
    }

    const group = new THREE.Group();
    group.userData.isProp = true;

    if (shapeType === ShapeType.Box) {
      const geo = new THREE.BoxGeometry(dim0 * 2, dim1 * 2, dim2 * 2);
      const mesh = new THREE.Mesh(geo, this.propBoxMat);
      mesh.userData.isProp = true;
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
      sphere.userData.isProp = true;
      sphere.castShadow = true;
      sphere.receiveShadow = true;
      group.add(sphere);

      const band = new THREE.Mesh(
        new THREE.TorusGeometry(r * 0.995, r * 0.045, 8, 28),
        this.propAccentMat
      );
      band.userData.isProp = true;
      band.rotation.x = Math.PI / 2;
      group.add(band);
    } else {
      const r = Math.max(0.03, dim0);
      const len = Math.max(0.04, dim1 * 2);
      const cap = new THREE.Mesh(
        new THREE.CapsuleGeometry(r, len, 8, 16),
        this.propBoxMat
      );
      cap.userData.isProp = true;
      cap.castShadow = true;
      cap.receiveShadow = true;
      group.add(cap);
    }

    return group;
  }

  private createDynamiteMesh(
    shapeType: number,
    dim0: number,
    dim1: number,
    dim2: number
  ): THREE.Object3D {
    const root = new THREE.Group();
    const bundle = new THREE.Group();

    const maxHalfLen =
      shapeType === ShapeType.Box
        ? Math.max(0.11, dim0, dim1, dim2)
        : Math.max(0.11, dim1 + dim0 * 0.5);
    const minHalfThick =
      shapeType === ShapeType.Box
        ? Math.max(0.042, Math.min(dim0, dim1, dim2))
        : Math.max(0.042, dim0);

    const stickLen = maxHalfLen * 2;
    const stickR = Math.max(0.022, minHalfThick * 0.52);
    const radialOffset = stickR * 1.06;

    const stickGeo = new THREE.CylinderGeometry(stickR, stickR, stickLen, 14);
    for (let i = 0; i < 3; i++) {
      const angle = (i * Math.PI * 2) / 3;
      const stick = new THREE.Mesh(stickGeo, this.dynamiteRedMat);
      stick.position.set(
        Math.cos(angle) * radialOffset,
        0,
        Math.sin(angle) * radialOffset
      );
      stick.castShadow = true;
      stick.receiveShadow = true;
      bundle.add(stick);
    }

    // Two black electrical-tape bands wrapping the 3 clustered sticks
    const bandR = radialOffset + stickR * 1.08;
    const bandGeo = new THREE.CylinderGeometry(bandR, bandR, stickLen * 0.12, 18);
    for (const bandY of [-stickLen * 0.25, stickLen * 0.25]) {
      const band = new THREE.Mesh(bandGeo, this.dynamiteTapeMat);
      band.position.y = bandY;
      band.castShadow = true;
      band.receiveShadow = true;
      bundle.add(band);
    }

    // Braided fuse cord emerging from the top end
    const cordLen = Math.max(0.055, stickLen * 0.24);
    const cordPivot = new THREE.Group();
    cordPivot.position.set(0, stickLen * 0.5, 0);
    cordPivot.rotation.z = -0.26;
    const cord = new THREE.Mesh(
      new THREE.CylinderGeometry(0.0045, 0.0055, cordLen, 8),
      this.dynamiteCordMat
    );
    cord.position.y = cordLen * 0.5;
    cordPivot.add(cord);

    // Glowing flickering amber/orange fuse tip
    const tipCoreMat = new THREE.MeshBasicMaterial({
      color: 0xffe484,
      transparent: true,
      opacity: 0.98,
    });
    const tipCore = new THREE.Mesh(
      new THREE.SphereGeometry(0.014, 10, 8),
      tipCoreMat
    );
    tipCore.position.y = cordLen;
    cordPivot.add(tipCore);

    const tipHaloMat = new THREE.MeshBasicMaterial({
      color: 0xff6b00,
      transparent: true,
      opacity: 0.78,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const tipHalo = new THREE.Mesh(
      new THREE.SphereGeometry(0.032, 10, 8),
      tipHaloMat
    );
    tipHalo.position.y = cordLen;
    cordPivot.add(tipHalo);

    bundle.add(cordPivot);

    // Align sticks along the collider's longest axis if it is a Box
    if (shapeType === ShapeType.Box) {
      if (dim0 >= dim1 && dim0 >= dim2) {
        bundle.rotation.z = Math.PI / 2;
      } else if (dim2 >= dim1 && dim2 >= dim0) {
        bundle.rotation.x = Math.PI / 2;
      }
    }

    root.add(bundle);

    const fuseTip: DynamiteFuseTip = {
      core: tipCore,
      halo: tipHalo,
      phase: Math.random() * Math.PI * 2,
    };
    this.dynamiteFuseTips.add(fuseTip);
    root.userData.fuseTip = fuseTip;

    return root;
  }

  private disposeObjectGeometries(root: THREE.Object3D): void {
    const fuseTip = root.userData.fuseTip as DynamiteFuseTip | undefined;
    if (fuseTip) {
      this.dynamiteFuseTips.delete(fuseTip);
      (fuseTip.core.material as THREE.Material).dispose();
      (fuseTip.halo.material as THREE.Material).dispose();
    }
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
   * Updates 3D debug markers (cyan CoM + dashed plumb line + amber Capture Point ring
   * + pulsing crimson Wound marker) from `get_ragdoll_telemetry()`.
   */
  public syncRagdollTelemetry(telemetry: Float32Array): void {
    const stride = getTelemetryStride(telemetry);
    const count = Math.floor(telemetry.length / stride);

    while (this.telemetryPool.length < count) {
      this.telemetryPool.push(this.createTelemetryVisual());
    }

    for (let i = 0; i < this.telemetryPool.length; i++) {
      const entry = this.telemetryPool[i];
      if (i >= count) {
        entry.group.visible = false;
        continue;
      }

      const o = i * stride;
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

      if (stride >= 16) {
        const woundRegion = Math.round(telemetry[o + 12]);
        if (woundRegion > 0) {
          entry.woundGroup.visible = true;
          entry.woundGroup.position.set(
            telemetry[o + 13],
            telemetry[o + 14],
            telemetry[o + 15]
          );
          entry.woundGroup.quaternion.copy(this.camera.quaternion);
        } else {
          entry.woundGroup.visible = false;
        }
      } else {
        entry.woundGroup.visible = false;
      }
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

    const woundGroup = new THREE.Group();
    const woundRing = new THREE.Mesh(this.woundRingGeo, this.woundRingMat);
    woundRing.renderOrder = 12;
    const woundCore = new THREE.Mesh(this.woundCoreGeo, this.woundCoreMat);
    woundCore.renderOrder = 13;
    woundGroup.add(woundRing, woundCore);
    woundGroup.visible = false;
    group.add(woundGroup);

    this.telemetryGroup.add(group);
    return {
      group,
      comMarker,
      plumbLine,
      plumbPositions,
      cpRing,
      cpTether,
      cpTetherPositions,
      woundGroup,
      woundRing,
    };
  }

  private buildBoxGhostPreview(
    sx: number,
    sy: number,
    sz: number,
    fillHex: number,
    wireHex: number
  ): THREE.Group {
    const group = new THREE.Group();
    const geo = new THREE.BoxGeometry(sx, sy, sz);
    const fill = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color: fillHex,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      })
    );
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({
        color: wireHex,
        transparent: true,
        opacity: 0.85,
      })
    );
    fill.position.y = sy * 0.5;
    wire.position.y = sy * 0.5;
    group.add(fill, wire);
    group.visible = false;
    return group;
  }

  private buildSphereGhostPreview(
    radius: number,
    fillHex: number,
    wireHex: number
  ): THREE.Group {
    const group = new THREE.Group();
    const geo = new THREE.SphereGeometry(radius, 20, 14);
    const fill = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color: fillHex,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      })
    );
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({
        color: wireHex,
        transparent: true,
        opacity: 0.55,
      })
    );
    fill.position.y = radius;
    wire.position.y = radius;
    group.add(fill, wire);
    group.visible = false;
    return group;
  }

  private buildDynamiteGhostPreview(): THREE.Group {
    const group = new THREE.Group();

    const fillMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.2,
      depthWrite: false,
    });
    const wireMat = new THREE.LineBasicMaterial({
      color: 0xf87171,
      transparent: true,
      opacity: 0.88,
    });

    // Translucent red wireframe 3-stick vertical dynamite bundle centered at y = 0.175
    // (matches vertical Capsule { radius: 0.055, half_length: 0.12 } whose bottom sits 0.175m below center)
    const bundle = new THREE.Group();
    bundle.position.y = 0.175;

    const stickR = 0.028;
    const stickLen = 0.295;
    const radialOffset = stickR * 1.06;
    const stickGeo = new THREE.CylinderGeometry(stickR, stickR, stickLen, 10, 1);
    const stickWireGeo = new THREE.WireframeGeometry(stickGeo);

    for (let i = 0; i < 3; i++) {
      const angle = (i * Math.PI * 2) / 3;
      const ox = Math.cos(angle) * radialOffset;
      const oz = Math.sin(angle) * radialOffset;

      const stickFill = new THREE.Mesh(stickGeo, fillMat);
      stickFill.position.set(ox, 0, oz);
      const stickWire = new THREE.LineSegments(stickWireGeo, wireMat);
      stickWire.position.set(ox, 0, oz);
      bundle.add(stickFill, stickWire);
    }

    // Fuse stub emerging from bundle end
    const cordGeo = new THREE.CylinderGeometry(0.004, 0.005, 0.055, 6);
    const cordMesh = new THREE.Mesh(cordGeo, fillMat);
    cordMesh.position.set(0, stickLen * 0.5 + 0.025, 0);
    cordMesh.rotation.z = -0.26;
    bundle.add(cordMesh);

    group.add(bundle);

    // Ground blast-hint rings (inner placement ring + outer danger perimeter)
    const innerRing = new THREE.Mesh(
      new THREE.RingGeometry(0.24, 0.28, 36),
      new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.82,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    innerRing.rotation.x = -Math.PI / 2;
    innerRing.position.y = 0.008;

    const blastRing = new THREE.Mesh(
      new THREE.RingGeometry(0.82, 0.86, 48),
      new THREE.MeshBasicMaterial({
        color: 0xf87171,
        transparent: true,
        opacity: 0.36,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    blastRing.rotation.x = -Math.PI / 2;
    blastRing.position.y = 0.008;

    group.add(innerRing, blastRing);
    group.visible = false;
    return group;
  }

  private buildRagdollGhostPreview(): THREE.Group {
    const group = new THREE.Group();
    const fillMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.15,
      depthWrite: false,
    });
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0x67e8f9,
      transparent: true,
      opacity: 0.85,
    });
    const wireMat = new THREE.LineBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.42,
    });

    const addBox = (
      sx: number,
      sy: number,
      sz: number,
      x: number,
      y: number,
      z: number,
      rz = 0
    ) => {
      const geo = new THREE.BoxGeometry(sx, sy, sz);
      const mesh = new THREE.Mesh(geo, fillMat);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
      const part = new THREE.Group();
      part.position.set(x, y, z);
      part.rotation.z = rz;
      part.add(mesh, edges);
      group.add(part);
    };

    const addLimb = (
      rTop: number,
      rBot: number,
      h: number,
      x: number,
      y: number,
      z: number,
      rz = 0
    ) => {
      const geo = new THREE.CylinderGeometry(rTop, rBot, h, 10, 1);
      const mesh = new THREE.Mesh(geo, fillMat);
      const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), wireMat);
      const part = new THREE.Group();
      part.position.set(x, y, z);
      part.rotation.z = rz;
      part.add(mesh, wire);
      group.add(part);
    };

    // Normalized 1.0m humanoid silhouette (scaled by heightM in setPlacePreview)
    // Feet
    addBox(0.052, 0.042, 0.13, -0.075, 0.021, 0.022);
    addBox(0.052, 0.042, 0.13, 0.075, 0.021, 0.022);
    // Shins
    addLimb(0.034, 0.026, 0.22, -0.075, 0.155, 0);
    addLimb(0.034, 0.026, 0.22, 0.075, 0.155, 0);
    // Thighs
    addLimb(0.046, 0.034, 0.23, -0.075, 0.385, 0);
    addLimb(0.046, 0.034, 0.23, 0.075, 0.385, 0);
    // Pelvis
    addBox(0.175, 0.09, 0.105, 0, 0.535, 0);
    // Lower spine & chest
    addBox(0.145, 0.09, 0.095, 0, 0.625, 0);
    addBox(0.215, 0.165, 0.12, 0, 0.745, 0);
    // Neck & Head
    addLimb(0.03, 0.032, 0.05, 0, 0.855, 0);
    const headGeo = new THREE.SphereGeometry(0.068, 12, 10);
    const headFill = new THREE.Mesh(headGeo, fillMat);
    const headWire = new THREE.LineSegments(
      new THREE.WireframeGeometry(headGeo),
      wireMat
    );
    headFill.position.set(0, 0.932, 0);
    headWire.position.set(0, 0.932, 0);
    group.add(headFill, headWire);

    // Arms (slightly relaxed A-pose)
    addLimb(0.032, 0.026, 0.165, -0.148, 0.72, 0, -0.14);
    addLimb(0.032, 0.026, 0.165, 0.148, 0.72, 0, 0.14);
    addLimb(0.025, 0.02, 0.15, -0.174, 0.565, 0.01, -0.08);
    addLimb(0.025, 0.02, 0.15, 0.174, 0.565, 0.01, 0.08);
    addBox(0.032, 0.052, 0.042, -0.184, 0.465, 0.015);
    addBox(0.032, 0.052, 0.042, 0.184, 0.465, 0.015);

    // Crisp floor placement ring
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.19, 0.225, 32),
      new THREE.MeshBasicMaterial({
        color: 0x22d3ee,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.008;
    group.add(ring);

    group.visible = false;
    return group;
  }

  /**
   * Updates or hides the translucent 3D placement ghost preview on the target surface.
   */
  public setPlacePreview(
    visible: boolean,
    mode: PlacerMode = 'curb',
    pos?: THREE.Vector3,
    heightM = 1.78
  ): void {
    if (!visible || !pos) {
      this.placePreviewGroup.visible = false;
      return;
    }

    this.placePreviewGroup.visible = true;
    this.placePreviewGroup.position.set(pos.x, Math.max(0, pos.y), pos.z);

    this.placePreviewRagdoll.visible = mode === 'ragdoll';
    this.placePreviewCurb.visible = mode === 'curb';
    this.placePreviewCube.visible = mode === 'cube';
    this.placePreviewSphere.visible = mode === 'sphere';
    this.placePreviewDynamite.visible = mode === 'dynamite';

    if (mode === 'ragdoll') {
      this.placePreviewRagdoll.scale.setScalar(Math.max(0.5, heightM));
    }
  }

  /**
   * Raycasts against static obstacle meshes (stairs, platforms, ramps), terrain heightfield,
   * and dynamic non-ragdoll prop meshes, returning hit point and outward world-space normal
   * so placed entities sit flush against tops, slopes, and vertical sides without overlapping.
   */
  public raycastPlacementSurface(
    raycaster: THREE.Raycaster
  ): { point: THREE.Vector3; normal: THREE.Vector3 } | null {
    const targets: THREE.Object3D[] = [...this.staticGroup.children];
    if (this.terrainSurfaceMesh) {
      targets.push(this.terrainSurfaceMesh);
    }
    for (const child of this.dynamicGroup.children) {
      if (child.userData.isProp === true) {
        targets.push(child);
      }
    }
    if (targets.length === 0) {
      return null;
    }

    const hits = raycaster.intersectObjects(targets, true);
    const meshHit = hits.find((h) => h.object instanceof THREE.Mesh);
    if (!meshHit) {
      return null;
    }

    const point = meshHit.point.clone();
    if (point.y < 0.0) {
      point.y = 0.0;
    }
    const normal = meshHit.face
      ? meshHit.face.normal
          .clone()
          .transformDirection(meshHit.object.matrixWorld)
          .normalize()
      : new THREE.Vector3(0, 1, 0);
    if (normal.dot(raycaster.ray.direction) > 0) {
      normal.negate();
    }
    return { point, normal };
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
    this.effects.setGrabTether(active, bodyPos, targetPos);
  }

  /**
   * Spawns a brief expanding shockwave FX at an impact point.
   */
  public spawnImpulseEffect(
    pos: THREE.Vector3,
    variant: ImpulseEffectVariant = 'shove'
  ): void {
    this.effects.spawnImpulseEffect(pos, variant);
  }

  /**
   * Spawns a crisp fast-fading glowing bullet tracer (0.12s lifetime) from `from` to `to`
   * and an impact spark burst at `to`.
   */
  public spawnTracer(
    from: THREE.Vector3,
    to: THREE.Vector3,
    colorHex: number,
    spawnSparks = true
  ): void {
    this.effects.spawnTracer(from, to, colorHex, spawnSparks);
  }

  /**
   * Spawns 3 expanding cyan-white concentric shockwave rings rushing along `dir`.
   */
  public spawnForceWaveEffect(origin: THREE.Vector3, dir: THREE.Vector3): void {
    this.effects.spawnForceWaveEffect(origin, dir);
  }

  /**
   * Spawns a 3D dynamite detonation: orange-white fireball sphere, expanding ground
   * shockwave ring, and 28 glowing ember/spark particles flying outward with gravity.
   */
  public spawnExplosionEffect(pos: THREE.Vector3, radius: number): void {
    this.effects.spawnExplosionEffect(pos, radius);
  }

  public setEarthquakeIntensity(magnitude: number): void {
    this.cameraController.setEarthquakeIntensity(magnitude);
  }

  public setSimulationPaused(paused: boolean): void {
    this.cameraController.setSimulationPaused(paused);
  }

  public updateEffects(dt: number): void {
    this.effects.update(dt, this.dynamiteFuseTips);

    // Pulse & billboard active wound rings
    if (this.telemetryGroup.visible) {
      const pulse = 1 + 0.24 * Math.sin(this.effects.getElapsedTime() * 9.5);
      for (const entry of this.telemetryPool) {
        if (entry.group.visible && entry.woundGroup.visible) {
          entry.woundGroup.quaternion.copy(this.camera.quaternion);
          entry.woundRing.scale.setScalar(pulse);
        }
      }
    }
  }

  public resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.cameraController.resize(w, h);
    this.renderer.setSize(w, h, false);
  }

  public render(): void {
    const didShake = this.cameraController.applyShakeBeforeRender();
    try {
      this.renderer.render(this.scene, this.camera);
    } finally {
      if (didShake) {
        this.cameraController.restoreAfterRender();
      }
    }
  }
}
