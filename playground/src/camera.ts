import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type CameraMode = 'orbit' | 'fly';
export type CameraPresetName = 'front' | 'back' | 'left' | 'right' | 'top';

interface CameraPreset {
  position: [number, number, number];
  target: [number, number, number];
}

export const CAMERA_PRESETS: Record<CameraPresetName, CameraPreset> = {
  front: { position: [0, 1.35, 3.6], target: [0, 1.0, 0] },
  back: { position: [0, 1.35, -3.6], target: [0, 1.0, 0] },
  left: { position: [3.6, 1.35, 0], target: [0, 1.0, 0] },
  right: { position: [-3.6, 1.35, 0], target: [0, 1.0, 0] },
  top: { position: [0, 5.2, 0.25], target: [0, 0.5, 0] },
};

interface CameraTransition {
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  startTarget: THREE.Vector3;
  endTarget: THREE.Vector3;
  elapsed: number;
  duration: number;
}

/**
 * Manages the Three.js perspective camera, Orbit/Fly controls, camera presets,
 * smooth framing transitions, and non-drifting earthquake camera shake.
 */
export class LabCameraController {
  public readonly camera: THREE.PerspectiveCamera;
  public readonly controls: OrbitControls;

  private cameraMode: CameraMode = 'orbit';
  private camAnim: CameraTransition | null = null;
  private flyYaw = 0;
  private flyPitch = 0;
  private flySpeed = 4.0;
  private readonly flyKeys = new Set<string>();

  private readonly tempVecA = new THREE.Vector3();
  private readonly tempVecB = new THREE.Vector3();
  private readonly tempVecC = new THREE.Vector3();
  private readonly tempEuler = new THREE.Euler(0, 0, 0, 'YXZ');

  private earthquakeMagnitude = 0;
  private simulationPaused = false;
  private shakeTime = 0;
  private readonly shakeOffset = new THREE.Vector3();
  private readonly preShakeCamPos = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.camera = new THREE.PerspectiveCamera(
      48,
      window.innerWidth / window.innerHeight,
      0.08,
      140
    );
    this.camera.position.set(3.2, 2.35, 4.6);

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
    this.frameCamera(preset.position, preset.target);
  }

  /**
   * Smoothly transitions the camera position and focus target.
   */
  public frameCamera(
    position: [number, number, number],
    target: [number, number, number],
    duration = 0.35
  ): void {
    let startTarget: THREE.Vector3;

    if (this.cameraMode === 'orbit') {
      startTarget = this.controls.target.clone();
    } else {
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(
        this.camera.quaternion
      );
      startTarget = this.camera.position.clone().addScaledVector(forward, 3.0);
    }

    this.camAnim = {
      startPos: this.camera.position.clone(),
      endPos: new THREE.Vector3(...position),
      startTarget,
      endTarget: new THREE.Vector3(...target),
      elapsed: 0,
      duration,
    };
  }

  public rotateFlyLook(dx: number, dy: number): void {
    if (this.cameraMode !== 'fly') return;
    this.camAnim = null;
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
      this.flyKeys.add(code);
      if (this.cameraMode === 'fly') {
        this.camAnim = null;
      }
    } else {
      this.flyKeys.delete(code);
    }
  }

  public clearFlyKeys(): void {
    this.flyKeys.clear();
  }

  public setEarthquakeIntensity(magnitude: number): void {
    this.earthquakeMagnitude = Math.max(0, Math.min(9, magnitude));
  }

  public setSimulationPaused(paused: boolean): void {
    this.simulationPaused = paused;
  }

  private syncFlyAnglesFromCamera(): void {
    this.tempEuler.setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.flyPitch = this.tempEuler.x;
    this.flyYaw = this.tempEuler.y;
  }

  public update(dt: number): void {
    if (this.earthquakeMagnitude > 0.1 && !this.simulationPaused) {
      this.shakeTime += dt;
    }

    if (this.camAnim) {
      const anim = this.camAnim;
      anim.elapsed += dt;
      const rawT = Math.min(1, anim.elapsed / anim.duration);
      const ease = 1 - Math.pow(1 - rawT, 3);

      this.camera.position.lerpVectors(anim.startPos, anim.endPos, ease);
      this.tempVecA.lerpVectors(anim.startTarget, anim.endTarget, ease);
      this.controls.target.copy(this.tempVecA);
      this.camera.lookAt(this.tempVecA);
      this.syncFlyAnglesFromCamera();

      if (rawT >= 1) {
        this.camAnim = null;
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

      if (this.flyKeys.has('KeyW') || this.flyKeys.has('ArrowUp')) {
        move.add(forward);
      }
      if (this.flyKeys.has('KeyS') || this.flyKeys.has('ArrowDown')) {
        move.sub(forward);
      }
      if (this.flyKeys.has('KeyD') || this.flyKeys.has('ArrowRight')) {
        move.add(right);
      }
      if (this.flyKeys.has('KeyA') || this.flyKeys.has('ArrowLeft')) {
        move.sub(right);
      }
      if (this.flyKeys.has('KeyE') || this.flyKeys.has('Space')) {
        move.y += 1;
      }
      if (
        this.flyKeys.has('KeyQ') ||
        this.flyKeys.has('ShiftLeft') ||
        this.flyKeys.has('ShiftRight')
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

  public resize(w: number, h: number): void {
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Applies a temporary, non-drifting earthquake shake offset before rendering.
   * Returns `true` if `restoreAfterRender()` must be called after the render pass.
   */
  public applyShakeBeforeRender(): boolean {
    if (this.earthquakeMagnitude <= 0.1 || this.simulationPaused) {
      return false;
    }

    const t = this.shakeTime;
    const amp = 0.0015 + Math.pow(this.earthquakeMagnitude / 9.0, 2.5) * 0.052;
    this.shakeOffset.set(
      (Math.sin(t * 47.3) * 0.6 + Math.sin(t * 73.1 + 1.3) * 0.4) * amp,
      (Math.sin(t * 53.7 + 0.7) * 0.65 + Math.cos(t * 89.2) * 0.35) * amp * 0.85,
      (Math.cos(t * 41.9 + 2.1) * 0.6 + Math.sin(t * 67.4) * 0.4) * amp
    );
    this.preShakeCamPos.copy(this.camera.position);
    this.camera.position.add(this.shakeOffset);
    return true;
  }

  public restoreAfterRender(): void {
    this.camera.position.copy(this.preShakeCamPos);
  }
}
