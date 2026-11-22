import * as THREE from 'three';

export type ImpulseEffectVariant = 'shove' | 'punch';

export interface DynamiteFuseTip {
  core: THREE.Mesh;
  halo: THREE.Mesh;
  phase: number;
}

export interface ActiveFireSource {
  pos: THREE.Vector3;
  radius: number;
}

interface ImpulseFx {
  group: THREE.Group;
  ring: THREE.Mesh;
  flashCore?: THREE.Mesh;
  age: number;
  duration: number;
  maxScale: number;
}

interface TracerFx {
  beam: THREE.Mesh;
  line: THREE.Line;
  age: number;
  duration: number;
}

interface SparkParticle {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  age: number;
  duration: number;
}

interface ForceWaveFx {
  mesh: THREE.Mesh;
  origin: THREE.Vector3;
  dir: THREE.Vector3;
  age: number;
  delay: number;
  duration: number;
  travelDist: number;
  startScale: number;
  endScale: number;
  maxOpacity: number;
}

interface ExplosionFx {
  core: THREE.Mesh;
  fireball: THREE.Mesh;
  groundRing: THREE.Mesh;
  age: number;
  duration: number;
  maxFireballRadius: number;
  maxRingRadius: number;
}

const TASER_WIRE_SEGMENTS = 8;
const TASER_WIRE_VERTS = TASER_WIRE_SEGMENTS + 1;
const TASER_ARC_SEGMENTS = 6;
const TASER_ARC_VERTS = TASER_ARC_SEGMENTS + 1;
const MAX_GROUND_FIRE_PATCHES = 6;
const MAX_FIRE_PARTICLES = 220;

interface TaserShotFx {
  wires: [THREE.Line, THREE.Line];
  wirePos: [Float32Array, Float32Array];
  arcs: [THREE.Line, THREE.Line] | null;
  arcPos: [Float32Array, Float32Array] | null;
  prongs: [THREE.Mesh, THREE.Mesh];
  isHit: boolean;
  bodyIdx: number;
  chestBodyIdx: number;
  localProngA: THREE.Vector3;
  localProngB: THREE.Vector3;
  missEndA: THREE.Vector3;
  missEndB: THREE.Vector3;
  age: number;
  duration: number;
  sparkAccum: number;
}

interface GroundFirePatch {
  group: THREE.Group;
  disc: THREE.Mesh;
  ring: THREE.Mesh;
  light: THREE.PointLight;
  pos: THREE.Vector3;
  radius: number;
  remaining: number;
  maxDuration: number;
  emitAccum: number;
  phase: number;
}

interface BurningPropFire {
  bodyIdx: number;
  pos: THREE.Vector3;
  radius: number;
  remaining: number;
  emitAccum: number;
}

interface FireParticle {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  age: number;
  duration: number;
  startScale: number;
  endScale: number;
  isEmber: boolean;
}

/**
 * Manages transient 3D visual FX: grab tether line & endpoints, impulse shockwaves,
 * ballistic tracers & spark bursts, Force Wave rings, 3D explosions, fuse-tip flicker,
 * 2-wire Taser arcs, ground fire patches, and burning prop/body flames.
 */
export class LabEffectsManager {
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.Camera;

  private readonly tetherLine: THREE.Line;
  private readonly tetherStartDot: THREE.Mesh;
  private readonly tetherEndDot: THREE.Mesh;
  private readonly tetherPositions = new Float32Array(6);

  private readonly impulseEffects: ImpulseFx[] = [];
  private readonly impulseRingGeo: THREE.RingGeometry;
  private readonly punchCoreGeo: THREE.SphereGeometry;

  private readonly tracerEffects: TracerFx[] = [];
  private readonly sparkParticles: SparkParticle[] = [];
  private readonly forceWaveEffects: ForceWaveFx[] = [];
  private readonly explosionEffects: ExplosionFx[] = [];
  private readonly taserEffects: TaserShotFx[] = [];
  private readonly groundFirePatches: GroundFirePatch[] = [];
  private readonly burningProps: BurningPropFire[] = [];
  private readonly fireParticles: FireParticle[] = [];

  private readonly tracerBeamGeo: THREE.CylinderGeometry;
  private readonly sparkGeo: THREE.BoxGeometry;
  private readonly forceWaveRingGeo: THREE.RingGeometry;
  private readonly explosionSphereGeo: THREE.SphereGeometry;
  private readonly explosionGroundRingGeo: THREE.RingGeometry;
  private readonly taserProngGeo: THREE.SphereGeometry;
  private readonly fireDiscGeo: THREE.CircleGeometry;
  private readonly fireRingGeo: THREE.RingGeometry;
  private readonly flameGeo: THREE.OctahedronGeometry;
  private readonly emberGeo: THREE.BoxGeometry;

  private readonly tempVecA = new THREE.Vector3();
  private readonly tempVecB = new THREE.Vector3();
  private readonly tempVecC = new THREE.Vector3();
  private readonly tempVecD = new THREE.Vector3();
  private readonly tempVecE = new THREE.Vector3();
  private readonly tempQuat = new THREE.Quaternion();
  private fxTime = 0;

  constructor(scene: THREE.Scene, camera: THREE.Camera) {
    this.scene = scene;
    this.camera = camera;

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

    this.impulseRingGeo = new THREE.RingGeometry(0.08, 0.13, 28);
    this.punchCoreGeo = new THREE.SphereGeometry(0.075, 14, 12);

    // Ballistic tracer & spark shared geometries (oriented along +Z for lookAt)
    this.tracerBeamGeo = new THREE.CylinderGeometry(0.0035, 0.006, 1, 6);
    this.tracerBeamGeo.rotateX(Math.PI / 2);
    this.sparkGeo = new THREE.BoxGeometry(0.007, 0.007, 0.065);
    this.forceWaveRingGeo = new THREE.RingGeometry(0.14, 0.195, 36);
    this.explosionSphereGeo = new THREE.SphereGeometry(1, 22, 16);
    this.explosionGroundRingGeo = new THREE.RingGeometry(0.68, 1.0, 40);
    this.taserProngGeo = new THREE.SphereGeometry(0.018, 8, 6);
    this.fireDiscGeo = new THREE.CircleGeometry(0.58, 24);
    this.fireDiscGeo.rotateX(-Math.PI / 2);
    this.fireRingGeo = new THREE.RingGeometry(0.45, 0.85, 32);
    this.fireRingGeo.rotateX(-Math.PI / 2);
    this.flameGeo = new THREE.OctahedronGeometry(0.075, 0);
    this.emberGeo = new THREE.BoxGeometry(0.02, 0.02, 0.02);
  }

  public getElapsedTime(): number {
    return this.fxTime;
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
    const dist = from.distanceTo(to);
    if (dist > 0.005) {
      const mid = this.tempVecA.addVectors(from, to).multiplyScalar(0.5);
      const beamMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.92,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const beam = new THREE.Mesh(this.tracerBeamGeo, beamMat);
      beam.position.copy(mid);
      beam.lookAt(to);
      beam.scale.set(1, 1, dist);
      this.scene.add(beam);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      this.scene.add(line);

      this.tracerEffects.push({
        beam,
        line,
        age: 0,
        duration: 0.12,
      });
    }

    if (spawnSparks) {
      this.spawnSparkBurst(to, from, colorHex);
    }
  }

  private spawnSparkBurst(
    pos: THREE.Vector3,
    origin: THREE.Vector3,
    colorHex: number
  ): void {
    // Brief bright impact flash core
    this.spawnImpulseEffect(pos, 'punch');

    const incomingDir = this.tempVecB.subVectors(origin, pos);
    if (incomingDir.lengthSq() > 1e-6) {
      incomingDir.normalize();
    } else {
      incomingDir.set(0, 1, 0);
    }

    const sparkCount = 10;
    for (let i = 0; i < sparkCount; i++) {
      const sparkMat = new THREE.MeshBasicMaterial({
        color: i % 2 === 0 ? 0xfff3b0 : colorHex,
        transparent: true,
        opacity: 0.98,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this.sparkGeo, sparkMat);
      mesh.position.copy(pos);

      const velocity = new THREE.Vector3(
        (Math.random() - 0.5) * 2.2 + incomingDir.x * 1.1,
        Math.random() * 1.6 + 0.35 + incomingDir.y * 0.6,
        (Math.random() - 0.5) * 2.2 + incomingDir.z * 1.1
      )
        .normalize()
        .multiplyScalar(2.6 + Math.random() * 4.4);

      const lookTarget = this.tempVecC.copy(pos).add(velocity);
      mesh.lookAt(lookTarget);
      this.scene.add(mesh);

      this.sparkParticles.push({
        mesh,
        velocity,
        age: 0,
        duration: 0.16 + Math.random() * 0.12,
      });
    }
  }

  /**
   * Spawns 3 expanding cyan-white concentric shockwave rings rushing along `dir`.
   */
  public spawnForceWaveEffect(origin: THREE.Vector3, dir: THREE.Vector3): void {
    const normDir = dir.clone();
    if (normDir.lengthSq() > 1e-6) {
      normDir.normalize();
    } else {
      normDir.set(0, 0, -1);
    }

    const baseOrigin = origin.clone().addScaledVector(normDir, 0.75);
    const ringColors = [0xe0faff, 0x67e8f9, 0x22d3ee];

    for (let i = 0; i < 3; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: ringColors[i],
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this.forceWaveRingGeo, mat);
      mesh.position.copy(baseOrigin);
      mesh.lookAt(this.tempVecA.copy(baseOrigin).add(normDir));
      mesh.visible = false;
      this.scene.add(mesh);

      this.forceWaveEffects.push({
        mesh,
        origin: baseOrigin.clone(),
        dir: normDir.clone(),
        age: 0,
        delay: i * 0.045,
        duration: 0.36,
        travelDist: 6.2 - i * 0.55,
        startScale: 0.9 + i * 0.28,
        endScale: 7.4 + i * 1.4,
        maxOpacity: 0.95 - i * 0.14,
      });
    }
  }

  /**
   * Spawns a 3D dynamite detonation: orange-white fireball sphere, expanding ground
   * shockwave ring, and 28 glowing ember/spark particles flying outward with gravity.
   */
  public spawnExplosionEffect(pos: THREE.Vector3, radius: number): void {
    const maxFireballRadius = Math.max(0.85, radius * 0.56);
    const maxRingRadius = Math.max(1.8, radius * 1.1);

    const coreMat = new THREE.MeshBasicMaterial({
      color: 0xfff5d6,
      transparent: true,
      opacity: 0.98,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const core = new THREE.Mesh(this.explosionSphereGeo, coreMat);
    core.position.copy(pos);
    core.scale.setScalar(0.12);
    this.scene.add(core);

    const fireballMat = new THREE.MeshBasicMaterial({
      color: 0xff6b1a,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const fireball = new THREE.Mesh(this.explosionSphereGeo, fireballMat);
    fireball.position.copy(pos);
    fireball.scale.setScalar(0.2);
    this.scene.add(fireball);

    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffaa33,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const groundRing = new THREE.Mesh(this.explosionGroundRingGeo, ringMat);
    groundRing.position.set(pos.x, 0.025, pos.z);
    groundRing.rotation.x = -Math.PI / 2;
    groundRing.scale.setScalar(0.25);
    this.scene.add(groundRing);

    this.explosionEffects.push({
      core,
      fireball,
      groundRing,
      age: 0,
      duration: 0.45,
      maxFireballRadius,
      maxRingRadius,
    });

    // 28 glowing ember/spark particles flying outward with gravity
    const emberColors = [0xfff3b0, 0xffa928, 0xff5418];
    const emberCount = 28;
    for (let i = 0; i < emberCount; i++) {
      const emberMat = new THREE.MeshBasicMaterial({
        color: emberColors[i % emberColors.length],
        transparent: true,
        opacity: 1.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this.sparkGeo, emberMat);
      mesh.position.copy(pos);

      const theta = Math.random() * Math.PI * 2;
      const upBias = Math.random() * 0.95 - 0.1;
      const speed = 3.8 + Math.random() * 6.8;
      const velocity = new THREE.Vector3(
        Math.cos(theta),
        upBias,
        Math.sin(theta)
      )
        .normalize()
        .multiplyScalar(speed);

      mesh.lookAt(this.tempVecA.copy(pos).add(velocity));
      this.scene.add(mesh);

      this.sparkParticles.push({
        mesh,
        velocity,
        age: 0,
        duration: 0.35 + Math.random() * 0.4,
      });
    }
  }

  /**
   * Computes the taser handheld muzzle origin slightly below-right of the camera:
   * `origin + right*0.18 - up*0.14 + fwd*0.45`.
   */
  public computeTaserMuzzle(out = new THREE.Vector3()): THREE.Vector3 {
    const fwd = this.tempVecA.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right = this.tempVecB.set(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = this.tempVecC.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return out
      .copy(this.camera.position)
      .addScaledVector(right, 0.18)
      .addScaledVector(up, -0.14)
      .addScaledVector(fwd, 0.45);
  }

  /**
   * Spawns a 3.0s twin-wire taser stun discharge attached to a live body (`bodyIdx`),
   * with crackling cyan-white lightning arcs around `chestBodyIdx` (or `bodyIdx`).
   */
  public spawnTaserHit(
    bodyIdx: number,
    chestBodyIdx: number,
    localProngA: THREE.Vector3,
    localProngB: THREE.Vector3
  ): void {
    const wires = this.createTaserWirePair(0.95);
    const arcs = this.createTaserArcPair();
    const prongs = this.createTaserProngs();

    this.taserEffects.push({
      wires: wires.lines,
      wirePos: wires.positions,
      arcs: arcs.lines,
      arcPos: arcs.positions,
      prongs,
      isHit: true,
      bodyIdx,
      chestBodyIdx: chestBodyIdx >= 0 ? chestBodyIdx : bodyIdx,
      localProngA: localProngA.clone(),
      localProngB: localProngB.clone(),
      missEndA: new THREE.Vector3(),
      missEndB: new THREE.Vector3(),
      age: 0,
      duration: 3.0,
      sparkAccum: 0,
    });
  }

  /**
   * Spawns a brief 0.25s twin-wire spark miss effect.
   */
  public spawnTaserMiss(endA: THREE.Vector3, endB: THREE.Vector3): void {
    const wires = this.createTaserWirePair(0.82);
    const prongs = this.createTaserProngs();
    prongs[0].position.copy(endA);
    prongs[1].position.copy(endB);

    this.emitElectricSparks(endA, 6, 0.35);
    this.emitElectricSparks(endB, 6, 0.35);

    this.taserEffects.push({
      wires: wires.lines,
      wirePos: wires.positions,
      arcs: null,
      arcPos: null,
      prongs,
      isHit: false,
      bodyIdx: -1,
      chestBodyIdx: -1,
      localProngA: new THREE.Vector3(),
      localProngB: new THREE.Vector3(),
      missEndA: endA.clone(),
      missEndB: endB.clone(),
      age: 0,
      duration: 0.25,
      sparkAccum: 0,
    });
  }

  private createTaserWirePair(opacity: number): {
    lines: [THREE.Line, THREE.Line];
    positions: [Float32Array, Float32Array];
  } {
    const colors = [0x38bdf8, 0xe0faff];
    const lines: THREE.Line[] = [];
    const positions: Float32Array[] = [];

    for (let i = 0; i < 2; i++) {
      const arr = new Float32Array(TASER_WIRE_VERTS * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      const mat = new THREE.LineBasicMaterial({
        color: colors[i],
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const line = new THREE.Line(geo, mat);
      line.frustumCulled = false;
      this.scene.add(line);
      lines.push(line);
      positions.push(arr);
    }

    return {
      lines: lines as [THREE.Line, THREE.Line],
      positions: positions as [Float32Array, Float32Array],
    };
  }

  private createTaserArcPair(): {
    lines: [THREE.Line, THREE.Line];
    positions: [Float32Array, Float32Array];
  } {
    const colors = [0xe0faff, 0x22d3ee];
    const lines: THREE.Line[] = [];
    const positions: Float32Array[] = [];

    for (let i = 0; i < 2; i++) {
      const arr = new Float32Array(TASER_ARC_VERTS * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      const mat = new THREE.LineBasicMaterial({
        color: colors[i],
        transparent: true,
        opacity: 0.96,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const line = new THREE.Line(geo, mat);
      line.frustumCulled = false;
      this.scene.add(line);
      lines.push(line);
      positions.push(arr);
    }

    return {
      lines: lines as [THREE.Line, THREE.Line],
      positions: positions as [Float32Array, Float32Array],
    };
  }

  private createTaserProngs(): [THREE.Mesh, THREE.Mesh] {
    const prongs: THREE.Mesh[] = [];
    for (let i = 0; i < 2; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: i === 0 ? 0xe0faff : 0x38bdf8,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this.taserProngGeo, mat);
      this.scene.add(mesh);
      prongs.push(mesh);
    }
    return prongs as [THREE.Mesh, THREE.Mesh];
  }

  private disposeTaserEffect(fx: TaserShotFx): void {
    for (const line of fx.wires) {
      this.scene.remove(line);
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    }
    if (fx.arcs) {
      for (const arc of fx.arcs) {
        this.scene.remove(arc);
        arc.geometry.dispose();
        (arc.material as THREE.Material).dispose();
      }
    }
    for (const prong of fx.prongs) {
      this.scene.remove(prong);
      (prong.material as THREE.Material).dispose();
    }
  }

  /**
   * Spawns a ground campfire / fire-pit patch (14s lifetime, radius 0.85m, capped at 6 patches).
   */
  public spawnGroundFirePatch(pos: THREE.Vector3): void {
    if (this.groundFirePatches.length >= MAX_GROUND_FIRE_PATCHES) {
      const oldest = this.groundFirePatches.shift()!;
      this.disposeGroundFirePatch(oldest);
    }

    const group = new THREE.Group();
    group.position.copy(pos);

    const discMat = new THREE.MeshBasicMaterial({
      color: 0xff5a12,
      transparent: true,
      opacity: 0.78,
      depthWrite: false,
    });
    const disc = new THREE.Mesh(this.fireDiscGeo, discMat);
    disc.position.y = 0.012;
    group.add(disc);

    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffa62b,
      transparent: true,
      opacity: 0.52,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(this.fireRingGeo, ringMat);
    ring.position.y = 0.018;
    group.add(ring);

    const light = new THREE.PointLight(0xff741c, 2.8, 6.2, 2.0);
    light.position.set(0, 0.38, 0);
    group.add(light);

    this.scene.add(group);
    this.groundFirePatches.push({
      group,
      disc,
      ring,
      light,
      pos: pos.clone(),
      radius: 0.85,
      remaining: 14.0,
      maxDuration: 14.0,
      emitAccum: 0,
      phase: Math.random() * Math.PI * 2,
    });

    this.emitFlameParticles(pos, 10, 0.42);
  }

  private disposeGroundFirePatch(patch: GroundFirePatch): void {
    this.scene.remove(patch.group);
    patch.light.dispose();
    (patch.disc.material as THREE.Material).dispose();
    (patch.ring.material as THREE.Material).dispose();
  }

  /**
   * Attaches or refreshes a 20s burning fire onto a dynamic prop body.
   */
  public igniteProp(bodyIdx: number, pos: THREE.Vector3): void {
    const existing = this.burningProps.find((p) => p.bodyIdx === bodyIdx);
    if (existing) {
      existing.remaining = 20.0;
      existing.pos.copy(pos);
    } else {
      this.burningProps.push({
        bodyIdx,
        pos: pos.clone(),
        radius: 0.65,
        remaining: 20.0,
        emitAccum: 0,
      });
    }
    this.emitFlameParticles(pos, 8, 0.25);
  }

  /**
   * Returns all currently active fire sources (ground patches + burning props)
   * for automatic ragdoll contact ignition.
   */
  public getActiveFireSources(): ActiveFireSource[] {
    const sources: ActiveFireSource[] = [];
    for (const p of this.groundFirePatches) {
      if (p.remaining > 0) {
        sources.push({ pos: p.pos, radius: p.radius });
      }
    }
    for (const prop of this.burningProps) {
      if (prop.remaining > 0) {
        sources.push({ pos: prop.pos, radius: prop.radius });
      }
    }
    return sources;
  }

  /**
   * Extinguishes any burning dynamic props whose Y position is submerged below `maxY`.
   */
  public extinguishPropsBelowY(maxY: number, transforms: Float32Array): void {
    for (let i = this.burningProps.length - 1; i >= 0; i--) {
      const prop = this.burningProps[i];
      const pOff = prop.bodyIdx * 7;
      const py =
        pOff + 1 < transforms.length ? transforms[pOff + 1] : prop.pos.y;
      if (py < maxY) {
        this.burningProps.splice(i, 1);
      }
    }
  }

  /**
   * Finds the nearest active ground fire patch or burning prop within `maxRadius` of `pos`.
   */
  public getNearestFireVisualSource(
    pos: THREE.Vector3,
    maxRadius: number
  ): {
    kind: 'patch' | 'prop';
    index: number;
    distSq: number;
    pos: THREE.Vector3;
  } | null {
    const maxSq = maxRadius * maxRadius;
    let best: {
      kind: 'patch' | 'prop';
      index: number;
      distSq: number;
      pos: THREE.Vector3;
    } | null = null;

    for (let i = 0; i < this.groundFirePatches.length; i++) {
      const p = this.groundFirePatches[i];
      if (p.remaining <= 0) continue;
      const dSq = p.pos.distanceToSquared(pos);
      if (dSq <= maxSq && (!best || dSq < best.distSq)) {
        best = { kind: 'patch', index: i, distSq: dSq, pos: p.pos.clone() };
      }
    }

    for (let i = 0; i < this.burningProps.length; i++) {
      const prop = this.burningProps[i];
      if (prop.remaining <= 0) continue;
      const dSq = prop.pos.distanceToSquared(pos);
      if (dSq <= maxSq && (!best || dSq < best.distSq)) {
        best = { kind: 'prop', index: i, distSq: dSq, pos: prop.pos.clone() };
      }
    }

    return best;
  }

  /**
   * Extinguishes a ground fire patch or burning prop by kind and array index.
   */
  public removeFireVisualSource(kind: 'patch' | 'prop', index: number): void {
    if (kind === 'patch') {
      if (index >= 0 && index < this.groundFirePatches.length) {
        const [removed] = this.groundFirePatches.splice(index, 1);
        this.disposeGroundFirePatch(removed);
      }
    } else if (index >= 0 && index < this.burningProps.length) {
      this.burningProps.splice(index, 1);
    }
  }

  /**
   * Emits rising orange/yellow flame & ember particles around `pos`.
   */
  public emitFlameParticles(
    pos: THREE.Vector3,
    count: number,
    spreadRadius = 0.22
  ): void {
    const flameColors = [0xfff3b0, 0xffae28, 0xff6214, 0xef3e18];
    for (let i = 0; i < count; i++) {
      if (this.fireParticles.length >= MAX_FIRE_PARTICLES) {
        const dropped = this.fireParticles.shift()!;
        this.scene.remove(dropped.mesh);
        (dropped.mesh.material as THREE.Material).dispose();
      }

      const isEmber = i % 3 === 2;
      const color = flameColors[(Math.random() * flameColors.length) | 0];
      const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: isEmber ? 0.95 : 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(
        isEmber ? this.emberGeo : this.flameGeo,
        mat
      );
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * spreadRadius;
      mesh.position.set(
        pos.x + Math.cos(angle) * r,
        pos.y + Math.random() * 0.08,
        pos.z + Math.sin(angle) * r
      );
      mesh.rotation.set(
        Math.random() * Math.PI,
        Math.random() * Math.PI,
        Math.random() * Math.PI
      );

      const startScale = isEmber
        ? 0.65 + Math.random() * 0.55
        : 0.7 + Math.random() * 0.85;
      const endScale = isEmber ? 0.2 : 0.18;
      mesh.scale.setScalar(startScale);
      this.scene.add(mesh);

      const velocity = new THREE.Vector3(
        (Math.random() - 0.5) * (isEmber ? 0.85 : 0.42),
        (isEmber ? 1.35 : 0.95) + Math.random() * 1.15,
        (Math.random() - 0.5) * (isEmber ? 0.85 : 0.42)
      );

      this.fireParticles.push({
        mesh,
        velocity,
        age: 0,
        duration: isEmber
          ? 0.45 + Math.random() * 0.45
          : 0.34 + Math.random() * 0.3,
        startScale,
        endScale,
        isEmber,
      });
    }
  }

  /**
   * Emits crackling cyan-white electric spark particles around `pos`.
   */
  public emitElectricSparks(
    pos: THREE.Vector3,
    count: number,
    spreadRadius = 0.24
  ): void {
    const colors = [0xe0faff, 0x67e8f9, 0x22d3ee, 0x38bdf8];
    for (let i = 0; i < count; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: colors[i % colors.length],
        transparent: true,
        opacity: 0.98,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this.sparkGeo, mat);
      mesh.position.set(
        pos.x + (Math.random() - 0.5) * spreadRadius,
        pos.y + (Math.random() - 0.5) * spreadRadius,
        pos.z + (Math.random() - 0.5) * spreadRadius
      );

      const velocity = new THREE.Vector3(
        Math.random() - 0.5,
        Math.random() * 0.85 - 0.15,
        Math.random() - 0.5
      )
        .normalize()
        .multiplyScalar(1.8 + Math.random() * 3.4);

      mesh.lookAt(this.tempVecD.copy(mesh.position).add(velocity));
      this.scene.add(mesh);

      this.sparkParticles.push({
        mesh,
        velocity,
        age: 0,
        duration: 0.1 + Math.random() * 0.1,
      });
    }
  }

  /**
   * Clears and disposes all active transient effects, taser wires, fire patches,
   * burning props, and particle systems (called on scenario reset).
   */
  public clearAll(): void {
    this.setGrabTether(false);

    while (this.impulseEffects.length > 0) {
      const fx = this.impulseEffects.pop()!;
      this.scene.remove(fx.group);
      (fx.ring.material as THREE.Material).dispose();
      if (fx.flashCore) {
        (fx.flashCore.material as THREE.Material).dispose();
      }
    }

    while (this.tracerEffects.length > 0) {
      const fx = this.tracerEffects.pop()!;
      this.scene.remove(fx.beam);
      this.scene.remove(fx.line);
      (fx.beam.material as THREE.Material).dispose();
      fx.line.geometry.dispose();
      (fx.line.material as THREE.Material).dispose();
    }

    while (this.sparkParticles.length > 0) {
      const spark = this.sparkParticles.pop()!;
      this.scene.remove(spark.mesh);
      (spark.mesh.material as THREE.Material).dispose();
    }

    while (this.forceWaveEffects.length > 0) {
      const fx = this.forceWaveEffects.pop()!;
      this.scene.remove(fx.mesh);
      (fx.mesh.material as THREE.Material).dispose();
    }

    while (this.explosionEffects.length > 0) {
      const fx = this.explosionEffects.pop()!;
      this.scene.remove(fx.core);
      this.scene.remove(fx.fireball);
      this.scene.remove(fx.groundRing);
      (fx.core.material as THREE.Material).dispose();
      (fx.fireball.material as THREE.Material).dispose();
      (fx.groundRing.material as THREE.Material).dispose();
    }

    while (this.taserEffects.length > 0) {
      this.disposeTaserEffect(this.taserEffects.pop()!);
    }

    while (this.groundFirePatches.length > 0) {
      this.disposeGroundFirePatch(this.groundFirePatches.pop()!);
    }

    this.burningProps.length = 0;

    while (this.fireParticles.length > 0) {
      const p = this.fireParticles.pop()!;
      this.scene.remove(p.mesh);
      (p.mesh.material as THREE.Material).dispose();
    }
  }

  private updateDynamiteFuseTips(
    dynamiteFuseTips: ReadonlySet<DynamiteFuseTip>
  ): void {
    for (const tip of dynamiteFuseTips) {
      const flicker =
        0.75 +
        0.35 * Math.sin(this.fxTime * 38.0 + tip.phase) +
        0.2 * Math.sin(this.fxTime * 71.0 + tip.phase * 1.7);
      const coreScale = Math.max(0.65, flicker);
      const haloScale = Math.max(0.75, flicker * 1.25);
      tip.core.scale.setScalar(coreScale);
      tip.halo.scale.setScalar(haloScale);
      (tip.halo.material as THREE.MeshBasicMaterial).opacity =
        THREE.MathUtils.clamp(0.45 + 0.4 * flicker, 0.3, 0.95);
    }
  }

  private fillCatenaryWire(
    arr: Float32Array,
    start: THREE.Vector3,
    end: THREE.Vector3,
    sag: number,
    jitter: number,
    phaseOffset: number
  ): void {
    for (let s = 0; s < TASER_WIRE_VERTS; s++) {
      const t = s / TASER_WIRE_SEGMENTS;
      const arch = 4 * t * (1 - t);
      const px = THREE.MathUtils.lerp(start.x, end.x, t);
      const py = THREE.MathUtils.lerp(start.y, end.y, t) - sag * arch;
      const pz = THREE.MathUtils.lerp(start.z, end.z, t);

      const jScale = arch * jitter;
      const jx =
        jScale *
        (Math.sin(this.fxTime * 68 + s * 2.1 + phaseOffset) +
          (Math.random() - 0.5) * 0.7);
      const jy =
        jScale *
        (Math.cos(this.fxTime * 82 + s * 1.7 + phaseOffset) +
          (Math.random() - 0.5) * 0.7);
      const jz =
        jScale *
        (Math.sin(this.fxTime * 74 + s * 2.9 + phaseOffset) +
          (Math.random() - 0.5) * 0.7);

      const o = s * 3;
      arr[o] = px + jx;
      arr[o + 1] = py + jy;
      arr[o + 2] = pz + jz;
    }
  }

  private fillLightningArc(
    arr: Float32Array,
    center: THREE.Vector3,
    radius: number
  ): void {
    const aTheta = Math.random() * Math.PI * 2;
    const bTheta = aTheta + (Math.random() * 1.6 + 1.0);
    const startX = center.x + Math.cos(aTheta) * radius;
    const startY = center.y + (Math.random() - 0.5) * radius * 1.2;
    const startZ = center.z + Math.sin(aTheta) * radius;
    const endX = center.x + Math.cos(bTheta) * radius;
    const endY = center.y + (Math.random() - 0.5) * radius * 1.2;
    const endZ = center.z + Math.sin(bTheta) * radius;

    for (let s = 0; s < TASER_ARC_VERTS; s++) {
      const t = s / TASER_ARC_SEGMENTS;
      const midEnv = Math.sin(t * Math.PI);
      const o = s * 3;
      arr[o] =
        THREE.MathUtils.lerp(startX, endX, t) +
        (Math.random() - 0.5) * 0.14 * midEnv;
      arr[o + 1] =
        THREE.MathUtils.lerp(startY, endY, t) +
        (Math.random() - 0.5) * 0.14 * midEnv;
      arr[o + 2] =
        THREE.MathUtils.lerp(startZ, endZ, t) +
        (Math.random() - 0.5) * 0.14 * midEnv;
    }
  }

  public update(
    dt: number,
    dynamiteFuseTips: ReadonlySet<DynamiteFuseTip>,
    transforms?: Float32Array,
    simDt = dt
  ): void {
    this.fxTime += dt;
    this.updateDynamiteFuseTips(dynamiteFuseTips);

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

    for (let i = this.tracerEffects.length - 1; i >= 0; i--) {
      const fx = this.tracerEffects[i];
      fx.age += dt;
      const t = Math.min(1, fx.age / fx.duration);
      const fade = 1 - t;
      (fx.beam.material as THREE.MeshBasicMaterial).opacity = fade * 0.9;
      (fx.line.material as THREE.LineBasicMaterial).opacity = fade * 0.95;

      if (t >= 1) {
        this.scene.remove(fx.beam);
        this.scene.remove(fx.line);
        (fx.beam.material as THREE.Material).dispose();
        fx.line.geometry.dispose();
        (fx.line.material as THREE.Material).dispose();
        this.tracerEffects.splice(i, 1);
      }
    }

    for (let i = this.sparkParticles.length - 1; i >= 0; i--) {
      const spark = this.sparkParticles[i];
      spark.age += dt;
      const t = Math.min(1, spark.age / spark.duration);

      spark.velocity.y -= 9.8 * dt;
      spark.mesh.position.addScaledVector(spark.velocity, dt);
      if (spark.mesh.position.y < 0.02) {
        spark.mesh.position.y = 0.02;
        spark.velocity.y = Math.abs(spark.velocity.y) * 0.4;
      }
      const lookTarget = this.tempVecA
        .copy(spark.mesh.position)
        .add(spark.velocity);
      spark.mesh.lookAt(lookTarget);
      spark.mesh.scale.set(1, 1, Math.max(0.25, 1 - t * 0.7));
      (spark.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.98;

      if (t >= 1) {
        this.scene.remove(spark.mesh);
        (spark.mesh.material as THREE.Material).dispose();
        this.sparkParticles.splice(i, 1);
      }
    }

    for (let i = this.explosionEffects.length - 1; i >= 0; i--) {
      const fx = this.explosionEffects[i];
      fx.age += dt;
      const t = Math.min(1, fx.age / fx.duration);
      const easeOut = 1 - Math.pow(1 - t, 3);

      const coreScale = THREE.MathUtils.lerp(0.14, fx.maxFireballRadius * 0.72, easeOut);
      fx.core.scale.setScalar(coreScale);
      (fx.core.material as THREE.MeshBasicMaterial).opacity = Math.max(
        0,
        (1 - t * 1.35) * 0.98
      );

      const fireballScale = THREE.MathUtils.lerp(0.22, fx.maxFireballRadius, easeOut);
      fx.fireball.scale.setScalar(fireballScale);
      (fx.fireball.material as THREE.MeshBasicMaterial).opacity =
        (1 - t) * (1 - t) * 0.9;

      const ringScale = THREE.MathUtils.lerp(0.25, fx.maxRingRadius, easeOut);
      fx.groundRing.scale.setScalar(ringScale);
      (fx.groundRing.material as THREE.MeshBasicMaterial).opacity =
        (1 - t) * 0.9;

      if (t >= 1) {
        this.scene.remove(fx.core);
        this.scene.remove(fx.fireball);
        this.scene.remove(fx.groundRing);
        (fx.core.material as THREE.Material).dispose();
        (fx.fireball.material as THREE.Material).dispose();
        (fx.groundRing.material as THREE.Material).dispose();
        this.explosionEffects.splice(i, 1);
      }
    }

    for (let i = this.forceWaveEffects.length - 1; i >= 0; i--) {
      const fx = this.forceWaveEffects[i];
      fx.age += dt;
      if (fx.age < fx.delay) {
        fx.mesh.visible = false;
        continue;
      }

      const t = Math.min(1, (fx.age - fx.delay) / fx.duration);
      const easeOut = 1 - Math.pow(1 - t, 2.2);

      fx.mesh.visible = true;
      fx.mesh.position
        .copy(fx.origin)
        .addScaledVector(fx.dir, easeOut * fx.travelDist);
      fx.mesh.scale.setScalar(
        THREE.MathUtils.lerp(fx.startScale, fx.endScale, easeOut)
      );
      (fx.mesh.material as THREE.MeshBasicMaterial).opacity =
        (1 - t) * (1 - t * 0.35) * fx.maxOpacity;

      if (t >= 1) {
        this.scene.remove(fx.mesh);
        (fx.mesh.material as THREE.Material).dispose();
        this.forceWaveEffects.splice(i, 1);
      }
    }

    // Taser 2-wire & crackling arc updates
    const muzzleOrigin = this.computeTaserMuzzle(new THREE.Vector3());
    for (let i = this.taserEffects.length - 1; i >= 0; i--) {
      const fx = this.taserEffects[i];
      fx.age += fx.isHit ? simDt : dt;
      const t = Math.min(1, fx.age / fx.duration);

      let endA = fx.missEndA;
      let endB = fx.missEndB;
      let validHit = !fx.isHit;

      if (fx.isHit && transforms && fx.bodyIdx * 7 + 6 < transforms.length) {
        const bOff = fx.bodyIdx * 7;
        const py = transforms[bOff + 1];
        if (py > -20) {
          this.tempVecD.set(transforms[bOff], py, transforms[bOff + 2]);
          this.tempQuat.set(
            transforms[bOff + 3],
            transforms[bOff + 4],
            transforms[bOff + 5],
            transforms[bOff + 6]
          );

          endA = fx.localProngA
            .clone()
            .applyQuaternion(this.tempQuat)
            .add(this.tempVecD);
          endB = fx.localProngB
            .clone()
            .applyQuaternion(this.tempQuat)
            .add(this.tempVecD);
          fx.prongs[0].position.copy(endA);
          fx.prongs[1].position.copy(endB);
          validHit = true;

          // Crackling lightning arcs around chest/torso
          const cIdx =
            fx.chestBodyIdx >= 0 && fx.chestBodyIdx * 7 + 2 < transforms.length
              ? fx.chestBodyIdx
              : fx.bodyIdx;
          const cOff = cIdx * 7;
          const chestPos = this.tempVecE.set(
            transforms[cOff],
            transforms[cOff + 1],
            transforms[cOff + 2]
          );

          if (fx.arcs && fx.arcPos) {
            for (let a = 0; a < 2; a++) {
              this.fillLightningArc(fx.arcPos[a], chestPos, 0.24);
              const attr = fx.arcs[a].geometry.getAttribute(
                'position'
              ) as THREE.BufferAttribute;
              attr.needsUpdate = true;
              (fx.arcs[a].material as THREE.LineBasicMaterial).opacity =
                0.55 + Math.random() * 0.45;
            }
          }

          if (simDt > 0) {
            fx.sparkAccum += dt;
            if (fx.sparkAccum >= 0.045) {
              fx.sparkAccum = 0;
              this.emitElectricSparks(chestPos, 3, 0.26);
            }
          }
        }
      }

      if (!validHit || t >= 1) {
        this.disposeTaserEffect(fx);
        this.taserEffects.splice(i, 1);
        continue;
      }

      const distA = muzzleOrigin.distanceTo(endA);
      const distB = muzzleOrigin.distanceTo(endB);
      const sagA = Math.min(0.45, distA * 0.085);
      const sagB = Math.min(0.45, distB * 0.085);
      const jitter = fx.isHit ? 0.014 : 0.022;
      const wireFade = fx.isHit
        ? THREE.MathUtils.clamp((1 - t) * 4.0, 0, 1)
        : 1 - t;

      this.fillCatenaryWire(fx.wirePos[0], muzzleOrigin, endA, sagA, jitter, 0);
      this.fillCatenaryWire(fx.wirePos[1], muzzleOrigin, endB, sagB, jitter, 2.3);

      for (let w = 0; w < 2; w++) {
        const attr = fx.wires[w].geometry.getAttribute(
          'position'
        ) as THREE.BufferAttribute;
        attr.needsUpdate = true;
        (fx.wires[w].material as THREE.LineBasicMaterial).opacity =
          (w === 0 ? 0.92 : 0.85) * wireFade;
        (fx.prongs[w].material as THREE.MeshBasicMaterial).opacity =
          0.95 * wireFade;
      }
    }

    // Ground fire patches (14s lifetime, warm flickering PointLight + rising flames)
    for (let i = this.groundFirePatches.length - 1; i >= 0; i--) {
      const patch = this.groundFirePatches[i];
      patch.remaining -= simDt;
      if (patch.remaining <= 0) {
        this.disposeGroundFirePatch(patch);
        this.groundFirePatches.splice(i, 1);
        continue;
      }

      const fade = THREE.MathUtils.clamp(patch.remaining / 2.5, 0, 1);
      const flicker =
        0.82 +
        0.24 * Math.sin(this.fxTime * 21.0 + patch.phase) +
        0.14 * Math.sin(this.fxTime * 47.0 + patch.phase * 1.9);

      patch.light.intensity = 2.75 * flicker * fade;
      patch.disc.scale.setScalar(0.95 + 0.08 * flicker);
      patch.ring.scale.setScalar(0.92 + 0.14 * flicker);
      (patch.disc.material as THREE.MeshBasicMaterial).opacity = 0.78 * fade;
      (patch.ring.material as THREE.MeshBasicMaterial).opacity =
        0.48 * flicker * fade;

      if (simDt > 0) {
        patch.emitAccum += dt;
        if (patch.emitAccum >= 0.045) {
          patch.emitAccum = 0;
          this.emitFlameParticles(patch.pos, 3, 0.45);
        }
      }
    }

    // Burning dynamic props (20s lifetime, tracking live body position)
    for (let i = this.burningProps.length - 1; i >= 0; i--) {
      const prop = this.burningProps[i];
      prop.remaining -= simDt;

      if (
        prop.remaining <= 0 ||
        !transforms ||
        prop.bodyIdx * 7 + 2 >= transforms.length
      ) {
        this.burningProps.splice(i, 1);
        continue;
      }

      const pOff = prop.bodyIdx * 7;
      const py = transforms[pOff + 1];
      if (py < -20) {
        this.burningProps.splice(i, 1);
        continue;
      }
      prop.pos.set(transforms[pOff], py, transforms[pOff + 2]);

      if (simDt > 0) {
        prop.emitAccum += dt;
        if (prop.emitAccum >= 0.05) {
          prop.emitAccum = 0;
          this.emitFlameParticles(prop.pos, 2, 0.22);
        }
      }
    }

    // Rising fire & ember particles
    for (let i = this.fireParticles.length - 1; i >= 0; i--) {
      const p = this.fireParticles[i];
      p.age += dt;
      const t = Math.min(1, p.age / p.duration);

      p.velocity.x += (Math.random() - 0.5) * 1.4 * dt;
      p.velocity.z += (Math.random() - 0.5) * 1.4 * dt;
      p.velocity.y += (p.isEmber ? 0.5 : 0.9) * dt;
      p.mesh.position.addScaledVector(p.velocity, dt);
      p.mesh.rotation.y += 3.2 * dt;
      p.mesh.rotation.x += 2.1 * dt;

      const s = THREE.MathUtils.lerp(p.startScale, p.endScale, t);
      p.mesh.scale.set(s, s * (p.isEmber ? 1.0 : 1.55), s);
      (p.mesh.material as THREE.MeshBasicMaterial).opacity =
        (1 - t) * (p.isEmber ? 0.95 : 0.8);

      if (t >= 1) {
        this.scene.remove(p.mesh);
        (p.mesh.material as THREE.Material).dispose();
        this.fireParticles.splice(i, 1);
      }
    }
  }
}
