import * as THREE from 'three';

export type ImpulseEffectVariant = 'shove' | 'punch';

export interface DynamiteFuseTip {
  core: THREE.Mesh;
  halo: THREE.Mesh;
  phase: number;
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

/**
 * Manages transient 3D visual FX: grab tether line & endpoints, impulse shockwaves,
 * ballistic tracers & spark bursts, Force Wave rings, 3D explosions, and fuse-tip flicker.
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
  private readonly tracerBeamGeo: THREE.CylinderGeometry;
  private readonly sparkGeo: THREE.BoxGeometry;
  private readonly forceWaveRingGeo: THREE.RingGeometry;
  private readonly explosionSphereGeo: THREE.SphereGeometry;
  private readonly explosionGroundRingGeo: THREE.RingGeometry;

  private readonly tempVecA = new THREE.Vector3();
  private readonly tempVecB = new THREE.Vector3();
  private readonly tempVecC = new THREE.Vector3();
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

  public update(
    dt: number,
    dynamiteFuseTips: ReadonlySet<DynamiteFuseTip>
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
  }
}
