import './style.css';
import * as THREE from 'three';
import init, { ArsaSim } from './wasm/arsa_wasm.js';
import { TacticalAudio } from './audio.ts';
import {
  CameraMode,
  CameraPresetName,
  getTelemetryStride,
  LabScene,
  PlacerMode,
} from './scene.ts';

type ArsaSimExt = ArsaSim & {
  spawn_dynamite?: (
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    fuse_secs: number
  ) => number;
  place_sphere?: (x: number, z: number, radius: number, mass: number) => number;
  take_explosions?: () => Float32Array;
  set_auto_get_up?: (enabled: boolean) => void;
  set_earthquake?: (magnitude: number) => void;
};

type WeaponToolMode = 'pistol' | 'shotgun' | 'rifle';
type ThrowerMode = 'cube' | 'crate' | 'sphere';
type ToolMode =
  | 'grab'
  | 'shove'
  | 'punch'
  | WeaponToolMode
  | 'placer'
  | 'thrower'
  | 'force_wave';
type ProfilePresetKey = 'light' | 'standard' | 'tall' | 'heavy';

interface BodyProfile {
  heightCm: number;
  massKg: number;
  strength: number;
}

interface WeaponSpec {
  kind: number;
  tracerColor: number;
  maxRange: number;
  fallbackImpulse: number;
}

const WEAPON_SPECS: Record<WeaponToolMode, WeaponSpec> = {
  pistol: { kind: 0, tracerColor: 0xfbbf24, maxRange: 32, fallbackImpulse: 18 },
  shotgun: { kind: 1, tracerColor: 0xff6b2b, maxRange: 24, fallbackImpulse: 45 },
  rifle: { kind: 2, tracerColor: 0x38bdf8, maxRange: 45, fallbackImpulse: 34 },
};

const PROFILE_PRESETS: Record<ProfilePresetKey, BodyProfile> = {
  light: { heightCm: 165, massKg: 60, strength: 0.9 },
  standard: { heightCm: 178, massKg: 75, strength: 1.0 },
  tall: { heightCm: 190, massKg: 70, strength: 1.0 },
  heavy: { heightCm: 217, massKg: 135, strength: 1.35 },
};

const FIXED_DT = 1 / 60;
const MAX_STEPS_PER_FRAME = 4;

const TOOL_LABELS: Record<ToolMode, string> = {
  grab: 'TOOL: GRAB',
  shove: 'TOOL: SHOVE',
  punch: 'TOOL: PUNCH',
  pistol: 'TOOL: PISTOL (9MM)',
  shotgun: 'TOOL: SHOTGUN (12GA)',
  rifle: 'TOOL: RIFLE (5.56)',
  placer: 'TOOL: PLACER',
  thrower: 'TOOL: THROWER',
  force_wave: 'TOOL: FORCE WAVE',
};

const PLACER_LABELS: Record<PlacerMode, string> = {
  ragdoll: 'Ragdoll',
  curb: 'Curb',
  cube: 'Cube',
  sphere: 'Sphere',
  dynamite: 'Dynamite',
};

const THROWER_LABELS: Record<ThrowerMode, string> = {
  cube: 'Cube',
  crate: 'Crate',
  sphere: 'Sphere',
};

const STATE_BADGES: Array<{ label: string; cls: string }> = [
  { label: 'BALANCED', cls: 'state-balanced' },
  { label: 'STUMBLING', cls: 'state-stumbling' },
  { label: 'FALLING', cls: 'state-falling' },
  { label: 'DOWNED', cls: 'state-downed' },
  { label: 'PASSIVE', cls: 'state-passive' },
  { label: 'UNCONSCIOUS', cls: 'state-unconscious' },
  { label: 'RECOVERING', cls: 'state-recovering' },
];

const WOUND_LABELS = [
  'INTACT',
  'LEFT ARM',
  'RIGHT ARM',
  'TORSO',
  'HEAD',
  'LEFT LEG',
  'RIGHT LEG',
];

class PlaygroundApp {
  private readonly canvas: HTMLCanvasElement;
  private readonly lab: LabScene;
  private readonly audio = new TacticalAudio();
  private sim: ArsaSimExt | null = null;

  private activeTool: ToolMode = 'grab';
  private activePlacerMode: PlacerMode = 'curb';
  private activeThrowerMode: ThrowerMode = 'cube';
  private openPopover: 'placer' | 'thrower' | null = null;
  private activeScenario = 0;
  private timeScale = 1.0;
  private paused = false;
  private activeMuscleMode = true;
  private autoGetUpEnabled = true;
  private earthquakeMagnitude = 0;
  private debugTelemetry = true;

  private bodyProfile: BodyProfile = { ...PROFILE_PRESETS.standard };

  // Fixed-timestep & FPS telemetry
  private accumulator = 0;
  private lastFrameTime = performance.now();
  private fpsFrames = 0;
  private fpsAccum = 0;
  private currentFps = 60;

  // Raycasting, Grab & Drag, Place Box & Free-Fly Look state
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointerNdc = new THREE.Vector2();
  private readonly dragPlane = new THREE.Plane();
  private readonly groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private readonly groundHit = new THREE.Vector3();
  private readonly dragTarget = new THREE.Vector3();
  private readonly grabbedBodyPos = new THREE.Vector3();
  private grabbedBodyIndex = -1;
  private isFlyLooking = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private hasLastPointer = false;

  // Cached DOM references
  private readonly statFps = document.getElementById('stat-fps')!;
  private readonly statBodies = document.getElementById('stat-bodies')!;
  private readonly statRagdolls = document.getElementById('stat-ragdolls')!;
  private readonly statTool = document.getElementById('stat-tool-readout')!;
  private readonly statScale = document.getElementById('stat-scale-readout')!;
  private readonly statRagdollState = document.getElementById('stat-ragdoll-state')!;
  private readonly contactLeft = document.getElementById('contact-left')!;
  private readonly contactRight = document.getElementById('contact-right')!;
  private readonly healthBarFill = document.getElementById('health-bar-fill')!;
  private readonly statHealth = document.getElementById('stat-health')!;
  private readonly statWound = document.getElementById('stat-wound')!;
  private readonly audioToggleBtn = document.getElementById('btn-audio-toggle')!;
  private readonly statusDot = document.getElementById('status-dot')!;
  private readonly reticle = document.getElementById('reticle')!;
  private readonly banner = document.getElementById('loading-banner')!;
  private readonly pauseBtn = document.getElementById('btn-pause')!;
  private readonly camModeBtn = document.getElementById('btn-cam-mode')!;
  private readonly camModeLabel = document.getElementById('cam-mode-label')!;
  private readonly debugTelemetryBtn = document.getElementById('btn-debug-telemetry')!;
  private readonly activeModeBtn = document.getElementById('btn-active-mode')!;
  private readonly muscleModeLabel = document.getElementById('muscle-mode-label')!;
  private readonly autoGetUpBtn = document.getElementById('btn-auto-getup')!;
  private readonly autoGetUpState = document.getElementById('auto-getup-state')!;
  private readonly earthquakePanel = document.getElementById('panel-earthquake')!;
  private readonly sliderQuake = document.getElementById(
    'slider-quake'
  ) as HTMLInputElement;
  private readonly valQuake = document.getElementById('val-quake')!;

  private readonly sliderHeight = document.getElementById('slider-height') as HTMLInputElement;
  private readonly sliderMass = document.getElementById('slider-mass') as HTMLInputElement;
  private readonly sliderStrength = document.getElementById('slider-strength') as HTMLInputElement;
  private readonly valHeight = document.getElementById('val-height')!;
  private readonly valMass = document.getElementById('val-mass')!;
  private readonly valStrength = document.getElementById('val-strength')!;
  private readonly sliderSimSpeed = document.getElementById(
    'slider-sim-speed'
  ) as HTMLInputElement;
  private readonly valSimSpeed = document.getElementById('val-sim-speed')!;
  private readonly slowmoBtn = document.getElementById(
    'btn-slowmo'
  ) as HTMLButtonElement;
  private readonly placerBtn = document.getElementById(
    'btn-tool-placer'
  ) as HTMLButtonElement;
  private readonly placerLabel = document.getElementById('label-tool-placer')!;
  private readonly placerChevron = document.getElementById('chevron-placer')!;
  private readonly placerPopover = document.getElementById('popover-placer')!;
  private readonly throwerBtn = document.getElementById(
    'btn-tool-thrower'
  ) as HTMLButtonElement;
  private readonly throwerLabel = document.getElementById('label-tool-thrower')!;
  private readonly throwerChevron = document.getElementById('chevron-thrower')!;
  private readonly throwerPopover = document.getElementById('popover-thrower')!;

  constructor() {
    this.canvas = document.getElementById('viewport') as HTMLCanvasElement;
    this.lab = new LabScene(this.canvas);

    this.bindUiEvents();
    this.bindPointerEvents();
    this.bindKeyboardShortcuts();
    this.setTool('grab');
    this.syncProfileUi();
    this.setActiveMuscleMode(true);
    this.setAutoGetUp(true);
    this.setEarthquakeMagnitude(0);
    this.setDebugTelemetry(true);
    this.setCameraMode('orbit');
  }

  public async start(): Promise<void> {
    try {
      await init();
      this.sim = new ArsaSim() as ArsaSimExt;
      this.loadScenario(0);
      this.banner.classList.add('hidden');
      this.lastFrameTime = performance.now();
      requestAnimationFrame(this.loop);
    } catch (err) {
      console.error('Failed to initialize ARSA WASM:', err);
      this.banner.textContent =
        'WASM module failed to load. Build `bindings/wasm` into `playground/src/wasm/` first.';
      this.banner.classList.add('error');
    }
  }

  private loadScenario(scenario: number): void {
    if (!this.sim) return;
    this.releaseGrab();
    this.activeScenario = scenario;

    const heightM = this.bodyProfile.heightCm / 100;
    if (typeof this.sim.reset_lab_scene_with_profile === 'function') {
      this.sim.reset_lab_scene_with_profile(
        scenario,
        heightM,
        this.bodyProfile.massKg,
        this.bodyProfile.strength
      );
    } else {
      this.sim.reset_lab_scene(scenario);
    }

    if (typeof this.sim.set_active_mode === 'function') {
      this.sim.set_active_mode(this.activeMuscleMode);
    }
    if (typeof this.sim.set_auto_get_up === 'function') {
      this.sim.set_auto_get_up(this.autoGetUpEnabled);
    }
    if (typeof this.sim.set_earthquake === 'function') {
      this.sim.set_earthquake(this.earthquakeMagnitude);
    }

    this.lab.syncStaticBoxes(this.sim.get_static_boxes());
    this.syncSimulationView();

    if (scenario === 4 || scenario === 5) {
      this.lab.frameCamera([0, 2.1, 5.4], [0, 1.1, 0]);
    }

    document
      .querySelectorAll<HTMLButtonElement>('[data-scenario]')
      .forEach((btn) => {
        btn.classList.toggle('active', Number(btn.dataset.scenario) === scenario);
      });
  }

  private setTool(tool: ToolMode): void {
    if (this.activeTool !== tool) {
      this.releaseGrab();
    }
    this.activeTool = tool;
    this.updateToolReadout();

    this.canvas.classList.toggle('tool-grab', tool === 'grab');
    this.canvas.classList.toggle('tool-place', tool === 'placer');
    this.reticle.classList.toggle(
      'visible',
      tool !== 'grab' && tool !== 'placer'
    );

    if (tool !== 'placer') {
      this.lab.setPlacePreview(false);
    } else if (this.hasLastPointer) {
      this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
    }

    document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tool === tool);
    });
  }

  private updateToolReadout(): void {
    if (this.activeTool === 'placer') {
      this.statTool.textContent = `TOOL: PLACER (${PLACER_LABELS[this.activePlacerMode].toUpperCase()})`;
    } else if (this.activeTool === 'thrower') {
      this.statTool.textContent = `TOOL: THROWER (${THROWER_LABELS[this.activeThrowerMode].toUpperCase()})`;
    } else {
      this.statTool.textContent = TOOL_LABELS[this.activeTool];
    }
  }

  private setPlacerMode(mode: PlacerMode): void {
    this.activePlacerMode = mode;
    this.placerLabel.textContent = `Placer: ${PLACER_LABELS[mode]}`;
    document
      .querySelectorAll<HTMLButtonElement>('[data-placer-item]')
      .forEach((item) => {
        item.classList.toggle('active', item.dataset.placerItem === mode);
      });
    this.setTool('placer');
    this.updateToolReadout();
    if (this.hasLastPointer) {
      this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
    }
  }

  private setThrowerMode(mode: ThrowerMode): void {
    this.activeThrowerMode = mode;
    this.throwerLabel.textContent = `Thrower: ${THROWER_LABELS[mode]}`;
    document
      .querySelectorAll<HTMLButtonElement>('[data-thrower-item]')
      .forEach((item) => {
        item.classList.toggle('active', item.dataset.throwerItem === mode);
      });
    this.setTool('thrower');
    this.updateToolReadout();
  }

  private setOpenPopover(target: 'placer' | 'thrower' | null): void {
    this.openPopover = target;

    const showPlacer = target === 'placer';
    this.placerPopover.hidden = !showPlacer;
    this.placerPopover.classList.toggle('open', showPlacer);
    this.placerBtn.setAttribute('aria-expanded', String(showPlacer));

    const showThrower = target === 'thrower';
    this.throwerPopover.hidden = !showThrower;
    this.throwerPopover.classList.toggle('open', showThrower);
    this.throwerBtn.setAttribute('aria-expanded', String(showThrower));
  }

  private setCameraMode(mode: CameraMode): void {
    this.isFlyLooking = false;
    this.lab.setCameraMode(mode);
    this.camModeLabel.textContent = mode === 'fly' ? 'Free Fly' : 'Orbit';
    this.camModeBtn.classList.toggle('active', mode === 'fly');
  }

  private setDebugTelemetry(enabled: boolean): void {
    this.debugTelemetry = enabled;
    this.lab.setDebugTelemetryVisible(enabled);
    this.debugTelemetryBtn.classList.toggle('active', enabled);
  }

  private setActiveMuscleMode(enabled: boolean): void {
    this.activeMuscleMode = enabled;
    if (this.sim && typeof this.sim.set_active_mode === 'function') {
      this.sim.set_active_mode(enabled);
    }
    this.activeModeBtn.classList.toggle('active', enabled);
    this.muscleModeLabel.textContent = enabled
      ? 'Active Muscle'
      : 'Passive Ragdoll';
  }

  private setAutoGetUp(enabled: boolean): void {
    this.autoGetUpEnabled = enabled;
    if (this.sim && typeof this.sim.set_auto_get_up === 'function') {
      this.sim.set_auto_get_up(enabled);
    }
    this.autoGetUpBtn.classList.toggle('active', enabled);
    this.autoGetUpBtn.setAttribute('aria-pressed', String(enabled));
    this.autoGetUpState.textContent = enabled ? 'ON' : 'OFF';
  }

  private setEarthquakeMagnitude(magnitude: number): void {
    const clamped = Math.max(0, Math.min(9, Math.round(magnitude * 2) / 2));
    this.earthquakeMagnitude = clamped;
    if (this.sim && typeof this.sim.set_earthquake === 'function') {
      this.sim.set_earthquake(clamped);
    }

    this.sliderQuake.value = clamped.toFixed(1);
    const active = clamped > 0.1;
    this.valQuake.textContent = active ? `M ${clamped.toFixed(1)}` : 'OFF';
    this.earthquakePanel.classList.toggle('quake-active', active);

    document
      .querySelectorAll<HTMLButtonElement>('[data-quake-preset]')
      .forEach((btn) => {
        const preset = Number(btn.dataset.quakePreset);
        btn.classList.toggle('active', Math.abs(preset - clamped) < 0.05);
      });

    this.syncEarthquakeEffects();
  }

  private syncEarthquakeEffects(): void {
    this.lab.setEarthquakeIntensity(this.earthquakeMagnitude);
    this.audio.setEarthquakeRumble(this.earthquakeMagnitude, !this.paused);
  }

  private applyProfilePreset(key: ProfilePresetKey): void {
    this.bodyProfile = { ...PROFILE_PRESETS[key] };
    this.syncProfileUi();
    this.applyLiveBodyProfile();
  }

  private applyLiveBodyProfile(): void {
    if (!this.sim) return;
    if (typeof this.sim.update_all_ragdolls_profile === 'function') {
      this.sim.update_all_ragdolls_profile(
        this.bodyProfile.heightCm / 100,
        this.bodyProfile.massKg,
        this.bodyProfile.strength
      );
    }
    this.syncSimulationView(true);
    if (this.activeTool === 'placer' && this.hasLastPointer) {
      this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
    }
  }

  private syncProfileUi(): void {
    this.sliderHeight.value = String(this.bodyProfile.heightCm);
    this.sliderMass.value = String(this.bodyProfile.massKg);
    this.sliderStrength.value = String(this.bodyProfile.strength);

    this.valHeight.textContent = `${Math.round(this.bodyProfile.heightCm)} cm`;
    this.valMass.textContent = `${Math.round(this.bodyProfile.massKg)} kg`;
    this.valStrength.textContent = `${this.bodyProfile.strength.toFixed(2)}×`;

    document
      .querySelectorAll<HTMLButtonElement>('[data-profile-preset]')
      .forEach((btn) => {
        const key = btn.dataset.profilePreset as ProfilePresetKey | undefined;
        if (!key) return;
        const p = PROFILE_PRESETS[key];
        const matches =
          Math.abs(p.heightCm - this.bodyProfile.heightCm) < 0.5 &&
          Math.abs(p.massKg - this.bodyProfile.massKg) < 0.5 &&
          Math.abs(p.strength - this.bodyProfile.strength) < 0.02;
        btn.classList.toggle('active', matches);
      });
  }

  private setTimeScale(scale: number): void {
    this.timeScale = scale;
    if (this.paused) {
      this.setPaused(false);
    }
    this.updateTimeReadout();

    document
      .querySelectorAll<HTMLButtonElement>('[data-timescale]')
      .forEach((btn) => {
        btn.classList.toggle(
          'active',
          Math.abs(Number(btn.dataset.timescale) - scale) < 1e-4
        );
      });
    if (this.sliderSimSpeed) {
      this.sliderSimSpeed.value = scale.toFixed(2);
    }
    if (this.valSimSpeed) {
      this.valSimSpeed.textContent = `${scale.toFixed(2)}×`;
    }
    if (this.slowmoBtn) {
      this.slowmoBtn.classList.toggle('active', scale < 0.5);
    }
  }

  private setPaused(paused: boolean): void {
    this.paused = paused;
    this.pauseBtn.classList.toggle('active', paused);
    this.statusDot.classList.toggle('paused', paused);
    this.lab.setSimulationPaused(paused);
    this.syncEarthquakeEffects();
    this.updateTimeReadout();
  }

  private updateTimeReadout(): void {
    this.statScale.textContent = this.paused
      ? 'PAUSED'
      : `${this.timeScale.toFixed(2)}×`;
  }

  private spawnRagdoll(): void {
    if (!this.sim) return;
    const jitterX = (Math.random() - 0.5) * 0.65;
    const jitterZ = (Math.random() - 0.5) * 0.65;
    const spawnY = this.activeScenario === 2 ? 3.2 : 1.35;
    const heightM = this.bodyProfile.heightCm / 100;

    if (typeof this.sim.spawn_ragdoll_with_profile === 'function') {
      this.sim.spawn_ragdoll_with_profile(
        jitterX,
        spawnY,
        jitterZ,
        heightM,
        this.bodyProfile.massKg,
        this.bodyProfile.strength
      );
    } else {
      this.sim.spawn_ragdoll(jitterX, spawnY, jitterZ);
    }

    if (typeof this.sim.set_active_mode === 'function') {
      this.sim.set_active_mode(this.activeMuscleMode);
    }
    if (typeof this.sim.set_auto_get_up === 'function') {
      this.sim.set_auto_get_up(this.autoGetUpEnabled);
    }
    this.syncSimulationView();
  }

  private updatePointerRay(clientX: number, clientY: number): THREE.Ray {
    const rect = this.canvas.getBoundingClientRect();
    this.pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.lab.camera);
    return this.raycaster.ray;
  }

  private updatePlacePreview(clientX: number, clientY: number): boolean {
    const ray = this.updatePointerRay(clientX, clientY);
    const surf = this.lab.raycastPlacementSurface(this.raycaster);
    let hit: THREE.Vector3 | null = null;
    if (surf) {
      this.groundHit.copy(surf);
      hit = this.groundHit;
    } else {
      hit = ray.intersectPlane(this.groundPlane, this.groundHit);
    }
    if (!hit) {
      this.lab.setPlacePreview(false);
      return false;
    }
    hit.x = Math.max(-12, Math.min(12, hit.x));
    hit.z = Math.max(-12, Math.min(12, hit.z));
    const dist = ray.origin.distanceTo(hit);
    if (dist > 28) {
      this.lab.setPlacePreview(false);
      return false;
    }
    this.lab.setPlacePreview(
      true,
      this.activePlacerMode,
      hit,
      this.bodyProfile.heightCm / 100
    );
    return true;
  }

  private releaseGrab(): void {
    if (this.grabbedBodyIndex >= 0 && this.sim) {
      this.sim.end_grab();
    }
    this.grabbedBodyIndex = -1;
    this.canvas.classList.remove('is-grabbing');
    this.lab.setGrabTether(false);
  }

  private syncSimulationView(forceRebuild = false): void {
    if (!this.sim) return;
    const count = this.sim.body_count();
    const transforms = this.sim.get_transforms();
    const descriptors = this.sim.get_body_descriptors();

    this.lab.syncBodies(transforms, descriptors, count, forceRebuild);
    this.lab.syncTethers(
      typeof this.sim.get_tethers === 'function'
        ? this.sim.get_tethers()
        : new Float32Array(0)
    );

    if (this.grabbedBodyIndex >= 0 && this.grabbedBodyIndex < count) {
      const tOff = this.grabbedBodyIndex * 7;
      const py = transforms[tOff + 1];
      if (py < -20.0) {
        this.releaseGrab();
      } else {
        this.grabbedBodyPos.set(
          transforms[tOff],
          py,
          transforms[tOff + 2]
        );
        this.lab.setGrabTether(true, this.grabbedBodyPos, this.dragTarget);
      }
    }

    const ragdollIds = new Set<number>();
    for (let i = 0; i < count; i++) {
      const ragdollId = Math.round(descriptors[i * 6 + 4]);
      if (ragdollId >= 0) ragdollIds.add(ragdollId);
    }

    this.statBodies.textContent = String(count);
    this.statRagdolls.textContent = String(ragdollIds.size);

    if (typeof this.sim.get_ragdoll_telemetry === 'function') {
      const telemetry = this.sim.get_ragdoll_telemetry();
      this.lab.syncRagdollTelemetry(telemetry);
      this.updateBrainTelemetryHud(telemetry);
    }
  }

  private updateBrainTelemetryHud(telemetry: Float32Array): void {
    const stride = getTelemetryStride(telemetry);
    if (telemetry.length < stride) {
      this.statRagdollState.textContent = 'NONE';
      this.statRagdollState.className = 'state-badge state-passive';
      this.contactLeft.classList.remove('grounded');
      this.contactRight.classList.remove('grounded');
      this.updateVitalHud(100, 0, true);
      return;
    }

    const leftGrounded = telemetry[5] > 0.5;
    const rightGrounded = telemetry[6] > 0.5;
    const stateCode = Math.round(telemetry[7]);
    const badge = STATE_BADGES[stateCode] ?? STATE_BADGES[4];

    this.statRagdollState.textContent = badge.label;
    this.statRagdollState.className = `state-badge ${badge.cls}`;
    this.contactLeft.classList.toggle('grounded', leftGrounded);
    this.contactRight.classList.toggle('grounded', rightGrounded);

    if (stride >= 16) {
      this.updateVitalHud(telemetry[11], Math.round(telemetry[12]), false);
    } else {
      this.updateVitalHud(100, 0, false);
    }
  }

  private updateVitalHud(
    health: number,
    woundRegion: number,
    empty: boolean
  ): void {
    const clampedHp = Math.max(0, Math.min(100, Math.round(health)));
    this.healthBarFill.style.width = `${empty ? 0 : clampedHp}%`;
    this.healthBarFill.classList.toggle('warn', !empty && clampedHp <= 60 && clampedHp > 25);
    this.healthBarFill.classList.toggle('critical', !empty && clampedHp <= 25);
    this.statHealth.textContent = empty ? '--' : `${clampedHp}%`;

    if (empty) {
      this.statWound.textContent = 'NONE';
      this.statWound.className = 'wound-badge intact';
      return;
    }

    if (woundRegion > 0) {
      const regionName = WOUND_LABELS[woundRegion] ?? `REGION ${woundRegion}`;
      this.statWound.textContent = `WOUND: ${regionName}`;
      this.statWound.className = 'wound-badge injured';
    } else {
      this.statWound.textContent = 'INTACT';
      this.statWound.className = 'wound-badge intact';
    }
  }

  private computeMuzzleOrigin(): THREE.Vector3 {
    const cam = this.lab.camera;
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion);

    return cam.position
      .clone()
      .addScaledVector(forward, 0.42)
      .addScaledVector(right, 0.22)
      .addScaledVector(up, -0.15);
  }

  private fireWeapon(ray: THREE.Ray, weapon: WeaponToolMode): void {
    if (!this.sim) return;

    const spec = WEAPON_SPECS[weapon];
    const muzzle = this.computeMuzzleOrigin();

    if (weapon === 'pistol') {
      this.audio.playPistol();
    } else if (weapon === 'shotgun') {
      this.audio.playShotgun();
    } else {
      this.audio.playRifle();
    }

    if (typeof this.sim.fire_weapon === 'function') {
      const result = this.sim.fire_weapon(
        ray.origin.x,
        ray.origin.y,
        ray.origin.z,
        ray.direction.x,
        ray.direction.y,
        ray.direction.z,
        spec.kind
      );
      const hitCount = Math.max(0, Math.round(result[0] ?? 0));

      if (hitCount > 0) {
        for (let i = 0; i < hitCount; i++) {
          const base = 1 + i * 4;
          if (base + 3 >= result.length) break;
          const hitPt = new THREE.Vector3(
            result[base],
            result[base + 1],
            result[base + 2]
          );
          this.lab.spawnTracer(muzzle, hitPt, spec.tracerColor, true);
        }
      } else {
        this.spawnMissedTracers(muzzle, ray, weapon, spec);
      }
      this.syncSimulationView();
      return;
    }

    // Graceful fallback if WASM binary has not yet been rebuilt for Phase 3
    const hitIdx = this.sim.apply_ray_impulse(
      ray.origin.x,
      ray.origin.y,
      ray.origin.z,
      ray.direction.x,
      ray.direction.y,
      ray.direction.z,
      spec.fallbackImpulse
    );
    if (hitIdx >= 0) {
      const hitPt = this.resolveHitPoint(this.sim.get_transforms(), hitIdx);
      this.lab.spawnTracer(muzzle, hitPt, spec.tracerColor, true);
    } else {
      this.spawnMissedTracers(muzzle, ray, weapon, spec);
    }
    this.syncSimulationView();
  }

  private spawnMissedTracers(
    muzzle: THREE.Vector3,
    ray: THREE.Ray,
    weapon: WeaponToolMode,
    spec: WeaponSpec
  ): void {
    if (weapon === 'shotgun') {
      for (let i = 0; i < 5; i++) {
        const spreadDir = ray.direction
          .clone()
          .add(
            new THREE.Vector3(
              (Math.random() - 0.5) * 0.08,
              (Math.random() - 0.5) * 0.08,
              (Math.random() - 0.5) * 0.08
            )
          )
          .normalize();
        const endPt = ray.origin.clone().addScaledVector(spreadDir, spec.maxRange);
        this.lab.spawnTracer(muzzle, endPt, spec.tracerColor, false);
      }
    } else {
      const endPt = ray.origin
        .clone()
        .addScaledVector(ray.direction, spec.maxRange);
      this.lab.spawnTracer(muzzle, endPt, spec.tracerColor, false);
    }
  }

  private resolveHitPoint(transforms: Float32Array, hitIdx: number): THREE.Vector3 {
    const surfaceHit = this.lab.raycastDynamicPoint(this.raycaster);
    if (surfaceHit) return surfaceHit;
    const tOff = hitIdx * 7;
    if (tOff + 2 < transforms.length) {
      return new THREE.Vector3(
        transforms[tOff],
        transforms[tOff + 1],
        transforms[tOff + 2]
      );
    }
    return new THREE.Vector3(0, 1, 0);
  }

  private readonly loop = (now: number): void => {
    requestAnimationFrame(this.loop);

    const rawDt = Math.min(0.05, Math.max(0, (now - this.lastFrameTime) / 1000));
    this.lastFrameTime = now;

    this.fpsFrames++;
    this.fpsAccum += rawDt;
    if (this.fpsAccum >= 0.35) {
      this.currentFps = Math.round(this.fpsFrames / this.fpsAccum);
      this.statFps.textContent = String(this.currentFps);
      this.fpsFrames = 0;
      this.fpsAccum = 0;
    }

    this.lab.updateCamera(rawDt);

    if (this.sim && !this.paused) {
      if (this.timeScale < 1.0) {
        // Silky-smooth 60 FPS cinematic bullet-time stepping
        const subStep = Math.min(FIXED_DT, rawDt * this.timeScale);
        if (subStep > 1e-6) {
          this.sim.step(subStep);
        }
        this.accumulator = 0;
      } else {
        this.accumulator += rawDt * this.timeScale;
        let steps = 0;
        while (this.accumulator >= FIXED_DT && steps < MAX_STEPS_PER_FRAME) {
          this.sim.step(FIXED_DT);
          this.accumulator -= FIXED_DT;
          steps++;
        }
        if (steps === MAX_STEPS_PER_FRAME && this.accumulator > FIXED_DT) {
          this.accumulator = 0;
        }
      }
      this.pollExplosions();
      this.syncSimulationView();
    } else if (this.sim && this.grabbedBodyIndex >= 0) {
      this.syncSimulationView();
    }

    this.lab.updateEffects(rawDt);
    this.lab.render();
  };

  private pollExplosions(): void {
    if (!this.sim || typeof this.sim.take_explosions !== 'function') return;
    const explosions = this.sim.take_explosions();
    for (let i = 0; i + 3 < explosions.length; i += 4) {
      const pos = new THREE.Vector3(
        explosions[i],
        explosions[i + 1],
        explosions[i + 2]
      );
      const radius = explosions[i + 3];
      this.lab.spawnExplosionEffect(pos, radius);
      this.audio.playExplosion();
    }
  }

  private executePlacerAt(x: number, y: number, z: number): void {
    if (!this.sim) return;

    switch (this.activePlacerMode) {
      case 'ragdoll': {
        const heightM = this.bodyProfile.heightCm / 100;
        if (typeof this.sim.spawn_ragdoll_with_profile === 'function') {
          this.sim.spawn_ragdoll_with_profile(
            x,
            y + 0.02,
            z,
            heightM,
            this.bodyProfile.massKg,
            this.bodyProfile.strength
          );
        } else {
          this.sim.spawn_ragdoll(x, y + 0.02, z);
        }
        if (typeof this.sim.set_active_mode === 'function') {
          this.sim.set_active_mode(this.activeMuscleMode);
        }
        if (typeof this.sim.set_auto_get_up === 'function') {
          this.sim.set_auto_get_up(this.autoGetUpEnabled);
        }
        this.audio.playBoxDrop();
        break;
      }
      case 'curb':
        this.placeBoxAt(x, y, z, 0.35, 0.14, 0.18, 30.0);
        this.audio.playBoxDrop();
        break;
      case 'cube':
        this.placeBoxAt(x, y, z, 0.24, 0.24, 0.24, 22.0);
        this.audio.playBoxDrop();
        break;
      case 'sphere':
        this.sim.spawn_sphere(x, y + 0.28 + 0.002, z, 0.28, 30.0, 0, 0, 0);
        this.audio.playBoxDrop();
        break;
      case 'dynamite':
        if (typeof this.sim.spawn_dynamite === 'function') {
          this.sim.spawn_dynamite(x, y + 0.176, z, 0, 0, 0, 1.8);
        } else {
          this.sim.spawn_box(x, y + 0.142, z, 0.06, 0.14, 0.06, 1.5, 0, 0, 0);
        }
        this.audio.playBoxDrop();
        break;
    }
  }

  private placeBoxAt(
    x: number,
    y: number,
    z: number,
    hx: number,
    hy: number,
    hz: number,
    mass: number
  ): void {
    if (!this.sim) return;
    this.sim.spawn_box(x, y + hy + 0.002, z, hx, hy, hz, mass, 0, 0, 0);
  }

  private executeThrower(ray: THREE.Ray): void {
    if (!this.sim) return;
    const spawn = ray.origin.clone().addScaledVector(ray.direction, 1.35);

    switch (this.activeThrowerMode) {
      case 'cube': {
        const vel = ray.direction.clone().multiplyScalar(15.5);
        vel.y += 1.25;
        this.sim.spawn_box(
          spawn.x,
          spawn.y,
          spawn.z,
          0.22,
          0.22,
          0.22,
          16.0,
          vel.x,
          vel.y,
          vel.z
        );
        break;
      }
      case 'crate': {
        const vel = ray.direction.clone().multiplyScalar(14.5);
        vel.y += 1.35;
        this.sim.spawn_box(
          spawn.x,
          spawn.y,
          spawn.z,
          0.34,
          0.18,
          0.22,
          22.0,
          vel.x,
          vel.y,
          vel.z
        );
        break;
      }
      case 'sphere': {
        const vel = ray.direction.clone().multiplyScalar(19.5);
        vel.y += 1.05;
        this.sim.spawn_sphere(
          spawn.x,
          spawn.y,
          spawn.z,
          0.24,
          32.0,
          vel.x,
          vel.y,
          vel.z
        );
        break;
      }
    }
  }

  private bindUiEvents(): void {
    window.addEventListener('resize', () => this.lab.resize());

    document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((btn) => {
      const tool = btn.dataset.tool as ToolMode | undefined;
      if (!tool || tool === 'placer' || tool === 'thrower') return;
      btn.addEventListener('click', () => {
        this.setOpenPopover(null);
        this.setTool(tool);
        btn.blur();
      });
    });

    this.placerChevron.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setTool('placer');
      this.setOpenPopover(this.openPopover === 'placer' ? null : 'placer');
      this.placerBtn.blur();
    });

    this.placerBtn.addEventListener('click', () => {
      if (this.activeTool !== 'placer') {
        this.setTool('placer');
        this.setOpenPopover(null);
      } else {
        this.setOpenPopover(this.openPopover === 'placer' ? null : 'placer');
      }
      this.placerBtn.blur();
    });

    this.throwerChevron.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setTool('thrower');
      this.setOpenPopover(this.openPopover === 'thrower' ? null : 'thrower');
      this.throwerBtn.blur();
    });

    this.throwerBtn.addEventListener('click', () => {
      if (this.activeTool !== 'thrower') {
        this.setTool('thrower');
        this.setOpenPopover('thrower');
      } else {
        this.setOpenPopover(this.openPopover === 'thrower' ? null : 'thrower');
      }
      this.throwerBtn.blur();
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-placer-item]')
      .forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const item = btn.dataset.placerItem as PlacerMode | undefined;
          if (item) {
            this.setPlacerMode(item);
            this.setOpenPopover(null);
          }
          btn.blur();
        });
      });

    document
      .querySelectorAll<HTMLButtonElement>('[data-thrower-item]')
      .forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const item = btn.dataset.throwerItem as ThrowerMode | undefined;
          if (item) {
            this.setThrowerMode(item);
            this.setOpenPopover(null);
          }
          btn.blur();
        });
      });

    document.addEventListener('pointerdown', (e) => {
      if (!this.openPopover) return;
      const target = e.target as Element | null;
      if (!target || !target.closest('.tool-selector-wrap')) {
        this.setOpenPopover(null);
      }
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-scenario]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const scenario = Number(btn.dataset.scenario);
          if (!Number.isNaN(scenario)) this.loadScenario(scenario);
          btn.blur();
        });
      });

    document
      .querySelectorAll<HTMLButtonElement>('[data-timescale]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const scale = Number(btn.dataset.timescale);
          if (!Number.isNaN(scale)) this.setTimeScale(scale);
          btn.blur();
        });
      });

    if (this.sliderSimSpeed) {
      this.sliderSimSpeed.addEventListener('input', () => {
        const scale = Number(this.sliderSimSpeed.value);
        if (!Number.isNaN(scale)) this.setTimeScale(scale);
      });
    }

    if (this.slowmoBtn) {
      this.slowmoBtn.addEventListener('click', () => {
        this.setTimeScale(this.timeScale < 0.5 ? 1.0 : 0.15);
        this.slowmoBtn.blur();
      });
    }

    document
      .querySelectorAll<HTMLButtonElement>('[data-cam-preset]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const preset = btn.dataset.camPreset as CameraPresetName | undefined;
          if (preset) this.lab.applyCameraPreset(preset);
          btn.blur();
        });
      });

    this.camModeBtn.addEventListener('click', () => {
      const next: CameraMode =
        this.lab.getCameraMode() === 'orbit' ? 'fly' : 'orbit';
      this.setCameraMode(next);
      this.camModeBtn.blur();
    });

    this.debugTelemetryBtn.addEventListener('click', () => {
      this.setDebugTelemetry(!this.debugTelemetry);
      this.debugTelemetryBtn.blur();
    });

    this.activeModeBtn.addEventListener('click', () => {
      this.setActiveMuscleMode(!this.activeMuscleMode);
      this.activeModeBtn.blur();
    });

    this.autoGetUpBtn.addEventListener('click', () => {
      this.setAutoGetUp(!this.autoGetUpEnabled);
      this.autoGetUpBtn.blur();
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-quake-preset]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          this.audio.unlock();
          const mag = Number(btn.dataset.quakePreset);
          if (!Number.isNaN(mag)) this.setEarthquakeMagnitude(mag);
          btn.blur();
        });
      });

    this.sliderQuake.addEventListener('input', () => {
      this.audio.unlock();
      const mag = Number(this.sliderQuake.value);
      if (!Number.isNaN(mag)) this.setEarthquakeMagnitude(mag);
    });

    this.audioToggleBtn.addEventListener('click', () => {
      this.audio.unlock();
      const muted = this.audio.toggleMute();
      this.audioToggleBtn.textContent = muted ? '🔇' : '🔊';
      this.audioToggleBtn.classList.toggle('muted', muted);
      this.audioToggleBtn.title = muted
        ? 'Unmute Procedural Audio'
        : 'Mute Procedural Audio';
      this.audioToggleBtn.blur();
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-profile-preset]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const key = btn.dataset.profilePreset as ProfilePresetKey | undefined;
          if (key) this.applyProfilePreset(key);
          btn.blur();
        });
      });

    const onSliderInput = () => {
      this.bodyProfile = {
        heightCm: Number(this.sliderHeight.value),
        massKg: Number(this.sliderMass.value),
        strength: Number(this.sliderStrength.value),
      };
      this.syncProfileUi();
      this.applyLiveBodyProfile();
    };
    this.sliderHeight.addEventListener('input', onSliderInput);
    this.sliderMass.addEventListener('input', onSliderInput);
    this.sliderStrength.addEventListener('input', onSliderInput);

    this.bindCollapsible('btn-profile-collapse', 'profile-body');
    this.bindCollapsible('btn-quake-collapse', 'quake-body');
    this.bindCollapsible('btn-legend-collapse', 'legend-body');

    this.pauseBtn.addEventListener('click', () => {
      this.setPaused(!this.paused);
      this.pauseBtn.blur();
    });
    document.getElementById('btn-reset')!.addEventListener('click', (e) => {
      this.loadScenario(this.activeScenario);
      (e.currentTarget as HTMLElement).blur();
    });
    document.getElementById('btn-spawn')!.addEventListener('click', (e) => {
      this.spawnRagdoll();
      (e.currentTarget as HTMLElement).blur();
    });
  }

  private bindCollapsible(btnId: string, bodyId: string): void {
    const btn = document.getElementById(btnId)!;
    const body = document.getElementById(bodyId)!;
    btn.addEventListener('click', () => {
      const isCollapsed = body.classList.toggle('collapsed');
      btn.textContent = isCollapsed ? '▸' : '▾';
      btn.setAttribute('aria-expanded', String(!isCollapsed));
      btn.blur();
    });
  }

  private bindPointerEvents(): void {
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.canvas.addEventListener(
      'wheel',
      (e: WheelEvent) => {
        if (this.lab.getCameraMode() === 'fly') {
          e.preventDefault();
          this.lab.adjustFlySpeed(e.deltaY);
        }
      },
      { passive: false }
    );

    this.canvas.addEventListener('pointerdown', (e: PointerEvent) => {
      this.audio.unlock();
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.hasLastPointer = true;

      if (e.button === 2 && this.lab.getCameraMode() === 'fly') {
        this.isFlyLooking = true;
        this.canvas.setPointerCapture(e.pointerId);
        return;
      }

      if (e.button !== 0 || !this.sim) return;
      const ray = this.updatePointerRay(e.clientX, e.clientY);

      if (this.activeTool === 'grab') {
        const hitIdx = this.sim.raycast_and_grab(
          ray.origin.x,
          ray.origin.y,
          ray.origin.z,
          ray.direction.x,
          ray.direction.y,
          ray.direction.z
        );
        if (hitIdx >= 0) {
          this.grabbedBodyIndex = hitIdx;
          const transforms = this.sim.get_transforms();
          const tOff = hitIdx * 7;
          this.grabbedBodyPos.set(
            transforms[tOff],
            transforms[tOff + 1],
            transforms[tOff + 2]
          );

          const camDir = new THREE.Vector3();
          this.lab.camera.getWorldDirection(camDir);
          this.dragPlane.setFromNormalAndCoplanarPoint(camDir, this.grabbedBodyPos);
          this.dragTarget.copy(this.grabbedBodyPos);

          this.canvas.setPointerCapture(e.pointerId);
          this.canvas.classList.add('is-grabbing');
          this.lab.setGrabTether(true, this.grabbedBodyPos, this.dragTarget);
        }
      } else if (this.activeTool === 'shove') {
        const hitIdx =
          typeof this.sim.apply_shove === 'function'
            ? this.sim.apply_shove(
                ray.origin.x,
                ray.origin.y,
                ray.origin.z,
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
                190.0
              )
            : this.sim.apply_ray_impulse(
                ray.origin.x,
                ray.origin.y,
                ray.origin.z,
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
                38.0
              );
        if (hitIdx >= 0) {
          const hitPt = this.resolveHitPoint(this.sim.get_transforms(), hitIdx);
          this.lab.spawnImpulseEffect(hitPt, 'shove');
          this.audio.playImpact(false);
        }
      } else if (this.activeTool === 'punch') {
        const hitIdx =
          typeof this.sim.apply_punch === 'function'
            ? this.sim.apply_punch(
                ray.origin.x,
                ray.origin.y,
                ray.origin.z,
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
                140.0
              )
            : this.sim.apply_ray_impulse(
                ray.origin.x,
                ray.origin.y,
                ray.origin.z,
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
                52.0
              );
        if (hitIdx >= 0) {
          const hitPt = this.resolveHitPoint(this.sim.get_transforms(), hitIdx);
          this.lab.spawnImpulseEffect(hitPt, 'punch');
          this.audio.playImpact(true);
        }
      } else if (
        this.activeTool === 'pistol' ||
        this.activeTool === 'shotgun' ||
        this.activeTool === 'rifle'
      ) {
        this.fireWeapon(ray, this.activeTool);
      } else if (this.activeTool === 'placer') {
        if (this.updatePlacePreview(e.clientX, e.clientY)) {
          this.executePlacerAt(
            this.groundHit.x,
            this.groundHit.y,
            this.groundHit.z
          );
          this.syncSimulationView();
        }
      } else if (this.activeTool === 'thrower') {
        this.executeThrower(ray);
        this.syncSimulationView();
      } else if (this.activeTool === 'force_wave') {
        if (typeof this.sim.apply_force_wave === 'function') {
          this.sim.apply_force_wave(
            ray.origin.x,
            ray.origin.y,
            ray.origin.z,
            ray.direction.x,
            ray.direction.y,
            ray.direction.z
          );
        } else {
          this.sim.apply_ray_impulse(
            ray.origin.x,
            ray.origin.y,
            ray.origin.z,
            ray.direction.x,
            ray.direction.y,
            ray.direction.z,
            65.0
          );
        }
        this.lab.spawnForceWaveEffect(ray.origin, ray.direction);
        this.audio.playForceWave();
        this.syncSimulationView();
      }
    });

    this.canvas.addEventListener('pointermove', (e: PointerEvent) => {
      const dx = this.hasLastPointer ? e.clientX - this.lastPointerX : e.movementX;
      const dy = this.hasLastPointer ? e.clientY - this.lastPointerY : e.movementY;
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.hasLastPointer = true;

      if (this.isFlyLooking && this.lab.getCameraMode() === 'fly') {
        this.lab.rotateFlyCamera(dx, dy);
      }

      if (this.activeTool === 'placer') {
        this.updatePlacePreview(e.clientX, e.clientY);
      }

      if (this.grabbedBodyIndex < 0 || !this.sim) return;
      const ray = this.updatePointerRay(e.clientX, e.clientY);
      if (ray.intersectPlane(this.dragPlane, this.dragTarget)) {
        this.dragTarget.y = Math.max(0.06, this.dragTarget.y);
        this.sim.move_grab(this.dragTarget.x, this.dragTarget.y, this.dragTarget.z);
        this.lab.setGrabTether(true, this.grabbedBodyPos, this.dragTarget);
      }
    });

    this.canvas.addEventListener('pointerleave', () => {
      if (this.activeTool === 'placer') {
        this.lab.setPlacePreview(false);
      }
    });

    const onEndPointer = (e: PointerEvent) => {
      if (this.isFlyLooking && (e.button === 2 || e.type === 'pointercancel')) {
        this.isFlyLooking = false;
        if (this.canvas.hasPointerCapture(e.pointerId)) {
          this.canvas.releasePointerCapture(e.pointerId);
        }
      }
      if (this.grabbedBodyIndex >= 0 && (e.button === 0 || e.type === 'pointercancel')) {
        if (this.canvas.hasPointerCapture(e.pointerId)) {
          this.canvas.releasePointerCapture(e.pointerId);
        }
        this.releaseGrab();
      }
    };

    this.canvas.addEventListener('pointerup', onEndPointer);
    this.canvas.addEventListener('pointercancel', onEndPointer);
    window.addEventListener('blur', () => {
      this.releaseGrab();
      this.isFlyLooking = false;
      this.lab.clearFlyKeys();
    });
  }

  private bindKeyboardShortcuts(): void {
    document
      .querySelectorAll<HTMLInputElement>('input[type="range"]')
      .forEach((slider) => {
        const releaseFocus = () => slider.blur();
        slider.addEventListener('change', releaseFocus);
        slider.addEventListener('pointerup', releaseFocus);
      });

    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (
        (e.target instanceof HTMLInputElement && e.target.type !== 'range') ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (e.target instanceof HTMLInputElement && e.target.type === 'range') {
        e.target.blur();
      }

      this.lab.setFlyKey(e.code, true);

      if (e.ctrlKey || e.metaKey || e.altKey) {
        return;
      }

      switch (e.code) {
        case 'Digit1':
          this.setOpenPopover(null);
          this.setTool('grab');
          break;
        case 'Digit2':
          this.setOpenPopover(null);
          this.setTool('shove');
          break;
        case 'Digit3':
          this.setOpenPopover(null);
          this.setTool('punch');
          break;
        case 'Digit4':
          this.setOpenPopover(null);
          this.setTool('pistol');
          break;
        case 'Digit5':
          this.setOpenPopover(null);
          this.setTool('shotgun');
          break;
        case 'Digit6':
          this.setOpenPopover(null);
          this.setTool('rifle');
          break;
        case 'Digit7':
          this.setOpenPopover(null);
          this.setTool('placer');
          break;
        case 'Digit8':
          this.setOpenPopover(null);
          this.setTool('thrower');
          break;
        case 'Digit9':
        case 'Digit0':
          this.setOpenPopover(null);
          this.setTool('force_wave');
          break;
        case 'Escape':
          this.setOpenPopover(null);
          break;
        case 'KeyF':
          this.setCameraMode(
            this.lab.getCameraMode() === 'orbit' ? 'fly' : 'orbit'
          );
          break;
        case 'KeyM':
          this.setActiveMuscleMode(!this.activeMuscleMode);
          break;
        case 'KeyD':
          this.setDebugTelemetry(!this.debugTelemetry);
          break;
        case 'KeyP':
          this.setPaused(!this.paused);
          break;
        case 'KeyT':
          this.setTimeScale(this.timeScale < 0.5 ? 1.0 : 0.15);
          break;
        case 'BracketLeft':
        case 'Minus':
          this.setTimeScale(Math.max(0.05, Number((this.timeScale - 0.1).toFixed(2))));
          break;
        case 'BracketRight':
          this.setTimeScale(Math.min(2.0, Number((this.timeScale + 0.1).toFixed(2))));
          break;
        case 'Space':
          e.preventDefault();
          if (this.lab.getCameraMode() === 'orbit') {
            this.setPaused(!this.paused);
          }
          break;
        case 'KeyR':
          this.loadScenario(this.activeScenario);
          break;
        case 'Equal':
        case 'NumpadAdd':
          this.spawnRagdoll();
          break;
        case 'Tab': {
          e.preventDefault();
          const hudLayer = document.querySelector<HTMLElement>('.hud-layer');
          const toast = document.getElementById('hud-hidden-toast');
          if (hudLayer) {
            const hidden = hudLayer.classList.toggle('hud-hidden');
            if (toast) {
              toast.classList.toggle('visible', hidden);
              if (hidden) {
                window.setTimeout(() => {
                  toast.classList.remove('visible');
                }, 1800);
              }
            }
          }
          break;
        }
      }
    });

    window.addEventListener('keyup', (e: KeyboardEvent) => {
      this.lab.setFlyKey(e.code, false);
    });
  }
}

const app = new PlaygroundApp();
void app.start();
