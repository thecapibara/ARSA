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
  ThemeMode,
} from './scene.ts';
import { BoneId } from './mannequin.ts';

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
  set_intoxication?: (level: number) => void;
  get_static_boxes_oriented?: () => Float32Array;
  get_heightfield_meta?: () => Float32Array;
  get_heightfield_heights?: () => Float32Array;
  set_lod_focus?: (
    x: number,
    y: number,
    z: number,
    reduced_dist: number,
    sleep_dist: number
  ) => void;
  give_ragdoll_held_box?: (
    ragdoll_id: number,
    hx: number,
    hy: number,
    hz: number,
    mass: number,
    break_force: number
  ) => number;
  release_hand_hold?: (ragdoll_id: number, is_left: boolean) => void;
  is_hand_holding?: (ragdoll_id: number, is_left: boolean) => boolean;
  set_ragdoll_status?: (
    ragdoll_id: number,
    effect: number,
    intensity: number,
    duration_sec: number
  ) => boolean;
  set_all_ragdolls_status?: (
    effect: number,
    intensity: number,
    duration_sec: number
  ) => void;
  get_ragdoll_status?: (ragdoll_id: number) => number;
  set_ragdoll_desired_velocity?: (
    ragdoll_id: number,
    vx: number,
    vz: number
  ) => boolean;
  set_all_ragdolls_desired_velocity?: (vx: number, vz: number) => void;
  spawn_box_oriented?: (
    x: number,
    y: number,
    z: number,
    hx: number,
    hy: number,
    hz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    mass: number,
    vx: number,
    vy: number,
    vz: number
  ) => number;
  get_water_level?: () => number;
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
  | 'force_wave'
  | 'taser'
  | 'ignite';
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
  taser: 'TOOL: TASER (ELECTROCUTE)',
  ignite: 'TOOL: IGNITE (FIRE)',
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
  private openPopover: 'placer' | 'thrower' | 'scenario' | null = null;
  private activeScenario = 0;
  private timeScale = 1.0;
  private paused = false;
  private activeMuscleMode = true;
  private autoGetUpEnabled = true;
  private walkForward = false;
  private earthquakeMagnitude = 0;
  private intoxicationLevel = 0;
  private debugTelemetry = true;
  private statusFxAccum = 0;
  private fireClockSec = 0;
  private readonly fireContactRefreshMap = new Map<number, number>();
  private theme: ThemeMode = 'dark';
  private placerYaw = 0;
  private selectedRagdollId = 0;
  private rmbDownX = 0;
  private rmbDownY = 0;
  private rmbTrackActive = false;

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
  private readonly sliderIntoxication = document.getElementById(
    'slider-intoxication'
  ) as HTMLInputElement;
  private readonly valHeight = document.getElementById('val-height')!;
  private readonly valMass = document.getElementById('val-mass')!;
  private readonly valStrength = document.getElementById('val-strength')!;
  private readonly valIntoxication = document.getElementById('val-intoxication')!;
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
  private readonly scenarioTriggerBtn = document.getElementById(
    'btn-scenario-trigger'
  ) as HTMLButtonElement;
  private readonly scenarioLabel = document.getElementById('scenario-label')!;
  private readonly scenarioPopover = document.getElementById('popover-scenario')!;
  private readonly giveBoxBtn = document.getElementById(
    'btn-give-box'
  ) as HTMLButtonElement;
  private readonly walkToggleBtn = document.getElementById(
    'btn-walk-toggle'
  ) as HTMLButtonElement;
  private readonly themeToggleBtn = document.getElementById(
    'btn-theme-toggle'
  ) as HTMLButtonElement;
  private readonly selectedMannequinBtn = document.getElementById(
    'btn-selected-mannequin'
  ) as HTMLButtonElement;

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
    this.setWalkForward(false);
    this.setIntoxicationPercent(0);
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
    this.lab.clearEffects();
    this.fireContactRefreshMap.clear();
    this.fireClockSec = 0;
    this.selectedRagdollId = 0;
    this.setWalkForward(false);
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

    this.applyRagdollControlModes();
    if (typeof this.sim.set_earthquake === 'function') {
      this.sim.set_earthquake(this.earthquakeMagnitude);
    }

    if (typeof this.sim.get_static_boxes_oriented === 'function') {
      this.lab.syncStaticBoxesOriented(this.sim.get_static_boxes_oriented());
    } else {
      this.lab.syncStaticBoxes(this.sim.get_static_boxes());
    }
    if (
      typeof this.sim.get_heightfield_meta === 'function' &&
      typeof this.sim.get_heightfield_heights === 'function'
    ) {
      this.lab.syncHeightfield(
        this.sim.get_heightfield_meta(),
        this.sim.get_heightfield_heights()
      );
    } else {
      this.lab.syncHeightfield([], []);
    }

    const waterLevel =
      typeof this.sim.get_water_level === 'function'
        ? this.sim.get_water_level()
        : -1000;
    this.lab.syncWaterLevel(waterLevel);
    this.lab.setIceFloorVisual(scenario === 11);

    this.syncSimulationView();

    if (scenario === 4 || scenario === 5) {
      this.lab.frameCamera([0, 2.1, 5.4], [0, 1.1, 0]);
    } else if (scenario === 6) {
      this.lab.frameCamera([7.2, 4.0, 3.5], [0, 1.5, -0.5]);
    } else if (scenario === 7) {
      this.lab.frameCamera([5.5, 3.8, 6.0], [0, 0.8, -0.5]);
    } else if (scenario === 8) {
      this.lab.frameCamera([4.2, 5.2, 4.2], [0, 4.2, 0.2]);
    } else if (scenario === 9) {
      this.lab.frameCamera([0.0, 7.0, 10.0], [0, 1.2, -10.0]);
    } else if (scenario === 10) {
      this.lab.frameCamera([5.8, 4.4, 5.8], [0, 1.35, -0.8]);
    } else if (scenario === 11) {
      this.lab.frameCamera([5.0, 3.2, 5.6], [0, 0.8, 0.2]);
    } else if (scenario === 12) {
      this.lab.frameCamera([6.2, 4.2, 4.4], [0, 2.2, 0.4]);
    }

    document
      .querySelectorAll<HTMLButtonElement>('[data-scenario]')
      .forEach((btn) => {
        const isActive = Number(btn.dataset.scenario) === scenario;
        btn.classList.toggle('active', isActive);
        if (isActive && btn.textContent) {
          this.scenarioLabel.textContent = btn.textContent.trim();
        }
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

  private setOpenPopover(target: 'placer' | 'thrower' | 'scenario' | null): void {
    this.openPopover = target;

    const showPlacer = target === 'placer';
    this.placerPopover.hidden = !showPlacer;
    this.placerPopover.classList.toggle('open', showPlacer);
    this.placerBtn.setAttribute('aria-expanded', String(showPlacer));

    const showThrower = target === 'thrower';
    this.throwerPopover.hidden = !showThrower;
    this.throwerPopover.classList.toggle('open', showThrower);
    this.throwerBtn.setAttribute('aria-expanded', String(showThrower));

    const showScenario = target === 'scenario';
    this.scenarioPopover.hidden = !showScenario;
    this.scenarioPopover.classList.toggle('open', showScenario);
    this.scenarioTriggerBtn.setAttribute('aria-expanded', String(showScenario));
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

  private setIntoxicationPercent(percent: number): void {
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    this.intoxicationLevel = clamped / 100;
    if (this.sim && typeof this.sim.set_intoxication === 'function') {
      this.sim.set_intoxication(this.intoxicationLevel);
    }
    this.sliderIntoxication.value = String(clamped);
    this.valIntoxication.textContent = `${clamped}%`;
  }

  private applyRagdollControlModes(): void {
    if (!this.sim) return;
    if (typeof this.sim.set_active_mode === 'function') {
      this.sim.set_active_mode(this.activeMuscleMode);
    }
    if (typeof this.sim.set_auto_get_up === 'function') {
      this.sim.set_auto_get_up(this.autoGetUpEnabled);
    }
    if (typeof this.sim.set_intoxication === 'function') {
      this.sim.set_intoxication(this.intoxicationLevel);
    }
    if (typeof this.sim.set_all_ragdolls_desired_velocity === 'function') {
      this.sim.set_all_ragdolls_desired_velocity(
        0.0,
        this.walkForward ? 1.35 : 0.0
      );
    }
  }

  private setWalkForward(enabled: boolean): void {
    this.walkForward = enabled;
    if (
      this.sim &&
      typeof this.sim.set_all_ragdolls_desired_velocity === 'function'
    ) {
      this.sim.set_all_ragdolls_desired_velocity(0.0, enabled ? 1.35 : 0.0);
    }
    this.walkToggleBtn.classList.toggle('active', enabled);
    this.walkToggleBtn.setAttribute('aria-pressed', String(enabled));
  }

  private giveBoxToRagdolls(all = false): void {
    if (!this.sim || typeof this.sim.give_ragdoll_held_box !== 'function') {
      return;
    }
    const count = this.sim.body_count();
    const descriptors = this.sim.get_body_descriptors();
    const ragdollIds = new Set<number>();
    for (let i = 0; i < count; i++) {
      const id = Math.round(descriptors[i * 6 + 4]);
      if (id >= 0) ragdollIds.add(id);
    }

    const sortedIds = Array.from(ragdollIds).sort((a, b) => a - b);
    const targets = all
      ? sortedIds
      : ragdollIds.has(this.selectedRagdollId)
        ? [this.selectedRagdollId]
        : sortedIds.slice(0, 1);

    let given = 0;
    for (const id of targets) {
      if (given >= (all ? 4 : 1)) break;
      const holdingLeft =
        typeof this.sim.is_hand_holding === 'function' &&
        this.sim.is_hand_holding(id, true);
      const holdingRight =
        typeof this.sim.is_hand_holding === 'function' &&
        this.sim.is_hand_holding(id, false);
      if (holdingLeft || holdingRight) continue;

      const boxIdx = this.sim.give_ragdoll_held_box(
        id,
        0.18,
        0.18,
        0.18,
        6.0,
        320.0
      );
      if (boxIdx >= 0) {
        given++;
      }
    }

    if (given > 0) {
      this.syncSimulationView();
      this.audio.playBoxDrop();
    }
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
    this.applyLiveBodyProfile(true);
  }

  private applyLiveBodyProfile(rebuildGeometry = true): void {
    if (!this.sim) return;
    if (typeof this.sim.update_all_ragdolls_profile === 'function') {
      this.sim.update_all_ragdolls_profile(
        this.bodyProfile.heightCm / 100,
        this.bodyProfile.massKg,
        this.bodyProfile.strength
      );
    }
    this.syncSimulationView(rebuildGeometry);
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

  private findFreeSpawnXZ(): [number, number] {
    if (!this.sim || typeof this.sim.get_ragdoll_telemetry !== 'function') {
      return [0, 0];
    }
    const telemetry = this.sim.get_ragdoll_telemetry();
    const stride = getTelemetryStride(telemetry);
    const count = Math.floor(telemetry.length / stride);
    const occupied: Array<[number, number]> = [];
    for (let i = 0; i < count; i++) {
      const o = i * stride;
      const px = stride >= 11 ? telemetry[o + 8] : telemetry[o];
      const pz = stride >= 11 ? telemetry[o + 10] : telemetry[o + 2];
      occupied.push([px, pz]);
    }

    const minDistSq = 0.9 * 0.9;
    const isFree = (x: number, z: number): boolean => {
      for (let i = 0; i < occupied.length; i++) {
        const dx = x - occupied[i][0];
        const dz = z - occupied[i][1];
        if (dx * dx + dz * dz < minDistSq) {
          return false;
        }
      }
      return true;
    };

    const rings = [0, 1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4];
    for (const r of rings) {
      if (r === 0) {
        if (isFree(0, 0)) return [0, 0];
        continue;
      }
      const steps = Math.max(8, Math.round(r * 6));
      for (let s = 0; s < steps; s++) {
        const angle = (s / steps) * Math.PI * 2;
        const x = Math.cos(angle) * r;
        const z = Math.sin(angle) * r;
        if (isFree(x, z)) return [x, z];
      }
    }

    return [0, 7.2];
  }

  private spawnRagdoll(): void {
    if (!this.sim) return;
    const [spawnX, spawnZ] = this.findFreeSpawnXZ();
    let spawnY = 0.02;
    if (this.activeScenario === 2) {
      spawnY = Math.abs(spawnX) <= 1.1 && Math.abs(spawnZ) <= 1.1 ? 2.15 : 0.02;
    } else if (this.activeScenario === 10) {
      if (Math.abs(spawnX) < 2.8 && Math.abs(spawnZ) < 2.8) {
        spawnY = 1.25;
      } else if (Math.abs(spawnX) <= 4.5 && Math.abs(spawnZ) <= 4.5) {
        spawnY = 1.97;
      }
    } else if (this.activeScenario === 7) {
      spawnY = 0.95;
    }
    const heightM = this.bodyProfile.heightCm / 100;

    if (typeof this.sim.spawn_ragdoll_with_profile === 'function') {
      this.sim.spawn_ragdoll_with_profile(
        spawnX,
        spawnY,
        spawnZ,
        heightM,
        this.bodyProfile.massKg,
        this.bodyProfile.strength
      );
    } else {
      this.sim.spawn_ragdoll(spawnX, spawnY, spawnZ);
    }

    this.applyRagdollControlModes();
    this.syncSimulationView();
  }

  private updatePointerRay(clientX: number, clientY: number): THREE.Ray {
    const rect = this.canvas.getBoundingClientRect();
    this.pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.lab.camera);
    return this.raycaster.ray;
  }

  private getPlacerHalfExtents(mode: PlacerMode): [number, number, number] {
    switch (mode) {
      case 'curb':
        return [0.35, 0.14, 0.18];
      case 'cube':
        return [0.24, 0.24, 0.24];
      case 'sphere':
        return [0.28, 0.28, 0.28];
      case 'dynamite':
        return [0.06, 0.174, 0.06];
      case 'ragdoll':
        return [0.28, 0.0, 0.22];
    }
  }

  private updatePlacePreview(clientX: number, clientY: number): boolean {
    const ray = this.updatePointerRay(clientX, clientY);
    const surf = this.lab.raycastPlacementSurface(this.raycaster);
    let hit: THREE.Vector3 | null = null;
    if (surf) {
      const [hx, hy, hz] = this.getPlacerHalfExtents(this.activePlacerMode);
      const n = surf.normal;
      const eps = 0.004;
      // Offset flush outside the hit surface along its outward normal so objects
      // never spawn embedded inside vertical walls, cube sides, or tilted ramps.
      const center = surf.point.clone();
      if (this.activePlacerMode === 'ragdoll') {
        center.x += n.x * (hx + eps);
        center.z += n.z * (hz + eps);
        center.y = Math.max(0.0, surf.point.y);
      } else {
        // Exact support distance of an axis-aligned or yaw-rotated box along normal `n`:
        // |localN.x|*hx + |n.y|*hy + |localN.z|*hz, then convert center -> bottom-center (y - hy)
        const c = Math.cos(this.placerYaw);
        const s = Math.sin(this.placerYaw);
        const localNx = n.x * c - n.z * s;
        const localNz = n.x * s + n.z * c;
        const support =
          Math.abs(localNx) * hx + Math.abs(n.y) * hy + Math.abs(localNz) * hz + eps;
        center.addScaledVector(n, support);
        center.y = Math.max(0.0, center.y - hy);
      }
      this.groundHit.copy(center);
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
      this.bodyProfile.heightCm / 100,
      this.placerYaw
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

  private selectRagdollFromBodyIdx(bodyIdx: number): void {
    if (!this.sim || bodyIdx < 0) return;
    const descriptors = this.sim.get_body_descriptors();
    const dOff = bodyIdx * 6;
    if (dOff + 4 < descriptors.length) {
      const ragdollId = Math.round(descriptors[dOff + 4]);
      if (ragdollId >= 0) {
        this.selectedRagdollId = ragdollId;
      }
    }
  }

  private updateBrainTelemetryHud(telemetry: Float32Array): void {
    const stride = getTelemetryStride(telemetry);
    const count = Math.floor(telemetry.length / stride);
    if (count === 0) {
      this.selectedMannequinBtn.textContent = 'MANNEQUIN #--';
      this.selectedMannequinBtn.disabled = true;
      this.statRagdollState.textContent = 'NONE';
      this.statRagdollState.className = 'state-badge state-passive';
      this.contactLeft.classList.remove('grounded');
      this.contactRight.classList.remove('grounded');
      this.updateVitalHud(100, 0, true);
      return;
    }

    this.selectedRagdollId = Math.max(
      0,
      Math.min(count - 1, this.selectedRagdollId)
    );
    this.selectedMannequinBtn.textContent = `MANNEQUIN #${this.selectedRagdollId + 1}`;
    this.selectedMannequinBtn.disabled = count <= 1;

    const base = this.selectedRagdollId * stride;
    const leftGrounded = telemetry[base + 5] > 0.5;
    const rightGrounded = telemetry[base + 6] > 0.5;
    const stateCode = Math.round(telemetry[base + 7]);
    const badge = STATE_BADGES[stateCode] ?? STATE_BADGES[4];

    this.statRagdollState.textContent = badge.label;
    this.statRagdollState.className = `state-badge ${badge.cls}`;
    this.contactLeft.classList.toggle('grounded', leftGrounded);
    this.contactRight.classList.toggle('grounded', rightGrounded);

    if (stride >= 16) {
      this.updateVitalHud(
        telemetry[base + 11],
        Math.round(telemetry[base + 12]),
        false
      );
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
          this.selectRagdollFromBodyIdx(Math.round(result[base + 3]));
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
      this.selectRagdollFromBodyIdx(hitIdx);
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

  private findRagdollBoneBodyIdx(
    descriptors: Float32Array,
    count: number,
    ragdollId: number,
    targetBoneId: number
  ): number {
    for (let i = 0; i < count; i++) {
      const dOff = i * 6;
      if (
        Math.round(descriptors[dOff + 4]) === ragdollId &&
        Math.round(descriptors[dOff + 5]) === targetBoneId
      ) {
        return i;
      }
    }
    return -1;
  }

  private fireTaser(ray: THREE.Ray): void {
    if (!this.sim) return;

    const hit = this.lab.raycastDynamicBody(this.raycaster, 25.0);
    if (hit && hit.ragdollId >= 0) {
      this.selectedRagdollId = hit.ragdollId;
      if (typeof this.sim.set_ragdoll_status === 'function') {
        this.sim.set_ragdoll_status(hit.ragdollId, 2, 1.0, 3.0);
      }
      this.audio.playTaser(true);

      const transforms = this.sim.get_transforms();
      const descriptors = this.sim.get_body_descriptors();
      const count = this.sim.body_count();

      const tOff = hit.bodyIdx * 7;
      let localProngA = new THREE.Vector3(0.035, 0.02, 0.0);
      let localProngB = new THREE.Vector3(-0.035, -0.02, 0.0);

      if (tOff + 6 < transforms.length) {
        const bodyPos = new THREE.Vector3(
          transforms[tOff],
          transforms[tOff + 1],
          transforms[tOff + 2]
        );
        const bodyQuat = new THREE.Quaternion(
          transforms[tOff + 3],
          transforms[tOff + 4],
          transforms[tOff + 5],
          transforms[tOff + 6]
        );
        const invQuat = bodyQuat.clone().invert();
        const localHit = hit.point.clone().sub(bodyPos).applyQuaternion(invQuat);

        const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(
          this.lab.camera.quaternion
        );
        const camUp = new THREE.Vector3(0, 1, 0).applyQuaternion(
          this.lab.camera.quaternion
        );
        const worldSep = camRight
          .multiplyScalar(0.032)
          .addScaledVector(camUp, 0.018);
        const localSep = worldSep.applyQuaternion(invQuat);

        localProngA = localHit.clone().add(localSep);
        localProngB = localHit.clone().sub(localSep);
      }

      const chestIdx = this.findRagdollBoneBodyIdx(
        descriptors,
        count,
        hit.ragdollId,
        BoneId.Chest
      );
      this.lab.spawnTaserHit(hit.bodyIdx, chestIdx, localProngA, localProngB);
      this.syncSimulationView();
      return;
    }

    // Missed ragdoll — fire brief 0.25s twin-wire spark miss effect
    this.audio.playTaser(false);
    let endCenter: THREE.Vector3 | null = hit ? hit.point : null;
    if (!endCenter) {
      const surf = this.lab.raycastPlacementSurface(this.raycaster);
      if (surf && ray.origin.distanceTo(surf.point) <= 20.0) {
        endCenter = surf.point;
      } else if (
        ray.intersectPlane(this.groundPlane, this.groundHit) &&
        ray.origin.distanceTo(this.groundHit) <= 20.0
      ) {
        endCenter = this.groundHit.clone();
      } else {
        endCenter = ray.origin.clone().addScaledVector(ray.direction, 12.0);
      }
    }

    const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(
      this.lab.camera.quaternion
    );
    const camUp = new THREE.Vector3(0, 1, 0).applyQuaternion(
      this.lab.camera.quaternion
    );
    const spread = camRight
      .multiplyScalar(0.11)
      .addScaledVector(camUp, 0.04);
    this.lab.spawnTaserMiss(
      endCenter.clone().add(spread),
      endCenter.clone().sub(spread)
    );
  }

  private executeIgnite(ray: THREE.Ray): void {
    if (!this.sim) return;

    const hit = this.lab.raycastDynamicBody(this.raycaster, 28.0);
    if (hit) {
      if (hit.ragdollId >= 0) {
        this.selectedRagdollId = hit.ragdollId;
        if (typeof this.sim.set_ragdoll_status === 'function') {
          this.sim.set_ragdoll_status(hit.ragdollId, 1, 1.0, 6.0);
        }
        this.lab.emitFlameParticles(hit.point, 12, 0.24);
        this.audio.playIgnite();
        this.syncSimulationView();
        return;
      }

      if (hit.bodyIdx >= 0) {
        const transforms = this.sim.get_transforms();
        const tOff = hit.bodyIdx * 7;
        const propPos =
          tOff + 2 < transforms.length
            ? new THREE.Vector3(
                transforms[tOff],
                transforms[tOff + 1],
                transforms[tOff + 2]
              )
            : hit.point;
        this.lab.igniteProp(hit.bodyIdx, propPos);
        this.audio.playIgnite();
        return;
      }
    }

    // Otherwise raycast placement surface / ground plane for a Ground Fire Patch
    const surf = this.lab.raycastPlacementSurface(this.raycaster);
    let firePos: THREE.Vector3 | null = null;
    if (surf && ray.origin.distanceTo(surf.point) <= 32.0) {
      firePos = surf.point.clone();
    } else if (
      ray.intersectPlane(this.groundPlane, this.groundHit) &&
      ray.origin.distanceTo(this.groundHit) <= 32.0
    ) {
      firePos = this.groundHit.clone();
    }

    if (firePos) {
      firePos.x = Math.max(-14, Math.min(14, firePos.x));
      firePos.z = Math.max(-14, Math.min(14, firePos.z));
      this.lab.spawnGroundFirePatch(firePos);
      this.audio.playIgnite();
    }
  }

  private executeExtinguish(ray: THREE.Ray): void {
    if (!this.sim) return;

    let hitPos: THREE.Vector3 | null = null;
    const dynHit = this.lab.raycastDynamicBody(this.raycaster, 32.0);
    if (dynHit) {
      hitPos = dynHit.point.clone();
    } else {
      const surf = this.lab.raycastPlacementSurface(this.raycaster);
      if (surf && ray.origin.distanceTo(surf.point) <= 32.0) {
        hitPos = surf.point.clone();
      } else if (
        ray.intersectPlane(this.groundPlane, this.groundHit) &&
        ray.origin.distanceTo(this.groundHit) <= 32.0
      ) {
        hitPos = this.groundHit.clone();
      }
    }
    if (!hitPos) return;

    const maxRadius = 2.5;
    const maxRadiusSq = maxRadius * maxRadius;
    const visualBest = this.lab.getNearestFireVisualSource(hitPos, maxRadius);

    let bestRagdollId = -1;
    let bestRagdollDistSq = maxRadiusSq;
    if (
      typeof this.sim.get_ragdoll_status === 'function' &&
      typeof this.sim.set_ragdoll_status === 'function'
    ) {
      const count = this.sim.body_count();
      const descriptors = this.sim.get_body_descriptors();
      const transforms = this.sim.get_transforms();
      const statusCache = new Map<number, number>();
      const bodyPos = new THREE.Vector3();

      for (let i = 0; i < count; i++) {
        const dOff = i * 6;
        const ragdollId = Math.round(descriptors[dOff + 4]);
        if (ragdollId < 0) continue;

        let st = statusCache.get(ragdollId);
        if (st === undefined) {
          st = this.sim.get_ragdoll_status(ragdollId);
          statusCache.set(ragdollId, st);
        }
        if (st !== 1) continue;

        const tOff = i * 7;
        if (tOff + 2 >= transforms.length) continue;
        bodyPos.set(transforms[tOff], transforms[tOff + 1], transforms[tOff + 2]);
        const dSq = bodyPos.distanceToSquared(hitPos);
        if (dSq <= bestRagdollDistSq) {
          bestRagdollDistSq = dSq;
          bestRagdollId = ragdollId;
        }
      }
    }

    if (
      bestRagdollId >= 0 &&
      (!visualBest || bestRagdollDistSq <= visualBest.distSq)
    ) {
      this.sim.set_ragdoll_status?.(bestRagdollId, 0, 0, 0);
      this.fireContactRefreshMap.delete(bestRagdollId);
      this.selectedRagdollId = bestRagdollId;
      this.syncSimulationView();
    } else if (visualBest) {
      this.lab.removeFireVisualSource(visualBest.kind, visualBest.index);
    }
  }

  private updateFireAndStatusEffects(
    rawDt: number,
    simDt: number,
    transforms: Float32Array
  ): void {
    if (!this.sim) return;
    if (simDt <= 0) return;

    const waterLevel =
      typeof this.sim.get_water_level === 'function'
        ? this.sim.get_water_level()
        : -1000;
    if (waterLevel > -500) {
      this.lab.extinguishPropsBelowY(waterLevel + 0.15, transforms);
    }

    this.fireClockSec += simDt;
    const count = this.sim.body_count();
    const descriptors = this.sim.get_body_descriptors();
    const fireSources = this.lab.getActiveFireSources();

    const getStatus =
      typeof this.sim.get_ragdoll_status === 'function'
        ? (id: number) => this.sim!.get_ragdoll_status!(id)
        : () => 0;

    // Track per-ragdoll status & contact ignition
    const ragdollStatusCache = new Map<number, number>();
    const readStatus = (id: number): number => {
      let s = ragdollStatusCache.get(id);
      if (s === undefined) {
        s = getStatus(id);
        ragdollStatusCache.set(id, s);
      }
      return s;
    };

    const emitStatusVfx =
      (this.statusFxAccum += rawDt) >= 0.045;
    if (emitStatusVfx) {
      this.statusFxAccum = 0;
    }

    const pos = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      const dOff = i * 6;
      const ragdollId = Math.round(descriptors[dOff + 4]);
      if (ragdollId < 0) continue;

      const boneId = Math.round(descriptors[dOff + 5]);
      const tOff = i * 7;
      if (tOff + 2 >= transforms.length) continue;

      const py = transforms[tOff + 1];
      if (py < -15) continue;
      pos.set(transforms[tOff], py, transforms[tOff + 2]);

      // Automatic contact ignition against pelvis, chest, or feet
      if (
        fireSources.length > 0 &&
        (boneId === BoneId.Pelvis ||
          boneId === BoneId.Chest ||
          boneId === BoneId.LeftFoot ||
          boneId === BoneId.RightFoot)
      ) {
        for (let s = 0; s < fireSources.length; s++) {
          const src = fireSources[s];
          const dx = pos.x - src.pos.x;
          const dy = pos.y - src.pos.y;
          const dz = pos.z - src.pos.z;
          if (dx * dx + dy * dy + dz * dz <= src.radius * src.radius) {
            const curStatus = readStatus(ragdollId);
            const lastRefresh = this.fireContactRefreshMap.get(ragdollId) ?? -10;
            if (
              curStatus !== 1 ||
              this.fireClockSec - lastRefresh >= 0.4
            ) {
              if (typeof this.sim.set_ragdoll_status === 'function') {
                this.sim.set_ragdoll_status(ragdollId, 1, 1.0, 5.5);
                ragdollStatusCache.set(ragdollId, 1);
                this.fireContactRefreshMap.set(ragdollId, this.fireClockSec);
              }
            }
            break;
          }
        }
      }

      if (emitStatusVfx) {
        const status = readStatus(ragdollId);
        if (
          status === 1 &&
          (boneId === BoneId.Pelvis ||
            boneId === BoneId.Chest ||
            boneId === BoneId.Head)
        ) {
          this.lab.emitFlameParticles(pos, 1, 0.14);
        } else if (
          status === 2 &&
          (boneId === BoneId.Chest || boneId === BoneId.Pelvis)
        ) {
          this.lab.emitElectricSparks(pos, 2, 0.18);
        }
      }
    }
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

    if (
      this.sim &&
      this.activeScenario === 9 &&
      typeof this.sim.set_lod_focus === 'function'
    ) {
      const cam = this.lab.camera.position;
      this.sim.set_lod_focus(cam.x, cam.y, cam.z, 12.0, 24.0);
    }

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

    const simDt = this.sim && !this.paused ? rawDt * this.timeScale : 0;
    const transforms = this.sim ? this.sim.get_transforms() : undefined;
    if (transforms) {
      this.updateFireAndStatusEffects(rawDt, simDt, transforms);
    }

    this.lab.updateEffects(rawDt, transforms, simDt);
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
        this.applyRagdollControlModes();
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
    if (typeof this.sim.spawn_box_oriented === 'function') {
      const q = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        this.placerYaw
      );
      this.sim.spawn_box_oriented(
        x,
        y + hy + 0.002,
        z,
        hx,
        hy,
        hz,
        q.x,
        q.y,
        q.z,
        q.w,
        mass,
        0,
        0,
        0
      );
    } else {
      this.sim.spawn_box(x, y + hy + 0.002, z, hx, hy, hz, mass, 0, 0, 0);
    }
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

    this.scenarioTriggerBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setOpenPopover(this.openPopover === 'scenario' ? null : 'scenario');
      this.scenarioTriggerBtn.blur();
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-scenario]')
      .forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const scenario = Number(btn.dataset.scenario);
          if (!Number.isNaN(scenario)) {
            this.loadScenario(scenario);
            this.setOpenPopover(null);
          }
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

    const onSliderInput = (rebuildGeometry: boolean) => {
      this.bodyProfile = {
        heightCm: Number(this.sliderHeight.value),
        massKg: Number(this.sliderMass.value),
        strength: Number(this.sliderStrength.value),
      };
      this.syncProfileUi();
      this.applyLiveBodyProfile(rebuildGeometry);
    };
    this.sliderHeight.addEventListener('input', () => onSliderInput(true));
    this.sliderMass.addEventListener('input', () => onSliderInput(true));
    this.sliderStrength.addEventListener('input', () => onSliderInput(false));
    this.sliderIntoxication.addEventListener('input', () => {
      const pct = Number(this.sliderIntoxication.value);
      if (!Number.isNaN(pct)) this.setIntoxicationPercent(pct);
    });

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
    this.giveBoxBtn.addEventListener('click', (e: MouseEvent) => {
      this.audio.unlock();
      this.giveBoxToRagdolls(e.shiftKey);
      this.giveBoxBtn.blur();
    });
    this.walkToggleBtn.addEventListener('click', () => {
      this.setWalkForward(!this.walkForward);
      this.walkToggleBtn.blur();
    });
    this.themeToggleBtn.addEventListener('click', () => {
      this.theme = this.theme === 'dark' ? 'day' : 'dark';
      this.lab.setTheme(this.theme);
      this.themeToggleBtn.textContent =
        this.theme === 'dark' ? '🌙 Dark' : '☀️ Day';
      this.themeToggleBtn.blur();
    });
    this.selectedMannequinBtn.addEventListener('click', () => {
      if (!this.sim || typeof this.sim.get_ragdoll_telemetry !== 'function') {
        return;
      }
      const telemetry = this.sim.get_ragdoll_telemetry();
      const stride = getTelemetryStride(telemetry);
      const count = Math.floor(telemetry.length / stride);
      if (count > 1) {
        this.selectedRagdollId = (this.selectedRagdollId + 1) % count;
        this.updateBrainTelemetryHud(telemetry);
      }
      this.selectedMannequinBtn.blur();
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
        if (
          e.shiftKey &&
          this.activeTool === 'placer' &&
          this.lab.getCameraMode() === 'orbit'
        ) {
          e.preventDefault();
          e.stopPropagation();
          this.placerYaw -= e.deltaY * 0.0025;
          if (this.hasLastPointer) {
            this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
          }
          return;
        }
        if (this.lab.getCameraMode() === 'fly') {
          e.preventDefault();
          this.lab.adjustFlySpeed(e.deltaY);
        }
      },
      { passive: false, capture: true }
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

      if (
        e.button === 2 &&
        this.lab.getCameraMode() === 'orbit' &&
        this.activeTool === 'ignite'
      ) {
        this.rmbDownX = e.clientX;
        this.rmbDownY = e.clientY;
        this.rmbTrackActive = true;
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
          this.selectRagdollFromBodyIdx(hitIdx);
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
          this.syncSimulationView();
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
          this.selectRagdollFromBodyIdx(hitIdx);
          const hitPt = this.resolveHitPoint(this.sim.get_transforms(), hitIdx);
          this.lab.spawnImpulseEffect(hitPt, 'shove');
          this.audio.playImpact(false);
          this.syncSimulationView();
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
          this.selectRagdollFromBodyIdx(hitIdx);
          const hitPt = this.resolveHitPoint(this.sim.get_transforms(), hitIdx);
          this.lab.spawnImpulseEffect(hitPt, 'punch');
          this.audio.playImpact(true);
          this.syncSimulationView();
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
      } else if (this.activeTool === 'taser') {
        this.fireTaser(ray);
      } else if (this.activeTool === 'ignite') {
        this.executeIgnite(ray);
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
      if (this.rmbTrackActive && e.button === 2) {
        this.rmbTrackActive = false;
        const dist = Math.hypot(
          e.clientX - this.rmbDownX,
          e.clientY - this.rmbDownY
        );
        if (
          dist < 6 &&
          this.activeTool === 'ignite' &&
          this.lab.getCameraMode() === 'orbit'
        ) {
          const ray = this.updatePointerRay(e.clientX, e.clientY);
          this.executeExtinguish(ray);
        }
      } else if (e.type === 'pointercancel') {
        this.rmbTrackActive = false;
      }
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
      this.rmbTrackActive = false;
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
        case 'KeyQ':
          if (
            this.activeTool === 'placer' &&
            this.lab.getCameraMode() === 'orbit'
          ) {
            this.placerYaw += Math.PI / 12;
            if (this.hasLastPointer) {
              this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
            }
          }
          break;
        case 'KeyE':
          if (
            this.activeTool === 'placer' &&
            this.lab.getCameraMode() === 'orbit'
          ) {
            this.placerYaw -= Math.PI / 12;
            if (this.hasLastPointer) {
              this.updatePlacePreview(this.lastPointerX, this.lastPointerY);
            }
          }
          break;
        case 'KeyF':
          if (!e.repeat) {
            this.setCameraMode(
              this.lab.getCameraMode() === 'orbit' ? 'fly' : 'orbit'
            );
          }
          break;
        case 'KeyM':
          if (!e.repeat) {
            this.setActiveMuscleMode(!this.activeMuscleMode);
          }
          break;
        case 'KeyD':
          if (this.lab.getCameraMode() === 'orbit' && !e.repeat) {
            this.setDebugTelemetry(!this.debugTelemetry);
          }
          break;
        case 'KeyP':
          if (!e.repeat) {
            this.setPaused(!this.paused);
          }
          break;
        case 'KeyT':
          if (!e.repeat) {
            this.setTimeScale(this.timeScale < 0.5 ? 1.0 : 0.15);
          }
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
