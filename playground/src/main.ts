import * as THREE from 'three';
import init, { ArsaSim } from './wasm/arsa_wasm.js';
import { LabScene } from './scene.ts';

type ToolMode = 'grab' | 'box' | 'sphere' | 'impulse';

const FIXED_DT = 1 / 60;
const MAX_STEPS_PER_FRAME = 4;

const TOOL_LABELS: Record<ToolMode, string> = {
  grab: 'TOOL: GRAB & DRAG',
  box: 'TOOL: THROW BOX',
  sphere: 'TOOL: THROW HEAVY SPHERE',
  impulse: 'TOOL: IMPULSE PUSH',
};

class PlaygroundApp {
  private readonly canvas: HTMLCanvasElement;
  private readonly lab: LabScene;
  private sim: ArsaSim | null = null;

  private activeTool: ToolMode = 'grab';
  private activeScenario = 0;
  private timeScale = 1.0;
  private paused = false;

  // Fixed-timestep & FPS telemetry
  private accumulator = 0;
  private lastFrameTime = performance.now();
  private fpsFrames = 0;
  private fpsAccum = 0;
  private currentFps = 60;

  // Grab & Drag interaction state
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointerNdc = new THREE.Vector2();
  private readonly dragPlane = new THREE.Plane();
  private readonly dragTarget = new THREE.Vector3();
  private readonly grabbedBodyPos = new THREE.Vector3();
  private grabbedBodyIndex = -1;

  // Cached DOM references
  private readonly statFps = document.getElementById('stat-fps')!;
  private readonly statBodies = document.getElementById('stat-bodies')!;
  private readonly statRagdolls = document.getElementById('stat-ragdolls')!;
  private readonly statTool = document.getElementById('stat-tool-readout')!;
  private readonly statScale = document.getElementById('stat-scale-readout')!;
  private readonly statusDot = document.getElementById('status-dot')!;
  private readonly reticle = document.getElementById('reticle')!;
  private readonly banner = document.getElementById('loading-banner')!;
  private readonly pauseBtn = document.getElementById('btn-pause')!;

  constructor() {
    this.canvas = document.getElementById('viewport') as HTMLCanvasElement;
    this.lab = new LabScene(this.canvas);

    this.bindUiEvents();
    this.bindPointerEvents();
    this.bindKeyboardShortcuts();
    this.setTool('grab');
  }

  public async start(): Promise<void> {
    try {
      await init();
      this.sim = new ArsaSim();
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
    this.sim.reset_lab_scene(scenario);
    this.lab.syncStaticBoxes(this.sim.get_static_boxes());
    this.syncSimulationView();

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
    this.statTool.textContent = TOOL_LABELS[tool];

    this.canvas.classList.toggle('tool-grab', tool === 'grab');
    this.reticle.classList.toggle('visible', tool !== 'grab');

    document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tool === tool);
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
  }

  private setPaused(paused: boolean): void {
    this.paused = paused;
    this.pauseBtn.classList.toggle('active', paused);
    this.statusDot.classList.toggle('paused', paused);
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
    this.sim.spawn_ragdoll(jitterX, spawnY, jitterZ);
    this.syncSimulationView();
  }

  private updatePointerRay(clientX: number, clientY: number): THREE.Ray {
    const rect = this.canvas.getBoundingClientRect();
    this.pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.lab.camera);
    return this.raycaster.ray;
  }

  private releaseGrab(): void {
    if (this.grabbedBodyIndex >= 0 && this.sim) {
      this.sim.end_grab();
    }
    this.grabbedBodyIndex = -1;
    this.canvas.classList.remove('is-grabbing');
    this.lab.setGrabTether(false);
  }

  private syncSimulationView(): void {
    if (!this.sim) return;
    const count = this.sim.body_count();
    const transforms = this.sim.get_transforms();
    const descriptors = this.sim.get_body_descriptors();

    this.lab.syncBodies(transforms, descriptors, count);

    // Update live grab tether anchor from the grabbed body's latest world position
    if (this.grabbedBodyIndex >= 0 && this.grabbedBodyIndex < count) {
      const tOff = this.grabbedBodyIndex * 7;
      this.grabbedBodyPos.set(
        transforms[tOff],
        transforms[tOff + 1],
        transforms[tOff + 2]
      );
      this.lab.setGrabTether(true, this.grabbedBodyPos, this.dragTarget);
    }

    // Count unique active ragdoll IDs
    const ragdollIds = new Set<number>();
    for (let i = 0; i < count; i++) {
      const ragdollId = Math.round(descriptors[i * 6 + 4]);
      if (ragdollId >= 0) ragdollIds.add(ragdollId);
    }

    this.statBodies.textContent = String(count);
    this.statRagdolls.textContent = String(ragdollIds.size);
  }

  private readonly loop = (now: number): void => {
    requestAnimationFrame(this.loop);

    const rawDt = Math.min(0.05, Math.max(0, (now - this.lastFrameTime) / 1000));
    this.lastFrameTime = now;

    // Smoothed FPS readout
    this.fpsFrames++;
    this.fpsAccum += rawDt;
    if (this.fpsAccum >= 0.35) {
      this.currentFps = Math.round(this.fpsFrames / this.fpsAccum);
      this.statFps.textContent = String(this.currentFps);
      this.fpsFrames = 0;
      this.fpsAccum = 0;
    }

    if (this.sim && !this.paused) {
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
      this.syncSimulationView();
    } else if (this.sim && this.grabbedBodyIndex >= 0) {
      this.syncSimulationView();
    }

    this.lab.updateEffects(rawDt);
    this.lab.render();
  };

  private bindUiEvents(): void {
    window.addEventListener('resize', () => this.lab.resize());

    document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const tool = btn.dataset.tool as ToolMode | undefined;
        if (tool) this.setTool(tool);
      });
    });

    document
      .querySelectorAll<HTMLButtonElement>('[data-scenario]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const scenario = Number(btn.dataset.scenario);
          if (!Number.isNaN(scenario)) this.loadScenario(scenario);
        });
      });

    document
      .querySelectorAll<HTMLButtonElement>('[data-timescale]')
      .forEach((btn) => {
        btn.addEventListener('click', () => {
          const scale = Number(btn.dataset.timescale);
          if (!Number.isNaN(scale)) this.setTimeScale(scale);
        });
      });

    this.pauseBtn.addEventListener('click', () => this.setPaused(!this.paused));
    document
      .getElementById('btn-reset')!
      .addEventListener('click', () => this.loadScenario(this.activeScenario));
    document
      .getElementById('btn-spawn')!
      .addEventListener('click', () => this.spawnRagdoll());
  }

  private bindPointerEvents(): void {
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.canvas.addEventListener('pointerdown', (e: PointerEvent) => {
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
      } else if (this.activeTool === 'box') {
        const spawn = ray.origin.clone().addScaledVector(ray.direction, 1.35);
        const vel = ray.direction.clone().multiplyScalar(15.5);
        vel.y += 1.25;
        this.sim.spawn_box(
          spawn.x,
          spawn.y,
          spawn.z,
          0.22,
          0.22,
          0.22,
          8.0,
          vel.x,
          vel.y,
          vel.z
        );
        this.syncSimulationView();
      } else if (this.activeTool === 'sphere') {
        const spawn = ray.origin.clone().addScaledVector(ray.direction, 1.35);
        const vel = ray.direction.clone().multiplyScalar(19.5);
        vel.y += 1.05;
        this.sim.spawn_sphere(
          spawn.x,
          spawn.y,
          spawn.z,
          0.26,
          26.0,
          vel.x,
          vel.y,
          vel.z
        );
        this.syncSimulationView();
      } else if (this.activeTool === 'impulse') {
        const hitIdx = this.sim.apply_ray_impulse(
          ray.origin.x,
          ray.origin.y,
          ray.origin.z,
          ray.direction.x,
          ray.direction.y,
          ray.direction.z,
          38.0
        );
        if (hitIdx >= 0) {
          const transforms = this.sim.get_transforms();
          const tOff = hitIdx * 7;
          if (tOff + 2 < transforms.length) {
            this.lab.spawnImpulseEffect(
              new THREE.Vector3(
                transforms[tOff],
                transforms[tOff + 1],
                transforms[tOff + 2]
              )
            );
          }
        }
      }
    });

    this.canvas.addEventListener('pointermove', (e: PointerEvent) => {
      if (this.grabbedBodyIndex < 0 || !this.sim) return;
      const ray = this.updatePointerRay(e.clientX, e.clientY);
      if (ray.intersectPlane(this.dragPlane, this.dragTarget)) {
        this.dragTarget.y = Math.max(0.06, this.dragTarget.y);
        this.sim.move_grab(this.dragTarget.x, this.dragTarget.y, this.dragTarget.z);
        this.lab.setGrabTether(true, this.grabbedBodyPos, this.dragTarget);
      }
    });

    const onEndGrab = (e: PointerEvent) => {
      if (this.grabbedBodyIndex >= 0) {
        if (this.canvas.hasPointerCapture(e.pointerId)) {
          this.canvas.releasePointerCapture(e.pointerId);
        }
        this.releaseGrab();
      }
    };

    this.canvas.addEventListener('pointerup', onEndGrab);
    this.canvas.addEventListener('pointercancel', onEndGrab);
    window.addEventListener('blur', () => this.releaseGrab());
  }

  private bindKeyboardShortcuts(): void {
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case '1':
          this.setTool('grab');
          break;
        case '2':
          this.setTool('box');
          break;
        case '3':
          this.setTool('sphere');
          break;
        case '4':
          this.setTool('impulse');
          break;
        case ' ':
          e.preventDefault();
          this.setPaused(!this.paused);
          break;
        case 'r':
        case 'R':
          this.loadScenario(this.activeScenario);
          break;
        case '+':
        case '=':
          this.spawnRagdoll();
          break;
      }
    });
  }
}

const app = new PlaygroundApp();
void app.start();
