//! WebAssembly bindings (`arsa-wasm`) for the ARSA Active Ragdoll Engine.

use arsa_core::glam::Vec3;
use arsa_core::ArsaWorld;
use wasm_bindgen::prelude::*;

/// WebAssembly simulation wrapper around `ArsaWorld`.
#[wasm_bindgen]
pub struct ArsaSim {
    world: ArsaWorld,
}

impl Default for ArsaSim {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ArsaSim {
    /// Creates a new `ArsaSim` initialized with the Flat Lab scenario (scenario 0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut sim = Self {
            world: ArsaWorld::new(),
        };
        sim.reset_lab_scene(0);
        sim
    }

    /// Resets the world to a built-in test scenario:
    /// - `0`: Flat Lab
    /// - `1`: Staircase Test
    /// - `2`: Drop Platform
    pub fn reset_lab_scene(&mut self, scenario: u32) {
        self.world.clear();

        match scenario {
            1 => {
                // Staircase Test: 6 ascending steps leading up in -Z, ragdoll at top
                let step_h = 0.18;
                let step_d = 0.36;
                let num_steps = 6;
                for i in 0..num_steps {
                    let h = (i + 1) as f32 * step_h;
                    let z = -0.35 - (i as f32) * step_d;
                    self.world.add_static_box(
                        Vec3::new(0.0, h * 0.5, z),
                        Vec3::new(1.5, h * 0.5, step_d * 0.5),
                        0.75,
                        0.05,
                    );
                }
                let top_y = (num_steps as f32) * step_h;
                let top_z = -0.35 - ((num_steps - 1) as f32) * step_d;
                self.world.spawn_ragdoll(Vec3::new(0.0, top_y + 0.05, top_z));
            }
            2 => {
                // Drop Platform: elevated platform + low bumper block
                self.world.add_static_box(
                    Vec3::new(0.0, 2.0, 0.0),
                    Vec3::new(1.1, 0.1, 1.1),
                    0.75,
                    0.05,
                );
                self.world.add_static_box(
                    Vec3::new(0.0, 0.35, 1.6),
                    Vec3::new(1.2, 0.35, 0.3),
                    0.75,
                    0.05,
                );
                self.world.spawn_ragdoll(Vec3::new(0.65, 2.35, 0.25));
            }
            _ => {
                // 0: Flat Lab
                self.world.spawn_ragdoll(Vec3::new(0.0, 0.05, 0.0));
            }
        }
    }

    /// Advances the physics world by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        self.world.step(dt);
    }

    /// Returns the total number of dynamic bodies currently in the world.
    pub fn body_count(&self) -> usize {
        self.world.bodies.len()
    }

    /// Returns packed body transforms (7 floats per body: `px, py, pz, qx, qy, qz, qw`).
    pub fn get_transforms(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.world.write_transforms(&mut out);
        out
    }

    /// Returns packed body metadata descriptors (6 floats per body:
    /// `type, dim0, dim1, dim2, ragdoll_id, bone_id`).
    pub fn get_body_descriptors(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.world.write_body_meta(&mut out);
        out
    }

    /// Returns packed static box obstacles (6 floats per box: `cx, cy, cz, hx, hy, hz`).
    pub fn get_static_boxes(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.world.static_boxes.len() * 6);
        for sb in &self.world.static_boxes {
            out.push(sb.center.x);
            out.push(sb.center.y);
            out.push(sb.center.z);
            out.push(sb.half_extents.x);
            out.push(sb.half_extents.y);
            out.push(sb.half_extents.z);
        }
        out
    }

    /// Spawns a humanoid ragdoll at `(x, y, z)` and returns its ragdoll ID.
    pub fn spawn_ragdoll(&mut self, x: f32, y: f32, z: f32) -> usize {
        self.world.spawn_ragdoll(Vec3::new(x, y, z))
    }

    /// Spawns a dynamic box and returns its body index.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_box(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        hx: f32,
        hy: f32,
        hz: f32,
        mass: f32,
        vx: f32,
        vy: f32,
        vz: f32,
    ) -> usize {
        self.world.spawn_box(
            Vec3::new(x, y, z),
            Vec3::new(hx, hy, hz),
            mass,
            Vec3::new(vx, vy, vz),
        )
    }

    /// Spawns a dynamic sphere and returns its body index.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_sphere(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        r: f32,
        mass: f32,
        vx: f32,
        vy: f32,
        vz: f32,
    ) -> usize {
        self.world
            .spawn_sphere(Vec3::new(x, y, z), r, mass, Vec3::new(vx, vy, vz))
    }

    /// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and starts grabbing the hit body.
    /// Returns the hit body index, or `-1` if nothing was hit.
    pub fn raycast_and_grab(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> i32 {
        let origin = Vec3::new(ox, oy, oz);
        let dir = Vec3::new(dx, dy, dz);
        if let Some(hit) = self.world.raycast(origin, dir, 100.0) {
            self.world
                .start_grab(hit.body_idx, hit.local_point, hit.point);
            hit.body_idx as i32
        } else {
            -1
        }
    }

    /// Moves the active grab target to `(tx, ty, tz)`.
    pub fn move_grab(&mut self, tx: f32, ty: f32, tz: f32) {
        self.world.move_grab(Vec3::new(tx, ty, tz));
    }

    /// Releases the active grab constraint.
    pub fn end_grab(&mut self) {
        self.world.end_grab();
    }

    /// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and applies an impulse of `strength` N*s
    /// at the hit point. Returns the hit body index, or `-1` if nothing was hit.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_ray_impulse(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        strength: f32,
    ) -> i32 {
        let origin = Vec3::new(ox, oy, oz);
        let dir = Vec3::new(dx, dy, dz);
        let Some(hit) = self.world.raycast(origin, dir, 100.0) else {
            return -1;
        };
        let dir_n = dir.normalize_or_zero();
        self.world
            .apply_impulse_at_point(hit.body_idx, dir_n * strength, hit.point);
        hit.body_idx as i32
    }
}
