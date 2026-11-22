//! WebAssembly bindings (`arsa-wasm`) for the ARSA Active Ragdoll Engine.

use arsa_core::glam::{Quat, Vec3};
use arsa_core::{ArsaWorld, BodyProfile, Heightfield, StatusEffect, WeaponKind};
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

    /// Resets the world to a built-in test scenario with default body profile:
    /// - `0`: Flat Lab
    /// - `1`: Staircase Test
    /// - `2`: Drop Platform
    /// - `3`: Trip Test
    /// - `4`: Wrecking Ball
    /// - `5`: Crowd (4x Diverse Builds)
    /// - `6`: Slope Ramp (Ramp & Slide)
    /// - `7`: Rolling Hills (Terrain Hills)
    /// - `8`: Cliff Ledge (Cliff Teeter)
    /// - `9`: Crowd 16x LOD
    pub fn reset_lab_scene(&mut self, scenario: u32) {
        self.reset_lab_scene_with_profile(scenario, 1.78, 75.0, 1.0);
    }

    /// Resets the world to a built-in test scenario using a custom anthropometric `BodyProfile`:
    /// - `0`: Flat Lab
    /// - `1`: Staircase Test
    /// - `2`: Drop Platform
    /// - `3`: Trip Test (standing mannequin + low obstacle box behind heels)
    /// - `4`: Wrecking Ball (standing mannequin + gantry beam + tethered 85kg wrecking ball)
    /// - `5`: Crowd (4x diverse builds in tight staggered formation facing +Z)
    /// - `6`: Slope Ramp (elevated start platform + 20° ramp + 2 tumbling crates)
    /// - `7`: Rolling Hills (33x33 procedural heightfield terrain + 2 ragdolls + 3 crates)
    /// - `8`: Cliff Ledge (4.4m tower + mid-cliff ledge + ragdoll near front brink)
    /// - `9`: Crowd 16x LOD (4x4 formation with spatial camera-distance LOD)
    pub fn reset_lab_scene_with_profile(
        &mut self,
        scenario: u32,
        height_m: f32,
        mass_kg: f32,
        muscle_ratio: f32,
    ) {
        self.world.clear();
        self.world.set_heightfield(None);
        self.world.set_lod_focus(None, 15.0, 35.0);
        let profile = BodyProfile {
            height_m,
            mass_kg,
            muscle_ratio,
        };

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
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, top_y + 0.05, top_z), profile);
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
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.65, 2.35, 0.25), profile);
            }
            3 => {
                // Trip Test: standing mannequin + low obstacle box behind heels
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, 0.02, 0.0), profile);
                self.world
                    .place_obstacle_box(0.0, -0.52, 0.45, 0.14, 0.16, 35.0);
            }
            4 => {
                // Wrecking Ball: standing mannequin + overhead gantry + tethered 85kg steel ball
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, 0.02, 0.0), profile);
                self.world.add_static_box(
                    Vec3::new(0.0, 4.35, 0.0),
                    Vec3::new(1.4, 0.12, 0.18),
                    0.75,
                    0.05,
                );
                let ball_idx = self.world.spawn_sphere(
                    Vec3::new(-2.75, 2.35, 0.0),
                    0.42,
                    85.0,
                    Vec3::new(1.8, 0.0, 0.0),
                );
                self.world
                    .add_tether(Vec3::new(0.0, 4.25, 0.0), ball_idx, 3.15);
            }
            5 => {
                // Crowd (4x Diverse Builds) in tight staggered formation facing +Z
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(0.0, 0.02, 0.75),
                    BodyProfile {
                        height_m: 1.78,
                        mass_kg: 75.0,
                        muscle_ratio: 1.0,
                    },
                );
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(-0.75, 0.02, -0.05),
                    BodyProfile {
                        height_m: 1.65,
                        mass_kg: 60.0,
                        muscle_ratio: 0.9,
                    },
                );
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(0.75, 0.02, -0.05),
                    BodyProfile {
                        height_m: 1.90,
                        mass_kg: 70.0,
                        muscle_ratio: 1.0,
                    },
                );
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(0.0, 0.02, -0.95),
                    BodyProfile {
                        height_m: 2.17,
                        mass_kg: 135.0,
                        muscle_ratio: 1.35,
                    },
                );
            }
            6 => {
                // Slope Ramp: elevated start platform + 20-deg sloped ramp + 2 tumbling crates
                self.world.add_static_box(
                    Vec3::new(0.0, 1.55, -2.6),
                    Vec3::new(1.6, 0.15, 1.0),
                    0.75,
                    0.05,
                );
                self.world.add_static_oriented_box(
                    Vec3::new(0.0, 0.78, 0.85),
                    Vec3::new(1.5, 0.12, 2.6),
                    Quat::from_rotation_x(20.0_f32.to_radians()),
                    0.55,
                    0.05,
                );
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, 1.72, -2.0), profile);
                self.world.spawn_box(
                    Vec3::new(-0.75, 1.98, -1.4),
                    Vec3::splat(0.22),
                    12.0,
                    Vec3::ZERO,
                );
                self.world.spawn_box(
                    Vec3::new(0.75, 1.98, -1.4),
                    Vec3::splat(0.22),
                    12.0,
                    Vec3::ZERO,
                );
            }
            7 => {
                // Rolling Hills: 33x33 procedural heightfield terrain + 2 ragdolls + 3 crates
                let cols = 33usize;
                let rows = 33usize;
                let origin_x = -10.0_f32;
                let origin_z = -10.0_f32;
                let cell_size = 0.625_f32;
                let hill_height = |x: f32, z: f32| -> f32 {
                    0.65 * (0.5 + 0.5 * (x * 0.45).sin() * (z * 0.45).cos())
                        + 0.25 * ((x * 0.25 + z * 0.3).sin() + 1.0) * 0.5
                };
                let mut heights = Vec::with_capacity(cols * rows);
                for r in 0..rows {
                    let z = origin_z + r as f32 * cell_size;
                    for c in 0..cols {
                        let x = origin_x + c as f32 * cell_size;
                        heights.push(hill_height(x, z));
                    }
                }
                self.world.set_heightfield(Some(Heightfield {
                    origin_x,
                    origin_z,
                    cell_size,
                    cols,
                    rows,
                    heights,
                    friction: 0.75,
                    restitution: 0.05,
                }));
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(-1.2, hill_height(-1.2, 0.0) + 0.03, 0.0),
                    profile,
                );
                self.world.spawn_ragdoll_with_profile(
                    Vec3::new(1.6, hill_height(1.6, -1.2) + 0.03, -1.2),
                    BodyProfile {
                        height_m: 1.88,
                        mass_kg: 92.0,
                        muscle_ratio: 1.1,
                    },
                );
                for (cx, cz) in [(3.2, 0.2), (-2.8, -3.5), (0.4, 3.2)] {
                    self.world.spawn_box(
                        Vec3::new(cx, hill_height(cx, cz) + 0.55, cz),
                        Vec3::splat(0.24),
                        16.0,
                        Vec3::ZERO,
                    );
                }
            }
            8 => {
                // Cliff Ledge: 4.4m tower + mid-cliff ledge + ragdoll near front brink
                self.world.add_static_box(
                    Vec3::new(0.0, 2.2, -0.8),
                    Vec3::new(1.35, 2.2, 1.35),
                    0.75,
                    0.05,
                );
                self.world.add_static_box(
                    Vec3::new(0.0, 2.4, 1.05),
                    Vec3::new(1.1, 0.12, 0.45),
                    0.75,
                    0.05,
                );
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, 4.42, 0.36), profile);
            }
            9 => {
                // Crowd 16x LOD: 4x4 formation spanning Full / Reduced / Sleep LOD tiers
                let xs = [-6.0_f32, -2.0, 2.0, 6.0];
                let zs = [0.0_f32, -6.0, -16.0, -28.0];
                for (row, &z) in zs.iter().enumerate() {
                    for (col, &x) in xs.iter().enumerate() {
                        let idx = row * 4 + col;
                        let h = 1.62 + ((idx * 7) % 16) as f32 / 15.0 * (2.05 - 1.62);
                        let m = 58.0 + ((idx * 11 + 3) % 16) as f32 / 15.0 * (115.0 - 58.0);
                        let s = 0.85 + ((idx * 5 + 1) % 16) as f32 / 15.0 * (1.35 - 0.85);
                        self.world.spawn_ragdoll_with_profile(
                            Vec3::new(x, 0.03, z),
                            BodyProfile {
                                height_m: h,
                                mass_kg: m,
                                muscle_ratio: s,
                            },
                        );
                    }
                }
                self.world
                    .set_lod_focus(Some(Vec3::new(0.0, 1.5, 4.0)), 10.0, 22.0);
            }
            _ => {
                // 0: Flat Lab
                self.world
                    .spawn_ragdoll_with_profile(Vec3::new(0.0, 0.05, 0.0), profile);
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

    /// Returns packed oriented static box obstacles (10 floats per box:
    /// `cx, cy, cz, hx, hy, hz, qx, qy, qz, qw`).
    pub fn get_static_boxes_oriented(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.world.static_boxes.len() * 10);
        for sb in &self.world.static_boxes {
            out.push(sb.center.x);
            out.push(sb.center.y);
            out.push(sb.center.z);
            out.push(sb.half_extents.x);
            out.push(sb.half_extents.y);
            out.push(sb.half_extents.z);
            out.push(sb.orientation.x);
            out.push(sb.orientation.y);
            out.push(sb.orientation.z);
            out.push(sb.orientation.w);
        }
        out
    }

    /// Returns heightfield metadata `[origin_x, origin_z, cell_size, cols, rows]`,
    /// or an empty `Vec` if no heightfield is active.
    pub fn get_heightfield_meta(&self) -> Vec<f32> {
        let Some(hf) = &self.world.heightfield else {
            return Vec::new();
        };
        vec![
            hf.origin_x,
            hf.origin_z,
            hf.cell_size,
            hf.cols as f32,
            hf.rows as f32,
        ]
    }

    /// Returns a copy of the active heightfield's row-major height samples,
    /// or an empty `Vec` if no heightfield is active.
    pub fn get_heightfield_heights(&self) -> Vec<f32> {
        let Some(hf) = &self.world.heightfield else {
            return Vec::new();
        };
        hf.heights.clone()
    }

    /// Returns packed active tether segments (6 floats per tether: `ax, ay, az, bx, by, bz`).
    pub fn get_tethers(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.world.write_tethers(&mut out);
        out
    }

    /// Spawns a humanoid ragdoll at `(x, y, z)` and returns its ragdoll ID.
    pub fn spawn_ragdoll(&mut self, x: f32, y: f32, z: f32) -> usize {
        self.world.spawn_ragdoll(Vec3::new(x, y, z))
    }

    /// Spawns a humanoid ragdoll at `(x, y, z)` with a custom anthropometric profile.
    pub fn spawn_ragdoll_with_profile(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        height_m: f32,
        mass_kg: f32,
        muscle_ratio: f32,
    ) -> usize {
        self.world.spawn_ragdoll_with_profile(
            Vec3::new(x, y, z),
            BodyProfile {
                height_m,
                mass_kg,
                muscle_ratio,
            },
        )
    }

    /// Enables or disables active balance & muscle motors for all ragdolls.
    pub fn set_active_mode(&mut self, enabled: bool) {
        self.world.set_all_ragdolls_active(enabled);
    }

    /// Enables or disables automatic procedural get-up from `Downed` for all ragdolls.
    pub fn set_auto_get_up(&mut self, enabled: bool) {
        self.world.set_all_ragdolls_auto_get_up(enabled);
    }

    /// Sets earthquake seismic magnitude (`0.0..=9.0`, `0.0` disables).
    pub fn set_earthquake(&mut self, magnitude: f32) {
        self.world.set_earthquake(magnitude);
    }

    /// Sets procedural stagger/intoxication level (`0.0..=1.0`, `0.0` disables) for all ragdolls.
    pub fn set_intoxication(&mut self, level: f32) {
        self.world.set_all_ragdolls_intoxication(level);
    }

    /// Returns the current earthquake seismic magnitude (`0.0..=9.0`).
    pub fn get_earthquake(&self) -> f32 {
        self.world.earthquake_magnitude
    }

    /// Sets or clears the global horizontal water surface level (`y` in meters).
    pub fn set_water_level(&mut self, has_water: bool, level: f32) {
        self.world
            .set_water_level(if has_water { Some(level) } else { None });
    }

    /// Returns the active water surface level in meters, or `-1000.0` if disabled.
    pub fn get_water_level(&self) -> f32 {
        self.world.water_level.unwrap_or(-1000.0)
    }

    /// Sets the baseline `y = 0` floor friction coefficient (`0.02..=1.5`, default `0.75`).
    pub fn set_floor_friction(&mut self, friction: f32) {
        self.world.set_floor_friction(friction);
    }

    /// Returns the baseline `y = 0` floor friction coefficient.
    pub fn get_floor_friction(&self) -> f32 {
        self.world.floor_friction
    }

    /// Updates the anthropometric body profile of all spawned ragdolls in-place
    /// without resetting the scene.
    pub fn update_all_ragdolls_profile(&mut self, height_m: f32, mass_kg: f32, muscle_ratio: f32) {
        self.world.update_all_ragdolls_profile(BodyProfile {
            height_m,
            mass_kg,
            muscle_ratio,
        });
    }

    /// Places a dynamic obstacle box resting on `y = 0` at `(x, hy, z)`.
    pub fn place_obstacle_box(
        &mut self,
        x: f32,
        z: f32,
        hx: f32,
        hy: f32,
        hz: f32,
        mass: f32,
    ) -> usize {
        self.world.place_obstacle_box(x, z, hx, hy, hz, mass)
    }

    /// Returns per-ragdoll balance & reflex telemetry (16 floats per ragdoll).
    pub fn get_ragdoll_telemetry(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.world.write_ragdoll_telemetry(&mut out);
        out
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

    /// Places a dynamic sphere resting cleanly on `y = 0` at `(x, radius, z)` with zero velocity.
    pub fn place_sphere(&mut self, x: f32, z: f32, radius: f32, mass: f32) -> usize {
        self.world
            .spawn_sphere(Vec3::new(x, radius, z), radius, mass, Vec3::ZERO)
    }

    /// Spawns a lit dynamite bundle at `(x, y, z)` with velocity `(vx, vy, vz)` and `fuse_secs`.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_dynamite(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        fuse_secs: f32,
    ) -> usize {
        self.world
            .spawn_dynamite(Vec3::new(x, y, z), Vec3::new(vx, vy, vz), fuse_secs)
    }

    /// Drains pending explosion events into `[x, y, z, radius, ...]`.
    pub fn take_explosions(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        self.world.take_explosions(&mut out);
        out
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

    /// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and applies a broad two-handed shove.
    /// Returns the hit body index, or `-1` if nothing was hit.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_shove(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        strength: f32,
    ) -> i32 {
        match self
            .world
            .apply_shove(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), strength)
        {
            Some(hit) => hit.body_idx as i32,
            None => -1,
        }
    }

    /// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and applies a sharp localized punch.
    /// Returns the hit body index, or `-1` if nothing was hit.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_punch(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        strength: f32,
    ) -> i32 {
        match self
            .world
            .apply_punch(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), strength)
        {
            Some(hit) => hit.body_idx as i32,
            None => -1,
        }
    }

    /// Fires a kinetic force wave from `(ox, oy, oz)` along `(dx, dy, dz)` with
    /// `radius = 3.2` m and `max_impulse = 42.0` N*s. Returns the count of affected bodies.
    pub fn apply_force_wave(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> usize {
        self.world
            .apply_force_wave(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), 3.2, 42.0)
    }

    /// Fires a ballistic weapon (`0 = Pistol`, `1 = Shotgun`, `2 = Rifle`) from `(ox, oy, oz)`
    /// along `(dx, dy, dz)` and returns packed hit points:
    /// `[hit_count, x0, y0, z0, body0, x1, y1, z1, body1, ...]`.
    #[allow(clippy::too_many_arguments)]
    pub fn fire_weapon(
        &mut self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        weapon_kind: u32,
    ) -> Vec<f32> {
        let weapon = match weapon_kind {
            1 => WeaponKind::Shotgun,
            2 => WeaponKind::Rifle,
            _ => WeaponKind::Pistol,
        };
        let res = self
            .world
            .fire_weapon(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), weapon);
        let mut out = Vec::with_capacity(1 + res.hits.len() * 4);
        out.push(res.hits.len() as f32);
        for hit in &res.hits {
            out.push(hit.point.x);
            out.push(hit.point.y);
            out.push(hit.point.z);
            out.push(hit.body_idx as f32);
        }
        out
    }

    /// Adds a static oriented box obstacle (XYZW quaternion) and returns its index.
    #[allow(clippy::too_many_arguments)]
    pub fn add_static_oriented_box(
        &mut self,
        cx: f32,
        cy: f32,
        cz: f32,
        hx: f32,
        hy: f32,
        hz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
        friction: f32,
        restitution: f32,
    ) -> usize {
        self.world.add_static_oriented_box(
            Vec3::new(cx, cy, cz),
            Vec3::new(hx, hy, hz),
            Quat::from_xyzw(qx, qy, qz, qw),
            friction,
            restitution,
        )
    }

    /// Sets a uniform-grid heightfield terrain collider from row-major `heights` (`cols * rows` floats).
    #[allow(clippy::too_many_arguments)]
    pub fn set_heightfield(
        &mut self,
        origin_x: f32,
        origin_z: f32,
        cell_size: f32,
        cols: usize,
        rows: usize,
        heights: &[f32],
        friction: f32,
        restitution: f32,
    ) -> bool {
        if cols < 2
            || rows < 2
            || !cell_size.is_finite()
            || cell_size <= 1e-4
            || !origin_x.is_finite()
            || !origin_z.is_finite()
            || !friction.is_finite()
            || !restitution.is_finite()
        {
            return false;
        }
        let Some(expected) = cols.checked_mul(rows) else {
            return false;
        };
        if heights.len() < expected || heights[..expected].iter().any(|v| !v.is_finite()) {
            return false;
        }
        self.world.set_heightfield(Some(Heightfield {
            origin_x,
            origin_z,
            cell_size,
            cols,
            rows,
            heights: heights[..expected].to_vec(),
            friction,
            restitution,
        }));
        true
    }

    /// Clears the world's heightfield terrain collider.
    pub fn clear_heightfield(&mut self) {
        self.world.set_heightfield(None);
    }

    /// Configures the spatial crowd-scaling LOD focus point and distance thresholds (in meters).
    pub fn set_lod_focus(&mut self, x: f32, y: f32, z: f32, reduced_dist: f32, sleep_dist: f32) {
        self.world
            .set_lod_focus(Some(Vec3::new(x, y, z)), reduced_dist, sleep_dist);
    }

    /// Clears the LOD focus point and wakes all sleeping ragdolls back to `Full`.
    pub fn clear_lod_focus(&mut self) {
        self.world
            .set_lod_focus(None, self.world.lod_reduced_dist, self.world.lod_sleep_dist);
    }

    /// Returns the current LOD tier (`0 = Full, 1 = Reduced, 2 = Sleep`) for `ragdoll_id`,
    /// or `u32::MAX` if `ragdoll_id` is out of range.
    pub fn get_ragdoll_lod(&self, ragdoll_id: usize) -> u32 {
        self.world
            .ragdoll_lod(ragdoll_id)
            .map(|l| l as u32)
            .unwrap_or(u32::MAX)
    }

    /// Attaches a ragdoll's left or right hand to `body_idx` at local anchor `(lx, ly, lz)`
    /// with grip break threshold `break_force` (Newtons).
    #[allow(clippy::too_many_arguments)]
    pub fn grab_body_with_hand(
        &mut self,
        ragdoll_id: usize,
        is_left: bool,
        body_idx: usize,
        lx: f32,
        ly: f32,
        lz: f32,
        break_force: f32,
    ) -> bool {
        self.world.grab_body_with_hand(
            ragdoll_id,
            is_left,
            body_idx,
            Vec3::new(lx, ly, lz),
            break_force,
        )
    }

    /// Releases any held object on the specified hand (`is_left`) of `ragdoll_id`.
    pub fn release_hand_hold(&mut self, ragdoll_id: usize, is_left: bool) {
        self.world.release_hand_hold(ragdoll_id, is_left);
    }

    /// Spawns a dynamic box directly in front of `ragdoll_id`'s chest and attaches both hands.
    /// Returns the spawned box body index, or `-1` on failure.
    pub fn give_ragdoll_held_box(
        &mut self,
        ragdoll_id: usize,
        hx: f32,
        hy: f32,
        hz: f32,
        mass: f32,
        break_force: f32,
    ) -> i32 {
        self.world
            .give_ragdoll_held_box(ragdoll_id, Vec3::new(hx, hy, hz), mass, break_force)
            .map(|idx| idx as i32)
            .unwrap_or(-1)
    }

    /// Returns `true` if the specified hand (`is_left`) of `ragdoll_id` is currently holding an object.
    pub fn is_hand_holding(&self, ragdoll_id: usize, is_left: bool) -> bool {
        self.world.is_hand_holding(ragdoll_id, is_left)
    }

    /// Applies a procedural status effect (`0 = None`, `1 = OnFire`, `2 = Electrocuted`)
    /// with `intensity` (`0.0..=1.0`) and `duration_sec` (`>= 0.0`) to `ragdoll_id`.
    pub fn set_ragdoll_status(
        &mut self,
        ragdoll_id: usize,
        effect: u32,
        intensity: f32,
        duration_sec: f32,
    ) -> bool {
        let Some(status) = StatusEffect::from_u32(effect) else {
            return false;
        };
        self.world
            .set_ragdoll_status(ragdoll_id, status, intensity, duration_sec)
    }

    /// Applies a procedural status effect (`0 = None`, `1 = OnFire`, `2 = Electrocuted`)
    /// to all ragdolls in the world.
    pub fn set_all_ragdolls_status(&mut self, effect: u32, intensity: f32, duration_sec: f32) {
        if let Some(status) = StatusEffect::from_u32(effect) {
            self.world
                .set_all_ragdolls_status(status, intensity, duration_sec);
        }
    }

    /// Returns the active status effect (`0 = None`, `1 = OnFire`, `2 = Electrocuted`)
    /// for `ragdoll_id`, or `u32::MAX` if `ragdoll_id` is out of range.
    pub fn get_ragdoll_status(&self, ragdoll_id: usize) -> u32 {
        self.world
            .ragdoll_status(ragdoll_id)
            .map(|(s, _)| s as u32)
            .unwrap_or(u32::MAX)
    }

    /// Sets the horizontal desired locomotion velocity `(vx, vz)` in m/s (clamped `<= 3.5` m/s)
    /// for `ragdoll_id`. Returns `false` if `ragdoll_id` is out of range.
    pub fn set_ragdoll_desired_velocity(&mut self, ragdoll_id: usize, vx: f32, vz: f32) -> bool {
        self.world.set_ragdoll_desired_velocity(ragdoll_id, vx, vz)
    }

    /// Sets the horizontal desired locomotion velocity `(vx, vz)` in m/s (clamped `<= 3.5` m/s)
    /// for all ragdolls in the world.
    pub fn set_all_ragdolls_desired_velocity(&mut self, vx: f32, vz: f32) {
        self.world.set_all_ragdolls_desired_velocity(vx, vz);
    }
}
