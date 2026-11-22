//! Top-level simulation world, sub-stepped XPBD loop, raycasting, and grab constraint.

use std::collections::HashSet;

use glam::{Quat, Vec3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::balance::{update_ragdoll_balance, BehaviorState, RecoveryStage};
use crate::body::{ColliderShape, RigidBody};
use crate::collision::{
    body_world_aabb, detect_contacts, solve_contacts_position_ordered,
    solve_contacts_velocity_ordered, Contact, Heightfield, StaticBox,
};
use crate::joint::Joint;
use crate::reflex::{
    apply_procedural_reflexes, HeldObject, IncomingThreat, StatusEffect, WoundRegion,
};
use crate::retarget::{convert_angular_vec_to_target, CoordinateSpace};
use crate::skeleton::{
    spawn_humanoid_with_profile, update_humanoid_profile, BodyProfile, BoneId, JointId,
    RagdollInstance, RagdollLod,
};

/// Ballistic weapon presets supported by [`ArsaWorld::fire_weapon`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaponKind {
    Pistol = 0,
    Shotgun = 1,
    Rifle = 2,
}

/// Summary of all ray hits and ragdoll wound metadata produced by [`ArsaWorld::fire_weapon`].
#[derive(Debug, Clone)]
pub struct WeaponShotResult {
    pub hits: Vec<RaycastHit>,
    pub hit_ragdoll_id: Option<usize>,
    pub hit_region: WoundRegion,
}

/// Interactive mouse/pointer spring-damper grab constraint attached to a body point.
#[derive(Debug, Clone)]
pub struct GrabConstraint {
    pub body_idx: usize,
    pub local_point: Vec3,
    pub target_world: Vec3,
    pub stiffness: f32,
    pub damping: f32,
}

/// Pendulum / cable distance tether constraint keeping a body within `length` of `anchor_world`.
#[derive(Debug, Clone)]
pub struct TetherConstraint {
    pub anchor_world: Vec3,
    pub body_idx: usize,
    pub length: f32,
}

/// Result of a raycast query against dynamic bodies in `ArsaWorld`.
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    pub body_idx: usize,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub local_point: Vec3,
}

/// Active fuse timer for a thrown/placed dynamite bundle.
#[derive(Debug, Clone)]
pub struct DynamiteTimer {
    pub body_idx: usize,
    pub fuse_remaining: f32,
    pub radius: f32,
    pub impulse: f32,
    pub damage: f32,
    pub detonated: bool,
}

/// Recorded radial explosion event to be consumed by the renderer/frontend.
#[derive(Debug, Clone, Copy)]
pub struct ExplosionEvent {
    pub position: Vec3,
    pub radius: f32,
}

/// Host engine ground-height query callback.
///
/// Given `(user_data, x, z, max_y)`, writes surface height into `out_y` and upward unit normal
/// `[nx, ny, nz]` into `out_normal`, returning `true` if a surface exists at `<= max_y`.
pub type GroundHeightFn = extern "C" fn(
    user_data: *mut std::ffi::c_void,
    x: f32,
    z: f32,
    max_y: f32,
    out_y: *mut f32,
    out_normal: *mut [f32; 3],
) -> bool;

/// Host engine static-world raycast occlusion callback.
///
/// Given `(user_data, origin, dir, max_dist)`, writes hit distance into `out_dist` and hit normal
/// into `out_normal`, returning `true` if the ray hit static host geometry closer than `max_dist`.
pub type RaycastWorldFn = extern "C" fn(
    user_data: *mut std::ffi::c_void,
    origin: *const [f32; 3],
    dir: *const [f32; 3],
    max_dist: f32,
    out_dist: *mut f32,
    out_normal: *mut [f32; 3],
) -> bool;

/// Optional host engine world-query callbacks for terrain height and raycast occlusion.
#[derive(Debug, Clone, Copy)]
pub struct WorldQueryHooks {
    pub user_data: *mut std::ffi::c_void,
    pub ground_height: Option<GroundHeightFn>,
    pub raycast: Option<RaycastWorldFn>,
}

unsafe impl Send for WorldQueryHooks {}
unsafe impl Sync for WorldQueryHooks {}

/// Main ARSA physics simulation world.
#[derive(Debug, Clone)]
pub struct ArsaWorld {
    pub gravity: Vec3,
    pub substeps: usize,
    pub solver_iterations: usize,
    pub bodies: Vec<RigidBody>,
    pub joints: Vec<Joint>,
    pub tethers: Vec<TetherConstraint>,
    pub static_boxes: Vec<StaticBox>,
    pub heightfield: Option<Heightfield>,
    pub world_query_hooks: Option<WorldQueryHooks>,
    pub ragdolls: Vec<RagdollInstance>,
    pub grab: Option<GrabConstraint>,
    pub dynamites: Vec<DynamiteTimer>,
    pub pending_explosions: Vec<ExplosionEvent>,
    pub earthquake_magnitude: f32,
    earthquake_time: f32,
    pub lod_focus: Option<Vec3>,
    pub lod_reduced_dist: f32,
    pub lod_sleep_dist: f32,
    connected_pairs: HashSet<(usize, usize)>,
    held_skip_pairs_buf: Vec<(usize, usize)>,
    contacts_buf: Vec<Contact>,
    foot_contacts_buf: Vec<(bool, bool)>,
    pre_solve_vels_buf: Vec<Vec3>,
    pre_solve_ang_vels_buf: Vec<Vec3>,
    body_aabbs_buf: Vec<(Vec3, Vec3)>,
}

impl Default for ArsaWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl ArsaWorld {
    /// Creates an empty `ArsaWorld` with standard Earth gravity (`-9.81` m/s^2),
    /// 10 substeps, and 4 solver iterations.
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            substeps: 10,
            solver_iterations: 4,
            bodies: Vec::new(),
            joints: Vec::new(),
            tethers: Vec::new(),
            static_boxes: Vec::new(),
            heightfield: None,
            world_query_hooks: None,
            ragdolls: Vec::new(),
            grab: None,
            dynamites: Vec::new(),
            pending_explosions: Vec::new(),
            earthquake_magnitude: 0.0,
            earthquake_time: 0.0,
            lod_focus: None,
            lod_reduced_dist: 15.0,
            lod_sleep_dist: 35.0,
            connected_pairs: HashSet::new(),
            held_skip_pairs_buf: Vec::new(),
            contacts_buf: Vec::new(),
            foot_contacts_buf: Vec::new(),
            pre_solve_vels_buf: Vec::new(),
            pre_solve_ang_vels_buf: Vec::new(),
            body_aabbs_buf: Vec::new(),
        }
    }

    /// Clears all bodies, joints, tethers, static obstacles, ragdolls, dynamites, and active grabs.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.joints.clear();
        self.tethers.clear();
        self.static_boxes.clear();
        self.heightfield = None;
        self.ragdolls.clear();
        self.grab = None;
        self.dynamites.clear();
        self.pending_explosions.clear();
        self.earthquake_time = 0.0;
        self.connected_pairs.clear();
        self.held_skip_pairs_buf.clear();
        self.contacts_buf.clear();
        self.foot_contacts_buf.clear();
        self.pre_solve_vels_buf.clear();
        self.pre_solve_ang_vels_buf.clear();
        self.body_aabbs_buf.clear();
    }

    /// Adds a distance tether constraint anchoring `body_idx` within `length` meters of `anchor_world`.
    pub fn add_tether(&mut self, anchor_world: Vec3, body_idx: usize, length: f32) -> usize {
        let safe_length = if length.is_finite() {
            length.max(0.0)
        } else {
            0.0
        };
        if let Some(body) = self.bodies.get_mut(body_idx) {
            if body.inv_mass > 0.0 {
                let delta = body.position - anchor_world;
                let dist = delta.length();
                if dist > safe_length && dist > 1e-6 {
                    let clamped_pos = anchor_world + (delta / dist) * safe_length;
                    body.position = clamped_pos;
                    body.prev_position = clamped_pos;
                }
            }
        }
        let idx = self.tethers.len();
        self.tethers.push(TetherConstraint {
            anchor_world,
            body_idx,
            length: safe_length,
        });
        idx
    }

    /// Adds a static axis-aligned box obstacle and returns its index.
    pub fn add_static_box(
        &mut self,
        center: Vec3,
        half_extents: Vec3,
        friction: f32,
        restitution: f32,
    ) -> usize {
        self.add_static_oriented_box(center, half_extents, Quat::IDENTITY, friction, restitution)
    }

    /// Adds a static oriented box obstacle and returns its index.
    pub fn add_static_oriented_box(
        &mut self,
        center: Vec3,
        half_extents: Vec3,
        orientation: Quat,
        friction: f32,
        restitution: f32,
    ) -> usize {
        let norm_orientation = if orientation.is_finite() && orientation.length_squared() >= 1e-12 {
            orientation.normalize()
        } else {
            Quat::IDENTITY
        };
        let idx = self.static_boxes.len();
        self.static_boxes.push(StaticBox {
            center,
            half_extents,
            orientation: norm_orientation,
            friction,
            restitution,
        });
        idx
    }

    /// Sets or clears the world's heightfield terrain collider.
    pub fn set_heightfield(&mut self, hf: Option<Heightfield>) {
        self.heightfield = hf;
    }

    /// Sets or clears host engine world-query callbacks (terrain height & raycast occlusion).
    pub fn set_world_query_callbacks(&mut self, hooks: Option<WorldQueryHooks>) {
        self.world_query_hooks = hooks;
    }

    /// Samples the highest static, heightfield, host-terrain, or heavy-dynamic support surface `y <= max_y` at `(x, z)`.
    pub fn sample_support_height(&self, x: f32, z: f32, max_y: f32) -> f32 {
        sample_support_height_parts(
            &self.static_boxes,
            &self.bodies,
            self.heightfield.as_ref(),
            self.world_query_hooks.as_ref(),
            x,
            z,
            max_y,
        )
    }

    /// Spawns a 17-bone humanoid ragdoll at `origin` with the default profile and returns its ragdoll ID.
    pub fn spawn_ragdoll(&mut self, origin: Vec3) -> usize {
        self.spawn_ragdoll_with_profile(origin, BodyProfile::default())
    }

    /// Spawns a 17-bone humanoid ragdoll at `origin` scaled to `profile` and returns its ragdoll ID.
    pub fn spawn_ragdoll_with_profile(&mut self, origin: Vec3, profile: BodyProfile) -> usize {
        let id = self.ragdolls.len();
        let start_joint = self.joints.len();
        let instance =
            spawn_humanoid_with_profile(&mut self.bodies, &mut self.joints, id, origin, profile);
        for j in &self.joints[start_joint..] {
            let pair = canonical_pair(j.parent, j.child);
            self.connected_pairs.insert(pair);
        }
        self.ragdolls.push(instance);
        id
    }

    /// Enables or disables active balance & muscle motors for a specific ragdoll.
    pub fn set_ragdoll_active(&mut self, ragdoll_id: usize, active: bool) {
        if let Some(r) = self.ragdolls.get_mut(ragdoll_id) {
            r.active_enabled = active;
            if !active {
                r.reflex.clear_status();
            }
        }
    }

    /// Enables or disables active balance & muscle motors for all ragdolls in the world.
    pub fn set_all_ragdolls_active(&mut self, active: bool) {
        for r in &mut self.ragdolls {
            r.active_enabled = active;
            if !active {
                r.reflex.clear_status();
            }
        }
    }

    /// Applies a procedural [`StatusEffect`] (`OnFire`, `Electrocuted`, or `None`) to `ragdoll_id`
    /// with clamped `intensity` (`0.0..=1.0`) and `duration_sec` (`>= 0.0`).
    /// Returns `false` if `ragdoll_id` is out of range.
    pub fn set_ragdoll_status(
        &mut self,
        ragdoll_id: usize,
        effect: StatusEffect,
        intensity: f32,
        duration_sec: f32,
    ) -> bool {
        let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) else {
            return false;
        };
        let clamped_intensity = if intensity.is_finite() {
            intensity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let clamped_duration = if duration_sec.is_finite() {
            duration_sec.max(0.0)
        } else {
            0.0
        };

        if effect == StatusEffect::None
            || clamped_intensity <= 0.0
            || clamped_duration <= 0.0
            || !ragdoll.active_enabled
            || ragdoll.reflex.unconscious
        {
            ragdoll.reflex.clear_status();
            return true;
        }

        if ragdoll.reflex.status_effect != effect {
            ragdoll.reflex.status_phase = 0.0;
        }
        ragdoll.reflex.status_effect = effect;
        ragdoll.reflex.status_intensity = clamped_intensity;
        ragdoll.reflex.status_timer = clamped_duration;

        if (effect == StatusEffect::Electrocuted && clamped_intensity > 0.25)
            || (effect == StatusEffect::OnFire && clamped_intensity > 0.7)
        {
            ragdoll.reflex.left_held = None;
            ragdoll.reflex.right_held = None;
            ragdoll.reflex.left_grab = None;
            ragdoll.reflex.right_grab = None;
        }

        self.wake_ragdoll(ragdoll_id, clamped_duration.max(2.5));
        true
    }

    /// Applies a procedural [`StatusEffect`] to all ragdolls in the world.
    pub fn set_all_ragdolls_status(
        &mut self,
        effect: StatusEffect,
        intensity: f32,
        duration_sec: f32,
    ) {
        for ragdoll_id in 0..self.ragdolls.len() {
            self.set_ragdoll_status(ragdoll_id, effect, intensity, duration_sec);
        }
    }

    /// Returns the active `(StatusEffect, intensity)` for `ragdoll_id`, or `None` if out of range.
    pub fn ragdoll_status(&self, ragdoll_id: usize) -> Option<(StatusEffect, f32)> {
        self.ragdolls
            .get(ragdoll_id)
            .map(|r| (r.reflex.status_effect, r.reflex.status_intensity))
    }

    /// Enables or disables automatic procedural get-up from `Downed` for all ragdolls in the world.
    pub fn set_all_ragdolls_auto_get_up(&mut self, enabled: bool) {
        for r in &mut self.ragdolls {
            r.auto_get_up = enabled;
        }
    }

    /// Sets the procedural stagger/intoxication level (`0.0..=1.0`) for all ragdolls in the world.
    pub fn set_all_ragdolls_intoxication(&mut self, level: f32) {
        let clamped = if level.is_finite() {
            level.clamp(0.0, 1.0)
        } else {
            0.0
        };
        for r in &mut self.ragdolls {
            r.balance.intoxication = clamped;
        }
    }

    /// Sets an external target pose (16 parent-relative joint orientations in `JointId` order)
    /// blended onto the ragdoll's active motor targets with weight `blend_weight` (`0.0..=1.0`).
    /// Returns `false` if `ragdoll_id` is out of range or any input quaternion / weight is non-finite or zero.
    pub fn set_ragdoll_target_pose(
        &mut self,
        ragdoll_id: usize,
        targets: &[Quat; 16],
        blend_weight: f32,
    ) -> bool {
        if !blend_weight.is_finite() {
            return false;
        }
        let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) else {
            return false;
        };
        let mut normalized = [Quat::IDENTITY; 16];
        for (dst, &q) in normalized.iter_mut().zip(targets.iter()) {
            if !q.is_finite() || q.length_squared() < 1e-12 {
                return false;
            }
            *dst = q.normalize();
        }
        ragdoll.external_pose = Some((normalized, blend_weight.clamp(0.0, 1.0)));
        self.wake_ragdoll(ragdoll_id, 2.5);
        true
    }

    /// Configures the spatial crowd-scaling LOD focus point and distance thresholds (in meters).
    /// Passing `None` clears the focus and wakes all sleeping ragdolls back to `RagdollLod::Full`.
    /// Ignores non-finite `focus` vectors; clamps `reduced_dist >= 1.0` and `sleep_dist >= reduced_dist + 1.0`.
    pub fn set_lod_focus(&mut self, focus: Option<Vec3>, reduced_dist: f32, sleep_dist: f32) {
        if let Some(f) = focus {
            if !f.is_finite() {
                return;
            }
        }
        let clamped_reduced = if reduced_dist.is_finite() {
            reduced_dist.max(1.0)
        } else {
            1.0
        };
        let clamped_sleep = if sleep_dist.is_finite() {
            sleep_dist.max(clamped_reduced + 1.0)
        } else {
            clamped_reduced + 1.0
        };
        self.lod_focus = focus;
        self.lod_reduced_dist = clamped_reduced;
        self.lod_sleep_dist = clamped_sleep;

        if focus.is_none() {
            for r in &mut self.ragdolls {
                if r.lod != RagdollLod::Full {
                    r.lod = RagdollLod::Full;
                    for &b_idx in &r.body_indices {
                        if let Some(b) = self.bodies.get_mut(b_idx) {
                            b.wake_up();
                        }
                    }
                }
            }
        }
    }

    /// Returns the current [`RagdollLod`] tier for `ragdoll_id`, or `None` if out of range.
    pub fn ragdoll_lod(&self, ragdoll_id: usize) -> Option<RagdollLod> {
        self.ragdolls.get(ragdoll_id).map(|r| r.lod)
    }

    /// Forces `ragdoll_id` to `RagdollLod::Full` for at least `duration` seconds and wakes all 17 bones.
    pub fn wake_ragdoll(&mut self, ragdoll_id: usize, duration: f32) {
        let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) else {
            return;
        };
        let safe_duration = if duration.is_finite() {
            duration.max(0.0)
        } else {
            0.0
        };
        ragdoll.lod_awake_override = ragdoll.lod_awake_override.max(safe_duration);
        ragdoll.lod = RagdollLod::Full;
        for &b_idx in &ragdoll.body_indices {
            if let Some(b) = self.bodies.get_mut(b_idx) {
                b.wake_up();
            }
        }
    }

    /// Clears any external animation target pose on `ragdoll_id`.
    /// Returns `false` if `ragdoll_id` is out of range.
    pub fn clear_ragdoll_target_pose(&mut self, ragdoll_id: usize) -> bool {
        let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) else {
            return false;
        };
        ragdoll.external_pose = None;
        true
    }

    /// Sets the earthquake seismic magnitude (`0.0..=9.0`, `0.0` disables).
    pub fn set_earthquake(&mut self, magnitude: f32) {
        self.earthquake_magnitude = if magnitude.is_finite() {
            magnitude.clamp(0.0, 9.0)
        } else {
            0.0
        };
        if self.earthquake_magnitude > 0.1 {
            for b in &mut self.bodies {
                if b.is_sleeping {
                    b.wake_up();
                }
            }
        }
    }

    /// Updates the anthropometric body profile of all ragdolls in-place without resetting the scene.
    pub fn update_all_ragdolls_profile(&mut self, profile: BodyProfile) {
        for ragdoll in &mut self.ragdolls {
            update_humanoid_profile(ragdoll, &mut self.bodies, &mut self.joints, profile);
        }
    }

    /// Spawns a dynamic obstacle box resting cleanly on `y = 0` at `(x, hy, z)` with zero initial velocity.
    pub fn place_obstacle_box(
        &mut self,
        x: f32,
        z: f32,
        hx: f32,
        hy: f32,
        hz: f32,
        mass: f32,
    ) -> usize {
        self.spawn_box(Vec3::new(x, hy, z), Vec3::new(hx, hy, hz), mass, Vec3::ZERO)
    }

    /// Spawns a dynamic box rigid body and returns its body index.
    pub fn spawn_box(&mut self, pos: Vec3, half_extents: Vec3, mass: f32, vel: Vec3) -> usize {
        let safe_pos = if pos.is_finite() { pos } else { Vec3::ZERO };
        let safe_half = Vec3::new(
            if half_extents.x.is_finite() && half_extents.x > 0.0 {
                half_extents.x
            } else {
                0.25
            },
            if half_extents.y.is_finite() && half_extents.y > 0.0 {
                half_extents.y
            } else {
                0.25
            },
            if half_extents.z.is_finite() && half_extents.z > 0.0 {
                half_extents.z
            } else {
                0.25
            },
        );
        let safe_mass = if mass.is_finite() && mass > 0.0 {
            mass
        } else {
            0.0
        };
        let safe_vel = if vel.is_finite() { vel } else { Vec3::ZERO };

        let idx = self.bodies.len();
        let mut rb = RigidBody::new(
            safe_pos,
            Quat::IDENTITY,
            safe_mass,
            ColliderShape::Box {
                half_extents: safe_half,
            },
        );
        rb.linear_velocity = safe_vel;
        self.bodies.push(rb);
        idx
    }

    /// Spawns a dynamic sphere rigid body and returns its body index.
    pub fn spawn_sphere(&mut self, pos: Vec3, radius: f32, mass: f32, vel: Vec3) -> usize {
        let safe_pos = if pos.is_finite() { pos } else { Vec3::ZERO };
        let safe_radius = if radius.is_finite() && radius > 0.0 {
            radius
        } else {
            0.25
        };
        let safe_mass = if mass.is_finite() && mass > 0.0 {
            mass
        } else {
            0.0
        };
        let safe_vel = if vel.is_finite() { vel } else { Vec3::ZERO };

        let idx = self.bodies.len();
        let mut rb = RigidBody::new(
            safe_pos,
            Quat::IDENTITY,
            safe_mass,
            ColliderShape::Sphere {
                radius: safe_radius,
            },
        );
        rb.linear_velocity = safe_vel;
        self.bodies.push(rb);
        idx
    }

    /// Spawns a lit dynamite capsule (`bone_id = Some(250)`) with a countdown fuse and returns its body index.
    pub fn spawn_dynamite(&mut self, pos: Vec3, vel: Vec3, fuse_secs: f32) -> usize {
        let idx = self.bodies.len();
        let mut rb = RigidBody::new(
            pos,
            Quat::IDENTITY,
            1.5,
            ColliderShape::Capsule {
                radius: 0.055,
                half_length: 0.12,
            },
        );
        rb.linear_velocity = vel;
        rb.bone_id = Some(250);
        self.bodies.push(rb);

        let safe_fuse = if fuse_secs.is_finite() && fuse_secs > 0.0 {
            fuse_secs
        } else {
            1.8
        };
        self.dynamites.push(DynamiteTimer {
            body_idx: idx,
            fuse_remaining: safe_fuse,
            radius: 4.2,
            impulse: 72.0,
            damage: 75.0,
            detonated: false,
        });
        idx
    }

    /// Drains all pending radial explosion events into `out` as `[x, y, z, radius, ...]`.
    pub fn take_explosions(&mut self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.pending_explosions.len() * 4);
        for ev in self.pending_explosions.drain(..) {
            out.push(ev.position.x);
            out.push(ev.position.y);
            out.push(ev.position.z);
            out.push(ev.radius);
        }
    }

    /// Applies a radial explosion centered at `center` with `radius`, `max_impulse`, and `max_damage`,
    /// returning the count of affected dynamic bodies.
    pub fn apply_explosion(
        &mut self,
        center: Vec3,
        radius: f32,
        max_impulse: f32,
        max_damage: f32,
    ) -> usize {
        if !center.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
            || !max_impulse.is_finite()
            || !max_damage.is_finite()
        {
            return 0;
        }

        const MAX_BLAST_DV: f32 = 24.0;
        let mut affected = 0usize;
        let track_damage = max_damage > 0.0;
        // Track the strongest blast hit per ragdoll so 17 bones don't multiply health damage 17x.
        let mut ragdoll_best_hit: Vec<Option<(BoneId, Vec3, f32)>> = if track_damage {
            vec![None; self.ragdolls.len()]
        } else {
            Vec::new()
        };
        let mut ragdoll_bone_falloffs: Vec<[f32; 17]> = if track_damage {
            vec![[0.0; 17]; self.ragdolls.len()]
        } else {
            Vec::new()
        };
        let mut ragdoll_max_falloffs = vec![0.0_f32; self.ragdolls.len()];

        for b in &mut self.bodies {
            if b.inv_mass == 0.0 {
                continue;
            }
            let offset = b.position - center;
            let dist = offset.length();
            if dist >= radius {
                continue;
            }
            let falloff = (1.0 - dist / radius).clamp(0.0, 1.0).powi(2);
            if falloff <= 0.0 {
                continue;
            }

            let blast_dir = (offset.normalize_or_zero() + Vec3::Y * 0.45).normalize_or_zero();
            let mass_scale = (b.mass / 6.0).sqrt().clamp(0.25, 2.5);
            let impulse_mag = max_impulse * falloff * mass_scale;
            let dv = (impulse_mag * b.inv_mass).min(MAX_BLAST_DV);
            b.wake_up();
            b.linear_velocity += blast_dir * dv;
            affected += 1;

            if let (Some(r_id), Some(bone)) = (b.ragdoll_id, b.bone_id.and_then(BoneId::from_u8)) {
                if r_id < self.ragdolls.len() {
                    ragdoll_max_falloffs[r_id] = ragdoll_max_falloffs[r_id].max(falloff);
                    if track_damage {
                        ragdoll_bone_falloffs[r_id][bone as usize] =
                            ragdoll_bone_falloffs[r_id][bone as usize].max(falloff);
                        let local_pt =
                            b.orientation.conjugate() * (-offset.normalize_or_zero() * 0.05);
                        let replace = match ragdoll_best_hit[r_id] {
                            Some((_, _, best_f)) => falloff > best_f,
                            None => true,
                        };
                        if replace {
                            ragdoll_best_hit[r_id] = Some((bone, local_pt, falloff));
                        }
                    }
                }
            }
        }

        if track_damage {
            for (r_id, hit_opt) in ragdoll_best_hit.into_iter().enumerate() {
                if let Some((best_bone, local_pt, falloff)) = hit_opt {
                    let ragdoll = &mut self.ragdolls[r_id];
                    let best_idx = best_bone as usize;
                    for (bone_idx, &f) in ragdoll_bone_falloffs[r_id].iter().enumerate() {
                        if bone_idx != best_idx && f > 0.0 {
                            ragdoll.reflex.bone_damage[bone_idx] =
                                (ragdoll.reflex.bone_damage[bone_idx] + f * 0.45).min(1.0);
                        }
                    }
                    let damage = falloff * max_damage;
                    let severity = (falloff * (max_damage / 75.0)).clamp(0.0, 1.0);
                    ragdoll.register_hit(best_bone, local_pt, severity, damage);
                }
            }
        }

        for (r_id, &falloff) in ragdoll_max_falloffs.iter().enumerate() {
            if falloff > 0.0 {
                self.wake_ragdoll(r_id, 2.5);
                let ragdoll = &mut self.ragdolls[r_id];
                if !ragdoll.reflex.unconscious {
                    ragdoll.balance.concussion_timer =
                        (ragdoll.balance.concussion_timer + 3.5 * falloff).min(5.0);
                }
            }
        }

        affected
    }

    /// Advances the simulation by `dt` seconds using sub-stepped XPBD.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let clamped_dt = dt.min(0.05);

        // Update dynamite fuse timers and trigger any detonations before substeps.
        let mut detonations: Vec<(Vec3, f32, f32, f32)> = Vec::new();
        for d in &mut self.dynamites {
            if d.detonated {
                continue;
            }
            d.fuse_remaining -= clamped_dt;
            if d.fuse_remaining <= 0.0 {
                d.detonated = true;
                if let Some(body) = self.bodies.get_mut(d.body_idx) {
                    let pos = body.position;
                    let buried = Vec3::new(0.0, -50.0, 0.0);
                    body.position = buried;
                    body.prev_position = buried;
                    body.mass = 0.0;
                    body.inv_mass = 0.0;
                    body.local_inertia = Vec3::ZERO;
                    body.inv_local_inertia = Vec3::ZERO;
                    body.update_world_inertia();
                    body.linear_velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                    body.collision_group = 0;
                    body.collision_mask = 0;
                    if let Some(grab) = &self.grab {
                        if grab.body_idx == d.body_idx {
                            self.grab = None;
                        }
                    }
                    self.pending_explosions.push(ExplosionEvent {
                        position: pos,
                        radius: d.radius,
                    });
                    detonations.push((pos, d.radius, d.impulse, d.damage));
                }
            }
        }
        for (pos, radius, impulse, damage) in detonations {
            self.apply_explosion(pos, radius, impulse, damage);
        }

        // Update crowd-scaling LOD tiers before substeps when an LOD focus is active.
        if let Some(focus) = self.lod_focus {
            for ragdoll in &mut self.ragdolls {
                ragdoll.lod_awake_override = (ragdoll.lod_awake_override - clamped_dt).max(0.0);
                let pelvis_pos =
                    self.bodies[ragdoll.body_indices[BoneId::Pelvis as usize]].position;
                let dx = pelvis_pos.x - focus.x;
                let dz = pelvis_pos.z - focus.z;
                let dist = (dx * dx + dz * dz).sqrt();

                let next_lod = if self.earthquake_magnitude >= 2.0
                    || ragdoll.lod_awake_override > 0.0
                    || matches!(
                        ragdoll.balance.state,
                        BehaviorState::Falling
                            | BehaviorState::Stumbling
                            | BehaviorState::Recovering
                    ) {
                    RagdollLod::Full
                } else if dist >= self.lod_sleep_dist && ragdoll.balance.com_vel.length() < 0.25 {
                    RagdollLod::Sleep
                } else if dist >= self.lod_reduced_dist {
                    RagdollLod::Reduced
                } else {
                    RagdollLod::Full
                };

                if ragdoll.lod != RagdollLod::Sleep && next_lod == RagdollLod::Sleep {
                    for &b_idx in &ragdoll.body_indices {
                        if let Some(b) = self.bodies.get_mut(b_idx) {
                            b.is_sleeping = true;
                            b.sleep_timer = 0.0;
                            b.linear_velocity = Vec3::ZERO;
                            b.angular_velocity = Vec3::ZERO;
                        }
                    }
                } else if ragdoll.lod == RagdollLod::Sleep && next_lod != RagdollLod::Sleep {
                    for &b_idx in &ragdoll.body_indices {
                        if let Some(b) = self.bodies.get_mut(b_idx) {
                            b.wake_up();
                        }
                    }
                }
                ragdoll.lod = next_lod;
            }
        }

        let substeps = self.substeps.max(1);
        let sub_dt = clamped_dt / (substeps as f32);
        let inv_dt = 1.0 / sub_dt;
        let gravity_mag = self.gravity.length();

        if self.earthquake_magnitude > 0.1 {
            for b in &mut self.bodies {
                if b.is_sleeping {
                    let in_sleep_lod = b
                        .ragdoll_id
                        .and_then(|id| self.ragdolls.get(id))
                        .is_some_and(|r| r.lod == RagdollLod::Sleep);
                    if !in_sleep_lod {
                        b.wake_up();
                    }
                }
            }
        }
        if let Some(grab) = &self.grab {
            let grab_ragdoll_id = self.bodies.get(grab.body_idx).and_then(|b| b.ragdoll_id);
            if let Some(r_id) = grab_ragdoll_id {
                self.wake_ragdoll(r_id, 2.5);
            } else if let Some(body) = self.bodies.get_mut(grab.body_idx) {
                if body.is_sleeping {
                    body.wake_up();
                }
            }
        }

        self.foot_contacts_buf.clear();
        self.foot_contacts_buf.extend(
            self.ragdolls
                .iter()
                .map(|r| (r.balance.left_foot_grounded, r.balance.right_foot_grounded)),
        );

        for substep in 0..substeps {
            // 1. Save previous state & integrate external forces
            let gravity = self.gravity;
            #[cfg(feature = "parallel")]
            if self.bodies.len() >= 64 {
                self.bodies
                    .par_iter_mut()
                    .for_each(|b| integrate_body_step(b, gravity, sub_dt));
            } else {
                for b in &mut self.bodies {
                    integrate_body_step(b, gravity, sub_dt);
                }
            }
            #[cfg(not(feature = "parallel"))]
            for b in &mut self.bodies {
                b.prev_position = b.position;
                b.prev_orientation = b.orientation;
                if b.inv_mass > 0.0 && !b.is_sleeping {
                    b.linear_velocity += gravity * sub_dt;
                }
            }

            // Apply seismic ground shaking to near-ground dynamic bodies
            if self.earthquake_magnitude > 0.1 {
                self.earthquake_time += sub_dt;
                let t = self.earthquake_time;
                let norm = ((self.earthquake_magnitude - 2.0) / 7.0).clamp(0.0, 1.0);
                let amp = norm * norm * 22.0;
                let ax = amp
                    * (0.55 * (t * 3.8).sin()
                        + 0.30 * (t * 9.7 + 0.7).cos()
                        + 0.25 * (t * 19.3).sin());
                let az = amp
                    * (0.55 * (t * 3.1 + 1.4).cos()
                        + 0.30 * (t * 8.3).sin()
                        + 0.25 * (t * 17.1 + 2.1).cos());
                let ay = amp * 0.28 * (t * 11.5).sin();
                let base_dv = Vec3::new(ax, ay, az) * sub_dt;
                for b in &mut self.bodies {
                    if b.inv_mass > 0.0 && !b.is_sleeping && b.position.y < 2.2 {
                        let horiz_scale = if b.position.y < 0.6 { 1.25 } else { 0.65 };
                        let vert_scale = if b.position.y < 0.6 { 1.0 } else { 0.65 };
                        b.linear_velocity += Vec3::new(
                            base_dv.x * horiz_scale,
                            base_dv.y * vert_scale,
                            base_dv.z * horiz_scale,
                        );
                    }
                }
            }

            // Apply interactive grab constraint spring-damper force
            if let Some(grab) = &self.grab {
                if let Some(body) = self.bodies.get_mut(grab.body_idx) {
                    if body.inv_mass > 0.0 {
                        let world_pt = body.world_point(grab.local_point);
                        let vel_pt = body.velocity_at_point(world_pt);
                        let mut force =
                            (grab.target_world - world_pt) * grab.stiffness - vel_pt * grab.damping;
                        let max_force = body.mass * 250.0;
                        let f_len = force.length();
                        if f_len > max_force && f_len > 1e-6 {
                            force *= max_force / f_len;
                        }
                        let impulse = force * sub_dt;
                        let r = world_pt - body.position;
                        body.linear_velocity += impulse * body.inv_mass;
                        let dw = body.inv_inertia_world_mul(r.cross(impulse));
                        body.angular_velocity += dw;
                    }
                }
            }

            // 1b. Update active balance controllers, procedural reflexes & apply PD muscle motors
            for (r_idx, ragdoll) in self.ragdolls.iter_mut().enumerate() {
                if ragdoll.lod == RagdollLod::Sleep {
                    continue;
                }
                if ragdoll.lod == RagdollLod::Reduced && substep % 2 != 0 {
                    continue;
                }
                let ctrl_dt = if ragdoll.lod == RagdollLod::Reduced {
                    sub_dt * 2.0
                } else {
                    sub_dt
                };
                step_ragdoll_controllers(
                    ragdoll,
                    &mut self.bodies,
                    &mut self.joints,
                    &self.static_boxes,
                    self.heightfield.as_ref(),
                    self.world_query_hooks.as_ref(),
                    self.foot_contacts_buf[r_idx],
                    gravity_mag,
                    self.earthquake_magnitude,
                    ctrl_dt,
                );
            }
            let j_len = self.joints.len();
            let rev_sub = substep % 2 == 1;
            for idx in 0..j_len {
                let j = &self.joints[if rev_sub { j_len - 1 - idx } else { idx }];
                j.apply_muscle_motor(&mut self.bodies, sub_dt);
            }

            self.pre_solve_vels_buf.clear();
            self.pre_solve_vels_buf
                .extend(self.bodies.iter().map(|b| b.linear_velocity));
            self.pre_solve_ang_vels_buf.clear();
            self.pre_solve_ang_vels_buf
                .extend(self.bodies.iter().map(|b| b.angular_velocity));

            // Predict positions and orientations
            for b in &mut self.bodies {
                if b.inv_mass == 0.0 || b.is_sleeping {
                    continue;
                }
                b.position += b.linear_velocity * sub_dt;
                let w = b.angular_velocity;
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * b.orientation;
                b.orientation = (b.orientation + dq * (0.5 * sub_dt)).normalize();
                b.update_world_inertia();
            }

            // 2. Collision detection
            self.held_skip_pairs_buf.clear();
            for r in &self.ragdolls {
                if let Some(held) = r.reflex.left_held {
                    self.held_skip_pairs_buf.push(canonical_pair(
                        r.body_indices[BoneId::LeftHand as usize],
                        held.body_idx,
                    ));
                    self.held_skip_pairs_buf.push(canonical_pair(
                        r.body_indices[BoneId::LeftForearm as usize],
                        held.body_idx,
                    ));
                }
                if let Some(held) = r.reflex.right_held {
                    self.held_skip_pairs_buf.push(canonical_pair(
                        r.body_indices[BoneId::RightHand as usize],
                        held.body_idx,
                    ));
                    self.held_skip_pairs_buf.push(canonical_pair(
                        r.body_indices[BoneId::RightForearm as usize],
                        held.body_idx,
                    ));
                }
            }
            detect_contacts(
                &self.bodies,
                &self.static_boxes,
                self.heightfield.as_ref(),
                self.world_query_hooks.as_ref(),
                &self.connected_pairs,
                &self.held_skip_pairs_buf,
                &mut self.body_aabbs_buf,
                &mut self.contacts_buf,
            );

            // Wake sleeping bodies contacted by an awake body
            let mut wake_ragdoll_ids: Vec<usize> = Vec::new();
            for c in &self.contacts_buf {
                if let Some(b_idx) = c.body_b {
                    let a_sleep = self.bodies[c.body_a].is_sleeping;
                    let b_sleep = self.bodies[b_idx].is_sleeping;
                    if a_sleep != b_sleep {
                        if a_sleep {
                            self.bodies[c.body_a].wake_up();
                            if let Some(r_id) = self.bodies[c.body_a].ragdoll_id {
                                wake_ragdoll_ids.push(r_id);
                            }
                        }
                        if b_sleep {
                            self.bodies[b_idx].wake_up();
                            if let Some(r_id) = self.bodies[b_idx].ragdoll_id {
                                wake_ragdoll_ids.push(r_id);
                            }
                        }
                    }
                }
            }
            for r_id in wake_ragdoll_ids {
                self.wake_ragdoll(r_id, 2.5);
            }

            // Track per-ragdoll foot grounding from detected contacts & near-support fallback
            for (r_idx, ragdoll) in self.ragdolls.iter_mut().enumerate() {
                if ragdoll.lod == RagdollLod::Sleep {
                    continue;
                }
                let lf_idx = ragdoll.body_indices[BoneId::LeftFoot as usize];
                let rf_idx = ragdoll.body_indices[BoneId::RightFoot as usize];
                let mut lf_grounded = false;
                let mut rf_grounded = false;
                for c in &self.contacts_buf {
                    if (c.body_a == lf_idx && c.normal.y > 0.45)
                        || (c.body_b == Some(lf_idx) && -c.normal.y > 0.45)
                    {
                        lf_grounded = true;
                    }
                    if (c.body_a == rf_idx && c.normal.y > 0.45)
                        || (c.body_b == Some(rf_idx) && -c.normal.y > 0.45)
                    {
                        rf_grounded = true;
                    }
                }
                let s_h = ragdoll.profile.height_m / 1.78;
                if !lf_grounded {
                    lf_grounded = foot_near_support_parts(
                        &self.bodies[lf_idx],
                        s_h,
                        &self.static_boxes,
                        &self.bodies,
                        self.heightfield.as_ref(),
                        self.world_query_hooks.as_ref(),
                    );
                }
                if !rf_grounded {
                    rf_grounded = foot_near_support_parts(
                        &self.bodies[rf_idx],
                        s_h,
                        &self.static_boxes,
                        &self.bodies,
                        self.heightfield.as_ref(),
                        self.world_query_hooks.as_ref(),
                    );
                }
                self.foot_contacts_buf[r_idx] = (lf_grounded, rf_grounded);
                ragdoll.balance.left_foot_grounded = lf_grounded;
                ragdoll.balance.right_foot_grounded = rf_grounded;
            }

            // 3. XPBD positional & angular constraint iterations (symmetric sweep eliminates L/R bias)
            for iter in 0..self.solver_iterations.max(1) {
                let reverse = (iter + substep) % 2 == 1;
                let j_len = self.joints.len();
                for idx in 0..j_len {
                    let j = &self.joints[if reverse { j_len - 1 - idx } else { idx }];
                    j.solve_position(&mut self.bodies, sub_dt);
                    j.solve_limits(&mut self.bodies, sub_dt);
                }
                for tether in &self.tethers {
                    let Some(body) = self.bodies.get_mut(tether.body_idx) else {
                        continue;
                    };
                    if body.inv_mass == 0.0 {
                        continue;
                    }
                    let delta = body.position - tether.anchor_world;
                    let dist = delta.length();
                    if dist > tether.length && dist > 1e-6 {
                        let n = delta / dist;
                        let correction = -n * (dist - tether.length);
                        body.position += correction;
                    }
                }
                solve_contacts_position_ordered(&mut self.bodies, &mut self.contacts_buf, reverse);
            }

            // 4. Derive velocities from position & orientation changes
            #[cfg(feature = "parallel")]
            if self.bodies.len() >= 64 {
                let pre_vels = &self.pre_solve_vels_buf;
                let pre_ang_vels = &self.pre_solve_ang_vels_buf;
                self.bodies
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(b_idx, b)| {
                        derive_body_velocity_step(
                            b,
                            pre_vels[b_idx],
                            pre_ang_vels[b_idx],
                            inv_dt,
                            sub_dt,
                        );
                    });
            } else {
                for (b_idx, b) in self.bodies.iter_mut().enumerate() {
                    derive_body_velocity_step(
                        b,
                        self.pre_solve_vels_buf[b_idx],
                        self.pre_solve_ang_vels_buf[b_idx],
                        inv_dt,
                        sub_dt,
                    );
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                const MAX_DEPENETRATION_DV: f32 = 6.0;
                const MAX_DEPENETRATION_DW: f32 = 25.0;
                for (b_idx, b) in self.bodies.iter_mut().enumerate() {
                    if b.inv_mass == 0.0 || b.is_sleeping {
                        continue;
                    }
                    let pre_solve_vel = self.pre_solve_vels_buf[b_idx];
                    let derived_vel = (b.position - b.prev_position) * inv_dt;
                    let dv = derived_vel - pre_solve_vel;
                    let dv_len = dv.length();
                    b.linear_velocity = if dv_len > MAX_DEPENETRATION_DV {
                        pre_solve_vel + dv * (MAX_DEPENETRATION_DV / dv_len)
                    } else {
                        derived_vel
                    };

                    let pre_solve_ang_vel = self.pre_solve_ang_vels_buf[b_idx];
                    let mut dq = b.orientation * b.prev_orientation.conjugate();
                    if dq.w < 0.0 {
                        dq = -dq;
                    }
                    let derived_ang_vel = Vec3::new(dq.x, dq.y, dq.z) * (2.0 * inv_dt);
                    let dw = derived_ang_vel - pre_solve_ang_vel;
                    let dw_len = dw.length();
                    b.angular_velocity = if dw_len > MAX_DEPENETRATION_DW {
                        pre_solve_ang_vel + dw * (MAX_DEPENETRATION_DW / dw_len)
                    } else {
                        derived_ang_vel
                    };

                    // Subtle air drag for numerical cleanliness
                    let drag = (-0.15 * sub_dt).exp();
                    b.linear_velocity *= drag;
                    b.angular_velocity *= drag;
                }
            }

            // 5. Velocity pass (joint damping + dynamic friction & restitution)
            for idx in 0..j_len {
                let j = &self.joints[if rev_sub { j_len - 1 - idx } else { idx }];
                j.solve_velocity_damping(&mut self.bodies, sub_dt);
            }
            for v_iter in 0..self.solver_iterations.max(1) {
                solve_contacts_velocity_ordered(
                    &mut self.bodies,
                    &self.contacts_buf,
                    sub_dt,
                    gravity_mag,
                    (substep + v_iter) % 2 == 1,
                );
            }
        }

        // Put calm non-ragdoll, non-dynamite dynamic props to sleep when no earthquake is active
        if self.earthquake_magnitude <= 0.1 {
            let grabbed_idx = self.grab.as_ref().map(|g| g.body_idx);
            for (idx, b) in self.bodies.iter_mut().enumerate() {
                let held_by_ragdoll = self.ragdolls.iter().any(|r| {
                    r.reflex.left_held.is_some_and(|h| h.body_idx == idx)
                        || r.reflex.right_held.is_some_and(|h| h.body_idx == idx)
                });
                if b.ragdoll_id.is_none()
                    && b.bone_id != Some(250)
                    && b.inv_mass > 0.0
                    && grabbed_idx != Some(idx)
                    && !held_by_ragdoll
                {
                    if b.is_sleeping {
                        continue;
                    }
                    if b.linear_velocity.length_squared() < 0.0025
                        && b.angular_velocity.length_squared() < 0.01
                    {
                        b.sleep_timer += clamped_dt;
                        if b.sleep_timer >= 0.75 {
                            b.is_sleeping = true;
                            b.linear_velocity = Vec3::ZERO;
                            b.angular_velocity = Vec3::ZERO;
                        }
                    } else {
                        b.sleep_timer = 0.0;
                    }
                }
            }
        }
    }

    #[inline]
    fn interrupt_ragdoll_recovery(&mut self, ragdoll_id: usize) {
        if let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) {
            ragdoll.balance.downed_still_time = 0.0;
            if ragdoll.balance.state == BehaviorState::Recovering {
                ragdoll.balance.state = BehaviorState::Falling;
                ragdoll.balance.recovery_phase = 0.0;
                ragdoll.balance.recovery_stage = RecoveryStage::RollOrPrep;
                ragdoll.balance.recovery_stage_time = 0.0;
            }
        }
    }

    /// Applies an instantaneous world-space `impulse` to `body_idx` at `world_point`.
    pub fn apply_impulse_at_point(&mut self, body_idx: usize, impulse: Vec3, world_point: Vec3) {
        if !impulse.is_finite() || !world_point.is_finite() {
            return;
        }
        let Some(body) = self.bodies.get_mut(body_idx) else {
            return;
        };
        if body.inv_mass == 0.0 {
            return;
        }
        body.wake_up();
        let r = world_point - body.position;
        body.linear_velocity += impulse * body.inv_mass;
        let dw = body.inv_inertia_world_mul(r.cross(impulse));
        body.angular_velocity += dw;
    }

    /// Applies an instantaneous world-space `impulse` to a ragdoll bone while capping
    /// per-bone `delta_v` and `delta_omega` (preventing light bones like `Neck` 1.5kg or
    /// `Hand` 0.45kg from launching into supersonic backflips). Any clipped excess impulse
    /// scaled by `overflow_to_torso_ratio` is transferred to the ragdoll's `Chest`.
    fn apply_clamped_ragdoll_impulse(
        &mut self,
        body_idx: usize,
        impulse: Vec3,
        world_point: Vec3,
        max_dv: f32,
        max_dw: f32,
        overflow_to_torso_ratio: f32,
    ) {
        if !impulse.is_finite() || !world_point.is_finite() {
            return;
        }
        let Some(body) = self.bodies.get(body_idx) else {
            return;
        };
        if body.inv_mass == 0.0 {
            return;
        }
        let ragdoll_id = body.ragdoll_id;
        let Some(r_id) = ragdoll_id else {
            self.apply_impulse_at_point(body_idx, impulse, world_point);
            return;
        };

        let imp_len = impulse.length();
        if imp_len < 1e-9 {
            return;
        }
        self.wake_ragdoll(r_id, 2.5);
        let body = &self.bodies[body_idx];
        let max_imp = (body.mass * max_dv).max(1e-4);
        let (applied_imp, overflow_imp) = if imp_len > max_imp {
            let capped = impulse * (max_imp / imp_len);
            (capped, impulse - capped)
        } else {
            (impulse, Vec3::ZERO)
        };

        {
            let b = &mut self.bodies[body_idx];
            let r = world_point - b.position;
            b.linear_velocity += applied_imp * b.inv_mass;
            let mut dw = b.inv_inertia_world_mul(r.cross(applied_imp));
            let dw_len = dw.length();
            if dw_len > max_dw && dw_len > 1e-6 {
                dw *= max_dw / dw_len;
            }
            b.angular_velocity += dw;
        }

        if overflow_to_torso_ratio > 0.0 && overflow_imp.length_squared() > 1e-8 {
            let chest_idx = self.ragdolls[r_id].body_indices[BoneId::Chest as usize];
            if chest_idx != body_idx {
                let chest_pos = self.bodies[chest_idx].position;
                self.apply_impulse_at_point(
                    chest_idx,
                    overflow_imp * overflow_to_torso_ratio,
                    chest_pos,
                );
            }
        }
    }

    /// Raycasts from `origin` along `dir` and applies a broad two-handed shove impulse
    /// distributed across the hit bone, Chest, and Pelvis when hitting a ragdoll.
    pub fn apply_shove(&mut self, origin: Vec3, dir: Vec3, strength: f32) -> Option<RaycastHit> {
        if !strength.is_finite() {
            return None;
        }
        let hit = self.raycast(origin, dir, 100.0)?;
        let dir_n = dir.normalize_or_zero();
        let total_impulse = dir_n * strength;

        if let Some(ragdoll_id) = self.bodies[hit.body_idx].ragdoll_id {
            self.wake_ragdoll(ragdoll_id, 2.5);
            let chest_idx = self.ragdolls[ragdoll_id].body_indices[BoneId::Chest as usize];
            let pelvis_idx = self.ragdolls[ragdoll_id].body_indices[BoneId::Pelvis as usize];
            let chest_pos = self.bodies[chest_idx].position;
            let pelvis_pos = self.bodies[pelvis_idx].position;

            self.apply_clamped_ragdoll_impulse(
                hit.body_idx,
                total_impulse * 0.40,
                hit.point,
                6.5,
                14.0,
                0.85,
            );
            self.apply_impulse_at_point(chest_idx, total_impulse * 0.35, chest_pos);
            self.apply_impulse_at_point(pelvis_idx, total_impulse * 0.25, pelvis_pos);
            if total_impulse.length() > 12.0 {
                self.interrupt_ragdoll_recovery(ragdoll_id);
            }
        } else {
            self.apply_impulse_at_point(hit.body_idx, total_impulse, hit.point);
        }
        Some(hit)
    }

    /// Raycasts from `origin` along `dir` and applies a sharp, localized punch impulse
    /// at the exact hit point.
    pub fn apply_punch(&mut self, origin: Vec3, dir: Vec3, strength: f32) -> Option<RaycastHit> {
        if !strength.is_finite() {
            return None;
        }
        let hit = self.raycast(origin, dir, 100.0)?;
        let dir_n = dir.normalize_or_zero();
        let impulse = dir_n * strength;
        self.apply_clamped_ragdoll_impulse(hit.body_idx, impulse, hit.point, 10.0, 22.0, 0.65);
        if impulse.length() > 12.0 {
            if let Some(ragdoll_id) = self.bodies[hit.body_idx].ragdoll_id {
                self.interrupt_ragdoll_recovery(ragdoll_id);
            }
        }
        Some(hit)
    }

    /// Fires a kinetic force wave along `dir` from `origin`, locating an epicenter via
    /// raycast (or `y = 1.0` plane intersection clamped within `1.5..8.0m` of `origin`) and
    /// applying a distance-attenuated radial + directional + upward impulse to every dynamic
    /// body within `radius`. Returns the count of affected dynamic bodies.
    pub fn apply_force_wave(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        radius: f32,
        max_impulse: f32,
    ) -> usize {
        if !origin.is_finite() || !dir.is_finite() {
            return 0;
        }
        let dir_n = dir.normalize_or_zero();
        if dir_n.length_squared() < 1e-12
            || !radius.is_finite()
            || radius <= 0.0
            || !max_impulse.is_finite()
        {
            return 0;
        }

        let impact_center = if let Some(hit) = self.raycast(origin, dir_n, 100.0) {
            hit.point
        } else if dir_n.y.abs() > 1e-5 {
            let t = (1.0 - origin.y) / dir_n.y;
            let dist = if t > 0.0 { t.clamp(1.5, 8.0) } else { 4.5 };
            origin + dir_n * dist
        } else {
            origin + dir_n * 4.5
        };

        let mut affected = 0usize;
        for idx in 0..self.bodies.len() {
            if self.bodies[idx].inv_mass == 0.0 {
                continue;
            }
            let pos = self.bodies[idx].position;
            let offset = pos - impact_center;
            let dist = offset.length();
            if dist >= radius {
                continue;
            }
            let falloff = (1.0 - dist / radius).powi(2);
            let radial = if dist > 1e-4 {
                offset / dist
            } else {
                Vec3::ZERO
            };
            let impulse_dir = (radial * 0.55 + dir_n * 0.45 + Vec3::Y * 0.25).normalize_or_zero();
            let impulse = impulse_dir * (max_impulse * falloff);
            if let Some(ragdoll_id) = self.bodies[idx].ragdoll_id {
                self.wake_ragdoll(ragdoll_id, 2.5);
            }
            self.apply_clamped_ragdoll_impulse(idx, impulse, pos, 10.0, 18.0, 0.0);
            if impulse.length() > 12.0 {
                if let Some(ragdoll_id) = self.bodies[idx].ragdoll_id {
                    self.interrupt_ragdoll_recovery(ragdoll_id);
                }
            }
            affected += 1;
        }
        affected
    }

    /// Fires a ballistic weapon (`Pistol`, `Shotgun`, or `Rifle`) from `origin` along `dir`,
    /// applying physical impulses at exact hit points and registering ragdoll wounds/damage.
    pub fn fire_weapon(&mut self, origin: Vec3, dir: Vec3, weapon: WeaponKind) -> WeaponShotResult {
        let dir_n = dir.normalize_or_zero();
        if dir_n.length_squared() < 1e-12 {
            return WeaponShotResult {
                hits: Vec::new(),
                hit_ragdoll_id: None,
                hit_region: WoundRegion::None,
            };
        }

        let (impulse_ns, severity, damage, is_shotgun) = match weapon {
            WeaponKind::Pistol => (38.0, 0.58, 22.0, false),
            WeaponKind::Shotgun => (28.0, 0.95, 14.0, true),
            WeaponKind::Rifle => (90.0, 0.82, 42.0, false),
        };

        let mut rays = Vec::with_capacity(if is_shotgun { 7 } else { 1 });
        rays.push(dir_n);
        if is_shotgun {
            let ref_up = if dir_n.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            };
            let u = ref_up.cross(dir_n).normalize();
            let v = dir_n.cross(u).normalize();
            let cone = 0.065_f32;
            let cos_c = cone.cos();
            let sin_c = cone.sin();
            for i in 0..6 {
                let angle = (i as f32) * (std::f32::consts::TAU / 6.0);
                let pellet_dir =
                    (dir_n * cos_c + (u * angle.cos() + v * angle.sin()) * sin_c).normalize();
                rays.push(pellet_dir);
            }
        }

        let mut hits = Vec::with_capacity(rays.len());
        let mut hit_ragdoll_id = None;
        let mut hit_region = WoundRegion::None;

        for ray_dir in rays {
            if let Some(hit) = self.raycast(origin, ray_dir, 100.0) {
                self.apply_clamped_ragdoll_impulse(
                    hit.body_idx,
                    ray_dir * impulse_ns,
                    hit.point,
                    11.0,
                    22.0,
                    0.35,
                );

                let ragdoll_info = {
                    let b = &self.bodies[hit.body_idx];
                    b.ragdoll_id.zip(b.bone_id.and_then(BoneId::from_u8))
                };
                if let Some((r_id, bone)) = ragdoll_info {
                    self.wake_ragdoll(r_id, 2.5);
                    let chest_idx = self.ragdolls[r_id].body_indices[BoneId::Chest as usize];
                    let chest_fwd = self.bodies[chest_idx].orientation * Vec3::Z;
                    let hit_from_back = ray_dir.dot(chest_fwd) > 0.25;
                    self.ragdolls[r_id].register_hit_with_dir(
                        bone,
                        hit.local_point,
                        severity,
                        damage,
                        hit_from_back,
                    );
                    if hit_ragdoll_id.is_none() {
                        hit_ragdoll_id = Some(r_id);
                        hit_region = WoundRegion::from_bone(bone);
                    }
                }
                hits.push(hit);
            }
        }

        WeaponShotResult {
            hits,
            hit_ragdoll_id,
            hit_region,
        }
    }

    /// Raycasts against all dynamic bodies up to `max_dist` (occluded by the `y = 0` ground plane
    /// and all `static_boxes`) and returns the closest hit.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
        if !origin.is_finite()
            || !dir.is_finite()
            || !max_dist.is_finite()
            || max_dist <= 0.0
            || dir.length_squared() < 1e-12
        {
            return None;
        }
        let d = dir.normalize();

        // 1. Compute nearest static occlusion distance (ground plane y = 0 and static boxes)
        let mut best_t = max_dist;
        if d.y < -1e-6 {
            let t = -origin.y / d.y;
            if t > 0.0 && t < best_t {
                best_t = t;
            }
        }
        for sb in &self.static_boxes {
            if let Some(t) = raycast_static_box(origin, d, sb, best_t) {
                if t > 0.0 && t < best_t {
                    best_t = t;
                }
            }
        }
        if let Some(hooks) = &self.world_query_hooks {
            if let Some(cb) = hooks.raycast {
                let o_arr = origin.to_array();
                let d_arr = d.to_array();
                let mut out_dist = 0.0_f32;
                let mut out_normal = [0.0_f32; 3];
                if cb(
                    hooks.user_data,
                    &o_arr,
                    &d_arr,
                    best_t,
                    &mut out_dist,
                    &mut out_normal,
                ) && out_dist.is_finite()
                    && out_dist > 0.0
                    && out_dist < best_t
                {
                    let raw_n = Vec3::from_array(out_normal);
                    let _normal = if raw_n.is_finite() && raw_n.length_squared() > 1e-12 {
                        raw_n.normalize()
                    } else {
                        Vec3::Y
                    };
                    best_t = out_dist;
                }
            }
        }

        // 2. Test dynamic bodies strictly closer than best_t
        let mut best: Option<RaycastHit> = None;
        let mut best_dist = best_t;

        for (idx, body) in self.bodies.iter().enumerate() {
            if body.inv_mass == 0.0 && body.collision_group == 0 {
                continue;
            }
            let hit = match body.collider {
                ColliderShape::Sphere { radius } => {
                    ray_sphere(origin, d, body.position, radius, best_dist)
                }
                ColliderShape::Box { half_extents } => ray_obb(
                    origin,
                    d,
                    body.position,
                    body.orientation,
                    half_extents,
                    best_dist,
                ),
                ColliderShape::Capsule {
                    radius,
                    half_length,
                } => ray_capsule(
                    origin,
                    d,
                    body.position,
                    body.orientation,
                    radius,
                    half_length,
                    best_dist,
                ),
            };

            if let Some((t, normal)) = hit {
                if t < best_dist {
                    best_dist = t;
                    let point = origin + d * t;
                    let local_point = body.orientation.conjugate() * (point - body.position);
                    best = Some(RaycastHit {
                        body_idx: idx,
                        distance: t,
                        point,
                        normal,
                        local_point,
                    });
                }
            }
        }

        best
    }

    /// Synchronizes a body's simulated pose and velocities from an external host physics engine
    /// (e.g. UE5 Chaos / Unity PhysX) so ARSA balance/reflex controllers can compute updated
    /// `joint.motor_target` values. Returns `false` if `body_idx` is invalid or any input is
    /// non-finite / degenerate.
    pub fn sync_bone_pose(
        &mut self,
        body_idx: usize,
        pos: Vec3,
        rot: Quat,
        lin_vel: Vec3,
        ang_vel: Vec3,
    ) -> bool {
        if !pos.is_finite()
            || !rot.is_finite()
            || rot.length_squared() < 1e-12
            || !lin_vel.is_finite()
            || !ang_vel.is_finite()
        {
            return false;
        }
        let Some(body) = self.bodies.get_mut(body_idx) else {
            return false;
        };
        let q = rot.normalize();
        body.position = pos;
        body.prev_position = pos;
        body.orientation = q;
        body.prev_orientation = q;
        body.update_world_inertia();
        body.linear_velocity = lin_vel;
        body.angular_velocity = ang_vel;
        true
    }

    /// Runs a brain-only controller tick for Bidirectional Host-Physics Mode (UE5 Chaos / Unity
    /// PhysX) over currently synced bone transforms, updating foot grounding, balance state,
    /// procedural reflexes, and all joint motor targets/gains without integrating positions or
    /// orientations or running XPBD positional/velocity solvers.
    pub fn step_brain_only(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let clamped_dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);

        for ragdoll in &self.ragdolls {
            for &b_idx in &ragdoll.body_indices {
                if let Some(b) = self.bodies.get_mut(b_idx) {
                    b.update_world_inertia();
                }
            }
        }

        self.foot_contacts_buf.clear();
        for ragdoll in &mut self.ragdolls {
            let s_h = ragdoll.profile.height_m / 1.78;
            let lf_idx = ragdoll.body_indices[BoneId::LeftFoot as usize];
            let rf_idx = ragdoll.body_indices[BoneId::RightFoot as usize];
            let lf_grounded = foot_near_support_parts(
                &self.bodies[lf_idx],
                s_h,
                &self.static_boxes,
                &self.bodies,
                self.heightfield.as_ref(),
                self.world_query_hooks.as_ref(),
            );
            let rf_grounded = foot_near_support_parts(
                &self.bodies[rf_idx],
                s_h,
                &self.static_boxes,
                &self.bodies,
                self.heightfield.as_ref(),
                self.world_query_hooks.as_ref(),
            );
            ragdoll.balance.left_foot_grounded = lf_grounded;
            ragdoll.balance.right_foot_grounded = rf_grounded;
            self.foot_contacts_buf.push((lf_grounded, rf_grounded));
        }

        let gravity_mag = self.gravity.length();
        for (r_idx, ragdoll) in self.ragdolls.iter_mut().enumerate() {
            if ragdoll.lod == RagdollLod::Sleep {
                continue;
            }
            step_ragdoll_controllers(
                ragdoll,
                &mut self.bodies,
                &mut self.joints,
                &self.static_boxes,
                self.heightfield.as_ref(),
                self.world_query_hooks.as_ref(),
                self.foot_contacts_buf[r_idx],
                gravity_mag,
                self.earthquake_magnitude,
                clamped_dt,
            );
        }
    }

    /// Computes the 16 world-space PD muscle motor torques (in SI `N·m`) for `ragdoll_id`
    /// converted into `space` coordinate axes for Bidirectional Host-Physics Mode.
    pub fn compute_host_joint_torques(
        &self,
        ragdoll_id: usize,
        space: CoordinateSpace,
    ) -> Option<[Vec3; 16]> {
        let ragdoll = self.ragdolls.get(ragdoll_id)?;
        let mut torques = [Vec3::ZERO; 16];
        for (i, &j_idx) in ragdoll.joint_indices.iter().enumerate() {
            let j = &self.joints[j_idx];
            if !ragdoll.active_enabled
                || !j.motor_stiffness.is_finite()
                || j.motor_stiffness <= 0.0
                || !j.max_torque.is_finite()
                || j.max_torque <= 0.0
                || !j.damage_scale.is_finite()
                || j.damage_scale <= 0.0
            {
                torques[i] = Vec3::ZERO;
                continue;
            }

            let parent = &self.bodies[j.parent];
            let child = &self.bodies[j.child];
            let q_p = parent.orientation * j.frame_parent;
            let q_c = child.orientation * j.frame_child;
            let q_desired = (q_p * j.motor_target).normalize();
            let mut q_err = q_desired * q_c.conjugate();
            if q_err.w < 0.0 {
                q_err = -q_err;
            }
            let (axis, angle) = q_err.to_axis_angle();
            let rot_err = if angle.is_finite() && axis.is_finite() {
                axis * angle
            } else {
                Vec3::ZERO
            };
            let rel_omega = child.angular_velocity - parent.angular_velocity;
            let k_eff = j.motor_stiffness * j.damage_scale;
            let d_eff = j.motor_damping * j.damage_scale;
            let max_t = j.max_torque * j.damage_scale;
            let raw_tau = rot_err * k_eff - rel_omega * d_eff;
            let tau = if raw_tau.is_finite() {
                let mag = raw_tau.length();
                if mag > max_t && mag > 1e-9 {
                    raw_tau * (max_t / mag)
                } else {
                    raw_tau
                }
            } else {
                Vec3::ZERO
            };
            torques[i] = convert_angular_vec_to_target(space, tau);
        }
        Some(torques)
    }

    /// Begins grabbing `body_idx` at `local_point` toward `target`.
    pub fn start_grab(&mut self, body_idx: usize, local_point: Vec3, target: Vec3) {
        if !local_point.is_finite() || !target.is_finite() {
            return;
        }
        if let Some(body) = self.bodies.get_mut(body_idx) {
            body.wake_up();
            let ragdoll_id = body.ragdoll_id;
            let m = body.mass.max(1.0);
            self.grab = Some(GrabConstraint {
                body_idx,
                local_point,
                target_world: target,
                stiffness: m * 220.0,
                damping: m * 22.0,
            });
            if let Some(r_id) = ragdoll_id {
                self.wake_ragdoll(r_id, 2.5);
            }
        }
    }

    /// Updates the active grab target position.
    pub fn move_grab(&mut self, target: Vec3) {
        if !target.is_finite() {
            return;
        }
        if let Some(grab) = &mut self.grab {
            grab.target_world = target;
        }
    }

    /// Releases the active grab constraint.
    pub fn end_grab(&mut self) {
        self.grab = None;
    }

    /// Attaches a ragdoll's left or right hand to `body_idx` at `local_anchor` with `break_force`.
    pub fn grab_body_with_hand(
        &mut self,
        ragdoll_id: usize,
        is_left: bool,
        body_idx: usize,
        local_anchor: Vec3,
        break_force: f32,
    ) -> bool {
        if !local_anchor.is_finite() || !break_force.is_finite() || break_force <= 0.0 {
            return false;
        }
        if body_idx >= self.bodies.len() || self.bodies[body_idx].inv_mass <= 0.0 {
            return false;
        }
        let Some(ragdoll) = self.ragdolls.get(ragdoll_id) else {
            return false;
        };
        if !ragdoll.active_enabled || ragdoll.reflex.unconscious || ragdoll.reflex.health <= 18.0 {
            return false;
        }
        let (ua_bid, fa_bid, ha_bid) = if is_left {
            (BoneId::LeftUpperArm, BoneId::LeftForearm, BoneId::LeftHand)
        } else {
            (
                BoneId::RightUpperArm,
                BoneId::RightForearm,
                BoneId::RightHand,
            )
        };
        if ragdoll.reflex.bone_damage[ha_bid as usize] >= 0.20
            || ragdoll.reflex.bone_damage[fa_bid as usize] >= 0.25
            || ragdoll.reflex.bone_damage[ua_bid as usize] >= 0.35
        {
            return false;
        }
        let hand_idx = ragdoll.body_indices[ha_bid as usize];
        let forearm_idx = ragdoll.body_indices[fa_bid as usize];
        if body_idx == hand_idx || body_idx == forearm_idx {
            return false;
        }

        let hand_mass = self.bodies[hand_idx].mass.max(0.1);
        let mass_ratio = hand_mass / 0.45;
        let held = HeldObject {
            body_idx,
            local_anchor,
            break_force,
            stiffness: 2200.0 * mass_ratio,
            damping: 48.0 * mass_ratio,
            hold_time: 0.0,
        };

        self.bodies[body_idx].wake_up();
        self.wake_ragdoll(ragdoll_id, 2.5);
        let ragdoll = &mut self.ragdolls[ragdoll_id];
        if is_left {
            ragdoll.reflex.left_held = Some(held);
            ragdoll.reflex.left_grab = None;
        } else {
            ragdoll.reflex.right_held = Some(held);
            ragdoll.reflex.right_grab = None;
        }
        true
    }

    /// Releases any held object on the specified hand (`is_left`) of `ragdoll_id`.
    pub fn release_hand_hold(&mut self, ragdoll_id: usize, is_left: bool) {
        if let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) {
            if is_left {
                ragdoll.reflex.left_held = None;
            } else {
                ragdoll.reflex.right_held = None;
            }
        }
    }

    /// Releases held objects from both hands of `ragdoll_id`.
    pub fn release_all_holds(&mut self, ragdoll_id: usize) {
        if let Some(ragdoll) = self.ragdolls.get_mut(ragdoll_id) {
            ragdoll.reflex.left_held = None;
            ragdoll.reflex.right_held = None;
        }
    }

    /// Returns `true` if the specified hand (`is_left`) of `ragdoll_id` is currently holding an object.
    pub fn is_hand_holding(&self, ragdoll_id: usize, is_left: bool) -> bool {
        self.ragdolls.get(ragdoll_id).is_some_and(|r| {
            if is_left {
                r.reflex.left_held.is_some()
            } else {
                r.reflex.right_held.is_some()
            }
        })
    }

    /// Spawns a dynamic box directly in front of `ragdoll_id`'s chest aligned with the chest's
    /// yaw orientation and attaches both hands to its left/right side anchors with `break_force`.
    pub fn give_ragdoll_held_box(
        &mut self,
        ragdoll_id: usize,
        half_extents: Vec3,
        mass: f32,
        break_force: f32,
    ) -> Option<usize> {
        if !half_extents.is_finite()
            || half_extents.x <= 0.0
            || half_extents.y <= 0.0
            || half_extents.z <= 0.0
            || !mass.is_finite()
            || mass <= 0.0
            || !break_force.is_finite()
            || break_force <= 0.0
        {
            return None;
        }
        let (chest_idx, s_h, can_hold) = {
            let r = self.ragdolls.get(ragdoll_id)?;
            let can = r.active_enabled && !r.reflex.unconscious && r.reflex.health > 18.0;
            (
                r.body_indices[BoneId::Chest as usize],
                r.profile.sanitized().height_m / 1.78,
                can,
            )
        };
        if !can_hold {
            return None;
        }

        let chest_pos = self.bodies[chest_idx].position;
        let chest_rot = self.bodies[chest_idx].orientation;
        let chest_fwd_raw = chest_rot * Vec3::Z;
        let chest_fwd_xz = Vec3::new(chest_fwd_raw.x, 0.0, chest_fwd_raw.z).normalize_or_zero();
        let chest_fwd = if chest_fwd_xz.length_squared() > 1e-6 {
            chest_fwd_xz
        } else {
            Vec3::Z
        };
        let yaw = chest_fwd.x.atan2(chest_fwd.z);
        let box_rot = Quat::from_rotation_y(yaw);
        let box_pos =
            chest_pos + chest_fwd * (0.38 * s_h + half_extents.z) - Vec3::Y * (0.10 * s_h);

        let box_idx = self.spawn_box(box_pos, half_extents, mass, Vec3::ZERO);
        self.bodies[box_idx].orientation = box_rot;
        self.bodies[box_idx].prev_orientation = box_rot;
        self.bodies[box_idx].update_world_inertia();

        let arm_configs = [
            (
                true,
                1.0_f32,
                BoneId::LeftUpperArm,
                BoneId::LeftForearm,
                BoneId::LeftHand,
                JointId::LeftShoulder,
                JointId::LeftElbow,
                JointId::LeftWrist,
            ),
            (
                false,
                -1.0_f32,
                BoneId::RightUpperArm,
                BoneId::RightForearm,
                BoneId::RightHand,
                JointId::RightShoulder,
                JointId::RightElbow,
                JointId::RightWrist,
            ),
        ];
        for (is_left, side_sign, ua_bid, fa_bid, ha_bid, sh_jid, el_jid, wr_jid) in arm_configs {
            let r = &self.ragdolls[ragdoll_id];
            let healthy = r.reflex.bone_damage[ha_bid as usize] < 0.20
                && r.reflex.bone_damage[fa_bid as usize] < 0.25
                && r.reflex.bone_damage[ua_bid as usize] < 0.35;
            if !healthy {
                continue;
            }
            let ua_idx = r.body_indices[ua_bid as usize];
            let fa_idx = r.body_indices[fa_bid as usize];
            let ha_idx = r.body_indices[ha_bid as usize];
            let sh_j_idx = r.joint_indices[sh_jid as usize];
            let el_j_idx = r.joint_indices[el_jid as usize];
            let wr_j_idx = r.joint_indices[wr_jid as usize];

            let q_sh = Quat::from_euler(
                glam::EulerRot::XYZ,
                -0.58,
                -side_sign * 0.22,
                -side_sign * 0.14,
            );
            let q_el = Quat::from_rotation_x(-0.92);
            let q_wr = Quat::from_rotation_y(-side_sign * 0.18);

            for (j_idx, p_idx, c_idx, q_rel) in [
                (sh_j_idx, chest_idx, ua_idx, q_sh),
                (el_j_idx, ua_idx, fa_idx, q_el),
                (wr_j_idx, fa_idx, ha_idx, q_wr),
            ] {
                let p_pos = self.bodies[p_idx].position;
                let p_rot = self.bodies[p_idx].orientation;
                let j = &self.joints[j_idx];
                let anchor_w = p_pos + p_rot * j.anchor_parent;
                let c_rot = (p_rot * q_rel).normalize();
                let c_pos = anchor_w - c_rot * j.anchor_child;
                let cb = &mut self.bodies[c_idx];
                cb.orientation = c_rot;
                cb.prev_orientation = c_rot;
                cb.position = c_pos;
                cb.prev_position = c_pos;
                cb.update_world_inertia();
            }

            let anchor_x = if is_left {
                half_extents.x
            } else {
                -half_extents.x
            };
            self.grab_body_with_hand(
                ragdoll_id,
                is_left,
                box_idx,
                Vec3::new(anchor_x, 0.0, 0.0),
                break_force,
            );
        }

        Some(box_idx)
    }

    /// Writes body transforms into `out` (7 floats per body: `px, py, pz, qx, qy, qz, qw`).
    pub fn write_transforms(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.bodies.len() * 7);
        for b in &self.bodies {
            out.push(b.position.x);
            out.push(b.position.y);
            out.push(b.position.z);
            out.push(b.orientation.x);
            out.push(b.orientation.y);
            out.push(b.orientation.z);
            out.push(b.orientation.w);
        }
    }

    /// Writes body descriptor metadata into `out` (6 floats per body:
    /// `type, dim0, dim1, dim2, ragdoll_id, bone_id`).
    /// - `type = 0.0` (Capsule): `dim0 = radius, dim1 = half_length, dim2 = 0.0`
    /// - `type = 1.0` (Box): `dim0 = hx, dim1 = hy, dim2 = hz`
    /// - `type = 2.0` (Sphere): `dim0 = radius, dim1 = 0.0, dim2 = 0.0`
    pub fn write_body_meta(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.bodies.len() * 6);
        for b in &self.bodies {
            match b.collider {
                ColliderShape::Capsule {
                    radius,
                    half_length,
                } => {
                    out.push(0.0);
                    out.push(radius);
                    out.push(half_length);
                    out.push(0.0);
                }
                ColliderShape::Box { half_extents } => {
                    out.push(1.0);
                    out.push(half_extents.x);
                    out.push(half_extents.y);
                    out.push(half_extents.z);
                }
                ColliderShape::Sphere { radius } => {
                    out.push(2.0);
                    out.push(radius);
                    out.push(0.0);
                    out.push(0.0);
                }
            }
            out.push(b.ragdoll_id.map(|id| id as f32).unwrap_or(-1.0));
            out.push(b.bone_id.map(|id| id as f32).unwrap_or(-1.0));
        }
    }

    /// Writes active tether segments into `out` (6 floats per tether:
    /// `ax, ay, az, bx, by, bz`).
    pub fn write_tethers(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.tethers.len() * 6);
        for t in &self.tethers {
            let Some(body) = self.bodies.get(t.body_idx) else {
                continue;
            };
            out.push(t.anchor_world.x);
            out.push(t.anchor_world.y);
            out.push(t.anchor_world.z);
            out.push(body.position.x);
            out.push(body.position.y);
            out.push(body.position.z);
        }
    }

    /// Writes per-ragdoll balance & reflex telemetry into `out` (16 floats per ragdoll:
    /// `com(3), cp(2), left_grounded, right_grounded, state_u32, pelvis(3), health, wound_region, wound_world(3)`).
    pub fn write_ragdoll_telemetry(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.ragdolls.len() * 16);
        for r in &self.ragdolls {
            let pelvis_pos = self.bodies[r.body_indices[BoneId::Pelvis as usize]].position;
            let (wound_region, wound_world) = if let Some(w) = &r.reflex.active_wound {
                let w_pos = self.bodies[r.body_indices[w.bone as usize]].world_point(w.local_point);
                (w.region, w_pos)
            } else {
                (WoundRegion::None, Vec3::ZERO)
            };
            out.push(r.balance.com_pos.x);
            out.push(r.balance.com_pos.y);
            out.push(r.balance.com_pos.z);
            out.push(r.balance.capture_point.x);
            out.push(r.balance.capture_point.z);
            out.push(if r.balance.left_foot_grounded {
                1.0
            } else {
                0.0
            });
            out.push(if r.balance.right_foot_grounded {
                1.0
            } else {
                0.0
            });
            out.push(r.balance.state as u32 as f32);
            out.push(pelvis_pos.x);
            out.push(pelvis_pos.y);
            out.push(pelvis_pos.z);
            out.push(r.reflex.health);
            out.push(wound_region as u32 as f32);
            out.push(wound_world.x);
            out.push(wound_world.y);
            out.push(wound_world.z);
        }
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn integrate_body_step(b: &mut RigidBody, gravity: Vec3, sub_dt: f32) {
    b.prev_position = b.position;
    b.prev_orientation = b.orientation;
    if b.inv_mass > 0.0 && !b.is_sleeping {
        b.linear_velocity += gravity * sub_dt;
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn derive_body_velocity_step(
    b: &mut RigidBody,
    pre_solve_vel: Vec3,
    pre_solve_ang_vel: Vec3,
    inv_dt: f32,
    sub_dt: f32,
) {
    const MAX_DEPENETRATION_DV: f32 = 6.0;
    const MAX_DEPENETRATION_DW: f32 = 25.0;
    if b.inv_mass == 0.0 || b.is_sleeping {
        return;
    }
    let derived_vel = (b.position - b.prev_position) * inv_dt;
    let dv = derived_vel - pre_solve_vel;
    let dv_len = dv.length();
    b.linear_velocity = if dv_len > MAX_DEPENETRATION_DV {
        pre_solve_vel + dv * (MAX_DEPENETRATION_DV / dv_len)
    } else {
        derived_vel
    };

    let mut dq = b.orientation * b.prev_orientation.conjugate();
    if dq.w < 0.0 {
        dq = -dq;
    }
    let derived_ang_vel = Vec3::new(dq.x, dq.y, dq.z) * (2.0 * inv_dt);
    let dw = derived_ang_vel - pre_solve_ang_vel;
    let dw_len = dw.length();
    b.angular_velocity = if dw_len > MAX_DEPENETRATION_DW {
        pre_solve_ang_vel + dw * (MAX_DEPENETRATION_DW / dw_len)
    } else {
        derived_ang_vel
    };

    let drag = (-0.15 * sub_dt).exp();
    b.linear_velocity *= drag;
    b.angular_velocity *= drag;
    b.update_world_inertia();
}

#[inline]
fn canonical_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn ray_sphere(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let mut t = -b - sqrt_d;
    if t < 0.0 {
        t = -b + sqrt_d;
    }
    if t < 0.0 || t > max_dist {
        return None;
    }
    let hit_pt = origin + dir * t;
    let normal = (hit_pt - center).normalize_or_zero();
    Some((t, normal))
}

fn ray_aabb(origin: Vec3, dir: Vec3, center: Vec3, half: Vec3, max_dist: f32) -> Option<f32> {
    let ro = origin - center;
    let mut tmin = 0.0_f32;
    let mut tmax = max_dist;

    for i in 0..3 {
        let o = ro[i];
        let d = dir[i];
        let h = half[i];
        if d.abs() < 1e-8 {
            if o < -h || o > h {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (-h - o) * inv_d;
            let mut t2 = (h - o) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }

    if tmin > 0.0 && tmin < max_dist {
        Some(tmin)
    } else {
        None
    }
}

fn ray_obb(
    origin: Vec3,
    dir: Vec3,
    pos: Vec3,
    rot: Quat,
    half: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let inv = rot.conjugate();
    let ro = inv * (origin - pos);
    let rd = inv * dir;

    let mut tmin = 0.0_f32;
    let mut tmax = max_dist;
    let mut hit_normal = Vec3::ZERO;

    for i in 0..3 {
        let o = ro[i];
        let d = rd[i];
        let h = half[i];
        if d.abs() < 1e-8 {
            if o < -h || o > h {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (-h - o) * inv_d;
            let mut t2 = (h - o) * inv_d;
            let mut n_sign = -1.0_f32;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
                n_sign = 1.0;
            }
            if t1 > tmin {
                tmin = t1;
                hit_normal = Vec3::ZERO;
                hit_normal[i] = n_sign;
            }
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }

    if tmin <= 0.0 || tmin > max_dist {
        return None;
    }
    if hit_normal.length_squared() < 0.5 {
        hit_normal = -rd.normalize_or_zero();
    }
    Some((tmin, (rot * hit_normal).normalize()))
}

fn ray_capsule(
    origin: Vec3,
    dir: Vec3,
    pos: Vec3,
    rot: Quat,
    radius: f32,
    half_length: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let inv = rot.conjugate();
    let ro = inv * (origin - pos);
    let rd = inv * dir;

    let mut best_t = max_dist;
    let mut best_n = Vec3::ZERO;
    let mut found = false;

    // 1. Infinite Y-aligned cylinder x^2 + z^2 = r^2
    let a = rd.x * rd.x + rd.z * rd.z;
    if a > 1e-8 {
        let b = ro.x * rd.x + ro.z * rd.z;
        let c = ro.x * ro.x + ro.z * ro.z - radius * radius;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let sqrt_d = disc.sqrt();
            for t in [(-b - sqrt_d) / a, (-b + sqrt_d) / a] {
                if t > 0.0 && t < best_t {
                    let y = ro.y + rd.y * t;
                    if y.abs() <= half_length {
                        best_t = t;
                        let hit = ro + rd * t;
                        best_n = Vec3::new(hit.x, 0.0, hit.z).normalize();
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    // 2. Endcap spheres at (0, +-half_length, 0)
    for cap_y in [half_length, -half_length] {
        let cap_center = Vec3::new(0.0, cap_y, 0.0);
        if let Some((t, n_local)) = ray_sphere(ro, rd, cap_center, radius, best_t) {
            if t < best_t {
                best_t = t;
                best_n = n_local;
                found = true;
            }
        }
    }

    if found {
        Some((best_t, (rot * best_n).normalize()))
    } else {
        None
    }
}

fn raycast_static_box(origin: Vec3, dir: Vec3, sb: &StaticBox, max_dist: f32) -> Option<f32> {
    if sb.orientation.w.abs() > 0.99999 {
        ray_aabb(origin, dir, sb.center, sb.half_extents, max_dist)
    } else {
        let inv_q = sb.orientation.conjugate();
        let ro = inv_q * (origin - sb.center);
        let rd = inv_q * dir;
        ray_aabb(ro, rd, Vec3::ZERO, sb.half_extents, max_dist)
    }
}

fn foot_near_support_parts(
    foot: &RigidBody,
    s_h: f32,
    static_boxes: &[StaticBox],
    bodies: &[RigidBody],
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
) -> bool {
    if foot.linear_velocity.y.abs() >= 0.18 {
        return false;
    }
    let foot_hy = match foot.collider {
        ColliderShape::Box { half_extents } => half_extents.y,
        _ => 0.035 * s_h,
    };
    let support_y = sample_support_height_parts(
        static_boxes,
        bodies,
        heightfield,
        query_hooks,
        foot.position.x,
        foot.position.z,
        foot.position.y + 0.05,
    );
    ((foot.position.y - foot_hy) - support_y).abs() <= 0.018
}

#[allow(clippy::too_many_arguments)]
fn step_ragdoll_controllers(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    static_boxes: &[StaticBox],
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
    foot_contacts: (bool, bool),
    gravity_mag: f32,
    earthquake_magnitude: f32,
    ctrl_dt: f32,
) {
    let pelvis_idx = ragdoll.body_indices[BoneId::Pelvis as usize];
    let pelvis_pos = bodies[pelvis_idx].position;
    let pelvis_y = pelvis_pos.y;
    let s_h = ragdoll.profile.height_m / 1.78;
    let foot_half_y = 0.035 * s_h;
    let lf_pos = bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]].position;
    let rf_pos = bodies[ragdoll.body_indices[BoneId::RightFoot as usize]].position;
    let foot_ground_y = match foot_contacts {
        (true, true) => 0.5 * (lf_pos.y + rf_pos.y) - foot_half_y,
        (true, false) => lf_pos.y - foot_half_y,
        (false, true) => rf_pos.y - foot_half_y,
        (false, false) => (lf_pos.y.min(rf_pos.y) - foot_half_y).max(0.0),
    };

    if matches!(
        ragdoll.balance.state,
        BehaviorState::Balanced | BehaviorState::Stumbling
    ) && foot_ground_y > 0.35
    {
        let probe_radius = 0.42 * s_h;
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let mut escape = Vec3::ZERO;
        for d in dirs {
            let probe_h = sample_support_height_parts(
                static_boxes,
                bodies,
                heightfield,
                query_hooks,
                pelvis_pos.x + d.x * probe_radius,
                pelvis_pos.z + d.z * probe_radius,
                pelvis_y,
            );
            if probe_h < foot_ground_y - 0.50 {
                escape -= d;
            }
        }
        if escape.length_squared() > 0.1 {
            ragdoll.balance.teetering = true;
            ragdoll.balance.teeter_escape_dir = escape.normalize();
        } else {
            ragdoll.balance.teetering = false;
            ragdoll.balance.teeter_escape_dir = Vec3::ZERO;
        }
    } else {
        ragdoll.balance.teetering = false;
        ragdoll.balance.teeter_escape_dir = Vec3::ZERO;
    }

    let step_target_ground_y = sample_support_height_parts(
        static_boxes,
        bodies,
        heightfield,
        query_hooks,
        ragdoll.balance.step_target_xz.x,
        ragdoll.balance.step_target_xz.z,
        pelvis_y,
    );
    update_ragdoll_balance(
        ragdoll,
        bodies,
        joints,
        foot_contacts,
        step_target_ground_y,
        gravity_mag,
        earthquake_magnitude,
        ctrl_dt,
    );
    ragdoll.reflex.incoming_threat = detect_incoming_threat(bodies, ragdoll);
    let ground_below_pelvis = sample_support_height_parts(
        static_boxes,
        bodies,
        heightfield,
        query_hooks,
        pelvis_pos.x,
        pelvis_pos.z,
        pelvis_pos.y,
    );
    apply_procedural_reflexes(
        ragdoll,
        bodies,
        joints,
        static_boxes,
        ground_below_pelvis,
        ctrl_dt,
    );
}

fn sample_support_height_parts(
    static_boxes: &[StaticBox],
    bodies: &[RigidBody],
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
    x: f32,
    z: f32,
    max_y: f32,
) -> f32 {
    let mut best_y = 0.0_f32;
    for sb in static_boxes {
        if sb.orientation.w.abs() > 0.99999 {
            if (x - sb.center.x).abs() <= sb.half_extents.x + 0.05
                && (z - sb.center.z).abs() <= sb.half_extents.z + 0.05
            {
                let top = sb.center.y + sb.half_extents.y;
                if top <= max_y && top > best_y {
                    best_y = top;
                }
            }
        } else {
            let inv_q = sb.orientation.conjugate();
            let o = inv_q * (Vec3::new(x, 0.0, z) - sb.center);
            let d = inv_q * Vec3::Y;
            let mut y_min = f32::NEG_INFINITY;
            let mut y_max = f32::INFINITY;
            let mut hit = true;
            for i in 0..3 {
                let h = sb.half_extents[i];
                if d[i].abs() < 1e-8 {
                    if o[i].abs() > h {
                        hit = false;
                        break;
                    }
                } else {
                    let inv_d = 1.0 / d[i];
                    let mut t1 = (-h - o[i]) * inv_d;
                    let mut t2 = (h - o[i]) * inv_d;
                    if t1 > t2 {
                        std::mem::swap(&mut t1, &mut t2);
                    }
                    y_min = y_min.max(t1);
                    y_max = y_max.min(t2);
                    if y_min > y_max {
                        hit = false;
                        break;
                    }
                }
            }
            if hit && y_max.is_finite() && y_max <= max_y && y_max > best_y {
                best_y = y_max;
            }
        }
    }
    if let Some(hf) = heightfield {
        if let Some((hy, _)) = hf.sample(x, z) {
            if hy <= max_y && hy > best_y {
                best_y = hy;
            }
        }
    }
    if let Some(hooks) = query_hooks {
        if let Some(cb) = hooks.ground_height {
            let mut out_y = 0.0_f32;
            let mut out_normal = [0.0_f32; 3];
            if cb(hooks.user_data, x, z, max_y, &mut out_y, &mut out_normal)
                && out_y.is_finite()
                && out_y <= max_y
                && out_y > best_y
            {
                best_y = out_y;
            }
        }
    }
    for b in bodies {
        if b.ragdoll_id.is_none()
            && b.mass >= 15.0
            && b.bone_id != Some(250)
            && matches!(b.collider, ColliderShape::Box { .. })
        {
            let (b_min, b_max) = body_world_aabb(b);
            if x >= b_min.x - 0.05
                && x <= b_max.x + 0.05
                && z >= b_min.z - 0.05
                && z <= b_max.z + 0.05
            {
                let top = b_max.y;
                if top <= max_y && top > best_y {
                    best_y = top;
                }
            }
        }
    }
    best_y
}

fn detect_incoming_threat(
    bodies: &[RigidBody],
    ragdoll: &RagdollInstance,
) -> Option<IncomingThreat> {
    if !ragdoll.active_enabled
        || ragdoll.reflex.unconscious
        || ragdoll.reflex.health <= 20.0
        || !matches!(
            ragdoll.balance.state,
            BehaviorState::Balanced | BehaviorState::Stumbling
        )
    {
        return None;
    }

    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let chest_pos = bodies[chest_idx].position;
    let s_h = ragdoll.profile.sanitized().height_m / 1.78;
    let max_range_sq = (4.5 * s_h).powi(2);

    let mut best: Option<IncomingThreat> = None;

    for b in bodies {
        if b.ragdoll_id.is_some() || b.inv_mass <= 0.0 || b.is_sleeping || b.mass < 1.0 {
            continue;
        }

        let vel = b.linear_velocity;
        let v_sq = vel.length_squared();
        if v_sq < 4.0 {
            continue;
        }

        let to_chest = chest_pos - b.position;
        if to_chest.length_squared() > max_range_sq {
            continue;
        }

        let t_closest = to_chest.dot(vel) / v_sq;
        if t_closest <= 0.0 || t_closest > 0.80 {
            continue;
        }

        let closest_pt = b.position + vel * t_closest;
        let miss_dist = (chest_pos - closest_pt).length();
        let body_radius = match b.collider {
            ColliderShape::Sphere { radius } => radius,
            ColliderShape::Box { half_extents } => half_extents.length(),
            ColliderShape::Capsule {
                radius,
                half_length,
            } => radius + half_length,
        };

        if miss_dist <= 0.75 * s_h + body_radius {
            let urgency = (1.0 - t_closest / 0.80).clamp(0.0, 1.0);
            let is_more_urgent = match best {
                None => true,
                Some(prev) => urgency > prev.urgency,
            };
            if is_more_urgent {
                best = Some(IncomingThreat {
                    world_pos: b.position,
                    velocity: vel,
                    time_to_impact: t_closest,
                    urgency,
                });
            }
        }
    }

    best
}
