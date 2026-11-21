//! Top-level simulation world, sub-stepped XPBD loop, raycasting, and grab constraint.

use std::collections::HashSet;

use glam::{Quat, Vec3};

use crate::balance::update_ragdoll_balance;
use crate::body::{ColliderShape, RigidBody};
use crate::collision::{
    detect_contacts, solve_contacts_position, solve_contacts_velocity, StaticBox,
};
use crate::joint::Joint;
use crate::reflex::{apply_procedural_reflexes, WoundRegion};
use crate::skeleton::{
    spawn_humanoid_with_profile, update_humanoid_profile, BodyProfile, BoneId, RagdollInstance,
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
    pub ragdolls: Vec<RagdollInstance>,
    pub grab: Option<GrabConstraint>,
    connected_pairs: HashSet<(usize, usize)>,
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
            ragdolls: Vec::new(),
            grab: None,
            connected_pairs: HashSet::new(),
        }
    }

    /// Clears all bodies, joints, tethers, static obstacles, ragdolls, and active grabs.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.joints.clear();
        self.tethers.clear();
        self.static_boxes.clear();
        self.ragdolls.clear();
        self.grab = None;
        self.connected_pairs.clear();
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
        let idx = self.static_boxes.len();
        self.static_boxes.push(StaticBox {
            center,
            half_extents,
            friction,
            restitution,
        });
        idx
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
        }
    }

    /// Enables or disables active balance & muscle motors for all ragdolls in the world.
    pub fn set_all_ragdolls_active(&mut self, active: bool) {
        for r in &mut self.ragdolls {
            r.active_enabled = active;
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
        self.spawn_box(
            Vec3::new(x, hy, z),
            Vec3::new(hx, hy, hz),
            mass,
            Vec3::ZERO,
        )
    }

    /// Spawns a dynamic box rigid body and returns its body index.
    pub fn spawn_box(&mut self, pos: Vec3, half_extents: Vec3, mass: f32, vel: Vec3) -> usize {
        let idx = self.bodies.len();
        let mut rb = RigidBody::new(
            pos,
            Quat::IDENTITY,
            mass,
            ColliderShape::Box { half_extents },
        );
        rb.linear_velocity = vel;
        self.bodies.push(rb);
        idx
    }

    /// Spawns a dynamic sphere rigid body and returns its body index.
    pub fn spawn_sphere(&mut self, pos: Vec3, radius: f32, mass: f32, vel: Vec3) -> usize {
        let idx = self.bodies.len();
        let mut rb = RigidBody::new(
            pos,
            Quat::IDENTITY,
            mass,
            ColliderShape::Sphere { radius },
        );
        rb.linear_velocity = vel;
        self.bodies.push(rb);
        idx
    }

    /// Advances the simulation by `dt` seconds using sub-stepped XPBD.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let clamped_dt = dt.min(0.05);
        let substeps = self.substeps.max(1);
        let sub_dt = clamped_dt / (substeps as f32);
        let inv_dt = 1.0 / sub_dt;
        let gravity_mag = self.gravity.length();
        let mut foot_contacts: Vec<(bool, bool)> = self
            .ragdolls
            .iter()
            .map(|r| (r.balance.left_foot_grounded, r.balance.right_foot_grounded))
            .collect();

        for _ in 0..substeps {
            // 1. Save previous state & integrate external forces
            for b in &mut self.bodies {
                b.prev_position = b.position;
                b.prev_orientation = b.orientation;
                if b.inv_mass > 0.0 {
                    b.linear_velocity += self.gravity * sub_dt;
                }
            }

            // Apply interactive grab constraint spring-damper force
            if let Some(grab) = &self.grab {
                if let Some(body) = self.bodies.get_mut(grab.body_idx) {
                    if body.inv_mass > 0.0 {
                        let world_pt = body.world_point(grab.local_point);
                        let vel_pt = body.velocity_at_point(world_pt);
                        let mut force = (grab.target_world - world_pt) * grab.stiffness
                            - vel_pt * grab.damping;
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
                update_ragdoll_balance(
                    ragdoll,
                    &mut self.bodies,
                    &mut self.joints,
                    foot_contacts[r_idx],
                    gravity_mag,
                    sub_dt,
                );
                apply_procedural_reflexes(ragdoll, &mut self.bodies, &mut self.joints, sub_dt);
            }
            for j in &self.joints {
                j.apply_muscle_motor(&mut self.bodies, sub_dt);
            }

            // Predict positions and orientations
            for b in &mut self.bodies {
                if b.inv_mass == 0.0 {
                    continue;
                }
                b.position += b.linear_velocity * sub_dt;
                let w = b.angular_velocity;
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * b.orientation;
                b.orientation = (b.orientation + dq * (0.5 * sub_dt)).normalize();
            }

            // 2. Collision detection
            let mut contacts =
                detect_contacts(&self.bodies, &self.static_boxes, &self.connected_pairs);

            // Track per-ragdoll foot grounding from detected contacts
            for (r_idx, ragdoll) in self.ragdolls.iter_mut().enumerate() {
                let lf_idx = ragdoll.body_indices[BoneId::LeftFoot as usize];
                let rf_idx = ragdoll.body_indices[BoneId::RightFoot as usize];
                let mut lf_grounded = false;
                let mut rf_grounded = false;
                for c in &contacts {
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
                foot_contacts[r_idx] = (lf_grounded, rf_grounded);
                ragdoll.balance.left_foot_grounded = lf_grounded;
                ragdoll.balance.right_foot_grounded = rf_grounded;
            }

            // 3. XPBD positional & angular constraint iterations
            for _ in 0..self.solver_iterations.max(1) {
                for j in &self.joints {
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
                solve_contacts_position(&mut self.bodies, &mut contacts);
            }

            // 4. Derive velocities from position & orientation changes
            for b in &mut self.bodies {
                if b.inv_mass == 0.0 {
                    continue;
                }
                b.linear_velocity = (b.position - b.prev_position) * inv_dt;
                let mut dq = b.orientation * b.prev_orientation.conjugate();
                if dq.w < 0.0 {
                    dq = -dq;
                }
                b.angular_velocity = Vec3::new(dq.x, dq.y, dq.z) * (2.0 * inv_dt);

                // Subtle air drag for numerical cleanliness
                let drag = (-0.15 * sub_dt).exp();
                b.linear_velocity *= drag;
                b.angular_velocity *= drag;
            }

            // 5. Velocity pass (joint damping + dynamic friction & restitution)
            for j in &self.joints {
                j.solve_velocity_damping(&mut self.bodies, sub_dt);
            }
            solve_contacts_velocity(&mut self.bodies, &contacts, sub_dt, gravity_mag);
        }
    }

    /// Applies an instantaneous world-space `impulse` to `body_idx` at `world_point`.
    pub fn apply_impulse_at_point(&mut self, body_idx: usize, impulse: Vec3, world_point: Vec3) {
        let Some(body) = self.bodies.get_mut(body_idx) else {
            return;
        };
        if body.inv_mass == 0.0 {
            return;
        }
        let r = world_point - body.position;
        body.linear_velocity += impulse * body.inv_mass;
        let dw = body.inv_inertia_world_mul(r.cross(impulse));
        body.angular_velocity += dw;
    }

    /// Raycasts from `origin` along `dir` and applies a broad two-handed shove impulse
    /// distributed across the hit bone, Chest, and Pelvis when hitting a ragdoll.
    pub fn apply_shove(&mut self, origin: Vec3, dir: Vec3, strength: f32) -> Option<RaycastHit> {
        let hit = self.raycast(origin, dir, 100.0)?;
        let dir_n = dir.normalize_or_zero();
        let total_impulse = dir_n * strength;

        if let Some(ragdoll_id) = self.bodies[hit.body_idx].ragdoll_id {
            let chest_idx = self.ragdolls[ragdoll_id].body_indices[BoneId::Chest as usize];
            let pelvis_idx = self.ragdolls[ragdoll_id].body_indices[BoneId::Pelvis as usize];
            let chest_pos = self.bodies[chest_idx].position;
            let pelvis_pos = self.bodies[pelvis_idx].position;

            self.apply_impulse_at_point(hit.body_idx, total_impulse * 0.40, hit.point);
            self.apply_impulse_at_point(chest_idx, total_impulse * 0.35, chest_pos);
            self.apply_impulse_at_point(pelvis_idx, total_impulse * 0.25, pelvis_pos);
        } else {
            self.apply_impulse_at_point(hit.body_idx, total_impulse, hit.point);
        }
        Some(hit)
    }

    /// Raycasts from `origin` along `dir` and applies a sharp, localized punch impulse
    /// at the exact hit point.
    pub fn apply_punch(&mut self, origin: Vec3, dir: Vec3, strength: f32) -> Option<RaycastHit> {
        let hit = self.raycast(origin, dir, 100.0)?;
        let dir_n = dir.normalize_or_zero();
        self.apply_impulse_at_point(hit.body_idx, dir_n * strength, hit.point);
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
            self.apply_impulse_at_point(idx, impulse, pos);
            affected += 1;
        }
        affected
    }

    /// Fires a ballistic weapon (`Pistol`, `Shotgun`, or `Rifle`) from `origin` along `dir`,
    /// applying physical impulses at exact hit points and registering ragdoll wounds/damage.
    pub fn fire_weapon(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        weapon: WeaponKind,
    ) -> WeaponShotResult {
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
                self.apply_impulse_at_point(hit.body_idx, ray_dir * impulse_ns, hit.point);

                let ragdoll_info = {
                    let b = &self.bodies[hit.body_idx];
                    b.ragdoll_id.zip(b.bone_id.and_then(BoneId::from_u8))
                };
                if let Some((r_id, bone)) = ragdoll_info {
                    self.ragdolls[r_id].register_hit(bone, hit.local_point, severity, damage);
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

    /// Raycasts against all dynamic bodies up to `max_dist` and returns the closest hit.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
        let dir_len_sq = dir.length_squared();
        if dir_len_sq < 1e-12 || max_dist <= 0.0 {
            return None;
        }
        let d = dir / dir_len_sq.sqrt();

        let mut best: Option<RaycastHit> = None;
        let mut best_dist = max_dist;

        for (idx, body) in self.bodies.iter().enumerate() {
            let hit = match body.collider {
                ColliderShape::Sphere { radius } => {
                    ray_sphere(origin, d, body.position, radius, best_dist)
                }
                ColliderShape::Box { half_extents } => {
                    ray_obb(origin, d, body.position, body.orientation, half_extents, best_dist)
                }
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

        best
    }

    /// Begins grabbing `body_idx` at `local_point` toward `target`.
    pub fn start_grab(&mut self, body_idx: usize, local_point: Vec3, target: Vec3) {
        if let Some(body) = self.bodies.get(body_idx) {
            let m = body.mass.max(1.0);
            self.grab = Some(GrabConstraint {
                body_idx,
                local_point,
                target_world: target,
                stiffness: m * 220.0,
                damping: m * 22.0,
            });
        }
    }

    /// Updates the active grab target position.
    pub fn move_grab(&mut self, target: Vec3) {
        if let Some(grab) = &mut self.grab {
            grab.target_world = target;
        }
    }

    /// Releases the active grab constraint.
    pub fn end_grab(&mut self) {
        self.grab = None;
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
            out.push(if r.balance.left_foot_grounded { 1.0 } else { 0.0 });
            out.push(if r.balance.right_foot_grounded { 1.0 } else { 0.0 });
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
