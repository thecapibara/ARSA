//! Collision detection and XPBD positional/velocity contact solver.

mod narrowphase;

use std::collections::HashSet;

use glam::{Mat3, Quat, Vec3};

use crate::body::{ColliderShape, RigidBody};
use crate::world::WorldQueryHooks;

/// Static oriented box obstacle in the world.
#[derive(Debug, Clone, Copy)]
pub struct StaticBox {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub orientation: Quat,
    pub friction: f32,
    pub restitution: f32,
}

/// Uniform-grid heightfield terrain collider in the XZ plane.
#[derive(Debug, Clone)]
pub struct Heightfield {
    pub origin_x: f32,
    pub origin_z: f32,
    pub cell_size: f32,
    pub cols: usize,
    pub rows: usize,
    /// Row-major height samples `[row * cols + col]`, length `>= cols * rows`.
    pub heights: Vec<f32>,
    pub friction: f32,
    pub restitution: f32,
}

impl Heightfield {
    /// Bilinearly samples terrain height and upward finite-difference unit normal (`normal.y > 0.0`)
    /// at world `(x, z)`. Returns `None` if invalid or out of bounds.
    pub fn sample(&self, x: f32, z: f32) -> Option<(f32, Vec3)> {
        if !x.is_finite()
            || !z.is_finite()
            || !self.origin_x.is_finite()
            || !self.origin_z.is_finite()
            || !self.cell_size.is_finite()
            || self.cols < 2
            || self.rows < 2
            || self.cell_size <= 1e-4
        {
            return None;
        }
        let needed = self.cols.checked_mul(self.rows)?;
        if self.heights.len() < needed {
            return None;
        }

        let fx = (x - self.origin_x) / self.cell_size;
        let fz = (z - self.origin_z) / self.cell_size;
        let max_c = (self.cols - 1) as f32;
        let max_r = (self.rows - 1) as f32;
        if fx < 0.0 || fx > max_c || fz < 0.0 || fz > max_r {
            return None;
        }

        let c0 = (fx.floor() as usize).min(self.cols - 2);
        let r0 = (fz.floor() as usize).min(self.rows - 2);
        let tx = (fx - c0 as f32).clamp(0.0, 1.0);
        let tz = (fz - r0 as f32).clamp(0.0, 1.0);

        let h00 = self.heights[r0 * self.cols + c0];
        let h10 = self.heights[r0 * self.cols + c0 + 1];
        let h01 = self.heights[(r0 + 1) * self.cols + c0];
        let h11 = self.heights[(r0 + 1) * self.cols + c0 + 1];
        if !h00.is_finite() || !h10.is_finite() || !h01.is_finite() || !h11.is_finite() {
            return None;
        }

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        let height = h0 + (h1 - h0) * tz;

        let dh_dx = ((h10 - h00) * (1.0 - tz) + (h11 - h01) * tz) / self.cell_size;
        let dh_dz = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / self.cell_size;
        let normal = Vec3::new(-dh_dx, 1.0, -dh_dz).normalize();

        Some((height, normal))
    }
}

/// Contact manifold point between dynamic `body_a` and either dynamic `body_b` or static world (`None`).
#[derive(Debug, Clone)]
pub struct Contact {
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub point_a: Vec3,
    pub point_b: Vec3,
    /// Unit normal pointing from B to A.
    pub normal: Vec3,
    pub penetration: f32,
    pub friction: f32,
    pub restitution: f32,
    pub(crate) local_a: Vec3,
    pub(crate) local_b: Vec3,
    pub(crate) normal_lambda: f32,
    pub(crate) tangent_lambda: f32,
    pub(crate) pre_solve_vn: f32,
}

pub(super) struct ContactMaterial {
    pub(super) friction: f32,
    pub(super) restitution: f32,
}

impl Contact {
    pub(super) fn new(
        bodies: &[RigidBody],
        body_a: usize,
        body_b: Option<usize>,
        geom: (Vec3, Vec3, Vec3, f32),
        mat: ContactMaterial,
    ) -> Self {
        let (point_a, point_b, normal, penetration) = geom;
        let local_a = bodies[body_a].orientation.conjugate() * (point_a - bodies[body_a].position);
        let local_b = match body_b {
            Some(b) => bodies[b].orientation.conjugate() * (point_b - bodies[b].position),
            None => point_b,
        };
        let va = bodies[body_a].velocity_at_point(point_a);
        let vb = match body_b {
            Some(b) => bodies[b].velocity_at_point(point_b),
            None => Vec3::ZERO,
        };
        let pre_solve_vn = (va - vb).dot(normal);

        Self {
            body_a,
            body_b,
            point_a,
            point_b,
            normal,
            penetration,
            friction: mat.friction,
            restitution: mat.restitution,
            local_a,
            local_b,
            normal_lambda: 0.0,
            tangent_lambda: 0.0,
            pre_solve_vn,
        }
    }
}

/// Returns `true` for same-ragdoll bone pairs that should resolve self-collision.
/// Only distal-limb vs torso/head/opposite-distal-limb pairs that cannot overlap
/// at standing spawn (even at extreme dense profiles like 166 cm / 160 kg) are enabled.
#[inline]
pub fn should_ragdoll_bones_collide(bone_a: u8, bone_b: u8) -> bool {
    let is_distal_left_arm = |b: u8| matches!(b, 6 | 7);
    let is_distal_right_arm = |b: u8| matches!(b, 9 | 10);
    let is_distal_arm = |b: u8| is_distal_left_arm(b) || is_distal_right_arm(b);
    let is_torso_or_head = |b: u8| matches!(b, 0 | 1 | 2 | 4);
    let is_distal_left_leg = |b: u8| matches!(b, 12 | 13);
    let is_distal_right_leg = |b: u8| matches!(b, 15 | 16);
    let is_distal_leg = |b: u8| is_distal_left_leg(b) || is_distal_right_leg(b);
    let is_upper_torso_or_head = |b: u8| matches!(b, 2 | 4);

    (is_distal_arm(bone_a) && is_torso_or_head(bone_b))
        || (is_distal_arm(bone_b) && is_torso_or_head(bone_a))
        || (is_distal_left_arm(bone_a) && is_distal_right_arm(bone_b))
        || (is_distal_left_arm(bone_b) && is_distal_right_arm(bone_a))
        || (is_distal_left_leg(bone_a) && is_distal_right_leg(bone_b))
        || (is_distal_left_leg(bone_b) && is_distal_right_leg(bone_a))
        || (is_distal_leg(bone_a) && is_upper_torso_or_head(bone_b))
        || (is_distal_leg(bone_b) && is_upper_torso_or_head(bone_a))
}

#[inline]
pub(crate) fn body_world_aabb(b: &RigidBody) -> (Vec3, Vec3) {
    let ext = match b.collider {
        ColliderShape::Sphere { radius } => Vec3::splat(radius),
        ColliderShape::Capsule {
            radius,
            half_length,
        } => {
            let axis = b.orientation * Vec3::Y;
            axis.abs() * half_length + Vec3::splat(radius)
        }
        ColliderShape::Box { half_extents } => {
            let m = Mat3::from_quat(b.orientation);
            m.x_axis.abs() * half_extents.x
                + m.y_axis.abs() * half_extents.y
                + m.z_axis.abs() * half_extents.z
        }
    };
    let margin = Vec3::splat(1e-4);
    (b.position - ext - margin, b.position + ext + margin)
}

#[inline]
fn aabbs_overlap(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool {
    a_min.x <= b_max.x
        && a_max.x >= b_min.x
        && a_min.y <= b_max.y
        && a_max.y >= b_min.y
        && a_min.z <= b_max.z
        && a_max.z >= b_min.z
}

#[inline]
pub(crate) fn static_box_world_aabb(sb: &StaticBox) -> (Vec3, Vec3) {
    let ext = if sb.orientation.w.abs() > 0.99999 {
        sb.half_extents
    } else {
        let m = Mat3::from_quat(sb.orientation);
        m.x_axis.abs() * sb.half_extents.x
            + m.y_axis.abs() * sb.half_extents.y
            + m.z_axis.abs() * sb.half_extents.z
    };
    let margin = Vec3::splat(1e-4);
    (sb.center - ext - margin, sb.center + ext + margin)
}

/// Detects all contacts against ground/heightfield/host surfaces, static boxes, and dynamic body pairs.
#[allow(clippy::too_many_arguments)]
pub fn detect_contacts(
    bodies: &[RigidBody],
    static_boxes: &[StaticBox],
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
    floor_friction: f32,
    connected_pairs: &HashSet<(usize, usize)>,
    held_skip_pairs: &[(usize, usize)],
    body_aabbs: &mut Vec<(Vec3, Vec3)>,
    out: &mut Vec<Contact>,
) {
    out.clear();
    body_aabbs.clear();
    body_aabbs.extend(bodies.iter().map(body_world_aabb));

    let has_terrain_queries =
        heightfield.is_some() || query_hooks.and_then(|h| h.ground_height).is_some();

    for (idx, body) in bodies.iter().enumerate() {
        if body.inv_mass == 0.0 || body.is_sleeping {
            continue;
        }
        let (b_min, b_max) = body_aabbs[idx];
        if has_terrain_queries || b_min.y < 0.0 {
            narrowphase::detect_ground_contacts(
                bodies,
                idx,
                heightfield,
                query_hooks,
                floor_friction,
                out,
            );
        }

        for sb in static_boxes {
            let (sb_min, sb_max) = static_box_world_aabb(sb);
            if aabbs_overlap(b_min, b_max, sb_min, sb_max) {
                narrowphase::detect_static_box_contacts(bodies, idx, sb, out);
            }
        }
    }

    let n = bodies.len();
    for i in 0..n {
        let a = &bodies[i];
        let (a_min, a_max) = body_aabbs[i];
        for j in (i + 1)..n {
            let b = &bodies[j];
            if (a.inv_mass == 0.0 || a.is_sleeping) && (b.inv_mass == 0.0 || b.is_sleeping) {
                continue;
            }
            if (a.collision_group & b.collision_mask) == 0
                || (b.collision_group & a.collision_mask) == 0
            {
                continue;
            }
            let same_ragdoll = a.ragdoll_id.is_some() && a.ragdoll_id == b.ragdoll_id;
            if same_ragdoll {
                let allow = match (a.bone_id, b.bone_id) {
                    (Some(ba), Some(bb)) => should_ragdoll_bones_collide(ba, bb),
                    _ => false,
                };
                if !allow {
                    continue;
                }
            }
            let (b_min, b_max) = body_aabbs[j];
            if !aabbs_overlap(a_min, a_max, b_min, b_max) {
                continue;
            }
            if same_ragdoll && connected_pairs.contains(&(i, j)) {
                continue;
            }
            if !held_skip_pairs.is_empty() && held_skip_pairs.contains(&(i, j)) {
                continue;
            }
            narrowphase::detect_dynamic_pair(bodies, i, j, out);
        }
    }
}

/// Solves positional contact penetration and static friction for one solver iteration.
pub fn solve_contacts_position(bodies: &mut [RigidBody], contacts: &mut [Contact]) {
    solve_contacts_position_ordered(bodies, contacts, false);
}

/// Solves positional contact penetration and static friction for one solver iteration,
/// optionally iterating in reverse order to eliminate corner-order bias across iterations.
pub fn solve_contacts_position_ordered(
    bodies: &mut [RigidBody],
    contacts: &mut [Contact],
    reverse: bool,
) {
    let n = contacts.len();
    for idx in 0..n {
        let c = &mut contacts[if reverse { n - 1 - idx } else { idx }];
        let a = c.body_a;
        let pa = bodies[a].world_point(c.local_a);
        let pb = match c.body_b {
            Some(b) => bodies[b].world_point(c.local_b),
            None => c.local_b,
        };

        let pen = (pb - pa).dot(c.normal);
        if pen > 0.0 {
            // Soft depenetration cap per solver iteration so deep initial overlaps
            // (e.g. spawning an obstacle under a ragdoll's feet) resolve smoothly
            // over multiple substeps rather than launching bodies into the air.
            let effective_pen = pen.min(0.008);

            let r_a = pa - bodies[a].position;
            let w_a = bodies[a].generalized_inv_mass(r_a, c.normal);
            let (r_b, w_b) = match c.body_b {
                Some(b) => {
                    let rb = pb - bodies[b].position;
                    let wb = bodies[b].generalized_inv_mass(rb, c.normal);
                    (rb, wb)
                }
                None => (Vec3::ZERO, 0.0),
            };

            let w = w_a + w_b;
            if w >= 1e-12 {
                // Gently relax micro-penetrations so 4-corner box manifolds don't chatter/bounce
                let relax = if pen < 0.003 { 0.55 } else { 1.0 };
                let d_lambda_n = (effective_pen * relax) / w;
                c.normal_lambda += d_lambda_n;
                let p_n = c.normal * d_lambda_n;

                bodies[a].apply_pos_correction(p_n, r_a);
                if let Some(b) = c.body_b {
                    bodies[b].apply_pos_correction(-p_n, r_b);
                }
            }
        }

        // Static friction projection in position space (runs whenever the contact has
        // accumulated normal load in this substep, even after normal pen hits 0)
        if c.friction > 0.0 && c.normal_lambda > 0.0 {
            let pa_after = bodies[a].world_point(c.local_a);
            let prev_pa = bodies[a].prev_position + bodies[a].prev_orientation * c.local_a;
            let (pb_after, prev_pb) = match c.body_b {
                Some(b) => (
                    bodies[b].world_point(c.local_b),
                    bodies[b].prev_position + bodies[b].prev_orientation * c.local_b,
                ),
                None => (c.local_b, c.local_b),
            };

            let dp = (pa_after - prev_pa) - (pb_after - prev_pb);
            let dp_t = dp - c.normal * dp.dot(c.normal);
            let t_len = dp_t.length();
            if t_len > 1e-7 {
                let t_dir = dp_t / t_len;
                let r_a_t = pa_after - bodies[a].position;
                let w_a_t = bodies[a].generalized_inv_mass(r_a_t, t_dir);
                let (r_b_t, w_b_t) = match c.body_b {
                    Some(b) => {
                        let rb = pb_after - bodies[b].position;
                        (rb, bodies[b].generalized_inv_mass(rb, t_dir))
                    }
                    None => (Vec3::ZERO, 0.0),
                };
                let w_t = w_a_t + w_b_t;
                if w_t > 1e-12 {
                    let d_lambda_t = t_len / w_t;
                    let cone_remaining = (c.friction * c.normal_lambda - c.tangent_lambda).max(0.0);
                    let applied_t = d_lambda_t.min(cone_remaining);
                    if applied_t > 0.0 {
                        c.tangent_lambda += applied_t;
                        let p_t = -t_dir * applied_t;
                        bodies[a].apply_pos_correction(p_t, r_a_t);
                        if let Some(b) = c.body_b {
                            bodies[b].apply_pos_correction(-p_t, r_b_t);
                        }
                    }
                }
            }
        }
    }
}

/// Applies restitution and Coulomb dynamic friction in the velocity pass.
pub fn solve_contacts_velocity(
    bodies: &mut [RigidBody],
    contacts: &[Contact],
    dt: f32,
    gravity_mag: f32,
) {
    solve_contacts_velocity_ordered(bodies, contacts, dt, gravity_mag, false);
}

/// Applies restitution and Coulomb dynamic friction in the velocity pass,
/// optionally iterating in reverse order to avoid corner-order yaw bias.
pub fn solve_contacts_velocity_ordered(
    bodies: &mut [RigidBody],
    contacts: &[Contact],
    dt: f32,
    gravity_mag: f32,
    reverse: bool,
) {
    if dt <= 0.0 {
        return;
    }

    let restitution_threshold = (2.0 * gravity_mag * dt).max(0.25);

    let n = contacts.len();
    for idx in 0..n {
        let c = &contacts[if reverse { n - 1 - idx } else { idx }];
        if c.normal_lambda <= 0.0 {
            continue;
        }

        let a = c.body_a;
        let pa = bodies[a].world_point(c.local_a);
        let pb = match c.body_b {
            Some(b) => bodies[b].world_point(c.local_b),
            None => c.local_b,
        };

        let va = bodies[a].velocity_at_point(pa);
        let vb = match c.body_b {
            Some(b) => bodies[b].velocity_at_point(pb),
            None => Vec3::ZERO,
        };
        let v_rel = va - vb;
        let v_n = v_rel.dot(c.normal);

        let r_a = pa - bodies[a].position;
        let r_b = match c.body_b {
            Some(b) => pb - bodies[b].position,
            None => Vec3::ZERO,
        };

        // Dynamic Coulomb friction
        if c.friction > 0.0 {
            let v_t = v_rel - c.normal * v_n;
            let v_t_len = v_t.length();
            if v_t_len > 1e-6 {
                let t_dir = v_t / v_t_len;
                let w_a_t = bodies[a].generalized_inv_mass(r_a, t_dir);
                let w_b_t = match c.body_b {
                    Some(b) => bodies[b].generalized_inv_mass(r_b, t_dir),
                    None => 0.0,
                };
                let w_t = w_a_t + w_b_t;
                if w_t > 1e-12 {
                    let max_j_f = c.friction * (c.normal_lambda / dt);
                    let is_box_pair = matches!(
                        (bodies[a].collider, c.body_b.map(|b| bodies[b].collider)),
                        (ColliderShape::Box { .. }, Some(ColliderShape::Box { .. }))
                    );
                    let stiction = (c.body_b.is_none() && c.normal.y > 0.5 && v_t_len < 0.08)
                        || (is_box_pair && c.normal.y.abs() > 0.7 && v_t_len < 0.06);
                    let j_f_mag = if stiction {
                        // Static friction stiction so resting bodies on ground/stacks stop cleanly
                        v_t_len / w_t
                    } else {
                        (v_t_len / w_t).min(max_j_f)
                    };
                    let j_f = -t_dir * j_f_mag;
                    apply_velocity_impulse(bodies, a, c.body_b, j_f, r_a, r_b);
                }
            }
        }

        // Restitution (only above resting threshold)
        if c.restitution > 0.0 && c.pre_solve_vn < -restitution_threshold {
            let va_now = bodies[a].velocity_at_point(pa);
            let vb_now = match c.body_b {
                Some(b) => bodies[b].velocity_at_point(pb),
                None => Vec3::ZERO,
            };
            let vn_now = (va_now - vb_now).dot(c.normal);
            let target_vn = -c.restitution * c.pre_solve_vn;
            let dv_n = target_vn - vn_now;
            if dv_n > 0.0 {
                let w_a_n = bodies[a].generalized_inv_mass(r_a, c.normal);
                let w_b_n = match c.body_b {
                    Some(b) => bodies[b].generalized_inv_mass(r_b, c.normal),
                    None => 0.0,
                };
                let w_n = w_a_n + w_b_n;
                if w_n > 1e-12 {
                    let j_n = c.normal * (dv_n / w_n);
                    apply_velocity_impulse(bodies, a, c.body_b, j_n, r_a, r_b);
                }
            }
        }
    }
}

#[inline]
fn apply_velocity_impulse(
    bodies: &mut [RigidBody],
    a: usize,
    b_opt: Option<usize>,
    impulse: Vec3,
    r_a: Vec3,
    r_b: Vec3,
) {
    bodies[a].linear_velocity += impulse * bodies[a].inv_mass;
    let dw_a = bodies[a].inv_inertia_world_mul(r_a.cross(impulse));
    bodies[a].angular_velocity += dw_a;

    if let Some(b) = b_opt {
        bodies[b].linear_velocity -= impulse * bodies[b].inv_mass;
        let dw_b = bodies[b].inv_inertia_world_mul(r_b.cross(-impulse));
        bodies[b].angular_velocity += dw_b;
    }
}
