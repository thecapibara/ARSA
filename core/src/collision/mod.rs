//! Collision detection and XPBD positional/velocity contact solver.

mod narrowphase;

use std::collections::HashSet;

use glam::Vec3;

use crate::body::{ColliderShape, RigidBody};

/// Axis-aligned static box obstacle in the world.
#[derive(Debug, Clone, Copy)]
pub struct StaticBox {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub friction: f32,
    pub restitution: f32,
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

/// Detects all contacts against the `y = 0.0` ground plane, static boxes, and dynamic body pairs.
pub fn detect_contacts(
    bodies: &[RigidBody],
    static_boxes: &[StaticBox],
    connected_pairs: &HashSet<(usize, usize)>,
) -> Vec<Contact> {
    let mut contacts = Vec::with_capacity(bodies.len() * 2);

    const GROUND_FRICTION: f32 = 0.75;
    const GROUND_RESTITUTION: f32 = 0.05;

    for (idx, body) in bodies.iter().enumerate() {
        if body.inv_mass == 0.0 {
            continue;
        }
        narrowphase::detect_ground_contacts(
            bodies,
            idx,
            GROUND_FRICTION,
            GROUND_RESTITUTION,
            &mut contacts,
        );

        for sb in static_boxes {
            narrowphase::detect_static_box_contacts(bodies, idx, sb, &mut contacts);
        }
    }

    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let a = &bodies[i];
            let b = &bodies[j];
            if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                continue;
            }
            if (a.collision_group & b.collision_mask) == 0
                || (b.collision_group & a.collision_mask) == 0
            {
                continue;
            }
            if connected_pairs.contains(&(i, j)) {
                continue;
            }
            if a.ragdoll_id.is_some() && a.ragdoll_id == b.ragdoll_id {
                let allow = match (a.bone_id, b.bone_id) {
                    (Some(ba), Some(bb)) => should_ragdoll_bones_collide(ba, bb),
                    _ => false,
                };
                if !allow {
                    continue;
                }
            }
            narrowphase::detect_dynamic_pair(bodies, i, j, &mut contacts);
        }
    }

    contacts
}

/// Solves positional contact penetration and static friction for one solver iteration.
pub fn solve_contacts_position(bodies: &mut [RigidBody], contacts: &mut [Contact]) {
    for c in contacts.iter_mut() {
        let a = c.body_a;
        let pa = bodies[a].world_point(c.local_a);
        let pb = match c.body_b {
            Some(b) => bodies[b].world_point(c.local_b),
            None => c.local_b,
        };

        let pen = (pb - pa).dot(c.normal);
        if pen <= 0.0 {
            continue;
        }
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
        if w < 1e-12 {
            continue;
        }

        let d_lambda_n = effective_pen / w;
        c.normal_lambda += d_lambda_n;
        let p_n = c.normal * d_lambda_n;

        bodies[a].apply_pos_correction(p_n, r_a);
        if let Some(b) = c.body_b {
            bodies[b].apply_pos_correction(-p_n, r_b);
        }

        // Static friction projection in position space
        if c.friction > 0.0 {
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
    if dt <= 0.0 {
        return;
    }

    let restitution_threshold = (2.0 * gravity_mag * dt).max(0.25);

    for c in contacts {
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
