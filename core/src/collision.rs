//! Collision detection and XPBD positional/velocity contact solver.

use std::collections::HashSet;

use glam::{Quat, Vec3};

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

struct ContactMaterial {
    friction: f32,
    restitution: f32,
}

impl Contact {
    fn new(
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
        detect_ground_contacts(bodies, idx, GROUND_FRICTION, GROUND_RESTITUTION, &mut contacts);

        for sb in static_boxes {
            detect_static_box_contacts(bodies, idx, sb, &mut contacts);
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
                continue;
            }
            detect_dynamic_pair(bodies, i, j, &mut contacts);
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

        let d_lambda_n = pen / w;
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
                    let j_f_mag = if c.body_b.is_none() && c.normal.y > 0.5 && v_t_len < 0.08 {
                        // Static friction stiction on ground contacts so resting bodies stop cleanly
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

#[inline]
fn combine_material(f1: f32, r1: f32, f2: f32, r2: f32) -> ContactMaterial {
    ContactMaterial {
        friction: (f1 * f2).sqrt(),
        restitution: r1.max(r2),
    }
}

fn detect_ground_contacts(
    bodies: &[RigidBody],
    idx: usize,
    ground_friction: f32,
    ground_restitution: f32,
    out: &mut Vec<Contact>,
) {
    let b = &bodies[idx];
    let mat = combine_material(b.friction, b.restitution, ground_friction, ground_restitution);

    match b.collider {
        ColliderShape::Sphere { radius } => {
            let lowest_y = b.position.y - radius;
            if lowest_y < 0.0 {
                let pa = Vec3::new(b.position.x, lowest_y, b.position.z);
                let pb = Vec3::new(b.position.x, 0.0, b.position.z);
                out.push(Contact::new(bodies, idx, None, (pa, pb, Vec3::Y, -lowest_y), mat));
            }
        }
        ColliderShape::Capsule {
            radius,
            half_length,
        } => {
            let axis = b.orientation * Vec3::Y;
            let p0 = b.position + axis * half_length;
            let p1 = b.position - axis * half_length;
            for p in [p0, p1] {
                let lowest_y = p.y - radius;
                if lowest_y < 0.0 {
                    let pa = Vec3::new(p.x, lowest_y, p.z);
                    let pb = Vec3::new(p.x, 0.0, p.z);
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        (pa, pb, Vec3::Y, -lowest_y),
                        ContactMaterial { ..mat },
                    ));
                }
            }
        }
        ColliderShape::Box { half_extents } => {
            for corner_local in box_corners(half_extents) {
                let pa = b.world_point(corner_local);
                if pa.y < 0.0 {
                    let pb = Vec3::new(pa.x, 0.0, pa.z);
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        (pa, pb, Vec3::Y, -pa.y),
                        ContactMaterial { ..mat },
                    ));
                }
            }
        }
    }
}

fn detect_static_box_contacts(
    bodies: &[RigidBody],
    idx: usize,
    sb: &StaticBox,
    out: &mut Vec<Contact>,
) {
    let b = &bodies[idx];
    let mat = combine_material(b.friction, b.restitution, sb.friction, sb.restitution);

    match b.collider {
        ColliderShape::Sphere { radius } => {
            if let Some(geom) = sphere_vs_aabb(b.position, radius, sb.center, sb.half_extents) {
                out.push(Contact::new(bodies, idx, None, geom, mat));
            }
        }
        ColliderShape::Capsule {
            radius,
            half_length,
        } => {
            let axis = b.orientation * Vec3::Y;
            let p0 = b.position + axis * half_length;
            let p1 = b.position - axis * half_length;
            let mid = b.position;
            let closest = closest_point_on_segment_to_aabb(p0, p1, sb.center, sb.half_extents);

            let samples = [p0, p1, mid, closest];
            let count = if closest.distance_squared(p0) > 1e-6
                && closest.distance_squared(p1) > 1e-6
                && closest.distance_squared(mid) > 1e-6
            {
                4
            } else {
                3
            };
            for &pt in &samples[..count] {
                if let Some(geom) = sphere_vs_aabb(pt, radius, sb.center, sb.half_extents) {
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        geom,
                        ContactMaterial { ..mat },
                    ));
                }
            }
        }
        ColliderShape::Box { half_extents } => {
            for local_pt in box_sample_points(half_extents) {
                let world_pt = b.world_point(local_pt);
                if let Some((pb, normal, pen)) =
                    point_inside_aabb(world_pt, sb.center, sb.half_extents)
                {
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        (world_pt, pb, normal, pen),
                        ContactMaterial { ..mat },
                    ));
                }
            }
            let inv_rot = b.orientation.conjugate();
            for sb_corner_offset in box_corners(sb.half_extents) {
                let world_sb_corner = sb.center + sb_corner_offset;
                let local_in_b = inv_rot * (world_sb_corner - b.position);
                if let Some((local_pa, local_normal_out, pen)) =
                    point_inside_aabb(local_in_b, Vec3::ZERO, half_extents)
                {
                    let normal = -(b.orientation * local_normal_out);
                    let pa = b.world_point(local_pa);
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        (pa, world_sb_corner, normal, pen),
                        ContactMaterial { ..mat },
                    ));
                }
            }
        }
    }
}

fn detect_dynamic_pair(bodies: &[RigidBody], i: usize, j: usize, out: &mut Vec<Contact>) {
    let a = &bodies[i];
    let b = &bodies[j];
    let mat = combine_material(a.friction, a.restitution, b.friction, b.restitution);

    match (a.collider, b.collider) {
        (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
            if let Some(geom) = sphere_vs_sphere(a.position, ra, b.position, rb) {
                out.push(Contact::new(bodies, i, Some(j), geom, mat));
            }
        }
        (ColliderShape::Sphere { radius }, ColliderShape::Box { half_extents }) => {
            if let Some(geom) =
                sphere_vs_obb(a.position, radius, b.position, b.orientation, half_extents)
            {
                out.push(Contact::new(bodies, i, Some(j), geom, mat));
            }
        }
        (ColliderShape::Box { half_extents }, ColliderShape::Sphere { radius }) => {
            if let Some((pb, pa, neg_normal, pen)) =
                sphere_vs_obb(b.position, radius, a.position, a.orientation, half_extents)
            {
                out.push(Contact::new(
                    bodies,
                    i,
                    Some(j),
                    (pa, pb, -neg_normal, pen),
                    mat,
                ));
            }
        }
        (
            ColliderShape::Sphere { radius: rs },
            ColliderShape::Capsule {
                radius: rc,
                half_length,
            },
        ) => {
            let axis = b.orientation * Vec3::Y;
            let closest = closest_point_on_segment(
                a.position,
                b.position + axis * half_length,
                b.position - axis * half_length,
            );
            if let Some(geom) = sphere_vs_sphere(a.position, rs, closest, rc) {
                out.push(Contact::new(bodies, i, Some(j), geom, mat));
            }
        }
        (
            ColliderShape::Capsule {
                radius: rc,
                half_length,
            },
            ColliderShape::Sphere { radius: rs },
        ) => {
            let axis = a.orientation * Vec3::Y;
            let closest = closest_point_on_segment(
                b.position,
                a.position + axis * half_length,
                a.position - axis * half_length,
            );
            if let Some(geom) = sphere_vs_sphere(closest, rc, b.position, rs) {
                out.push(Contact::new(bodies, i, Some(j), geom, mat));
            }
        }
        (
            ColliderShape::Capsule {
                radius: ra,
                half_length: hla,
            },
            ColliderShape::Capsule {
                radius: rb,
                half_length: hlb,
            },
        ) => {
            let ax_a = a.orientation * Vec3::Y;
            let ax_b = b.orientation * Vec3::Y;
            let (ca, cb) = closest_points_segment_segment(
                a.position + ax_a * hla,
                a.position - ax_a * hla,
                b.position + ax_b * hlb,
                b.position - ax_b * hlb,
            );
            if let Some(geom) = sphere_vs_sphere(ca, ra, cb, rb) {
                out.push(Contact::new(bodies, i, Some(j), geom, mat));
            }
        }
        (
            ColliderShape::Capsule {
                radius,
                half_length,
            },
            ColliderShape::Box { half_extents },
        ) => {
            detect_capsule_vs_obb(
                bodies,
                i,
                j,
                (radius, half_length, half_extents, false),
                mat,
                out,
            );
        }
        (
            ColliderShape::Box { half_extents },
            ColliderShape::Capsule {
                radius,
                half_length,
            },
        ) => {
            detect_capsule_vs_obb(
                bodies,
                i,
                j,
                (radius, half_length, half_extents, true),
                mat,
                out,
            );
        }
        (
            ColliderShape::Box { half_extents: ha },
            ColliderShape::Box { half_extents: hb },
        ) => {
            detect_obb_vs_obb(bodies, i, j, ha, hb, mat, out);
        }
    }
}

fn detect_capsule_vs_obb(
    bodies: &[RigidBody],
    body_i: usize,
    body_j: usize,
    params: (f32, f32, Vec3, bool),
    mat: ContactMaterial,
    out: &mut Vec<Contact>,
) {
    let (radius, half_length, half_extents, flip_normal) = params;
    let (cap_body, box_body) = if flip_normal {
        (&bodies[body_j], &bodies[body_i])
    } else {
        (&bodies[body_i], &bodies[body_j])
    };

    let inv_box = box_body.orientation.conjugate();
    let ax_local = inv_box * (cap_body.orientation * Vec3::Y);
    let center_local = inv_box * (cap_body.position - box_body.position);
    let p0 = center_local + ax_local * half_length;
    let p1 = center_local - ax_local * half_length;
    let closest = closest_point_on_segment_to_aabb(p0, p1, Vec3::ZERO, half_extents);

    let samples = [p0, p1, center_local, closest];
    let count = if closest.distance_squared(p0) > 1e-6
        && closest.distance_squared(p1) > 1e-6
        && closest.distance_squared(center_local) > 1e-6
    {
        4
    } else {
        3
    };

    for &pt_local in &samples[..count] {
        if let Some((pa_cap_local, pb_box_local, n_box_to_cap_local, pen)) =
            sphere_vs_aabb(pt_local, radius, Vec3::ZERO, half_extents)
        {
            let pa_cap = box_body.position + box_body.orientation * pa_cap_local;
            let pb_box = box_body.position + box_body.orientation * pb_box_local;
            let n_box_to_cap = box_body.orientation * n_box_to_cap_local;

            let geom = if flip_normal {
                (pb_box, pa_cap, -n_box_to_cap, pen)
            } else {
                (pa_cap, pb_box, n_box_to_cap, pen)
            };

            out.push(Contact::new(
                bodies,
                body_i,
                Some(body_j),
                geom,
                ContactMaterial { ..mat },
            ));
        }
    }
}

fn detect_obb_vs_obb(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    ha: Vec3,
    hb: Vec3,
    mat: ContactMaterial,
    out: &mut Vec<Contact>,
) {
    let a = &bodies[i];
    let b = &bodies[j];
    let inv_b = b.orientation.conjugate();
    for local_a in box_sample_points(ha) {
        let world_a = a.world_point(local_a);
        let in_b = inv_b * (world_a - b.position);
        if let Some((pb_local, normal_b_local, pen)) = point_inside_aabb(in_b, Vec3::ZERO, hb) {
            let pb = b.world_point(pb_local);
            let normal = b.orientation * normal_b_local;
            out.push(Contact::new(
                bodies,
                i,
                Some(j),
                (world_a, pb, normal, pen),
                ContactMaterial { ..mat },
            ));
        }
    }

    let inv_a = a.orientation.conjugate();
    for local_b in box_sample_points(hb) {
        let world_b = b.world_point(local_b);
        let in_a = inv_a * (world_b - a.position);
        if let Some((pa_local, normal_a_local, pen)) = point_inside_aabb(in_a, Vec3::ZERO, ha) {
            let pa = a.world_point(pa_local);
            let normal = -(a.orientation * normal_a_local);
            out.push(Contact::new(
                bodies,
                i,
                Some(j),
                (pa, world_b, normal, pen),
                ContactMaterial { ..mat },
            ));
        }
    }
}

#[inline]
fn sphere_vs_sphere(pa: Vec3, ra: f32, pb: Vec3, rb: f32) -> Option<(Vec3, Vec3, Vec3, f32)> {
    let diff = pa - pb;
    let dist_sq = diff.length_squared();
    let r_sum = ra + rb;
    if dist_sq >= r_sum * r_sum {
        return None;
    }

    let (normal, dist) = if dist_sq > 1e-12 {
        let d = dist_sq.sqrt();
        (diff / d, d)
    } else {
        (Vec3::Y, 0.0)
    };
    let pen = r_sum - dist;
    let point_a = pa - normal * ra;
    let point_b = pb + normal * rb;
    Some((point_a, point_b, normal, pen))
}

#[inline]
fn sphere_vs_obb(
    sphere_pos: Vec3,
    radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
) -> Option<(Vec3, Vec3, Vec3, f32)> {
    let inv_rot = box_rot.conjugate();
    let local_s = inv_rot * (sphere_pos - box_pos);
    let (pa_local, pb_local, normal_local, pen) =
        sphere_vs_aabb(local_s, radius, Vec3::ZERO, half_extents)?;
    Some((
        box_pos + box_rot * pa_local,
        box_pos + box_rot * pb_local,
        box_rot * normal_local,
        pen,
    ))
}

/// Returns `(point_on_sphere, point_on_aabb, normal_aabb_to_sphere, penetration)`.
#[inline]
fn sphere_vs_aabb(
    center_s: Vec3,
    radius: f32,
    aabb_center: Vec3,
    half: Vec3,
) -> Option<(Vec3, Vec3, Vec3, f32)> {
    let delta = center_s - aabb_center;
    let clamped = delta.clamp(-half, half);
    let closest = aabb_center + clamped;
    let diff = center_s - closest;
    let dist_sq = diff.length_squared();

    if dist_sq > radius * radius {
        return None;
    }

    if dist_sq > 1e-10 {
        let dist = dist_sq.sqrt();
        let normal = diff / dist;
        let pen = radius - dist;
        let pa = center_s - normal * radius;
        Some((pa, closest, normal, pen))
    } else {
        // Sphere center is inside the AABB: push out along nearest face.
        let (pb, normal, face_pen) = point_inside_aabb(center_s, aabb_center, half)?;
        let pen = face_pen + radius;
        let pa = center_s - normal * radius;
        Some((pa, pb, normal, pen))
    }
}

/// Tests if `p` is strictly inside AABB `(center, half)` and returns
/// `(projected_surface_point, outward_normal, penetration)`.
#[inline]
fn point_inside_aabb(p: Vec3, center: Vec3, half: Vec3) -> Option<(Vec3, Vec3, f32)> {
    let d = p - center;
    let dx = half.x - d.x.abs();
    let dy = half.y - d.y.abs();
    let dz = half.z - d.z.abs();
    if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
        return None;
    }

    if dx <= dy && dx <= dz {
        let s = if d.x >= 0.0 { 1.0 } else { -1.0 };
        let pb = Vec3::new(center.x + s * half.x, p.y, p.z);
        Some((pb, Vec3::new(s, 0.0, 0.0), dx))
    } else if dy <= dz {
        let s = if d.y >= 0.0 { 1.0 } else { -1.0 };
        let pb = Vec3::new(p.x, center.y + s * half.y, p.z);
        Some((pb, Vec3::new(0.0, s, 0.0), dy))
    } else {
        let s = if d.z >= 0.0 { 1.0 } else { -1.0 };
        let pb = Vec3::new(p.x, p.y, center.z + s * half.z);
        Some((pb, Vec3::new(0.0, 0.0, s), dz))
    }
}

#[inline]
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a;
    }
    let t = (p - a).dot(ab) / len_sq;
    a + ab * t.clamp(0.0, 1.0)
}

fn closest_point_on_segment_to_aabb(a: Vec3, b: Vec3, center: Vec3, half: Vec3) -> Vec3 {
    let mut pt = 0.5 * (a + b);
    for _ in 0..5 {
        let on_box = center + (pt - center).clamp(-half, half);
        pt = closest_point_on_segment(on_box, a, b);
    }
    pt
}

/// Closest points `(c1, c2)` between segments `[p1, q1]` and `[p2, q2]`.
fn closest_points_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);

    let (s, t) = if a <= 1e-12 && e <= 1e-12 {
        (0.0, 0.0)
    } else if a <= 1e-12 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= 1e-12 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };

    (p1 + d1 * s, p2 + d2 * t)
}

#[inline]
fn box_corners(h: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(-h.x, -h.y, -h.z),
        Vec3::new(h.x, -h.y, -h.z),
        Vec3::new(-h.x, h.y, -h.z),
        Vec3::new(h.x, h.y, -h.z),
        Vec3::new(-h.x, -h.y, h.z),
        Vec3::new(h.x, -h.y, h.z),
        Vec3::new(-h.x, h.y, h.z),
        Vec3::new(h.x, h.y, h.z),
    ]
}

#[inline]
fn box_sample_points(h: Vec3) -> [Vec3; 14] {
    [
        Vec3::new(-h.x, -h.y, -h.z),
        Vec3::new(h.x, -h.y, -h.z),
        Vec3::new(-h.x, h.y, -h.z),
        Vec3::new(h.x, h.y, -h.z),
        Vec3::new(-h.x, -h.y, h.z),
        Vec3::new(h.x, -h.y, h.z),
        Vec3::new(-h.x, h.y, h.z),
        Vec3::new(h.x, h.y, h.z),
        Vec3::new(h.x, 0.0, 0.0),
        Vec3::new(-h.x, 0.0, 0.0),
        Vec3::new(0.0, h.y, 0.0),
        Vec3::new(0.0, -h.y, 0.0),
        Vec3::new(0.0, 0.0, h.z),
        Vec3::new(0.0, 0.0, -h.z),
    ]
}
