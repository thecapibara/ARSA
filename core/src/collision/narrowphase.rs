//! Narrowphase shape-pair contact detectors, SAT OBB clipping, and geometric primitives.

use glam::{Quat, Vec3};

use crate::body::{ColliderShape, RigidBody};
use crate::world::WorldQueryHooks;

use super::{Contact, ContactMaterial, Heightfield, StaticBox};

#[inline]
fn combine_material(f1: f32, r1: f32, f2: f32, r2: f32) -> ContactMaterial {
    ContactMaterial {
        friction: (f1 * f2).sqrt(),
        restitution: r1.max(r2),
    }
}

#[inline]
fn evaluate_ground_surface(
    pt: Vec3,
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
) -> (f32, Vec3, f32, f32) {
    let mut best_y = 0.0_f32;
    let mut best_normal = Vec3::Y;
    let mut best_friction = 0.75_f32;
    let mut best_restitution = 0.05_f32;

    if let Some(hf) = heightfield {
        if let Some((hy, hn)) = hf.sample(pt.x, pt.z) {
            if hy > best_y {
                best_y = hy;
                best_normal = hn;
                best_friction = hf.friction;
                best_restitution = hf.restitution;
            }
        }
    }

    if let Some(hooks) = query_hooks {
        if let Some(cb) = hooks.ground_height {
            let mut out_y = 0.0_f32;
            let mut out_normal = [0.0_f32, 1.0, 0.0];
            if cb(
                hooks.user_data,
                pt.x,
                pt.z,
                pt.y + 0.15,
                &mut out_y,
                &mut out_normal,
            ) && out_y.is_finite()
                && out_y > best_y
            {
                let raw_n = Vec3::from_array(out_normal);
                let n = if raw_n.is_finite() && raw_n.length_squared() > 1e-12 {
                    let norm = raw_n.normalize();
                    if norm.y > 0.0 {
                        norm
                    } else {
                        Vec3::Y
                    }
                } else {
                    Vec3::Y
                };
                best_y = out_y;
                best_normal = n;
                best_friction = 0.85;
                best_restitution = 0.05;
            }
        }
    }

    (best_y, best_normal, best_friction, best_restitution)
}

pub(super) fn detect_ground_contacts(
    bodies: &[RigidBody],
    idx: usize,
    heightfield: Option<&Heightfield>,
    query_hooks: Option<&WorldQueryHooks>,
    out: &mut Vec<Contact>,
) {
    let b = &bodies[idx];
    let mut emit_point = |pt: Vec3, radius: f32| {
        let (ground_y, ground_normal, gf, gr) =
            evaluate_ground_surface(pt, heightfield, query_hooks);
        let lowest_y = pt.y - radius;
        if lowest_y < ground_y {
            let pen = (ground_y - lowest_y) * ground_normal.y.max(0.25);
            let pa = pt - ground_normal * radius;
            let pb = pa + ground_normal * pen;
            let mat = combine_material(b.friction, b.restitution, gf, gr);
            out.push(Contact::new(
                bodies,
                idx,
                None,
                (pa, pb, ground_normal, pen),
                mat,
            ));
        }
    };

    match b.collider {
        ColliderShape::Sphere { radius } => {
            emit_point(b.position, radius);
        }
        ColliderShape::Capsule {
            radius,
            half_length,
        } => {
            let axis = b.orientation * Vec3::Y;
            let p0 = b.position + axis * half_length;
            let p1 = b.position - axis * half_length;
            emit_point(p0, radius);
            emit_point(p1, radius);
        }
        ColliderShape::Box { half_extents } => {
            for corner_local in box_corners(half_extents) {
                emit_point(b.world_point(corner_local), 0.0);
            }
        }
    }
}

pub(super) fn detect_static_box_contacts(
    bodies: &[RigidBody],
    idx: usize,
    sb: &StaticBox,
    out: &mut Vec<Contact>,
) {
    let b = &bodies[idx];
    let mat = combine_material(b.friction, b.restitution, sb.friction, sb.restitution);
    let is_axis_aligned = sb.orientation.w.abs() > 0.99999;

    if is_axis_aligned {
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
                detect_obb_vs_obb_poses(
                    b.position,
                    b.orientation,
                    half_extents,
                    sb.center,
                    Quat::IDENTITY,
                    sb.half_extents,
                    |geom| {
                        out.push(Contact::new(
                            bodies,
                            idx,
                            None,
                            geom,
                            ContactMaterial { ..mat },
                        ));
                    },
                );
            }
        }
        return;
    }

    let q = if sb.orientation.length_squared() > 1e-12 {
        sb.orientation.normalize()
    } else {
        Quat::IDENTITY
    };
    let inv_q = q.conjugate();
    let to_world_geom = |(pa_l, pb_l, n_l, pen): (Vec3, Vec3, Vec3, f32)| {
        (
            sb.center + q * pa_l,
            sb.center + q * pb_l,
            (q * n_l).normalize(),
            pen,
        )
    };

    match b.collider {
        ColliderShape::Sphere { radius } => {
            let local_s = inv_q * (b.position - sb.center);
            if let Some(geom_l) = sphere_vs_aabb(local_s, radius, Vec3::ZERO, sb.half_extents) {
                out.push(Contact::new(bodies, idx, None, to_world_geom(geom_l), mat));
            }
        }
        ColliderShape::Capsule {
            radius,
            half_length,
        } => {
            let ax_local = inv_q * (b.orientation * Vec3::Y);
            let center_local = inv_q * (b.position - sb.center);
            let p0 = center_local + ax_local * half_length;
            let p1 = center_local - ax_local * half_length;
            let closest = closest_point_on_segment_to_aabb(p0, p1, Vec3::ZERO, sb.half_extents);

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
                if let Some(geom_l) = sphere_vs_aabb(pt_local, radius, Vec3::ZERO, sb.half_extents)
                {
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        to_world_geom(geom_l),
                        ContactMaterial { ..mat },
                    ));
                }
            }
        }
        ColliderShape::Box { half_extents } => {
            let local_pos = inv_q * (b.position - sb.center);
            let local_rot = (inv_q * b.orientation).normalize();
            detect_obb_vs_obb_poses(
                local_pos,
                local_rot,
                half_extents,
                Vec3::ZERO,
                Quat::IDENTITY,
                sb.half_extents,
                |geom_l| {
                    out.push(Contact::new(
                        bodies,
                        idx,
                        None,
                        to_world_geom(geom_l),
                        ContactMaterial { ..mat },
                    ));
                },
            );
        }
    }
}

pub(super) fn detect_dynamic_pair(
    bodies: &[RigidBody],
    i: usize,
    j: usize,
    out: &mut Vec<Contact>,
) {
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
        (ColliderShape::Box { half_extents: ha }, ColliderShape::Box { half_extents: hb }) => {
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
    detect_obb_vs_obb_poses(
        a.position,
        a.orientation,
        ha,
        b.position,
        b.orientation,
        hb,
        |geom| {
            out.push(Contact::new(
                bodies,
                i,
                Some(j),
                geom,
                ContactMaterial { ..mat },
            ));
        },
    );
}

fn detect_obb_vs_obb_poses(
    pos_a: Vec3,
    rot_a: Quat,
    ha: Vec3,
    pos_b: Vec3,
    rot_b: Quat,
    hb: Vec3,
    mut emit: impl FnMut((Vec3, Vec3, Vec3, f32)),
) {
    let axes_a = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let axes_b = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    let delta = pos_a - pos_b;

    // 1. Test the 6 face axes (A.x, A.y, A.z, B.x, B.y, B.z) and 9 edge-edge cross axes
    //    for minimum penetration.
    let mut best_pen = f32::MAX;
    let mut best_axis = Vec3::Y;
    let mut best_local_idx = 1usize;
    let mut ref_is_b = false;
    let mut best_edge: Option<(usize, usize)> = None;

    for (idx, &axis) in axes_a.iter().enumerate() {
        let ra = ha.x * axes_a[0].dot(axis).abs()
            + ha.y * axes_a[1].dot(axis).abs()
            + ha.z * axes_a[2].dot(axis).abs();
        let rb = hb.x * axes_b[0].dot(axis).abs()
            + hb.y * axes_b[1].dot(axis).abs()
            + hb.z * axes_b[2].dot(axis).abs();
        let dist = delta.dot(axis).abs();
        let overlap = ra + rb - dist;
        if overlap <= 0.0 {
            return;
        }
        if overlap < best_pen {
            best_pen = overlap;
            best_axis = axis;
            best_local_idx = idx;
            ref_is_b = false;
        }
    }

    for (idx, &axis) in axes_b.iter().enumerate() {
        let ra = ha.x * axes_a[0].dot(axis).abs()
            + ha.y * axes_a[1].dot(axis).abs()
            + ha.z * axes_a[2].dot(axis).abs();
        let rb = hb.x * axes_b[0].dot(axis).abs()
            + hb.y * axes_b[1].dot(axis).abs()
            + hb.z * axes_b[2].dot(axis).abs();
        let dist = delta.dot(axis).abs();
        let overlap = ra + rb - dist;
        if overlap <= 0.0 {
            return;
        }
        if overlap < best_pen {
            best_pen = overlap;
            best_axis = axis;
            best_local_idx = idx;
            ref_is_b = true;
        }
    }

    for (p, &ax_a) in axes_a.iter().enumerate() {
        for (q, &ax_b) in axes_b.iter().enumerate() {
            let cross = ax_a.cross(ax_b);
            let len_sq = cross.length_squared();
            if len_sq < 1e-6 {
                continue;
            }
            let axis = cross / len_sq.sqrt();
            let ra = ha.x * axes_a[0].dot(axis).abs()
                + ha.y * axes_a[1].dot(axis).abs()
                + ha.z * axes_a[2].dot(axis).abs();
            let rb = hb.x * axes_b[0].dot(axis).abs()
                + hb.y * axes_b[1].dot(axis).abs()
                + hb.z * axes_b[2].dot(axis).abs();
            let dist = delta.dot(axis).abs();
            let overlap = ra + rb - dist;
            if overlap <= 0.0 {
                return;
            }
            let better = if best_edge.is_some() {
                overlap < best_pen
            } else {
                overlap * 1.02 < best_pen
            };
            if better {
                best_pen = overlap;
                best_axis = axis;
                best_edge = Some((p, q));
            }
        }
    }

    // Orient normal from B to A.
    let mut normal = best_axis;
    if delta.dot(normal) < 0.0 {
        normal = -normal;
    }

    if let Some((p, q)) = best_edge {
        let mut c_a = pos_a;
        for (k, &ax) in axes_a.iter().enumerate() {
            if k != p {
                let sign = if (-normal).dot(ax) >= 0.0 { 1.0 } else { -1.0 };
                c_a += ax * (sign * ha[k]);
            }
        }
        let a0 = c_a - axes_a[p] * ha[p];
        let a1 = c_a + axes_a[p] * ha[p];

        let mut c_b = pos_b;
        for (k, &ax) in axes_b.iter().enumerate() {
            if k != q {
                let sign = if normal.dot(ax) >= 0.0 { 1.0 } else { -1.0 };
                c_b += ax * (sign * hb[k]);
            }
        }
        let b0 = c_b - axes_b[q] * hb[q];
        let b1 = c_b + axes_b[q] * hb[q];

        let (pa, pb) = closest_points_segment_segment(a0, a1, b0, b1);
        emit((pa, pb, normal, best_pen));
        return;
    }

    // 2. Identify reference & incident boxes and clip incident face against reference side planes.
    let (
        ref_pos,
        ref_rot,
        inc_pos,
        inc_rot,
        ref_half,
        inc_half,
        ref_axes,
        inc_axes,
        ref_outward_normal,
    ) = if ref_is_b {
        (pos_b, rot_b, pos_a, rot_a, hb, ha, axes_b, axes_a, normal)
    } else {
        (pos_a, rot_a, pos_b, rot_b, ha, hb, axes_a, axes_b, -normal)
    };

    let n_axis = best_local_idx;
    let u_axis = (n_axis + 1) % 3;
    let v_axis = (n_axis + 2) % 3;
    let ref_sign = if ref_axes[n_axis].dot(ref_outward_normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };

    // Pick the incident face whose outward normal is most antiparallel to ref_outward_normal.
    let mut inc_n_axis = 0usize;
    let mut inc_sign = 1.0_f32;
    let mut min_dot = f32::MAX;
    for (k, &ax) in inc_axes.iter().enumerate() {
        let d = ax.dot(ref_outward_normal);
        if d < min_dot {
            min_dot = d;
            inc_n_axis = k;
            inc_sign = 1.0;
        }
        if -d < min_dot {
            min_dot = -d;
            inc_n_axis = k;
            inc_sign = -1.0;
        }
    }

    let iu = (inc_n_axis + 1) % 3;
    let iv = (inc_n_axis + 2) % 3;
    let inv_ref_rot = ref_rot.conjugate();

    let mut poly_a = [Vec3::ZERO; 8];
    let mut poly_b = [Vec3::ZERO; 8];
    let quad_signs = [(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    for (idx, &(su, sv)) in quad_signs.iter().enumerate() {
        let mut p_inc = Vec3::ZERO;
        p_inc[inc_n_axis] = inc_sign * inc_half[inc_n_axis];
        p_inc[iu] = su * inc_half[iu];
        p_inc[iv] = sv * inc_half[iv];
        let p_world = inc_pos + inc_rot * p_inc;
        poly_a[idx] = inv_ref_rot * (p_world - ref_pos);
    }

    let len = clip_polygon_against_plane(&poly_a[..4], &mut poly_b, u_axis, 1.0, ref_half[u_axis]);
    let len =
        clip_polygon_against_plane(&poly_b[..len], &mut poly_a, u_axis, -1.0, ref_half[u_axis]);
    let len =
        clip_polygon_against_plane(&poly_a[..len], &mut poly_b, v_axis, 1.0, ref_half[v_axis]);
    let len =
        clip_polygon_against_plane(&poly_b[..len], &mut poly_a, v_axis, -1.0, ref_half[v_axis]);

    // Collect penetrating points (deduplicating near-coincident clipped vertices).
    let mut cands = [(Vec3::ZERO, 0.0_f32); 8];
    let mut cand_len = 0usize;
    for &v_ref_local in &poly_a[..len] {
        let pen = ref_half[n_axis] - v_ref_local[n_axis] * ref_sign;
        if pen <= 0.0 {
            continue;
        }
        let mut merged = false;
        for existing in &mut cands[..cand_len] {
            let du = existing.0[u_axis] - v_ref_local[u_axis];
            let dv = existing.0[v_axis] - v_ref_local[v_axis];
            if du * du + dv * dv < 1e-8 {
                if pen > existing.1 {
                    *existing = (v_ref_local, pen);
                }
                merged = true;
                break;
            }
        }
        if !merged && cand_len < 8 {
            cands[cand_len] = (v_ref_local, pen);
            cand_len += 1;
        }
    }

    if cand_len == 0 {
        return;
    }

    let selected = reduce_manifold_to_4(&cands[..cand_len], u_axis, v_axis);
    for &(v_ref_local, pen) in selected.iter().flatten() {
        let v_world = ref_pos + ref_rot * v_ref_local;
        let (pa, pb) = if ref_is_b {
            (v_world, v_world + normal * pen)
        } else {
            (v_world - normal * pen, v_world)
        };
        emit((pa, pb, normal, pen));
    }
}

fn clip_polygon_against_plane(
    poly: &[Vec3],
    out: &mut [Vec3; 8],
    axis: usize,
    sign: f32,
    limit: f32,
) -> usize {
    let n = poly.len();
    if n == 0 {
        return 0;
    }
    const EPS: f32 = -1e-6;
    let mut out_len = 0usize;
    let mut prev = poly[n - 1];
    let mut prev_dist = limit - sign * prev[axis];

    for &curr in poly {
        let curr_dist = limit - sign * curr[axis];
        let prev_in = prev_dist >= EPS;
        let curr_in = curr_dist >= EPS;

        if curr_in {
            if !prev_in {
                let denom = prev_dist - curr_dist;
                let t = if denom.abs() > 1e-12 {
                    (prev_dist / denom).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                if out_len < 8 {
                    out[out_len] = prev.lerp(curr, t);
                    out_len += 1;
                }
            }
            if out_len < 8 {
                out[out_len] = curr;
                out_len += 1;
            }
        } else if prev_in {
            let denom = prev_dist - curr_dist;
            let t = if denom.abs() > 1e-12 {
                (prev_dist / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            if out_len < 8 {
                out[out_len] = prev.lerp(curr, t);
                out_len += 1;
            }
        }

        prev = curr;
        prev_dist = curr_dist;
    }

    out_len
}

fn reduce_manifold_to_4(
    cands: &[(Vec3, f32)],
    u_axis: usize,
    v_axis: usize,
) -> [Option<(Vec3, f32)>; 4] {
    let mut out = [None; 4];
    if cands.len() <= 4 {
        for (idx, &c) in cands.iter().enumerate() {
            out[idx] = Some(c);
        }
        return out;
    }

    // 1. Deepest point
    let mut i0 = 0usize;
    for (idx, &(_, pen)) in cands.iter().enumerate().skip(1) {
        if pen > cands[i0].1 {
            i0 = idx;
        }
    }

    // 2. Point farthest from i0 in reference face plane
    let u0 = cands[i0].0[u_axis];
    let v0 = cands[i0].0[v_axis];
    let mut i1 = i0;
    let mut max_d2 = -1.0_f32;
    for (idx, &(p, _)) in cands.iter().enumerate() {
        if idx == i0 {
            continue;
        }
        let du = p[u_axis] - u0;
        let dv = p[v_axis] - v0;
        let d2 = du * du + dv * dv;
        if d2 > max_d2 {
            max_d2 = d2;
            i1 = idx;
        }
    }

    // 3. Extremal triangle areas on each side of segment (i0 -> i1)
    let du01 = cands[i1].0[u_axis] - u0;
    let dv01 = cands[i1].0[v_axis] - v0;
    let mut i2: Option<usize> = None;
    let mut max_pos = 0.0_f32;
    let mut i3: Option<usize> = None;
    let mut min_neg = 0.0_f32;

    for (idx, &(p, _)) in cands.iter().enumerate() {
        if idx == i0 || idx == i1 {
            continue;
        }
        let cross = du01 * (p[v_axis] - v0) - dv01 * (p[u_axis] - u0);
        if cross > max_pos {
            max_pos = cross;
            i2 = Some(idx);
        }
        if cross < min_neg {
            min_neg = cross;
            i3 = Some(idx);
        }
    }

    let mut chosen = [Some(i0), Some(i1), i2, i3];
    for slot_idx in 2..4 {
        if chosen[slot_idx].is_some() {
            continue;
        }
        let mut best_idx = None;
        let mut best_min_d2 = -1.0_f32;
        for (idx, &(p, _)) in cands.iter().enumerate() {
            if chosen.iter().flatten().any(|&c| c == idx) {
                continue;
            }
            let min_d2 = chosen
                .iter()
                .flatten()
                .map(|&c| {
                    let du = p[u_axis] - cands[c].0[u_axis];
                    let dv = p[v_axis] - cands[c].0[v_axis];
                    du * du + dv * dv
                })
                .fold(f32::MAX, f32::min);
            if min_d2 > best_min_d2 {
                best_min_d2 = min_d2;
                best_idx = Some(idx);
            }
        }
        chosen[slot_idx] = best_idx;
    }

    for (k, idx_opt) in chosen.into_iter().enumerate() {
        if let Some(idx) = idx_opt {
            out[k] = Some(cands[idx]);
        }
    }
    out
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
