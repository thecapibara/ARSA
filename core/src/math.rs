//! Mathematical utilities for ARSA: swing-twist quaternion decomposition and
//! rigid-body inertia tensor diagonals.

use glam::{Quat, Vec3};

/// Decomposes a rotation quaternion `q` into `swing * twist`, where `twist` is a
/// pure rotation around `axis` and `swing` is the shortest-arc rotation around
/// an axis perpendicular to `axis`.
///
/// Both returned quaternions are normalized. Handles near-zero quaternions,
/// near-zero axes, and 180-degree rotations perpendicular to `axis` safely.
#[inline]
pub fn swing_twist_decomposition(q: Quat, axis: Vec3) -> (Quat, Quat) {
    let axis_len_sq = axis.length_squared();
    if axis_len_sq < 1e-12 {
        let qn = if q.length_squared() < 1e-12 {
            Quat::IDENTITY
        } else {
            q.normalize()
        };
        return (qn, Quat::IDENTITY);
    }

    let u = axis / axis_len_sq.sqrt();
    let qn = if q.length_squared() < 1e-12 {
        Quat::IDENTITY
    } else {
        q.normalize()
    };

    let q_vec = Vec3::new(qn.x, qn.y, qn.z);
    let proj = u * q_vec.dot(u);
    let mut twist = Quat::from_xyzw(proj.x, proj.y, proj.z, qn.w);

    let twist_len_sq = twist.length_squared();
    if twist_len_sq < 1e-12 {
        return (qn, Quat::IDENTITY);
    }

    twist = twist / twist_len_sq.sqrt();
    if twist.w < 0.0 {
        twist = -twist;
    }

    let swing = (qn * twist.conjugate()).normalize();
    (swing, twist)
}

/// Computes the local-space diagonal inertia tensor of a Y-aligned capsule
/// with the given `mass`, `radius`, and cylinder `half_length`.
#[inline]
pub fn inertia_capsule(mass: f32, radius: f32, half_length: f32) -> Vec3 {
    if mass <= 0.0 || radius <= 0.0 {
        return Vec3::ZERO;
    }

    let r = radius;
    let hl = half_length.max(0.0);
    let h = 2.0 * hl;

    let vol_cyl = std::f32::consts::PI * r * r * h;
    let vol_sph = (4.0 / 3.0) * std::f32::consts::PI * r * r * r;
    let total_vol = vol_cyl + vol_sph;
    if total_vol <= 1e-12 {
        return Vec3::ZERO;
    }

    let m_cyl = mass * (vol_cyl / total_vol);
    let m_sph = mass * (vol_sph / total_vol);

    let r2 = r * r;
    let iy = 0.5 * m_cyl * r2 + 0.4 * m_sph * r2;
    let ixz = m_cyl * (0.25 * r2 + (h * h) / 12.0)
        + m_sph * (0.4 * r2 + hl * hl + 0.75 * hl * r);

    Vec3::new(ixz, iy, ixz)
}

/// Computes the local-space diagonal inertia tensor of an axis-aligned box
/// with the given `mass` and `half_extents`.
#[inline]
pub fn inertia_box(mass: f32, half_extents: Vec3) -> Vec3 {
    if mass <= 0.0 {
        return Vec3::ZERO;
    }

    let hx2 = half_extents.x * half_extents.x;
    let hy2 = half_extents.y * half_extents.y;
    let hz2 = half_extents.z * half_extents.z;
    let factor = mass / 3.0;

    Vec3::new(
        factor * (hy2 + hz2),
        factor * (hx2 + hz2),
        factor * (hx2 + hy2),
    )
}

/// Computes the local-space diagonal inertia tensor of a solid sphere
/// with the given `mass` and `radius`.
#[inline]
pub fn inertia_sphere(mass: f32, radius: f32) -> Vec3 {
    if mass <= 0.0 || radius <= 0.0 {
        return Vec3::ZERO;
    }
    Vec3::splat(0.4 * mass * radius * radius)
}
