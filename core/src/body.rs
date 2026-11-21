//! Rigid body representation, collider shapes, and XPBD generalized inertia helpers.

use glam::{Quat, Vec3};

use crate::math::{inertia_box, inertia_capsule, inertia_sphere};

/// Geometric collider shapes supported by ARSA rigid bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    /// Local Y-aligned capsule with hemispherical caps of `radius` and cylinder `half_length`.
    Capsule { radius: f32, half_length: f32 },
    /// Oriented bounding box with local `half_extents`.
    Box { half_extents: Vec3 },
    /// Sphere with `radius`.
    Sphere { radius: f32 },
}

impl ColliderShape {
    /// Computes the local diagonal inertia tensor for this shape with `mass`.
    pub fn compute_inertia(&self, mass: f32) -> Vec3 {
        match *self {
            ColliderShape::Capsule {
                radius,
                half_length,
            } => inertia_capsule(mass, radius, half_length),
            ColliderShape::Box { half_extents } => inertia_box(mass, half_extents),
            ColliderShape::Sphere { radius } => inertia_sphere(mass, radius),
        }
    }
}

/// Dynamic or kinematic rigid body simulated by the XPBD solver.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub position: Vec3,
    pub orientation: Quat,
    pub prev_position: Vec3,
    pub prev_orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub local_inertia: Vec3,
    pub inv_local_inertia: Vec3,
    pub collider: ColliderShape,
    pub friction: f32,
    pub restitution: f32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub ragdoll_id: Option<usize>,
    pub bone_id: Option<u8>,
}

impl RigidBody {
    /// Creates a new `RigidBody` with inertia derived automatically from `mass` and `collider`.
    /// Passing `mass <= 0.0` creates an immovable/kinematic body.
    pub fn new(position: Vec3, orientation: Quat, mass: f32, collider: ColliderShape) -> Self {
        let q = orientation.normalize();
        let (safe_mass, inv_mass, local_inertia, inv_local_inertia) = if mass > 0.0 {
            let inertia = collider.compute_inertia(mass).max(Vec3::splat(1e-6));
            (mass, 1.0 / mass, inertia, Vec3::ONE / inertia)
        } else {
            (0.0, 0.0, Vec3::ZERO, Vec3::ZERO)
        };

        Self {
            position,
            orientation: q,
            prev_position: position,
            prev_orientation: q,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: safe_mass,
            inv_mass,
            local_inertia,
            inv_local_inertia,
            collider,
            friction: 0.65,
            restitution: 0.05,
            collision_group: 1,
            collision_mask: u32::MAX,
            ragdoll_id: None,
            bone_id: None,
        }
    }

    /// Multiplies a world-space vector `v` by the world-space inverse inertia tensor `I^-1_world`.
    #[inline]
    pub fn inv_inertia_world_mul(&self, v: Vec3) -> Vec3 {
        if self.inv_mass == 0.0 {
            return Vec3::ZERO;
        }
        let local_v = self.orientation.conjugate() * v;
        self.orientation * (local_v * self.inv_local_inertia)
    }

    /// Generalized inverse mass `w = 1/m + (r x n)^T I^-1 (r x n)` for positional constraints
    /// acting at offset `r` (world vector from body center of mass) along unit direction `n`.
    #[inline]
    pub fn generalized_inv_mass(&self, r: Vec3, n: Vec3) -> f32 {
        if self.inv_mass == 0.0 {
            return 0.0;
        }
        let rxn = r.cross(n);
        self.inv_mass + rxn.dot(self.inv_inertia_world_mul(rxn))
    }

    /// Generalized angular inverse mass `axis^T I^-1 axis` for pure orientation constraints.
    #[inline]
    pub fn generalized_angular_inv_mass(&self, axis: Vec3) -> f32 {
        if self.inv_mass == 0.0 {
            return 0.0;
        }
        axis.dot(self.inv_inertia_world_mul(axis))
    }

    /// Applies an XPBD positional impulse `corr` (`Δλ * n`) at lever arm `r` from center of mass.
    #[inline]
    pub fn apply_pos_correction(&mut self, corr: Vec3, r: Vec3) {
        if self.inv_mass == 0.0 {
            return;
        }
        self.position += corr * self.inv_mass;
        let delta_rot = self.inv_inertia_world_mul(r.cross(corr));
        self.apply_ang_correction(delta_rot);
    }

    /// Applies a world-space incremental rotation vector `delta_rot` (axis * angle in radians)
    /// to the body's orientation.
    #[inline]
    pub fn apply_ang_correction(&mut self, delta_rot: Vec3) {
        if self.inv_mass == 0.0 {
            return;
        }
        let angle_sq = delta_rot.length_squared();
        if angle_sq < 1e-16 {
            return;
        }
        let dq = Quat::from_scaled_axis(delta_rot);
        self.orientation = (dq * self.orientation).normalize();
    }

    /// Transforms a body-local point into world space.
    #[inline]
    pub fn world_point(&self, local: Vec3) -> Vec3 {
        self.position + self.orientation * local
    }

    /// Computes the linear velocity of the body at a given `world_point`.
    #[inline]
    pub fn velocity_at_point(&self, world_point: Vec3) -> Vec3 {
        let r = world_point - self.position;
        self.linear_velocity + self.angular_velocity.cross(r)
    }
}
