//! Articulated XPBD joints: Hinge and Spherical (elliptical swing cone + twist),
//! with Phase-2-ready PD muscle motor fields.

use std::f32::consts::{PI, TAU};

use glam::{Quat, Vec3};

use crate::body::RigidBody;
use crate::math::swing_twist_decomposition;

/// Constraint geometry and anatomical range of motion for a joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKind {
    /// Single-axis revolute joint (e.g. knees, elbows).
    /// `axis` is expressed in the joint frame (typically `Vec3::X`).
    Hinge {
        axis: Vec3,
        min_angle: f32,
        max_angle: f32,
    },
    /// 3-DOF ball-and-socket joint with elliptical swing cone around joint local `+Y`
    /// and axial twist limits around `+Y`.
    Spherical {
        swing_x_limit: f32,
        swing_z_limit: f32,
        twist_min: f32,
        twist_max: f32,
    },
}

/// Articulated joint connecting a `parent` and `child` rigid body.
#[derive(Debug, Clone)]
pub struct Joint {
    pub parent: usize,
    pub child: usize,
    pub anchor_parent: Vec3,
    pub anchor_child: Vec3,
    pub frame_parent: Quat,
    pub frame_child: Quat,
    pub kind: JointKind,
    pub compliance: f32,
    pub damping: f32,
    // Phase-2-ready active muscle motor parameters
    pub motor_target: Quat,
    pub motor_stiffness: f32,
    pub motor_damping: f32,
    pub max_torque: f32,
    pub damage_scale: f32,
}

impl Joint {
    /// Creates a new joint with default zero compliance, moderate viscous damping,
    /// and disabled Phase-2 muscle motors.
    pub fn new(
        parent: usize,
        child: usize,
        anchor_parent: Vec3,
        anchor_child: Vec3,
        kind: JointKind,
    ) -> Self {
        Self {
            parent,
            child,
            anchor_parent,
            anchor_child,
            frame_parent: Quat::IDENTITY,
            frame_child: Quat::IDENTITY,
            kind,
            compliance: 0.0,
            damping: 6.0,
            motor_target: Quat::IDENTITY,
            motor_stiffness: 0.0,
            motor_damping: 0.0,
            max_torque: 0.0,
            damage_scale: 1.0,
        }
    }

    /// Builder helper to override joint local frames.
    pub fn with_frames(mut self, frame_parent: Quat, frame_child: Quat) -> Self {
        self.frame_parent = frame_parent.normalize();
        self.frame_child = frame_child.normalize();
        self
    }

    /// Builder helper to set XPBD compliance and angular damping.
    pub fn with_solver_params(mut self, compliance: f32, damping: f32) -> Self {
        self.compliance = compliance;
        self.damping = damping;
        self
    }

    /// Projects the ball-and-socket anchor positions together via XPBD.
    pub fn solve_position(&self, bodies: &mut [RigidBody], dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        if bodies[self.parent].is_sleeping && bodies[self.child].is_sleeping {
            return;
        }
        let p_a = bodies[self.parent].world_point(self.anchor_parent);
        let p_b = bodies[self.child].world_point(self.anchor_child);
        let delta = p_a - p_b;
        let dist = delta.length();
        if dist < 1e-7 {
            return;
        }

        let n = delta / dist;
        let r_a = p_a - bodies[self.parent].position;
        let r_b = p_b - bodies[self.child].position;

        let w_a = bodies[self.parent].generalized_inv_mass(r_a, n);
        let w_b = bodies[self.child].generalized_inv_mass(r_b, n);
        let w = w_a + w_b;
        if w < 1e-12 {
            return;
        }

        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = -dist / (w + alpha_tilde);
        let impulse = n * delta_lambda;

        bodies[self.parent].apply_pos_correction(impulse, r_a);
        bodies[self.child].apply_pos_correction(-impulse, r_b);
    }

    /// Enforces hinge axis alignment + angle clamp, or spherical swing-twist limits.
    pub fn solve_limits(&self, bodies: &mut [RigidBody], dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        if bodies[self.parent].is_sleeping && bodies[self.child].is_sleeping {
            return;
        }
        let alpha_tilde = self.compliance / (dt * dt);
        match self.kind {
            JointKind::Hinge {
                axis,
                min_angle,
                max_angle,
            } => {
                let axis_norm = if axis.length_squared() > 1e-12 {
                    axis.normalize()
                } else {
                    Vec3::X
                };

                // 1. Align hinge axes (stiff structural constraint)
                let q_a = bodies[self.parent].orientation * self.frame_parent;
                let q_b = bodies[self.child].orientation * self.frame_child;
                let a_axis_w = q_a * axis_norm;
                let b_axis_w = q_b * axis_norm;

                let dot = a_axis_w.dot(b_axis_w).clamp(-1.0, 1.0);
                let angle_err = dot.acos();
                if angle_err > 1e-6 {
                    let cross = b_axis_w.cross(a_axis_w);
                    let rot_axis = if cross.length_squared() > 1e-10 {
                        cross.normalize()
                    } else {
                        any_perpendicular(a_axis_w)
                    };
                    // Use zero compliance for the structural hinge-axis lock so knees/elbows
                    // do not twist sideways under load.
                    apply_angular_xpbd_pair(
                        bodies,
                        self.parent,
                        self.child,
                        rot_axis,
                        angle_err,
                        0.0,
                    );
                }

                // 2. Clamp hinge angle within [min_angle, max_angle]
                let angle = self.measure_hinge_angle(bodies);
                let violation = if angle < min_angle {
                    angle - min_angle
                } else if angle > max_angle {
                    angle - max_angle
                } else {
                    0.0
                };

                if violation.abs() > 1e-6 {
                    let q_a_now = bodies[self.parent].orientation * self.frame_parent;
                    let n_axis = q_a_now * axis_norm;
                    // When violation > 0, child must rotate by -violation around n_axis.
                    apply_angular_xpbd_pair(
                        bodies,
                        self.parent,
                        self.child,
                        n_axis,
                        -violation,
                        alpha_tilde,
                    );
                }
            }
            JointKind::Spherical {
                swing_x_limit,
                swing_z_limit,
                twist_min,
                twist_max,
            } => {
                // 1. Elliptical swing cone around local +Y
                let q_a = bodies[self.parent].orientation * self.frame_parent;
                let q_b = bodies[self.child].orientation * self.frame_child;
                let q_rel = (q_a.conjugate() * q_b).normalize();
                let (mut swing, _) = swing_twist_decomposition(q_rel, Vec3::Y);
                if swing.w < 0.0 {
                    swing = -swing;
                }

                let swing_vec = swing.to_scaled_axis();
                let lx = swing_x_limit.max(1e-4);
                let lz = swing_z_limit.max(1e-4);
                let ell_sq = (swing_vec.x / lx).powi(2) + (swing_vec.z / lz).powi(2);

                if ell_sq > 1.0 {
                    let ell = ell_sq.sqrt();
                    let clamped_vec = Vec3::new(swing_vec.x / ell, 0.0, swing_vec.z / ell);
                    let swing_clamped = Quat::from_scaled_axis(clamped_vec);
                    let mut delta_swing_local = swing_clamped * swing.conjugate();
                    if delta_swing_local.w < 0.0 {
                        delta_swing_local = -delta_swing_local;
                    }
                    let err_local = delta_swing_local.to_scaled_axis();
                    let err_world = q_a * err_local;
                    let err_angle = err_world.length();
                    if err_angle > 1e-6 {
                        apply_angular_xpbd_pair(
                            bodies,
                            self.parent,
                            self.child,
                            err_world / err_angle,
                            err_angle,
                            alpha_tilde,
                        );
                    }
                }

                // 2. Axial twist clamp around child's local +Y
                let q_a_now = bodies[self.parent].orientation * self.frame_parent;
                let q_b_now = bodies[self.child].orientation * self.frame_child;
                let q_rel_now = (q_a_now.conjugate() * q_b_now).normalize();
                let (_, twist) = swing_twist_decomposition(q_rel_now, Vec3::Y);
                let raw_twist_angle = 2.0 * twist.y.atan2(twist.w);
                let mid = 0.5 * (twist_min + twist_max);
                let twist_angle = unwrap_near(raw_twist_angle, mid);

                let twist_violation = if twist_angle >= twist_min && twist_angle <= twist_max {
                    0.0
                } else {
                    let d_min = diff_angle(raw_twist_angle, twist_min);
                    let d_max = diff_angle(raw_twist_angle, twist_max);
                    if d_min.abs() < d_max.abs() {
                        d_min
                    } else {
                        d_max
                    }
                };

                if twist_violation.abs() > 1e-6 {
                    let twist_axis_w = q_b_now * Vec3::Y;
                    apply_angular_xpbd_pair(
                        bodies,
                        self.parent,
                        self.child,
                        twist_axis_w,
                        -twist_violation,
                        alpha_tilde,
                    );
                }
            }
        }
    }

    /// Measures the current signed hinge angle in radians (unwrapped around the limit midpoint).
    /// Returns `0.0` if the joint is not a `JointKind::Hinge`.
    pub fn measure_hinge_angle(&self, bodies: &[RigidBody]) -> f32 {
        let JointKind::Hinge {
            axis,
            min_angle,
            max_angle,
        } = self.kind
        else {
            return 0.0;
        };

        let axis_norm = if axis.length_squared() > 1e-12 {
            axis.normalize()
        } else {
            Vec3::X
        };
        let ref_local = hinge_reference_vector(axis_norm);

        let q_a = bodies[self.parent].orientation * self.frame_parent;
        let q_b = bodies[self.child].orientation * self.frame_child;

        let n_axis = q_a * axis_norm;
        let a_ref = q_a * ref_local;
        let b_ref = q_b * ref_local;

        let b_proj = b_ref - n_axis * b_ref.dot(n_axis);
        let len_sq = b_proj.length_squared();
        if len_sq < 1e-10 {
            return 0.0;
        }
        let b_proj_n = b_proj / len_sq.sqrt();
        let sin_a = a_ref.cross(b_proj_n).dot(n_axis);
        let cos_a = a_ref.dot(b_proj_n);
        let raw_angle = sin_a.atan2(cos_a);
        let mid = 0.5 * (min_angle + max_angle);
        unwrap_near(raw_angle, mid)
    }

    /// Smoothly damps relative angular velocity across the joint for non-jittery, fleshy motion.
    pub fn solve_velocity_damping(&self, bodies: &mut [RigidBody], dt: f32) {
        if self.damping <= 0.0
            || dt <= 0.0
            || (bodies[self.parent].is_sleeping && bodies[self.child].is_sleeping)
        {
            return;
        }

        let rel_omega = bodies[self.child].angular_velocity - bodies[self.parent].angular_velocity;
        let speed = rel_omega.length();
        if speed < 1e-6 {
            return;
        }

        let n = rel_omega / speed;
        let w_a = bodies[self.parent].generalized_angular_inv_mass(n);
        let w_b = bodies[self.child].generalized_angular_inv_mass(n);
        let w = w_a + w_b;
        if w < 1e-12 {
            return;
        }

        let damp_frac = 1.0 - (-self.damping * dt).exp();
        let j = -(speed * damp_frac) / w;
        let impulse = n * j;

        let dv_a = bodies[self.parent].inv_inertia_world_mul(-impulse);
        let dv_b = bodies[self.child].inv_inertia_world_mul(impulse);
        bodies[self.parent].angular_velocity += dv_a;
        bodies[self.child].angular_velocity += dv_b;
    }

    /// Applies a torque-limited PD muscle motor driving the child joint frame toward `motor_target`.
    pub fn apply_muscle_motor(&self, bodies: &mut [RigidBody], dt: f32) {
        if self.motor_stiffness <= 0.0
            || self.max_torque <= 0.0
            || self.damage_scale <= 0.0
            || dt <= 0.0
            || (bodies[self.parent].is_sleeping && bodies[self.child].is_sleeping)
        {
            return;
        }

        let q_p = bodies[self.parent].orientation * self.frame_parent;
        let q_c = bodies[self.child].orientation * self.frame_child;
        let q_des = (q_p * self.motor_target).normalize();

        let mut q_err = q_des * q_c.conjugate();
        if q_err.w < 0.0 {
            q_err = -q_err;
        }

        let angle = 2.0 * q_err.w.clamp(-1.0, 1.0).acos();
        let err_vec = if angle >= 1e-5 {
            let axis = Vec3::new(q_err.x, q_err.y, q_err.z).normalize_or_zero();
            axis * angle
        } else {
            Vec3::ZERO
        };

        let rel_omega = bodies[self.child].angular_velocity - bodies[self.parent].angular_velocity;
        let k_eff = self.motor_stiffness * self.damage_scale;
        let d_eff = self.motor_damping * self.damage_scale;
        let mut raw_tau = err_vec * k_eff - rel_omega * d_eff;

        if let JointKind::Hinge { axis, .. } = self.kind {
            let axis_w = (q_p * axis).normalize_or_zero();
            raw_tau = axis_w * raw_tau.dot(axis_w);
        }

        let raw_mag = raw_tau.length();
        if raw_mag < 1e-9 {
            return;
        }
        let dir = raw_tau / raw_mag;

        let w_ang = bodies[self.parent].generalized_angular_inv_mass(dir)
            + bodies[self.child].generalized_angular_inv_mass(dir);
        if w_ang < 1e-12 {
            return;
        }

        // Implicit / Stable-PD denominator along `dir` prevents stiffness/damping overshoot.
        let denom = 1.0 + w_ang * dt * (d_eff + dt * k_eff);
        let max_t = self.max_torque * self.damage_scale;
        let mut tau_mag = (raw_mag / denom).min(max_t);

        // Stability guard: prevent light bones (Hand/Foot) from exceeding 2.5 rad/s per substep.
        let delta_omega = tau_mag * dt * w_ang;
        if delta_omega > 2.5 {
            tau_mag = 2.5 / (dt * w_ang);
        }

        let impulse = dir * (tau_mag * dt);
        let dv_p = bodies[self.parent].inv_inertia_world_mul(-impulse);
        let dv_c = bodies[self.child].inv_inertia_world_mul(impulse);
        bodies[self.parent].angular_velocity += dv_p;
        bodies[self.child].angular_velocity += dv_c;
    }
}

/// Applies an XPBD angular correction rotating `child` by `+delta_angle` relative to `parent`
/// around world unit `axis` (so `parent` moves `-` and `child` moves `+`).
fn apply_angular_xpbd_pair(
    bodies: &mut [RigidBody],
    parent: usize,
    child: usize,
    axis: Vec3,
    delta_angle: f32,
    alpha_tilde: f32,
) {
    let w_a = bodies[parent].generalized_angular_inv_mass(axis);
    let w_b = bodies[child].generalized_angular_inv_mass(axis);
    let w = w_a + w_b;
    if w < 1e-12 {
        return;
    }

    let delta_lambda = delta_angle / (w + alpha_tilde);
    let ang_impulse = axis * delta_lambda;

    let d_a = bodies[parent].inv_inertia_world_mul(-ang_impulse);
    let d_b = bodies[child].inv_inertia_world_mul(ang_impulse);
    bodies[parent].apply_ang_correction(d_a);
    bodies[child].apply_ang_correction(d_b);
}

#[inline]
fn hinge_reference_vector(axis_norm: Vec3) -> Vec3 {
    let candidate = if axis_norm.y.abs() < 0.9 {
        Vec3::Y - axis_norm * axis_norm.y
    } else {
        Vec3::Z - axis_norm * axis_norm.z
    };
    candidate.normalize()
}

#[inline]
fn any_perpendicular(n: Vec3) -> Vec3 {
    let seed = if n.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
    n.cross(seed).normalize()
}

#[inline]
fn diff_angle(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    d
}

#[inline]
fn unwrap_near(angle: f32, target: f32) -> f32 {
    let mut diff = angle - target;
    while diff > PI {
        diff -= TAU;
    }
    while diff < -PI {
        diff += TAU;
    }
    target + diff
}
