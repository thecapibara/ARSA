//! Active balance, state machine, and capture-point stepping controller.

mod recovery;

use glam::{Quat, Vec3};

use crate::body::RigidBody;
use crate::joint::Joint;
use crate::skeleton::{BoneId, JointId, RagdollInstance};

/// Behavioral balance state of an active ragdoll.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorState {
    Balanced = 0,
    Stumbling = 1,
    Falling = 2,
    Downed = 3,
    Passive = 4,
    Unconscious = 5,
    Recovering = 6,
}

/// Per-ragdoll balance telemetry and capture-point stepping state.
#[derive(Debug, Clone)]
pub struct BalanceState {
    pub state: BehaviorState,
    pub com_pos: Vec3,
    pub com_vel: Vec3,
    pub capture_point: Vec3,
    pub left_foot_grounded: bool,
    pub right_foot_grounded: bool,
    pub airborne_time: f32,
    pub fallen_time: f32,
    pub downed_still_time: f32,
    pub recovery_phase: f32,
    pub recovery_prone: bool,
    pub step_active: bool,
    pub swing_left: bool,
    pub step_phase: f32,
    pub step_target_xz: Vec3,
    pub idle_time: f32,
}

impl Default for BalanceState {
    fn default() -> Self {
        Self {
            state: BehaviorState::Balanced,
            com_pos: Vec3::ZERO,
            com_vel: Vec3::ZERO,
            capture_point: Vec3::ZERO,
            left_foot_grounded: false,
            right_foot_grounded: false,
            airborne_time: 0.0,
            fallen_time: 0.0,
            downed_still_time: 0.0,
            recovery_phase: 0.0,
            recovery_prone: false,
            step_active: false,
            swing_left: false,
            step_phase: 0.0,
            step_target_xz: Vec3::ZERO,
            idle_time: 0.0,
        }
    }
}

/// Updates active balance state, muscle gains/targets, capture-point stepping, and
/// torque-limited virtual upright & CoM support for a humanoid ragdoll.
pub fn update_ragdoll_balance(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    foot_contacts: (bool, bool),
    gravity_mag: f32,
    quake_mag: f32,
    dt: f32,
) {
    if dt <= 0.0 {
        return;
    }

    // 1. Exact mass-weighted Center of Mass position & velocity across all 17 bones
    let mut total_mass = 0.0_f32;
    let mut com_pos = Vec3::ZERO;
    let mut com_vel = Vec3::ZERO;
    for &idx in &ragdoll.body_indices {
        let b = &bodies[idx];
        total_mass += b.mass;
        com_pos += b.position * b.mass;
        com_vel += b.linear_velocity * b.mass;
    }
    if total_mass > 1e-6 {
        com_pos /= total_mass;
        com_vel /= total_mass;
    }
    ragdoll.balance.com_pos = com_pos;
    ragdoll.balance.com_vel = com_vel;

    // 2. Foot contacts & local ground plane height
    ragdoll.balance.left_foot_grounded = foot_contacts.0;
    ragdoll.balance.right_foot_grounded = foot_contacts.1;
    let grounded_count = (foot_contacts.0 as u8) + (foot_contacts.1 as u8);

    let s_h = ragdoll.profile.height_m / 1.78;
    let s_r = ((ragdoll.profile.mass_kg / 75.0) / s_h)
        .sqrt()
        .clamp(0.72, 1.55);
    let s_w = 0.45 * s_h + 0.55 * s_r;
    let foot_half_y = 0.035 * s_h;

    let lf_pos = bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]].position;
    let rf_pos = bodies[ragdoll.body_indices[BoneId::RightFoot as usize]].position;
    let foot_ground_y = match foot_contacts {
        (true, true) => 0.5 * (lf_pos.y + rf_pos.y) - foot_half_y,
        (true, false) => lf_pos.y - foot_half_y,
        (false, true) => rf_pos.y - foot_half_y,
        (false, false) => (lf_pos.y.min(rf_pos.y) - foot_half_y).max(0.0),
    };

    // 3. Linear Inverted Pendulum Capture Point
    let pelvis_idx = ragdoll.body_indices[BoneId::Pelvis as usize];
    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let pelvis_pos = bodies[pelvis_idx].position;
    let pelvis_h = (pelvis_pos.y - foot_ground_y).max(0.25);
    let t_pend = (pelvis_h / gravity_mag.max(1.0)).sqrt();
    let cp_xz = Vec3::new(
        com_pos.x + com_vel.x * t_pend,
        foot_ground_y,
        com_pos.z + com_vel.z * t_pend,
    );
    ragdoll.balance.capture_point = cp_xz;

    // 4. Passive or unconscious early out
    if !ragdoll.active_enabled || ragdoll.reflex.unconscious {
        ragdoll.balance.state = if !ragdoll.active_enabled {
            BehaviorState::Passive
        } else {
            BehaviorState::Unconscious
        };
        ragdoll.balance.step_active = false;
        ragdoll.balance.downed_still_time = 0.0;
        ragdoll.balance.recovery_phase = 0.0;
        for &j_idx in &ragdoll.joint_indices {
            joints[j_idx].motor_stiffness = 0.0;
            joints[j_idx].motor_damping = 0.0;
            joints[j_idx].max_torque = 0.0;
        }
        return;
    }

    // 5. Auto-scale muscle gains from anthropometric profile
    let m_ratio = ragdoll.profile.mass_kg / 75.0;
    let h_ratio = s_h;
    let mu = ragdoll.profile.muscle_ratio;
    let grav_scale = m_ratio * h_ratio * mu;
    let damp_scale = (m_ratio * h_ratio * h_ratio * mu).sqrt();

    assign_nominal_gains(ragdoll, joints, grav_scale, damp_scale);

    // 6. State transitions
    if grounded_count == 0 {
        ragdoll.balance.airborne_time += dt;
    } else {
        ragdoll.balance.airborne_time = 0.0;
    }
    ragdoll.balance.idle_time += dt;

    let up_dot = (bodies[pelvis_idx].orientation * Vec3::Y).y;
    let chest_up_dot = (bodies[chest_idx].orientation * Vec3::Y).y;
    let pelvis_rel_y = pelvis_pos.y - foot_ground_y;

    let mid_feet_xz = Vec3::new(
        0.5 * (lf_pos.x + rf_pos.x),
        foot_ground_y,
        0.5 * (lf_pos.z + rf_pos.z),
    );
    let com_vel_xz = Vec3::new(com_vel.x, 0.0, com_vel.z);
    let cp_offset = Vec3::new(cp_xz.x - mid_feet_xz.x, 0.0, cp_xz.z - mid_feet_xz.z).length();

    let quake_crouch = if quake_mag >= 3.0 {
        ((quake_mag - 3.0) / 5.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let can_recover = ragdoll.auto_get_up
        && ragdoll.active_enabled
        && !ragdoll.reflex.unconscious
        && ragdoll.reflex.health > 18.0;

    let next_state = if ragdoll.balance.state == BehaviorState::Recovering {
        if !can_recover || com_vel.length() > 2.6 {
            ragdoll.balance.recovery_phase = 0.0;
            ragdoll.balance.downed_still_time = 0.0;
            BehaviorState::Falling
        } else {
            let recovery_duration = 2.1
                * s_h.sqrt()
                * (if ragdoll.reflex.health < 65.0 {
                    1.35
                } else {
                    1.0
                });
            ragdoll.balance.recovery_phase += dt / recovery_duration;
            if ragdoll.balance.recovery_phase >= 1.0 {
                ragdoll.balance.recovery_phase = 0.0;
                ragdoll.balance.downed_still_time = 0.0;
                ragdoll.balance.fallen_time = 0.0;
                let upright_and_high = pelvis_rel_y >= 0.68 * ragdoll.standing_pelvis_y
                    && chest_up_dot >= 0.65
                    && up_dot >= 0.65;
                if upright_and_high && cp_offset <= 0.16 * s_h && com_vel_xz.length() <= 0.45 {
                    BehaviorState::Balanced
                } else {
                    BehaviorState::Stumbling
                }
            } else {
                BehaviorState::Recovering
            }
        }
    } else if pelvis_rel_y < 0.56 * ragdoll.standing_pelvis_y
        || chest_up_dot < 0.45
        || up_dot < 0.35
    {
        ragdoll.balance.fallen_time += dt;
        if pelvis_rel_y < 0.40 * ragdoll.standing_pelvis_y && com_vel.length() < 0.6 {
            if ragdoll.balance.state == BehaviorState::Downed && com_vel.length() < 0.40 {
                ragdoll.balance.downed_still_time += dt;
            } else {
                ragdoll.balance.downed_still_time = 0.0;
            }
            let required_delay =
                if ragdoll.reflex.active_wound.is_some() || ragdoll.reflex.health < 75.0 {
                    2.8
                } else {
                    1.5
                };
            if can_recover && ragdoll.balance.downed_still_time >= required_delay {
                ragdoll.balance.recovery_phase = 0.0;
                ragdoll.balance.recovery_prone = (bodies[chest_idx].orientation * Vec3::Z).y < 0.0;
                BehaviorState::Recovering
            } else {
                BehaviorState::Downed
            }
        } else {
            ragdoll.balance.downed_still_time = 0.0;
            BehaviorState::Falling
        }
    } else if grounded_count == 0 && ragdoll.balance.airborne_time > 0.25 {
        ragdoll.balance.fallen_time += dt;
        ragdoll.balance.downed_still_time = 0.0;
        BehaviorState::Falling
    } else {
        ragdoll.balance.fallen_time = 0.0;
        ragdoll.balance.downed_still_time = 0.0;
        let cp_thresh = (0.16 - 0.08 * quake_crouch) * s_h;
        let vel_thresh = 0.45 - 0.23 * quake_crouch;
        if cp_offset > cp_thresh || com_vel_xz.length() > vel_thresh {
            BehaviorState::Stumbling
        } else {
            BehaviorState::Balanced
        }
    };
    ragdoll.balance.state = next_state;

    // 7. Behavioral control per state
    match next_state {
        BehaviorState::Balanced | BehaviorState::Stumbling => {
            let pelvis_fwd = bodies[pelvis_idx].orientation * Vec3::Z;
            let fwd_xz = Vec3::new(pelvis_fwd.x, 0.0, pelvis_fwd.z).normalize_or_zero();
            let fwd_xz = if fwd_xz.length_squared() > 1e-6 {
                fwd_xz
            } else {
                Vec3::Z
            };
            let left_xz = Vec3::Y.cross(fwd_xz).normalize();
            let yaw_inv = Quat::from_rotation_y(fwd_xz.x.atan2(fwd_xz.z)).conjugate();

            // Express horizontal CoM velocity in pelvis yaw frame (local +Z = fwd, +X = left)
            let local_vel = yaw_inv * com_vel_xz;
            let sway = (ragdoll.balance.idle_time * 1.6).sin() * 0.025;
            let counter_pitch = (-local_vel.z * 0.10).clamp(-0.22, 0.22);
            let counter_roll = (local_vel.x * 0.10).clamp(-0.20, 0.20);

            // Spine, chest, neck, head (forward bracing crouch lean during earthquakes)
            let quake_lean = 0.08 * quake_crouch;
            set_target(
                ragdoll,
                joints,
                JointId::LowerSpine,
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    counter_pitch * 0.5 + sway + quake_lean,
                    0.0,
                    counter_roll * 0.5,
                ),
            );
            set_target(
                ragdoll,
                joints,
                JointId::ChestSpine,
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    counter_pitch * 0.6 + sway * 0.5 + quake_lean,
                    0.0,
                    counter_roll * 0.6,
                ),
            );
            set_target(
                ragdoll,
                joints,
                JointId::Neck,
                Quat::from_rotation_x(-counter_pitch * 0.4 - quake_lean * 0.6),
            );
            set_target(
                ragdoll,
                joints,
                JointId::Head,
                Quat::from_rotation_x(-counter_pitch * 0.3 - quake_lean * 0.6),
            );

            // Arms: relaxed standing bend, flaring outward/opposite velocity when stumbling or bracing in quake
            let is_stumbling = next_state == BehaviorState::Stumbling;
            let mut flare = if is_stumbling { 0.48 } else { 0.10 };
            flare += 0.55 * quake_crouch;
            let arm_pitch_swing = (-local_vel.z * 0.25).clamp(-0.55, 0.55);
            let elbow_bend = if is_stumbling || quake_mag >= 3.0 {
                -0.55
            } else {
                -0.25
            };

            // Left arm (+X): +Z rotation flares outward
            set_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    arm_pitch_swing,
                    0.0,
                    flare + (-local_vel.x * 0.18).clamp(-0.35, 0.35),
                ),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(elbow_bend),
            );
            set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);

            // Right arm (-X): -Z rotation flares outward
            set_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    arm_pitch_swing,
                    0.0,
                    -flare + (-local_vel.x * 0.18).clamp(-0.35, 0.35),
                ),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightElbow,
                Quat::from_rotation_x(elbow_bend),
            );
            set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

            // Legs: capture-point stepping vs stance
            if is_stumbling && grounded_count > 0 {
                if !ragdoll.balance.step_active {
                    let swing_left = match foot_contacts {
                        (false, true) => true,
                        (true, false) => false,
                        _ => (cp_xz - mid_feet_xz).dot(left_xz) >= 0.0,
                    };
                    ragdoll.balance.step_active = true;
                    ragdoll.balance.swing_left = swing_left;
                    ragdoll.balance.step_phase = 0.0;
                }

                let step_duration = 0.34 * s_h.sqrt();
                ragdoll.balance.step_phase += dt / step_duration;

                // Update step target in XZ around pelvis
                let side_sign = if ragdoll.balance.swing_left {
                    1.0
                } else {
                    -1.0
                };
                let lat_offset = left_xz * (side_sign * (0.095 + 0.06 * quake_crouch) * s_w);
                let raw_target = cp_xz + com_vel_xz * 0.12 + lat_offset;
                let pelvis_xz = Vec3::new(pelvis_pos.x, foot_ground_y, pelvis_pos.z);
                let mut rel_target = raw_target - pelvis_xz;
                rel_target.y = 0.0;
                let max_stride = 0.65 * s_h;
                if rel_target.length() > max_stride {
                    rel_target = rel_target.normalize() * max_stride;
                }
                ragdoll.balance.step_target_xz = pelvis_xz + rel_target;

                let fwd_comp = (rel_target.dot(fwd_xz) / (0.50 * s_h)).clamp(-1.0, 1.0);
                let lat_comp = (rel_target.dot(left_xz) / (0.40 * s_h)).clamp(-0.55, 0.55);

                let phase = ragdoll.balance.step_phase;
                let (swing_hip, swing_knee, swing_ankle) = if phase < 0.55 {
                    let lift = (phase / 0.35).clamp(0.0, 1.0);
                    (
                        Quat::from_euler(
                            glam::EulerRot::XYZ,
                            -fwd_comp * 0.65 * lift,
                            0.0,
                            lat_comp * 0.45 * lift,
                        ),
                        Quat::from_rotation_x(0.95 * lift),
                        Quat::from_rotation_x(-0.28 * lift),
                    )
                } else {
                    let reach = ((phase - 0.55) / 0.45).clamp(0.0, 1.0);
                    (
                        Quat::from_euler(
                            glam::EulerRot::XYZ,
                            -fwd_comp * (0.65 - 0.10 * reach),
                            0.0,
                            lat_comp * (0.45 - 0.10 * reach),
                        ),
                        Quat::from_rotation_x(0.95 * (1.0 - reach) + 0.12 * reach),
                        Quat::from_rotation_x(-0.28 * (1.0 - reach) - 0.05 * reach),
                    )
                };

                // Stance leg keeps athletic knee bend & slight counter-brace (deepens during earthquakes)
                let stance_hip_x = -0.05 - 0.28 * quake_crouch;
                let stance_knee_x = 0.10 + 0.36 * quake_crouch;
                let stance_ankle_x = -0.05 - 0.08 * quake_crouch;
                let stance_hip = Quat::from_rotation_x(stance_hip_x);
                let stance_knee = Quat::from_rotation_x(stance_knee_x);
                let stance_ankle = Quat::from_rotation_x(stance_ankle_x);

                if ragdoll.balance.swing_left {
                    set_target(ragdoll, joints, JointId::LeftHip, swing_hip);
                    set_target(ragdoll, joints, JointId::LeftKnee, swing_knee);
                    set_target(ragdoll, joints, JointId::LeftAnkle, swing_ankle);
                    set_target(ragdoll, joints, JointId::RightHip, stance_hip);
                    set_target(ragdoll, joints, JointId::RightKnee, stance_knee);
                    set_target(ragdoll, joints, JointId::RightAnkle, stance_ankle);
                } else {
                    set_target(ragdoll, joints, JointId::RightHip, swing_hip);
                    set_target(ragdoll, joints, JointId::RightKnee, swing_knee);
                    set_target(ragdoll, joints, JointId::RightAnkle, swing_ankle);
                    set_target(ragdoll, joints, JointId::LeftHip, stance_hip);
                    set_target(ragdoll, joints, JointId::LeftKnee, stance_knee);
                    set_target(ragdoll, joints, JointId::LeftAnkle, stance_ankle);
                }

                if ragdoll.balance.step_phase >= 1.0 {
                    ragdoll.balance.step_active = false;
                    ragdoll.balance.step_phase = 0.0;
                }
            } else {
                ragdoll.balance.step_active = false;
                ragdoll.balance.step_phase = 0.0;
                let stance_hip_x = -0.04 - 0.28 * quake_crouch;
                let stance_knee_x = 0.08 + 0.36 * quake_crouch;
                let stance_ankle_x = -0.04 - 0.08 * quake_crouch;
                let hip_abduct = 0.08 * quake_crouch;
                let stance_knee = Quat::from_rotation_x(stance_knee_x);
                let stance_ankle = Quat::from_rotation_x(stance_ankle_x);
                set_target(
                    ragdoll,
                    joints,
                    JointId::LeftHip,
                    Quat::from_euler(glam::EulerRot::XYZ, stance_hip_x, 0.0, hip_abduct),
                );
                set_target(ragdoll, joints, JointId::LeftKnee, stance_knee);
                set_target(ragdoll, joints, JointId::LeftAnkle, stance_ankle);
                set_target(
                    ragdoll,
                    joints,
                    JointId::RightHip,
                    Quat::from_euler(glam::EulerRot::XYZ, stance_hip_x, 0.0, -hip_abduct),
                );
                set_target(ragdoll, joints, JointId::RightKnee, stance_knee);
                set_target(ragdoll, joints, JointId::RightAnkle, stance_ankle);
            }

            // Virtual upright & CoM support ONLY when grounded
            if grounded_count > 0 {
                let foot_scale = if grounded_count >= 2 { 1.0 } else { 0.55 };
                let strength_scale = foot_scale * mu;

                // Upright PD on Pelvis and Chest preserving yaw
                apply_upright_pd(
                    &mut bodies[pelvis_idx],
                    320.0 * m_ratio * h_ratio * strength_scale,
                    22.0 * damp_scale * foot_scale,
                    165.0 * m_ratio * h_ratio * strength_scale,
                    dt,
                );
                apply_upright_pd(
                    &mut bodies[chest_idx],
                    260.0 * m_ratio * h_ratio * strength_scale,
                    18.0 * damp_scale * foot_scale,
                    135.0 * m_ratio * h_ratio * strength_scale,
                    dt,
                );

                // Gentle vertical gravity-compensation support when near standing height
                let rel_frac = pelvis_rel_y / ragdoll.standing_pelvis_y.max(0.2);
                if (0.72..=1.08).contains(&rel_frac) && bodies[pelvis_idx].linear_velocity.y < 0.8 {
                    let height_fade = ((rel_frac - 0.72) / 0.12).clamp(0.0, 1.0);
                    let target_y =
                        foot_ground_y + (0.96 - 0.10 * quake_crouch) * ragdoll.standing_pelvis_y;
                    let height_err = target_y - pelvis_pos.y;
                    let pelvis = &mut bodies[pelvis_idx];
                    let fy_raw = (total_mass * gravity_mag * 0.55
                        + height_err * (total_mass * 28.0)
                        - pelvis.linear_velocity.y * (total_mass * 4.5))
                        * strength_scale
                        * height_fade;
                    let fy = fy_raw.clamp(0.0, total_mass * gravity_mag * 1.12);
                    pelvis.linear_velocity.y += (fy * dt) * pelvis.inv_mass;
                }

                // Horizontal ankle-strategy restoring force toward true ankle support polygon
                // (using ankle joint world positions, NOT forward-shifted toe box centers!)
                let la_j = &joints[ragdoll.joint_indices[JointId::LeftAnkle as usize]];
                let ra_j = &joints[ragdoll.joint_indices[JointId::RightAnkle as usize]];
                let la_pos = bodies[la_j.child].world_point(la_j.anchor_child);
                let ra_pos = bodies[ra_j.child].world_point(ra_j.anchor_child);
                let mid_ankles_xz = Vec3::new(
                    0.5 * (la_pos.x + ra_pos.x),
                    foot_ground_y,
                    0.5 * (la_pos.z + ra_pos.z),
                );
                let support_center = match foot_contacts {
                    (true, true) => mid_ankles_xz,
                    (true, false) => Vec3::new(la_pos.x, foot_ground_y, la_pos.z),
                    (false, true) => Vec3::new(ra_pos.x, foot_ground_y, ra_pos.z),
                    (false, false) => mid_ankles_xz,
                };
                let raw_horiz_err = Vec3::new(
                    support_center.x - com_pos.x,
                    0.0,
                    support_center.z - com_pos.z,
                );
                // Deadzone covering the real foot support footprint (heel-to-toe & stance width)
                // prevents phantom sliding when hunched forward (e.g. during Reach-for-Wound)
                let deadzone = 0.14 * s_h;
                let err_len = raw_horiz_err.length();
                let horiz_err = if err_len > deadzone {
                    raw_horiz_err * ((err_len - deadzone) / err_len)
                } else {
                    Vec3::ZERO
                };
                let stumble_scale = if is_stumbling { 0.35 } else { 1.0 };
                let mut f_horiz = (horiz_err * (total_mass * 6.0)
                    - com_vel_xz * (total_mass * 8.0))
                    * strength_scale
                    * stumble_scale;
                let max_f_horiz = total_mass * 2.2 * strength_scale * stumble_scale;
                let f_len = f_horiz.length();
                if f_len > max_f_horiz && f_len > 1e-6 {
                    f_horiz *= max_f_horiz / f_len;
                }
                let pelvis = &mut bodies[pelvis_idx];
                pelvis.linear_velocity += f_horiz * (dt * pelvis.inv_mass);

                // Damp idle yaw spin & plant feet firmly on the floor when Balanced (disabled during earthquakes >= 3.0)
                // (prevents "hovercraft" sliding caused by vertical pelvis support unloading foot normal_lambda)
                if !is_stumbling && quake_mag < 3.0 {
                    let foot_grip = (-35.0 * dt).exp();
                    let foot_yaw_grip = (-25.0 * dt).exp();
                    if foot_contacts.0 {
                        let lf = &mut bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]];
                        let v_xz = Vec3::new(lf.linear_velocity.x, 0.0, lf.linear_velocity.z);
                        if v_xz.length() < 0.65 {
                            lf.linear_velocity.x *= foot_grip;
                            lf.linear_velocity.z *= foot_grip;
                            lf.angular_velocity.y *= foot_yaw_grip;
                        }
                    }
                    if foot_contacts.1 {
                        let rf = &mut bodies[ragdoll.body_indices[BoneId::RightFoot as usize]];
                        let v_xz = Vec3::new(rf.linear_velocity.x, 0.0, rf.linear_velocity.z);
                        if v_xz.length() < 0.65 {
                            rf.linear_velocity.x *= foot_grip;
                            rf.linear_velocity.z *= foot_grip;
                            rf.angular_velocity.y *= foot_yaw_grip;
                        }
                    }
                    if grounded_count >= 2 {
                        let yaw_damp = (-12.0 * dt).exp();
                        bodies[pelvis_idx].angular_velocity.y *= yaw_damp;
                        bodies[chest_idx].angular_velocity.y *= yaw_damp;
                    }
                }
            }
        }
        BehaviorState::Falling => {
            ragdoll.balance.step_active = false;
            for &j_idx in &ragdoll.joint_indices {
                joints[j_idx].motor_stiffness *= 0.28;
                joints[j_idx].motor_damping *= 0.45;
                joints[j_idx].max_torque *= 0.35;
            }
            // Keep neck mildly stiff to protect head
            joints[ragdoll.joint_indices[JointId::Neck as usize]].motor_stiffness *= 1.8;
            joints[ragdoll.joint_indices[JointId::Head as usize]].motor_stiffness *= 1.8;
            set_target(ragdoll, joints, JointId::Neck, Quat::IDENTITY);
            set_target(ragdoll, joints, JointId::Head, Quat::IDENTITY);

            // Raise arms forward/downward to brace for impact
            set_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, -0.75, 0.0, 0.25),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, -0.75, 0.0, -0.25),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(-0.65),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightElbow,
                Quat::from_rotation_x(-0.65),
            );
        }
        BehaviorState::Downed => {
            ragdoll.balance.step_active = false;
            let is_wounded = ragdoll.reflex.active_wound.is_some();
            for &j_idx in &ragdoll.joint_indices {
                if is_wounded {
                    joints[j_idx].motor_stiffness *= 0.04;
                    joints[j_idx].motor_damping *= 0.25;
                    joints[j_idx].max_torque *= 0.08;
                } else {
                    joints[j_idx].motor_stiffness = 0.0;
                    joints[j_idx].motor_damping *= 0.25;
                    joints[j_idx].max_torque *= 0.08;
                }
            }
        }
        BehaviorState::Recovering => {
            ragdoll.balance.step_active = false;
            recovery::update_recovering_pose(
                ragdoll,
                bodies,
                joints,
                foot_contacts,
                foot_ground_y,
                total_mass,
                gravity_mag,
                s_h,
                m_ratio,
                h_ratio,
                mu,
                damp_scale,
                com_pos,
                com_vel_xz,
                dt,
            );
        }
        BehaviorState::Passive | BehaviorState::Unconscious => {}
    }
}

#[inline]
pub(super) fn set_target(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    joint_id: JointId,
    target: Quat,
) {
    let j_idx = ragdoll.joint_indices[joint_id as usize];
    let j = &mut joints[j_idx];
    j.motor_target = (j.frame_parent.conjugate() * target * j.frame_child).normalize();
}

fn assign_nominal_gains(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    grav_scale: f32,
    damp_scale: f32,
) {
    let mut set_gain = |jid: JointId, k: f32, d: f32, max_t: f32| {
        let j = &mut joints[ragdoll.joint_indices[jid as usize]];
        j.motor_stiffness = k * grav_scale;
        j.motor_damping = d * damp_scale;
        j.max_torque = max_t * grav_scale;
    };

    // Spine & Head
    set_gain(JointId::LowerSpine, 260.0, 14.0, 220.0);
    set_gain(JointId::ChestSpine, 240.0, 13.0, 200.0);
    set_gain(JointId::Neck, 35.0, 0.35, 30.0);
    set_gain(JointId::Head, 35.0, 0.35, 30.0);

    // Arms
    set_gain(JointId::LeftShoulder, 48.0, 0.60, 55.0);
    set_gain(JointId::RightShoulder, 48.0, 0.60, 55.0);
    set_gain(JointId::LeftElbow, 32.0, 0.90, 35.0);
    set_gain(JointId::RightElbow, 32.0, 0.90, 35.0);
    set_gain(JointId::LeftWrist, 8.0, 0.10, 10.0);
    set_gain(JointId::RightWrist, 8.0, 0.10, 10.0);

    // Legs (strongest)
    set_gain(JointId::LeftHip, 280.0, 6.0, 260.0);
    set_gain(JointId::RightHip, 280.0, 6.0, 260.0);
    set_gain(JointId::LeftKnee, 220.0, 7.0, 200.0);
    set_gain(JointId::RightKnee, 220.0, 7.0, 200.0);
    set_gain(JointId::LeftAnkle, 110.0, 0.45, 110.0);
    set_gain(JointId::RightAnkle, 110.0, 0.45, 110.0);
}

fn apply_upright_pd(body: &mut RigidBody, stiffness: f32, damping: f32, max_torque: f32, dt: f32) {
    apply_pitched_upright_pd(body, Vec3::Y, stiffness, damping, max_torque, dt);
}

pub(super) fn apply_pitched_upright_pd(
    body: &mut RigidBody,
    target_up: Vec3,
    stiffness: f32,
    damping: f32,
    max_torque: f32,
    dt: f32,
) {
    if body.inv_mass == 0.0 || stiffness <= 0.0 || max_torque <= 0.0 {
        return;
    }

    let current_up = (body.orientation * Vec3::Y).normalize_or_zero();
    let target_up_n = target_up.normalize_or_zero();
    if current_up.length_squared() < 1e-6 || target_up_n.length_squared() < 1e-6 {
        return;
    }

    let mut q_err = Quat::from_rotation_arc(current_up, target_up_n);
    if q_err.w < 0.0 {
        q_err = -q_err;
    }

    let angle = 2.0 * q_err.w.clamp(-1.0, 1.0).acos();
    let err_vec = if angle > 1e-5 {
        Vec3::new(q_err.x, q_err.y, q_err.z).normalize_or_zero() * angle
    } else {
        Vec3::ZERO
    };

    // Correct tilt perpendicular to target_up so yaw/spin around target axis turns freely
    let err_perp = err_vec - target_up_n * err_vec.dot(target_up_n);
    let omega_perp = body.angular_velocity - target_up_n * body.angular_velocity.dot(target_up_n);
    let raw_tau = err_perp * stiffness - omega_perp * damping;
    let raw_mag = raw_tau.length();
    if raw_mag < 1e-9 {
        return;
    }

    let dir = raw_tau / raw_mag;
    let w_ang = body.generalized_angular_inv_mass(dir);
    if w_ang < 1e-12 {
        return;
    }
    let denom = 1.0 + w_ang * dt * (damping + dt * stiffness);
    let mut tau_mag = (raw_mag / denom).min(max_torque);
    let delta_omega = tau_mag * dt * w_ang;
    if delta_omega > 2.0 {
        tau_mag = 2.0 / (dt * w_ang);
    }

    let impulse = dir * (tau_mag * dt);
    let dw = body.inv_inertia_world_mul(impulse);
    body.angular_velocity += dw;
}
