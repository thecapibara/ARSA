//! Continuous 6-keyframe biomechanical get-up recovery controller (`BehaviorState::Recovering`).

use glam::{EulerRot, Quat, Vec3};

use crate::body::RigidBody;
use crate::joint::Joint;
use crate::reflex::WoundRegion;
use crate::skeleton::{BoneId, JointId, RagdollInstance};

use super::{apply_pitched_upright_pd, set_target, BehaviorState, RecoveryStage, RecoveryStrategy};

#[derive(Clone, Copy)]
struct RecoveryKeyframe {
    t: f32,
    joints: [Quat; 16],
    torso_pitch: f32,
    pelvis_height_frac: f32,
    lift_weight: f32,
    gain_scale: f32,
}

#[inline]
fn smoothstep(u: f32) -> f32 {
    let x = u.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[inline]
fn heaviness_duration_scale(ragdoll: &RagdollInstance) -> f32 {
    let profile = ragdoll.profile.sanitized();
    let heaviness = ((profile.mass_kg / 75.0) / profile.muscle_ratio.max(0.4)).clamp(0.75, 2.2);
    let health_factor = (ragdoll.reflex.health / 100.0).clamp(0.45, 1.0);
    heaviness.sqrt() * (1.35 - 0.35 * health_factor)
}

/// Selects the appropriate [`RecoveryStrategy`] and lead leg when transitioning `Downed -> Recovering`.
pub(super) fn select_recovery_strategy(
    ragdoll: &RagdollInstance,
    bodies: &[RigidBody],
    chest_idx: usize,
    lf_pos: Vec3,
    rf_pos: Vec3,
    foot_ground_y: f32,
    s_h: f32,
) -> (RecoveryStrategy, bool, f32) {
    let profile = ragdoll.profile.sanitized();
    let wound_region = ragdoll
        .reflex
        .active_wound
        .map(|w| w.region)
        .unwrap_or(WoundRegion::None);

    let left_arm_hurt = wound_region == WoundRegion::LeftArm
        || ragdoll.reflex.bone_damage[BoneId::LeftUpperArm as usize] > 0.4
        || ragdoll.reflex.bone_damage[BoneId::LeftForearm as usize] > 0.4;
    let right_arm_hurt = wound_region == WoundRegion::RightArm
        || ragdoll.reflex.bone_damage[BoneId::RightUpperArm as usize] > 0.4
        || ragdoll.reflex.bone_damage[BoneId::RightForearm as usize] > 0.4;

    let chest_fwd_y = (bodies[chest_idx].orientation * Vec3::Z).y;
    let uneven_surface = (lf_pos.y - rf_pos.y).abs() > 0.10 * s_h || foot_ground_y > 0.12;
    let is_heavy_or_weak =
        profile.mass_kg >= 100.0 || profile.muscle_ratio < 0.75 || ragdoll.reflex.health < 55.0;

    let strategy = if right_arm_hurt && !left_arm_hurt {
        RecoveryStrategy::OneArmSideRoll { healthy_left: true }
    } else if left_arm_hurt && !right_arm_hurt {
        RecoveryStrategy::OneArmSideRoll {
            healthy_left: false,
        }
    } else if is_heavy_or_weak || uneven_surface || chest_fwd_y.abs() <= 0.25 {
        RecoveryStrategy::HeavyQuadruped
    } else if chest_fwd_y < -0.25 {
        RecoveryStrategy::PronePushUp
    } else {
        RecoveryStrategy::SupineAthletic
    };

    let lead_left = if wound_region == WoundRegion::RightLeg {
        true
    } else if wound_region == WoundRegion::LeftLeg {
        false
    } else if let RecoveryStrategy::OneArmSideRoll { healthy_left } = strategy {
        !healthy_left
    } else {
        true
    };

    (strategy, lead_left, chest_fwd_y)
}

/// Advances continuous `recovery_phase` (`0.0..=1.0`), updates [`RecoveryStage`] milestones,
/// and checks the pinned-under-debris stall abort.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance_recovery_stage(
    ragdoll: &mut RagdollInstance,
    pelvis_rel_y: f32,
    chest_up_dot: f32,
    up_dot: f32,
    cp_offset: f32,
    com_vel_xz: Vec3,
    s_h: f32,
    dt: f32,
) -> BehaviorState {
    let base_dur = match ragdoll.balance.recovery_strategy {
        RecoveryStrategy::SupineAthletic | RecoveryStrategy::PronePushUp => 2.1,
        RecoveryStrategy::OneArmSideRoll { .. } => 2.7,
        RecoveryStrategy::HeavyQuadruped => 3.1,
    };
    let total_dur = (base_dur * heaviness_duration_scale(ragdoll)).max(0.5);
    ragdoll.balance.recovery_phase =
        (ragdoll.balance.recovery_phase + dt / total_dur).clamp(0.0, 1.0);

    let phase = ragdoll.balance.recovery_phase;
    ragdoll.balance.recovery_stage = if phase < 0.22 {
        RecoveryStage::RollOrPrep
    } else if phase < 0.44 {
        RecoveryStage::QuadrupedBase
    } else if phase < 0.66 {
        RecoveryStage::HalfKneel
    } else {
        RecoveryStage::StandUp
    };

    // Pinned-under-debris stall abort: stuck low late in recovery (> 1.2s after phase 0.70)
    if phase > 0.70 && pelvis_rel_y < 0.38 * ragdoll.standing_pelvis_y {
        ragdoll.balance.recovery_stage_time += dt;
        if ragdoll.balance.recovery_stage_time > 1.2 {
            ragdoll.balance.recovery_phase = 0.0;
            ragdoll.balance.recovery_stage = RecoveryStage::RollOrPrep;
            ragdoll.balance.recovery_stage_time = 0.0;
            ragdoll.balance.downed_still_time = 0.0;
            return BehaviorState::Downed;
        }
        return BehaviorState::Recovering;
    }

    if phase >= 1.0 {
        if pelvis_rel_y < 0.76 * ragdoll.standing_pelvis_y {
            ragdoll.balance.recovery_stage_time += dt;
            if ragdoll.balance.recovery_stage_time <= 0.50 {
                return BehaviorState::Recovering;
            }
        }
        ragdoll.balance.recovery_phase = 0.0;
        ragdoll.balance.recovery_stage = RecoveryStage::RollOrPrep;
        ragdoll.balance.recovery_stage_time = 0.0;
        ragdoll.balance.downed_still_time = 0.0;
        ragdoll.balance.fallen_time = 0.0;
        let upright_and_high = pelvis_rel_y >= 0.68 * ragdoll.standing_pelvis_y
            && chest_up_dot >= 0.65
            && up_dot >= 0.65;
        if upright_and_high && cp_offset <= 0.26 * s_h && com_vel_xz.length() <= 0.45 {
            BehaviorState::Balanced
        } else {
            BehaviorState::Stumbling
        }
    } else {
        ragdoll.balance.recovery_stage_time = 0.0;
        BehaviorState::Recovering
    }
}

/// Builds a 16-joint pose array in [`JointId`] order from spine/head, left/right arm, and lead/trail leg tuples.
#[allow(clippy::too_many_arguments)]
fn build_pose_joints(
    spine_head: (f32, f32, f32, f32, f32),
    left_arm: ((f32, f32, f32), f32),
    right_arm: ((f32, f32, f32), f32),
    lead_left: bool,
    lead_leg: (f32, f32, f32, f32),
    trail_leg: (f32, f32, f32, f32),
) -> [Quat; 16] {
    let mut j = [Quat::IDENTITY; 16];
    let (lower_x, chest_x, roll_z, neck_x, head_x) = spine_head;
    j[JointId::LowerSpine as usize] = Quat::from_euler(EulerRot::XYZ, lower_x, 0.0, roll_z);
    j[JointId::ChestSpine as usize] = Quat::from_euler(EulerRot::XYZ, chest_x, 0.0, roll_z);
    j[JointId::Neck as usize] = Quat::from_rotation_x(neck_x);
    j[JointId::Head as usize] = Quat::from_rotation_x(head_x);

    j[JointId::LeftShoulder as usize] =
        Quat::from_euler(EulerRot::XYZ, left_arm.0 .0, left_arm.0 .1, left_arm.0 .2);
    j[JointId::LeftElbow as usize] = Quat::from_rotation_x(left_arm.1);
    j[JointId::RightShoulder as usize] = Quat::from_euler(
        EulerRot::XYZ,
        right_arm.0 .0,
        right_arm.0 .1,
        right_arm.0 .2,
    );
    j[JointId::RightElbow as usize] = Quat::from_rotation_x(right_arm.1);

    let (left, right) = if lead_left {
        (lead_leg, trail_leg)
    } else {
        (trail_leg, lead_leg)
    };
    j[JointId::LeftHip as usize] = Quat::from_euler(EulerRot::XYZ, left.0, 0.0, left.1);
    j[JointId::LeftKnee as usize] = Quat::from_rotation_x(left.2);
    j[JointId::LeftAnkle as usize] = Quat::from_rotation_x(left.3);
    j[JointId::RightHip as usize] = Quat::from_euler(EulerRot::XYZ, right.0, 0.0, -right.1);
    j[JointId::RightKnee as usize] = Quat::from_rotation_x(right.2);
    j[JointId::RightAnkle as usize] = Quat::from_rotation_x(right.3);
    j
}

fn build_pose_track(
    strategy: RecoveryStrategy,
    lead_left: bool,
    start_poses: [Quat; 16],
) -> [RecoveryKeyframe; 6] {
    let key0 = RecoveryKeyframe {
        t: 0.00,
        joints: start_poses,
        torso_pitch: 0.0,
        pelvis_height_frac: 0.18,
        lift_weight: 0.0,
        gain_scale: 0.30,
    };

    let (key1, key2) = match strategy {
        RecoveryStrategy::SupineAthletic => (
            RecoveryKeyframe {
                t: 0.22,
                joints: build_pose_joints(
                    (0.55, 0.55, 0.0, 0.15, 0.35),
                    ((0.55, 0.0, 0.28), -0.45),
                    ((0.55, 0.0, -0.28), -0.45),
                    lead_left,
                    (-1.45, 0.08, 1.85, -0.35),
                    (-1.25, 0.08, 1.70, -0.35),
                ),
                torso_pitch: 0.45,
                pelvis_height_frac: 0.22,
                lift_weight: 0.0,
                gain_scale: 0.74,
            },
            RecoveryKeyframe {
                t: 0.44,
                joints: build_pose_joints(
                    (0.35, 0.35, 0.0, -0.10, -0.15),
                    ((-0.85, 0.0, 0.32), -0.55),
                    ((-0.85, 0.0, -0.32), -0.55),
                    lead_left,
                    (-1.35, 0.10, 1.75, -0.35),
                    (-0.95, 0.10, 1.55, -0.30),
                ),
                torso_pitch: 0.82,
                pelvis_height_frac: 0.36,
                lift_weight: 0.35,
                gain_scale: 0.88,
            },
        ),
        RecoveryStrategy::PronePushUp => (
            RecoveryKeyframe {
                t: 0.22,
                joints: build_pose_joints(
                    (-0.20, -0.25, 0.0, -0.20, -0.35),
                    ((-1.05, 0.0, 0.30), -1.35),
                    ((-1.05, 0.0, -0.30), -1.35),
                    lead_left,
                    (-1.35, 0.08, 1.75, -0.35),
                    (-0.65, 0.08, 1.10, -0.25),
                ),
                torso_pitch: 0.50,
                pelvis_height_frac: 0.22,
                lift_weight: 0.0,
                gain_scale: 0.76,
            },
            RecoveryKeyframe {
                t: 0.44,
                joints: build_pose_joints(
                    (0.15, 0.15, 0.0, -0.20, -0.25),
                    ((-1.05, 0.0, 0.30), -0.35),
                    ((-1.05, 0.0, -0.30), -0.35),
                    lead_left,
                    (-1.20, 0.14, 1.65, -0.35),
                    (-1.05, 0.14, 1.65, -0.35),
                ),
                torso_pitch: 0.82,
                pelvis_height_frac: 0.36,
                lift_weight: 0.35,
                gain_scale: 0.90,
            },
        ),
        RecoveryStrategy::HeavyQuadruped => (
            RecoveryKeyframe {
                t: 0.22,
                joints: build_pose_joints(
                    (0.05, 0.05, 0.0, -0.20, -0.30),
                    ((-0.95, 0.0, 0.42), -0.95),
                    ((-0.95, 0.0, -0.42), -0.95),
                    lead_left,
                    (-1.15, 0.14, 1.65, -0.30),
                    (-1.10, 0.14, 1.60, -0.30),
                ),
                torso_pitch: 0.75,
                pelvis_height_frac: 0.22,
                lift_weight: 0.0,
                gain_scale: 0.78,
            },
            RecoveryKeyframe {
                t: 0.44,
                joints: build_pose_joints(
                    (0.15, 0.15, 0.0, -0.20, -0.25),
                    ((-1.05, 0.0, 0.30), -0.35),
                    ((-1.05, 0.0, -0.30), -0.35),
                    lead_left,
                    (-1.20, 0.15, 1.65, -0.35),
                    (-1.05, 0.15, 1.65, -0.35),
                ),
                torso_pitch: 0.82,
                pelvis_height_frac: 0.36,
                lift_weight: 0.35,
                gain_scale: 0.92,
            },
        ),
        RecoveryStrategy::OneArmSideRoll { healthy_left } => {
            let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
            let arm_pair = |healthy_sh: (f32, f32, f32),
                            healthy_el: f32,
                            hurt_sh: (f32, f32, f32),
                            hurt_el: f32| {
                if healthy_left {
                    ((healthy_sh, healthy_el), (hurt_sh, hurt_el))
                } else {
                    ((hurt_sh, hurt_el), (healthy_sh, healthy_el))
                }
            };
            let (l1, r1) = arm_pair(
                (-0.85, 0.0, h_sign * 0.55),
                -0.65,
                (-0.25, 0.0, -h_sign * 0.12),
                -0.95,
            );
            let (l2, r2) = arm_pair(
                (-0.95, 0.0, h_sign * 0.45),
                -0.35,
                (-0.25, 0.0, -h_sign * 0.10),
                -0.95,
            );
            (
                RecoveryKeyframe {
                    t: 0.22,
                    joints: build_pose_joints(
                        (0.20, 0.20, -h_sign * 0.12, -0.15, -0.20),
                        l1,
                        r1,
                        lead_left,
                        (-1.25, 0.12, 1.65, -0.35),
                        (-1.05, 0.12, 1.55, -0.30),
                    ),
                    torso_pitch: 0.65,
                    pelvis_height_frac: 0.22,
                    lift_weight: 0.0,
                    gain_scale: 0.78,
                },
                RecoveryKeyframe {
                    t: 0.44,
                    joints: build_pose_joints(
                        (0.15, 0.15, 0.0, -0.20, -0.25),
                        l2,
                        r2,
                        lead_left,
                        (-1.20, 0.15, 1.65, -0.35),
                        (-1.05, 0.15, 1.65, -0.35),
                    ),
                    torso_pitch: 0.82,
                    pelvis_height_frac: 0.36,
                    lift_weight: 0.35,
                    gain_scale: 0.90,
                },
            )
        }
    };

    let (k3_left_arm, k3_right_arm, k4_left_arm, k4_right_arm) = match strategy {
        RecoveryStrategy::OneArmSideRoll { healthy_left } => {
            let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
            let h3 = ((-0.75, 0.0, h_sign * 0.40), -0.65);
            let i3 = ((-0.20, 0.0, -h_sign * 0.10), -0.85);
            let h4 = ((-0.35, 0.0, h_sign * 0.22), -0.45);
            let i4 = ((-0.10, 0.0, -h_sign * 0.10), -0.55);
            if healthy_left {
                (h3, i3, h4, i4)
            } else {
                (i3, h3, i4, h4)
            }
        }
        _ => (
            ((-0.65, 0.0, 0.35), -0.65),
            ((-0.65, 0.0, -0.35), -0.65),
            ((-0.30, 0.0, 0.22), -0.45),
            ((-0.30, 0.0, -0.22), -0.45),
        ),
    };

    // Key 3 (t = 0.66): Deep Forward Half-Kneel / Lunge — chest strongly over lead knee
    let key3 = RecoveryKeyframe {
        t: 0.66,
        joints: build_pose_joints(
            (0.20, 0.20, 0.0, -0.15, -0.20),
            k3_left_arm,
            k3_right_arm,
            lead_left,
            (-1.25, 0.10, 1.50, -0.35),
            (-0.55, 0.10, 1.15, -0.25),
        ),
        torso_pitch: 0.58,
        pelvis_height_frac: 0.52,
        lift_weight: 0.68,
        gain_scale: 0.94,
    };

    // Key 4 (t = 0.85): Crouch Rise — hands lift for balance, legs half-extended
    let key4 = RecoveryKeyframe {
        t: 0.85,
        joints: build_pose_joints(
            (0.12, 0.12, 0.0, -0.08, -0.10),
            k4_left_arm,
            k4_right_arm,
            lead_left,
            (-0.45, 0.06, 0.62, -0.20),
            (-0.42, 0.06, 0.60, -0.18),
        ),
        torso_pitch: 0.22,
        pelvis_height_frac: 0.78,
        lift_weight: 0.90,
        gain_scale: 0.98,
    };

    // Key 5 (t = 1.00): Standing — upright stance matching Balanced controller
    let key5 = RecoveryKeyframe {
        t: 1.00,
        joints: build_pose_joints(
            (0.0, 0.0, 0.0, 0.0, 0.0),
            ((0.0, 0.0, 0.10), -0.25),
            ((0.0, 0.0, -0.10), -0.25),
            lead_left,
            (-0.04, 0.0, 0.08, -0.04),
            (-0.04, 0.0, 0.08, -0.04),
        ),
        torso_pitch: 0.0,
        pelvis_height_frac: 0.94,
        lift_weight: 1.0,
        gain_scale: 1.0,
    };

    [key0, key1, key2, key3, key4, key5]
}

fn sample_pose_track(keys: &[RecoveryKeyframe; 6], phase: f32) -> RecoveryKeyframe {
    let t = phase.clamp(0.0, 1.0);
    let mut idx = 0;
    while idx + 1 < keys.len() - 1 && t > keys[idx + 1].t {
        idx += 1;
    }
    let a = &keys[idx];
    let b = &keys[idx + 1];
    let span = (b.t - a.t).max(1e-6);
    let s = smoothstep((t - a.t) / span);

    let mut joints = [Quat::IDENTITY; 16];
    for (i, dst) in joints.iter_mut().enumerate() {
        *dst = a.joints[i]
            .normalize()
            .slerp(b.joints[i].normalize(), s)
            .normalize();
    }

    RecoveryKeyframe {
        t,
        joints,
        torso_pitch: a.torso_pitch + (b.torso_pitch - a.torso_pitch) * s,
        pelvis_height_frac: a.pelvis_height_frac
            + (b.pelvis_height_frac - a.pelvis_height_frac) * s,
        lift_weight: a.lift_weight + (b.lift_weight - a.lift_weight) * s,
        gain_scale: a.gain_scale + (b.gain_scale - a.gain_scale) * s,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_recovering_pose(
    ragdoll: &RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    foot_contacts: (bool, bool),
    foot_ground_y: f32,
    total_mass: f32,
    gravity_mag: f32,
    s_h: f32,
    m_ratio: f32,
    h_ratio: f32,
    mu: f32,
    damp_scale: f32,
    com_pos: Vec3,
    com_vel_xz: Vec3,
    dt: f32,
) {
    let phase = ragdoll.balance.recovery_phase.clamp(0.0, 1.0);
    let strategy = ragdoll.balance.recovery_strategy;
    let lead_left = ragdoll.balance.recovery_lead_left;
    let rec_strength = (ragdoll.reflex.health / 100.0).clamp(0.45, 1.0) * mu;

    let pelvis_idx = ragdoll.body_indices[BoneId::Pelvis as usize];
    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let pelvis_pos = bodies[pelvis_idx].position;
    let grounded_count = (foot_contacts.0 as u8) + (foot_contacts.1 as u8);
    let rec_ground_y = if grounded_count > 0 {
        foot_ground_y
    } else {
        (pelvis_pos.y - 0.14 * s_h).min(foot_ground_y).max(0.0)
    };

    // Determine horizontal facing direction even when torso is lying flat
    let chest_fwd = bodies[chest_idx].orientation * Vec3::Z;
    let chest_fwd_xz = Vec3::new(chest_fwd.x, 0.0, chest_fwd.z);
    let chest_up = bodies[chest_idx].orientation * Vec3::Y;
    let chest_up_xz = Vec3::new(chest_up.x, 0.0, chest_up.z);
    let pelvis_fwd = bodies[pelvis_idx].orientation * Vec3::Z;
    let pelvis_fwd_xz = Vec3::new(pelvis_fwd.x, 0.0, pelvis_fwd.z);
    let fwd_xz = if chest_fwd_xz.length_squared() > 0.12 {
        chest_fwd_xz.normalize()
    } else if pelvis_fwd_xz.length_squared() > 0.12 {
        pelvis_fwd_xz.normalize()
    } else if chest_up_xz.length_squared() > 1e-4 {
        if chest_fwd.y < 0.0 {
            chest_up_xz.normalize()
        } else {
            -chest_up_xz.normalize()
        }
    } else {
        Vec3::Z
    };

    // Sample continuous 6-keyframe PoseTrack
    let track = build_pose_track(strategy, lead_left, ragdoll.balance.recovery_start_poses);
    let sampled = sample_pose_track(&track, phase);

    const ALL_JOINTS: [JointId; 16] = [
        JointId::LowerSpine,
        JointId::ChestSpine,
        JointId::Neck,
        JointId::Head,
        JointId::LeftShoulder,
        JointId::LeftElbow,
        JointId::LeftWrist,
        JointId::RightShoulder,
        JointId::RightElbow,
        JointId::RightWrist,
        JointId::LeftHip,
        JointId::LeftKnee,
        JointId::LeftAnkle,
        JointId::RightHip,
        JointId::RightKnee,
        JointId::RightAnkle,
    ];
    for jid in ALL_JOINTS {
        set_target(ragdoll, joints, jid, sampled.joints[jid as usize]);
    }

    // Smoothly scale joint motor gains with sampled gain_scale
    let d_scale = 0.55 + 0.45 * sampled.gain_scale;
    scale_all_joint_gains(
        ragdoll,
        joints,
        sampled.gain_scale,
        d_scale,
        sampled.gain_scale,
    );

    // Smooth arm push boost early & injured arm soften in OneArmSideRoll
    let arm_boost_fade = 1.0 - smoothstep((phase - 0.35) / 0.25);
    if arm_boost_fade > 1e-3 {
        let boost_factor = 1.0 + 0.50 * arm_boost_fade;
        match strategy {
            RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                boost_arms(ragdoll, joints, healthy_left, !healthy_left, boost_factor);
            }
            _ => {
                boost_arms(ragdoll, joints, true, true, boost_factor);
            }
        }
    }
    if let RecoveryStrategy::OneArmSideRoll { healthy_left } = strategy {
        let soften_fade = 1.0 - smoothstep((phase - 0.66) / 0.25);
        if soften_fade > 1e-3 {
            let factor = 1.0 - 0.55 * soften_fade;
            soften_arm(ragdoll, joints, !healthy_left, factor);
        }
    }

    // Gentle axial roll torque on Chest/Pelvis toward floor during Roll & Hand-Plant -> Quadruped
    if phase < 0.44 {
        let roll_fade = 1.0 - smoothstep((phase - 0.22) / 0.22);
        match strategy {
            RecoveryStrategy::HeavyQuadruped => {
                apply_axial_roll_toward_floor(
                    &mut bodies[chest_idx],
                    Vec3::Z,
                    145.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    14.0 * damp_scale,
                    95.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    dt,
                );
                apply_axial_roll_toward_floor(
                    &mut bodies[pelvis_idx],
                    Vec3::Z,
                    120.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    14.0 * damp_scale,
                    80.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    dt,
                );
            }
            RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
                let prop_local = Vec3::new(h_sign * 0.70, 0.0, 0.71).normalize();
                apply_axial_roll_toward_floor(
                    &mut bodies[chest_idx],
                    prop_local,
                    160.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    14.0 * damp_scale,
                    105.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    dt,
                );
                apply_axial_roll_toward_floor(
                    &mut bodies[pelvis_idx],
                    prop_local,
                    130.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    14.0 * damp_scale,
                    85.0 * m_ratio * h_ratio * rec_strength * roll_fade,
                    dt,
                );
            }
            _ => {}
        }
    }

    // Smoothly ramped torso/pelvis upright PD toward continuously sampled torso_pitch
    let chest_target_up = pitched_up(fwd_xz, sampled.torso_pitch);
    let pelvis_target_up = pitched_up(fwd_xz, sampled.torso_pitch * 0.70);
    let chest_upright_w = match strategy {
        RecoveryStrategy::HeavyQuadruped => {
            0.75 * smoothstep((phase - 0.14) / 0.30) + 0.25 * smoothstep((phase - 0.44) / 0.41)
        }
        _ => 0.55 * smoothstep(phase / 0.22) + 0.45 * smoothstep((phase - 0.22) / 0.63),
    };
    let pelvis_upright_w =
        0.75 * smoothstep((phase - 0.08) / 0.28) + 0.25 * smoothstep((phase - 0.36) / 0.49);

    if chest_upright_w > 1e-3 {
        apply_pitched_upright_pd(
            &mut bodies[chest_idx],
            chest_target_up,
            260.0 * m_ratio * h_ratio * rec_strength * chest_upright_w,
            18.0 * damp_scale * (0.70 + 0.30 * chest_upright_w),
            135.0 * m_ratio * h_ratio * rec_strength * chest_upright_w,
            dt,
        );
    }
    if pelvis_upright_w > 1e-3 {
        apply_pitched_upright_pd(
            &mut bodies[pelvis_idx],
            pelvis_target_up,
            320.0 * m_ratio * h_ratio * rec_strength * pelvis_upright_w,
            22.0 * damp_scale * (0.70 + 0.30 * pelvis_upright_w),
            165.0 * m_ratio * h_ratio * rec_strength * pelvis_upright_w,
            dt,
        );
    }

    // Slow, heavy, CoM-gated pelvis lift (velocity capped at 0.42 m/s, zero lift while lift_weight == 0)
    let target_y = rec_ground_y + sampled.pelvis_height_frac * ragdoll.standing_pelvis_y;
    apply_recovery_lift(
        &mut bodies[pelvis_idx],
        target_y,
        sampled.lift_weight,
        total_mass,
        gravity_mag,
        rec_strength,
        dt,
    );

    // Ankle support coordinates for CoM-over-foot shift
    let la_j = &joints[ragdoll.joint_indices[JointId::LeftAnkle as usize]];
    let ra_j = &joints[ragdoll.joint_indices[JointId::RightAnkle as usize]];
    let la_pos = bodies[la_j.child].world_point(la_j.anchor_child);
    let ra_pos = bodies[ra_j.child].world_point(ra_j.anchor_child);
    let mid_ankles_xz = Vec3::new(
        0.5 * (la_pos.x + ra_pos.x),
        foot_ground_y,
        0.5 * (la_pos.z + ra_pos.z),
    );
    let lead_ankle_xz = if lead_left {
        Vec3::new(la_pos.x, foot_ground_y, la_pos.z)
    } else {
        Vec3::new(ra_pos.x, foot_ground_y, ra_pos.z)
    };
    let support_center = match foot_contacts {
        (true, true) => mid_ankles_xz,
        (true, false) => Vec3::new(la_pos.x, foot_ground_y, la_pos.z),
        (false, true) => Vec3::new(ra_pos.x, foot_ground_y, ra_pos.z),
        (false, false) => mid_ankles_xz,
    };

    // Pull Pelvis horizontally forward toward lead_ankle_xz during t = 0.35..0.85
    // so hips genuinely shift over the planted foot BEFORE straightening the legs.
    if (0.35..=0.85).contains(&phase) {
        let ramp_in = smoothstep((phase - 0.35) / 0.12);
        let ramp_out = 1.0 - smoothstep((phase - 0.72) / 0.13);
        let pull_weight = ramp_in * ramp_out;
        let pelvis = &mut bodies[pelvis_idx];
        let pelvis_vel_xz = Vec3::new(pelvis.linear_velocity.x, 0.0, pelvis.linear_velocity.z);
        let horiz_err = Vec3::new(
            lead_ankle_xz.x - pelvis.position.x,
            0.0,
            lead_ankle_xz.z - pelvis.position.z,
        );
        let mut f_pull = (horiz_err * (total_mass * 9.5) - pelvis_vel_xz * (total_mass * 5.0))
            * rec_strength
            * pull_weight;
        let max_f = total_mass * 4.5 * rec_strength * pull_weight;
        let f_len = f_pull.length();
        if f_len > max_f && f_len > 1e-6 {
            f_pull *= max_f / f_len;
        }
        pelvis.linear_velocity += f_pull * (dt * pelvis.inv_mass);
    }

    // Center CoM over support polygon once lifting / grounded
    if phase > 0.44 || grounded_count > 0 {
        let com_weight = smoothstep((phase - 0.22) / 0.30);
        if com_weight > 1e-3 {
            let (gain, max_accel) = if phase < 0.75 {
                (7.5 * com_weight, 4.0 * com_weight)
            } else {
                (5.0, 2.5)
            };
            apply_horizontal_com_shift(
                &mut bodies[pelvis_idx],
                support_center,
                com_pos,
                com_vel_xz,
                total_mass,
                rec_strength,
                gain,
                max_accel,
                dt,
            );
        }
    }
}

#[inline]
fn pitched_up(fwd_xz: Vec3, pitch_rad: f32) -> Vec3 {
    (Vec3::Y * pitch_rad.cos() + fwd_xz * pitch_rad.sin()).normalize()
}

fn scale_all_joint_gains(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    k_scale: f32,
    d_scale: f32,
    t_scale: f32,
) {
    for &j_idx in &ragdoll.joint_indices {
        joints[j_idx].motor_stiffness *= k_scale;
        joints[j_idx].motor_damping *= d_scale;
        joints[j_idx].max_torque *= t_scale;
    }
}

fn boost_arms(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    left: bool,
    right: bool,
    factor: f32,
) {
    if left {
        for jid in [JointId::LeftShoulder, JointId::LeftElbow] {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_stiffness *= factor;
            j.max_torque *= factor;
        }
    }
    if right {
        for jid in [JointId::RightShoulder, JointId::RightElbow] {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_stiffness *= factor;
            j.max_torque *= factor;
        }
    }
}

fn soften_arm(ragdoll: &RagdollInstance, joints: &mut [Joint], is_left: bool, factor: f32) {
    let pair = if is_left {
        [JointId::LeftShoulder, JointId::LeftElbow]
    } else {
        [JointId::RightShoulder, JointId::RightElbow]
    };
    for jid in pair {
        let j = &mut joints[ragdoll.joint_indices[jid as usize]];
        j.motor_stiffness *= factor;
        j.max_torque *= factor;
    }
}

/// Applies gentle axial roll torque around the body's local `+Y` spine axis so `local_prop_dir`
/// turns toward the floor (`-Vec3::Y`).
fn apply_axial_roll_toward_floor(
    body: &mut RigidBody,
    local_prop_dir: Vec3,
    stiffness: f32,
    damping: f32,
    max_torque: f32,
    dt: f32,
) {
    if body.inv_mass == 0.0 || stiffness <= 0.0 || max_torque <= 0.0 {
        return;
    }
    let spine_axis = (body.orientation * Vec3::Y).normalize_or_zero();
    if spine_axis.length_squared() < 0.5 {
        return;
    }
    let cur = (body.orientation * local_prop_dir).normalize_or_zero();
    let cur_perp = (cur - spine_axis * cur.dot(spine_axis)).normalize_or_zero();
    let down_perp = (-Vec3::Y - spine_axis * (-Vec3::Y).dot(spine_axis)).normalize_or_zero();
    if cur_perp.length_squared() < 0.05 || down_perp.length_squared() < 0.05 {
        return;
    }

    let sin_r = cur_perp.cross(down_perp).dot(spine_axis);
    let cos_r = cur_perp.dot(down_perp);
    let mut roll_err = sin_r.atan2(cos_r);
    // Break exact 180-deg supine symmetry cleanly toward one side
    if roll_err.abs() > 2.85 && sin_r.abs() < 0.15 {
        roll_err = 2.85;
    }

    let omega_axial = body.angular_velocity.dot(spine_axis);
    let w_ang = body.generalized_angular_inv_mass(spine_axis);
    if w_ang < 1e-12 {
        return;
    }
    let raw_tau = roll_err * stiffness - omega_axial * damping;
    let denom = 1.0 + w_ang * dt * (damping + dt * stiffness);
    let tau = (raw_tau / denom).clamp(-max_torque, max_torque);
    let dw = body.inv_inertia_world_mul(spine_axis * (tau * dt));
    body.angular_velocity += dw;
}

#[allow(clippy::too_many_arguments)]
fn apply_horizontal_com_shift(
    pelvis: &mut RigidBody,
    support_center: Vec3,
    com_pos: Vec3,
    com_vel_xz: Vec3,
    total_mass: f32,
    rec_strength: f32,
    gain: f32,
    max_accel: f32,
    dt: f32,
) {
    let horiz_err = Vec3::new(
        support_center.x - com_pos.x,
        0.0,
        support_center.z - com_pos.z,
    );
    let mut f_horiz =
        (horiz_err * (total_mass * gain) - com_vel_xz * (total_mass * 6.0)) * rec_strength;
    let max_f = total_mass * max_accel * rec_strength;
    let f_len = f_horiz.length();
    if f_len > max_f && f_len > 1e-6 {
        f_horiz *= max_f / f_len;
    }
    pelvis.linear_velocity += f_horiz * (dt * pelvis.inv_mass);
}

fn apply_recovery_lift(
    pelvis: &mut RigidBody,
    target_y: f32,
    lift_weight: f32,
    total_mass: f32,
    gravity_mag: f32,
    rec_strength: f32,
    dt: f32,
) {
    let lift_scale = lift_weight.clamp(0.0, 1.0);
    if pelvis.inv_mass == 0.0 || lift_scale <= 1e-4 || pelvis.linear_velocity.y >= 0.42 {
        return;
    }
    let height_err = target_y - pelvis.position.y;
    let eff_strength = 0.60 + 0.40 * rec_strength;
    let ramp = smoothstep(lift_scale / 0.18);
    let fy_raw = (total_mass * gravity_mag * 0.85 + height_err * (total_mass * 38.0)
        - pelvis.linear_velocity.y * (total_mass * 6.0))
        * eff_strength
        * ramp;
    let max_fy = total_mass * gravity_mag * 1.25 * ramp;
    let fy = fy_raw.clamp(0.0, max_fy);
    pelvis.linear_velocity.y += (fy * dt) * pelvis.inv_mass;
}
