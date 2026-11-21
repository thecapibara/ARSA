//! Multi-strategy, milestone-driven biomechanical get-up recovery controller
//! (`BehaviorState::Recovering`).

use glam::{EulerRot, Quat, Vec3};

use crate::body::RigidBody;
use crate::joint::Joint;
use crate::reflex::WoundRegion;
use crate::skeleton::{BoneId, JointId, RagdollInstance};

use super::{apply_pitched_upright_pd, set_target, BehaviorState, RecoveryStage, RecoveryStrategy};

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

#[inline]
pub(super) fn compute_stage_speed(ragdoll: &RagdollInstance) -> f32 {
    let profile = ragdoll.profile.sanitized();
    let heaviness = ((profile.mass_kg / 75.0) / profile.muscle_ratio.max(0.4)).clamp(0.75, 2.2);
    let health_factor = (ragdoll.reflex.health / 100.0).clamp(0.45, 1.0);
    1.0 / (heaviness.sqrt() * (1.35 - 0.35 * health_factor))
}

/// Advances [`RecoveryStage`] milestones, checks stall/pinned abort, and updates `recovery_phase`.
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
    ragdoll.balance.recovery_stage_time += dt;
    let stage_time = ragdoll.balance.recovery_stage_time;
    let stage_speed = compute_stage_speed(ragdoll);
    let strategy = ragdoll.balance.recovery_strategy;
    let uses_quadruped = matches!(
        strategy,
        RecoveryStrategy::HeavyQuadruped | RecoveryStrategy::OneArmSideRoll { .. }
    );

    // Stall / Pinned Abort: if stuck low in a late stage for > 2.2s, drop cleanly back to Downed
    if matches!(
        ragdoll.balance.recovery_stage,
        RecoveryStage::HalfKneel | RecoveryStage::StandUp
    ) && stage_time > 2.2
        && pelvis_rel_y < 0.42 * ragdoll.standing_pelvis_y
    {
        ragdoll.balance.recovery_phase = 0.0;
        ragdoll.balance.recovery_stage = RecoveryStage::RollOrPrep;
        ragdoll.balance.recovery_stage_time = 0.0;
        ragdoll.balance.downed_still_time = 0.0;
        return BehaviorState::Downed;
    }

    match ragdoll.balance.recovery_stage {
        RecoveryStage::RollOrPrep => {
            let prep_dur = 0.45 / stage_speed;
            let frac = (stage_time / prep_dur).clamp(0.0, 1.0);
            ragdoll.balance.recovery_phase = if uses_quadruped {
                0.22 * frac
            } else {
                0.30 * frac
            };
            if stage_time >= prep_dur {
                ragdoll.balance.recovery_stage = if uses_quadruped {
                    RecoveryStage::QuadrupedBase
                } else {
                    RecoveryStage::HalfKneel
                };
                ragdoll.balance.recovery_stage_time = 0.0;
            }
            BehaviorState::Recovering
        }
        RecoveryStage::QuadrupedBase => {
            let min_dur = 0.45 / stage_speed;
            let max_dur = 1.15 / stage_speed;
            let frac = (stage_time / max_dur).clamp(0.0, 1.0);
            ragdoll.balance.recovery_phase = 0.22 + 0.28 * frac;

            let reached_height =
                pelvis_rel_y >= 0.38 * ragdoll.standing_pelvis_y && stage_time >= min_dur;
            if reached_height || stage_time >= max_dur {
                ragdoll.balance.recovery_stage = RecoveryStage::HalfKneel;
                ragdoll.balance.recovery_stage_time = 0.0;
                ragdoll.balance.recovery_phase = 0.50;
            }
            BehaviorState::Recovering
        }
        RecoveryStage::HalfKneel => {
            let min_dur = 0.45 / stage_speed;
            let max_dur = 1.25 / stage_speed;
            let frac = (stage_time / max_dur).clamp(0.0, 1.0);
            ragdoll.balance.recovery_phase = if uses_quadruped {
                0.50 + 0.28 * frac
            } else {
                0.30 + 0.40 * frac
            };

            let reached_height =
                pelvis_rel_y >= 0.56 * ragdoll.standing_pelvis_y && stage_time >= min_dur;
            if reached_height || stage_time >= max_dur {
                ragdoll.balance.recovery_stage = RecoveryStage::StandUp;
                ragdoll.balance.recovery_stage_time = 0.0;
                ragdoll.balance.recovery_phase = if uses_quadruped { 0.78 } else { 0.70 };
            }
            BehaviorState::Recovering
        }
        RecoveryStage::StandUp => {
            let stand_dur = 0.65 / stage_speed;
            let t = (stage_time / stand_dur).clamp(0.0, 1.0);
            ragdoll.balance.recovery_phase = if uses_quadruped {
                0.78 + 0.22 * t
            } else {
                0.70 + 0.30 * t
            };

            if t >= 1.0 {
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
                BehaviorState::Recovering
            }
        }
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
    let stage = ragdoll.balance.recovery_stage;
    let stage_time = ragdoll.balance.recovery_stage_time;
    let stage_speed = compute_stage_speed(ragdoll);
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
        if ragdoll.balance.recovery_prone {
            chest_up_xz.normalize()
        } else {
            -chest_up_xz.normalize()
        }
    } else {
        Vec3::Z
    };

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

    match stage {
        RecoveryStage::RollOrPrep => {
            scale_all_joint_gains(ragdoll, joints, 0.78, 0.85, 0.82);

            match strategy {
                RecoveryStrategy::SupineAthletic => {
                    boost_arms(ragdoll, joints, true, true, 1.45);
                    set_spine_head(ragdoll, joints, 0.55, 0.55, 0.0, 0.15, 0.35);
                    set_asym_legs(
                        ragdoll,
                        joints,
                        lead_left,
                        (-1.45, 0.08, 1.85, -0.35),
                        (-1.25, 0.08, 1.70, -0.35),
                    );
                    set_arm_pose(ragdoll, joints, true, (0.55, 0.0, 0.28), -0.45);
                    set_arm_pose(ragdoll, joints, false, (0.55, 0.0, -0.28), -0.45);

                    let target_up = pitched_up(fwd_xz, 0.45);
                    apply_pitched_upright_pd(
                        &mut bodies[chest_idx],
                        target_up,
                        260.0 * m_ratio * h_ratio * rec_strength * 0.45,
                        18.0 * damp_scale * 0.70,
                        135.0 * m_ratio * h_ratio * rec_strength * 0.45,
                        dt,
                    );
                }
                RecoveryStrategy::PronePushUp => {
                    boost_arms(ragdoll, joints, true, true, 1.50);
                    set_spine_head(ragdoll, joints, -0.20, -0.25, 0.0, -0.20, -0.35);
                    set_asym_legs(
                        ragdoll,
                        joints,
                        lead_left,
                        (-1.35, 0.08, 1.75, -0.35),
                        (-0.65, 0.08, 1.10, -0.25),
                    );
                    set_arm_pose(ragdoll, joints, true, (-1.05, 0.0, 0.30), -1.35);
                    set_arm_pose(ragdoll, joints, false, (-1.05, 0.0, -0.30), -1.35);

                    let target_up = pitched_up(fwd_xz, 0.50);
                    apply_pitched_upright_pd(
                        &mut bodies[chest_idx],
                        target_up,
                        260.0 * m_ratio * h_ratio * rec_strength * 0.45,
                        18.0 * damp_scale * 0.70,
                        135.0 * m_ratio * h_ratio * rec_strength * 0.45,
                        dt,
                    );
                }
                RecoveryStrategy::HeavyQuadruped => {
                    boost_arms(ragdoll, joints, true, true, 1.50);
                    set_spine_head(ragdoll, joints, 0.05, 0.05, 0.0, -0.20, -0.30);
                    set_asym_legs(
                        ragdoll,
                        joints,
                        lead_left,
                        (-1.15, 0.14, 1.65, -0.30),
                        (-1.10, 0.14, 1.60, -0.30),
                    );
                    set_arm_pose(ragdoll, joints, true, (-0.95, 0.0, 0.42), -0.95);
                    set_arm_pose(ragdoll, joints, false, (-0.95, 0.0, -0.42), -0.95);

                    // Gentle axial roll torque on Chest and Pelvis toward prone / hands-and-knees
                    apply_axial_roll_toward_floor(
                        &mut bodies[chest_idx],
                        Vec3::Z,
                        145.0 * m_ratio * h_ratio * rec_strength,
                        14.0 * damp_scale,
                        95.0 * m_ratio * h_ratio * rec_strength,
                        dt,
                    );
                    apply_axial_roll_toward_floor(
                        &mut bodies[pelvis_idx],
                        Vec3::Z,
                        120.0 * m_ratio * h_ratio * rec_strength,
                        14.0 * damp_scale,
                        80.0 * m_ratio * h_ratio * rec_strength,
                        dt,
                    );
                }
                RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                    let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
                    boost_arms(ragdoll, joints, healthy_left, !healthy_left, 1.60);
                    set_spine_head(ragdoll, joints, 0.20, 0.20, -h_sign * 0.12, -0.15, -0.20);
                    set_asym_legs(
                        ragdoll,
                        joints,
                        lead_left,
                        (-1.25, 0.12, 1.65, -0.35),
                        (-1.05, 0.12, 1.55, -0.30),
                    );

                    // Plant healthy arm firmly down-forward onto floor; keep injured arm tucked lightly
                    set_arm_pose(
                        ragdoll,
                        joints,
                        healthy_left,
                        (-0.85, 0.0, h_sign * 0.55),
                        -0.65,
                    );
                    set_arm_pose(
                        ragdoll,
                        joints,
                        !healthy_left,
                        (-0.25, 0.0, -h_sign * 0.12),
                        -0.95,
                    );
                    soften_arm(ragdoll, joints, !healthy_left, 0.45);

                    // Roll onto the healthy side (between healthy shoulder +X/-X and chest front +Z)
                    let prop_local = Vec3::new(h_sign * 0.70, 0.0, 0.71).normalize();
                    apply_axial_roll_toward_floor(
                        &mut bodies[chest_idx],
                        prop_local,
                        160.0 * m_ratio * h_ratio * rec_strength,
                        14.0 * damp_scale,
                        105.0 * m_ratio * h_ratio * rec_strength,
                        dt,
                    );
                    apply_axial_roll_toward_floor(
                        &mut bodies[pelvis_idx],
                        prop_local,
                        130.0 * m_ratio * h_ratio * rec_strength,
                        14.0 * damp_scale,
                        85.0 * m_ratio * h_ratio * rec_strength,
                        dt,
                    );
                    let target_up = pitched_up(fwd_xz, 0.50);
                    apply_pitched_upright_pd(
                        &mut bodies[chest_idx],
                        target_up,
                        220.0 * m_ratio * h_ratio * rec_strength * 0.40,
                        16.0 * damp_scale * 0.70,
                        115.0 * m_ratio * h_ratio * rec_strength * 0.40,
                        dt,
                    );
                }
            }

            set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
            set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);
        }

        RecoveryStage::QuadrupedBase => {
            scale_all_joint_gains(ragdoll, joints, 0.90, 0.92, 0.92);
            set_spine_head(ragdoll, joints, 0.15, 0.15, 0.0, -0.20, -0.25);
            set_asym_legs(
                ragdoll,
                joints,
                lead_left,
                (-1.20, 0.15, 1.65, -0.35),
                (-1.05, 0.15, 1.65, -0.35),
            );

            match strategy {
                RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                    let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
                    boost_arms(ragdoll, joints, healthy_left, !healthy_left, 1.50);
                    set_arm_pose(
                        ragdoll,
                        joints,
                        healthy_left,
                        (-0.95, 0.0, h_sign * 0.45),
                        -0.35,
                    );
                    set_arm_pose(
                        ragdoll,
                        joints,
                        !healthy_left,
                        (-0.25, 0.0, -h_sign * 0.10),
                        -0.95,
                    );
                    soften_arm(ragdoll, joints, !healthy_left, 0.45);
                    if chest_fwd.y > 0.15 {
                        let prop_local = Vec3::new(h_sign * 0.70, 0.0, 0.71).normalize();
                        apply_axial_roll_toward_floor(
                            &mut bodies[chest_idx],
                            prop_local,
                            110.0 * m_ratio * h_ratio * rec_strength,
                            12.0 * damp_scale,
                            75.0 * m_ratio * h_ratio * rec_strength,
                            dt,
                        );
                    }
                }
                _ => {
                    boost_arms(ragdoll, joints, true, true, 1.45);
                    set_arm_pose(ragdoll, joints, true, (-1.05, 0.0, 0.30), -0.35);
                    set_arm_pose(ragdoll, joints, false, (-1.05, 0.0, -0.30), -0.35);
                }
            }
            set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
            set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

            let target_up = pitched_up(fwd_xz, 0.55);
            apply_pitched_upright_pd(
                &mut bodies[pelvis_idx],
                target_up,
                320.0 * m_ratio * h_ratio * rec_strength * 0.70,
                22.0 * damp_scale * 0.85,
                165.0 * m_ratio * h_ratio * rec_strength * 0.70,
                dt,
            );
            apply_pitched_upright_pd(
                &mut bodies[chest_idx],
                target_up,
                260.0 * m_ratio * h_ratio * rec_strength * 0.75,
                18.0 * damp_scale * 0.85,
                135.0 * m_ratio * h_ratio * rec_strength * 0.75,
                dt,
            );

            let target_y = rec_ground_y + 0.46 * ragdoll.standing_pelvis_y;
            apply_recovery_lift(
                &mut bodies[pelvis_idx],
                target_y,
                total_mass,
                gravity_mag,
                rec_strength,
                dt,
            );
            apply_horizontal_com_shift(
                &mut bodies[pelvis_idx],
                support_center,
                com_pos,
                com_vel_xz,
                total_mass,
                rec_strength,
                6.0,
                3.0,
                dt,
            );
        }

        RecoveryStage::HalfKneel => {
            scale_all_joint_gains(ragdoll, joints, 0.92, 0.92, 0.92);
            set_spine_head(ragdoll, joints, 0.20, 0.20, 0.0, -0.15, -0.20);
            set_asym_legs(
                ragdoll,
                joints,
                lead_left,
                (-1.20, 0.10, 1.45, -0.35),
                (-0.60, 0.10, 1.15, -0.25),
            );

            match strategy {
                RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                    let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
                    set_arm_pose(
                        ragdoll,
                        joints,
                        healthy_left,
                        (-0.75, 0.0, h_sign * 0.40),
                        -0.65,
                    );
                    set_arm_pose(
                        ragdoll,
                        joints,
                        !healthy_left,
                        (-0.20, 0.0, -h_sign * 0.10),
                        -0.85,
                    );
                    soften_arm(ragdoll, joints, !healthy_left, 0.40);
                }
                _ => {
                    set_arm_pose(ragdoll, joints, true, (-0.65, 0.0, 0.35), -0.65);
                    set_arm_pose(ragdoll, joints, false, (-0.65, 0.0, -0.35), -0.65);
                }
            }
            set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
            set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

            let target_up = pitched_up(fwd_xz, 0.38);
            apply_pitched_upright_pd(
                &mut bodies[pelvis_idx],
                target_up,
                320.0 * m_ratio * h_ratio * rec_strength * 0.80,
                22.0 * damp_scale * 0.85,
                165.0 * m_ratio * h_ratio * rec_strength * 0.80,
                dt,
            );
            apply_pitched_upright_pd(
                &mut bodies[chest_idx],
                target_up,
                260.0 * m_ratio * h_ratio * rec_strength * 0.80,
                18.0 * damp_scale * 0.85,
                135.0 * m_ratio * h_ratio * rec_strength * 0.80,
                dt,
            );

            let start_frac = match strategy {
                RecoveryStrategy::HeavyQuadruped | RecoveryStrategy::OneArmSideRoll { .. } => 0.46,
                _ => 0.38,
            };
            let ramp = (stage_time / (0.55 / stage_speed)).clamp(0.0, 1.0);
            let target_y = rec_ground_y
                + (start_frac + (0.66 - start_frac) * ramp) * ragdoll.standing_pelvis_y;
            apply_recovery_lift(
                &mut bodies[pelvis_idx],
                target_y,
                total_mass,
                gravity_mag,
                rec_strength,
                dt,
            );
            apply_horizontal_com_shift(
                &mut bodies[pelvis_idx],
                support_center,
                com_pos,
                com_vel_xz,
                total_mass,
                rec_strength,
                9.0,
                4.5,
                dt,
            );
        }

        RecoveryStage::StandUp => {
            let t = (stage_time / (0.65 / stage_speed)).clamp(0.0, 1.0);
            let spine_x = 0.19 * (1.0 - t);
            set_spine_head(
                ragdoll,
                joints,
                spine_x,
                spine_x,
                0.0,
                -0.15 * (1.0 - t),
                -0.20 * (1.0 - t),
            );

            let abduct = 0.10 * (1.0 - t);
            set_asym_legs(
                ragdoll,
                joints,
                lead_left,
                (
                    -1.20 * (1.0 - t) + (-0.05) * t,
                    abduct,
                    1.45 * (1.0 - t) + 0.10 * t,
                    -0.35 * (1.0 - t) + (-0.05) * t,
                ),
                (
                    -0.60 * (1.0 - t) + (-0.05) * t,
                    abduct,
                    1.15 * (1.0 - t) + 0.10 * t,
                    -0.25 * (1.0 - t) + (-0.05) * t,
                ),
            );

            match strategy {
                RecoveryStrategy::OneArmSideRoll { healthy_left } => {
                    let h_sign = if healthy_left { 1.0_f32 } else { -1.0_f32 };
                    set_arm_pose(
                        ragdoll,
                        joints,
                        healthy_left,
                        (
                            -0.75 * (1.0 - t),
                            0.0,
                            h_sign * (0.40 * (1.0 - t) + 0.10 * t),
                        ),
                        -0.65 * (1.0 - t) + (-0.25) * t,
                    );
                    set_arm_pose(
                        ragdoll,
                        joints,
                        !healthy_left,
                        (-0.20 * (1.0 - t), 0.0, -h_sign * 0.10),
                        -0.85 * (1.0 - t) + (-0.25) * t,
                    );
                }
                _ => {
                    let arm_pitch = -0.65 * (1.0 - t);
                    let arm_roll = 0.35 * (1.0 - t) + 0.10 * t;
                    let elbow_x = -0.65 * (1.0 - t) + (-0.25) * t;
                    set_arm_pose(ragdoll, joints, true, (arm_pitch, 0.0, arm_roll), elbow_x);
                    set_arm_pose(ragdoll, joints, false, (arm_pitch, 0.0, -arm_roll), elbow_x);
                }
            }
            set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
            set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

            let target_up = pitched_up(fwd_xz, 0.38 * (1.0 - t));
            apply_pitched_upright_pd(
                &mut bodies[pelvis_idx],
                target_up,
                320.0 * m_ratio * h_ratio * rec_strength,
                22.0 * damp_scale,
                165.0 * m_ratio * h_ratio * rec_strength,
                dt,
            );
            apply_pitched_upright_pd(
                &mut bodies[chest_idx],
                target_up,
                260.0 * m_ratio * h_ratio * rec_strength,
                18.0 * damp_scale,
                135.0 * m_ratio * h_ratio * rec_strength,
                dt,
            );

            let target_y = rec_ground_y + (0.66 + 0.28 * t) * ragdoll.standing_pelvis_y;
            apply_recovery_lift(
                &mut bodies[pelvis_idx],
                target_y,
                total_mass,
                gravity_mag,
                rec_strength,
                dt,
            );

            if grounded_count > 0 {
                apply_horizontal_com_shift(
                    &mut bodies[pelvis_idx],
                    support_center,
                    com_pos,
                    com_vel_xz,
                    total_mass,
                    rec_strength,
                    4.5,
                    2.0,
                    dt,
                );
            }
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

fn set_spine_head(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    lower_x: f32,
    chest_x: f32,
    roll_z: f32,
    neck_x: f32,
    head_x: f32,
) {
    set_target(
        ragdoll,
        joints,
        JointId::LowerSpine,
        Quat::from_euler(EulerRot::XYZ, lower_x, 0.0, roll_z),
    );
    set_target(
        ragdoll,
        joints,
        JointId::ChestSpine,
        Quat::from_euler(EulerRot::XYZ, chest_x, 0.0, roll_z),
    );
    set_target(
        ragdoll,
        joints,
        JointId::Neck,
        Quat::from_rotation_x(neck_x),
    );
    set_target(
        ragdoll,
        joints,
        JointId::Head,
        Quat::from_rotation_x(head_x),
    );
}

fn set_arm_pose(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    is_left: bool,
    shoulder_xyz: (f32, f32, f32),
    elbow_x: f32,
) {
    let (sh, el) = if is_left {
        (JointId::LeftShoulder, JointId::LeftElbow)
    } else {
        (JointId::RightShoulder, JointId::RightElbow)
    };
    set_target(
        ragdoll,
        joints,
        sh,
        Quat::from_euler(
            EulerRot::XYZ,
            shoulder_xyz.0,
            shoulder_xyz.1,
            shoulder_xyz.2,
        ),
    );
    set_target(ragdoll, joints, el, Quat::from_rotation_x(elbow_x));
}

/// Sets asymmetric lead/trail leg joint targets: `(hip_x, hip_abduct, knee_x, ankle_x)`.
fn set_asym_legs(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    lead_left: bool,
    lead: (f32, f32, f32, f32),
    trail: (f32, f32, f32, f32),
) {
    let (left, right) = if lead_left {
        (lead, trail)
    } else {
        (trail, lead)
    };
    set_target(
        ragdoll,
        joints,
        JointId::LeftHip,
        Quat::from_euler(EulerRot::XYZ, left.0, 0.0, left.1),
    );
    set_target(
        ragdoll,
        joints,
        JointId::LeftKnee,
        Quat::from_rotation_x(left.2),
    );
    set_target(
        ragdoll,
        joints,
        JointId::LeftAnkle,
        Quat::from_rotation_x(left.3),
    );
    set_target(
        ragdoll,
        joints,
        JointId::RightHip,
        Quat::from_euler(EulerRot::XYZ, right.0, 0.0, -right.1),
    );
    set_target(
        ragdoll,
        joints,
        JointId::RightKnee,
        Quat::from_rotation_x(right.2),
    );
    set_target(
        ragdoll,
        joints,
        JointId::RightAnkle,
        Quat::from_rotation_x(right.3),
    );
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
    total_mass: f32,
    gravity_mag: f32,
    rec_strength: f32,
    dt: f32,
) {
    if pelvis.inv_mass == 0.0 || pelvis.linear_velocity.y >= 0.75 {
        return;
    }
    let height_err = target_y - pelvis.position.y;
    let fy_raw = (total_mass * gravity_mag * 0.75 + height_err * (total_mass * 32.0)
        - pelvis.linear_velocity.y * (total_mass * 6.0))
        * rec_strength;
    let max_fy = total_mass * gravity_mag * 1.15 * rec_strength;
    let fy = fy_raw.clamp(0.0, max_fy);
    pelvis.linear_velocity.y += (fy * dt) * pelvis.inv_mass;
}
