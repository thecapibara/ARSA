//! Three-phase biomechanical get-up recovery controller (`BehaviorState::Recovering`).

use glam::{Quat, Vec3};

use crate::body::RigidBody;
use crate::joint::Joint;
use crate::skeleton::{BoneId, JointId, RagdollInstance};

use super::{apply_pitched_upright_pd, set_target};

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

    if phase < 0.38 {
        // Phase 1 (0.0..0.38): Sit-Up / Push-Up Prep — ZERO pelvis levitation!
        for &j_idx in &ragdoll.joint_indices {
            joints[j_idx].motor_stiffness *= 0.75;
            joints[j_idx].motor_damping *= 0.85;
            joints[j_idx].max_torque *= 0.80;
        }
        for jid in [
            JointId::LeftShoulder,
            JointId::RightShoulder,
            JointId::LeftElbow,
            JointId::RightElbow,
        ] {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_stiffness *= 1.45;
            j.max_torque *= 1.45;
        }

        if ragdoll.balance.recovery_prone {
            set_target(
                ragdoll,
                joints,
                JointId::LowerSpine,
                Quat::from_rotation_x(-0.20),
            );
            set_target(
                ragdoll,
                joints,
                JointId::ChestSpine,
                Quat::from_rotation_x(-0.25),
            );
            set_target(ragdoll, joints, JointId::Neck, Quat::from_rotation_x(-0.20));
            set_target(ragdoll, joints, JointId::Head, Quat::from_rotation_x(-0.35));

            set_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, -1.05, 0.0, 0.30),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, -1.05, 0.0, -0.30),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(-1.35),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightElbow,
                Quat::from_rotation_x(-1.35),
            );

            set_target(
                ragdoll,
                joints,
                JointId::LeftHip,
                Quat::from_rotation_x(-1.35),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftKnee,
                Quat::from_rotation_x(1.75),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftAnkle,
                Quat::from_rotation_x(-0.35),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightHip,
                Quat::from_rotation_x(-0.65),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightKnee,
                Quat::from_rotation_x(1.10),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightAnkle,
                Quat::from_rotation_x(-0.25),
            );
        } else {
            // Supine sit-up curl: tuck knees tightly and reach arms down/back toward floor
            set_target(
                ragdoll,
                joints,
                JointId::LowerSpine,
                Quat::from_rotation_x(0.55),
            );
            set_target(
                ragdoll,
                joints,
                JointId::ChestSpine,
                Quat::from_rotation_x(0.55),
            );
            set_target(ragdoll, joints, JointId::Neck, Quat::from_rotation_x(0.15));
            set_target(ragdoll, joints, JointId::Head, Quat::from_rotation_x(0.35));

            set_target(
                ragdoll,
                joints,
                JointId::LeftHip,
                Quat::from_rotation_x(-1.45),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftKnee,
                Quat::from_rotation_x(1.80),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftAnkle,
                Quat::from_rotation_x(-0.35),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightHip,
                Quat::from_rotation_x(-1.30),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightKnee,
                Quat::from_rotation_x(1.70),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightAnkle,
                Quat::from_rotation_x(-0.35),
            );

            set_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, 0.55, 0.0, 0.25),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_euler(glam::EulerRot::XYZ, 0.55, 0.0, -0.25),
            );
            set_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(-0.45),
            );
            set_target(
                ragdoll,
                joints,
                JointId::RightElbow,
                Quat::from_rotation_x(-0.45),
            );
        }
        set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
        set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

        // Gentle forward-pitch torque on Chest only (no pelvis lift, no full upright on pelvis)
        let pitch_p1 = 0.45_f32;
        let target_up_p1 = (Vec3::Y * pitch_p1.cos() + fwd_xz * pitch_p1.sin()).normalize();
        apply_pitched_upright_pd(
            &mut bodies[chest_idx],
            target_up_p1,
            260.0 * m_ratio * h_ratio * rec_strength * 0.45,
            18.0 * damp_scale * 0.70,
            135.0 * m_ratio * h_ratio * rec_strength * 0.45,
            dt,
        );
    } else if phase < 0.74 {
        // Phase 2 (0.38..0.74): Deep Crouch / Half-Kneel — Torso LEANS FORWARD over feet!
        let t2 = ((phase - 0.38) / 0.36).clamp(0.0, 1.0);
        for &j_idx in &ragdoll.joint_indices {
            joints[j_idx].motor_stiffness *= 0.90;
            joints[j_idx].motor_damping *= 0.92;
            joints[j_idx].max_torque *= 0.90;
        }

        // +0.42 rad total forward crouch lean
        set_target(
            ragdoll,
            joints,
            JointId::LowerSpine,
            Quat::from_rotation_x(0.21),
        );
        set_target(
            ragdoll,
            joints,
            JointId::ChestSpine,
            Quat::from_rotation_x(0.21),
        );
        set_target(ragdoll, joints, JointId::Neck, Quat::from_rotation_x(-0.15));
        set_target(ragdoll, joints, JointId::Head, Quat::from_rotation_x(-0.20));

        // Deep asymmetric crouch/lunge legs
        set_target(
            ragdoll,
            joints,
            JointId::LeftHip,
            Quat::from_rotation_x(-1.15),
        );
        set_target(
            ragdoll,
            joints,
            JointId::LeftKnee,
            Quat::from_rotation_x(1.45),
        );
        set_target(
            ragdoll,
            joints,
            JointId::LeftAnkle,
            Quat::from_rotation_x(-0.35),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightHip,
            Quat::from_rotation_x(-0.75),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightKnee,
            Quat::from_rotation_x(1.15),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightAnkle,
            Quat::from_rotation_x(-0.25),
        );

        // Arms reach forward/outward for counterbalance
        set_target(
            ragdoll,
            joints,
            JointId::LeftShoulder,
            Quat::from_euler(glam::EulerRot::XYZ, -0.65, 0.0, 0.35),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightShoulder,
            Quat::from_euler(glam::EulerRot::XYZ, -0.65, 0.0, -0.35),
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
        set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
        set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

        let pitch_p2 = 0.35_f32;
        let target_up_p2 = (Vec3::Y * pitch_p2.cos() + fwd_xz * pitch_p2.sin()).normalize();
        apply_pitched_upright_pd(
            &mut bodies[pelvis_idx],
            target_up_p2,
            320.0 * m_ratio * h_ratio * rec_strength * 0.75,
            22.0 * damp_scale * 0.85,
            165.0 * m_ratio * h_ratio * rec_strength * 0.75,
            dt,
        );
        apply_pitched_upright_pd(
            &mut bodies[chest_idx],
            target_up_p2,
            260.0 * m_ratio * h_ratio * rec_strength * 0.75,
            18.0 * damp_scale * 0.85,
            135.0 * m_ratio * h_ratio * rec_strength * 0.75,
            dt,
        );

        let target_y = rec_ground_y + (0.38 + 0.26 * t2) * ragdoll.standing_pelvis_y;
        apply_recovery_lift(
            &mut bodies[pelvis_idx],
            target_y,
            total_mass,
            gravity_mag,
            rec_strength,
            dt,
        );

        // Shift CoM horizontally toward mid_ankles_xz so torso stays over the feet
        let horiz_err = Vec3::new(
            mid_ankles_xz.x - com_pos.x,
            0.0,
            mid_ankles_xz.z - com_pos.z,
        );
        let mut f_horiz =
            (horiz_err * (total_mass * 9.0) - com_vel_xz * (total_mass * 6.0)) * rec_strength;
        let max_f = total_mass * 4.5 * rec_strength;
        let f_len = f_horiz.length();
        if f_len > max_f && f_len > 1e-6 {
            f_horiz *= max_f / f_len;
        }
        let pelvis = &mut bodies[pelvis_idx];
        pelvis.linear_velocity += f_horiz * (dt * pelvis.inv_mass);
    } else {
        // Phase 3 (0.74..1.0): Rise from Crouch to Standing
        let t3 = ((phase - 0.74) / 0.26).clamp(0.0, 1.0);
        let spine_pitch = 0.21 * (1.0 - t3);
        set_target(
            ragdoll,
            joints,
            JointId::LowerSpine,
            Quat::from_rotation_x(spine_pitch),
        );
        set_target(
            ragdoll,
            joints,
            JointId::ChestSpine,
            Quat::from_rotation_x(spine_pitch),
        );
        set_target(
            ragdoll,
            joints,
            JointId::Neck,
            Quat::from_rotation_x(-0.15 * (1.0 - t3)),
        );
        set_target(
            ragdoll,
            joints,
            JointId::Head,
            Quat::from_rotation_x(-0.20 * (1.0 - t3)),
        );

        let lh_x = -1.15 * (1.0 - t3) + (-0.05) * t3;
        let lk_x = 1.45 * (1.0 - t3) + 0.10 * t3;
        let la_x = -0.35 * (1.0 - t3) + (-0.05) * t3;
        let rh_x = -0.75 * (1.0 - t3) + (-0.05) * t3;
        let rk_x = 1.15 * (1.0 - t3) + 0.10 * t3;
        let ra_x = -0.25 * (1.0 - t3) + (-0.05) * t3;

        set_target(
            ragdoll,
            joints,
            JointId::LeftHip,
            Quat::from_rotation_x(lh_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::LeftKnee,
            Quat::from_rotation_x(lk_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::LeftAnkle,
            Quat::from_rotation_x(la_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightHip,
            Quat::from_rotation_x(rh_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightKnee,
            Quat::from_rotation_x(rk_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightAnkle,
            Quat::from_rotation_x(ra_x),
        );

        let arm_pitch = -0.65 * (1.0 - t3);
        let arm_roll = 0.35 * (1.0 - t3) + 0.10 * t3;
        let elbow_x = -0.65 * (1.0 - t3) + (-0.25) * t3;
        set_target(
            ragdoll,
            joints,
            JointId::LeftShoulder,
            Quat::from_euler(glam::EulerRot::XYZ, arm_pitch, 0.0, arm_roll),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightShoulder,
            Quat::from_euler(glam::EulerRot::XYZ, arm_pitch, 0.0, -arm_roll),
        );
        set_target(
            ragdoll,
            joints,
            JointId::LeftElbow,
            Quat::from_rotation_x(elbow_x),
        );
        set_target(
            ragdoll,
            joints,
            JointId::RightElbow,
            Quat::from_rotation_x(elbow_x),
        );
        set_target(ragdoll, joints, JointId::LeftWrist, Quat::IDENTITY);
        set_target(ragdoll, joints, JointId::RightWrist, Quat::IDENTITY);

        let pitch_p3 = 0.35 * (1.0 - t3);
        let target_up_p3 = (Vec3::Y * pitch_p3.cos() + fwd_xz * pitch_p3.sin()).normalize();
        apply_pitched_upright_pd(
            &mut bodies[pelvis_idx],
            target_up_p3,
            320.0 * m_ratio * h_ratio * rec_strength,
            22.0 * damp_scale,
            165.0 * m_ratio * h_ratio * rec_strength,
            dt,
        );
        apply_pitched_upright_pd(
            &mut bodies[chest_idx],
            target_up_p3,
            260.0 * m_ratio * h_ratio * rec_strength,
            18.0 * damp_scale,
            135.0 * m_ratio * h_ratio * rec_strength,
            dt,
        );

        let target_y = rec_ground_y + (0.64 + 0.30 * t3) * ragdoll.standing_pelvis_y;
        apply_recovery_lift(
            &mut bodies[pelvis_idx],
            target_y,
            total_mass,
            gravity_mag,
            rec_strength,
            dt,
        );

        if grounded_count > 0 {
            let horiz_err = Vec3::new(
                support_center.x - com_pos.x,
                0.0,
                support_center.z - com_pos.z,
            );
            let mut f_horiz =
                (horiz_err * (total_mass * 4.5) - com_vel_xz * (total_mass * 6.5)) * rec_strength;
            let max_f = total_mass * 2.0 * rec_strength;
            let f_len = f_horiz.length();
            if f_len > max_f && f_len > 1e-6 {
                f_horiz *= max_f / f_len;
            }
            let pelvis = &mut bodies[pelvis_idx];
            pelvis.linear_velocity += f_horiz * (dt * pelvis.inv_mass);
        }
    }
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
