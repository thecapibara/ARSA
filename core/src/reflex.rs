//! Procedural reflexes, injury / reach-for-wound, vestibular head stabilization,
//! and brace-for-impact behaviors.

use glam::{EulerRot, Quat, Vec3};

use crate::balance::{BehaviorState, RecoveryStage};
use crate::body::RigidBody;
use crate::joint::Joint;
use crate::skeleton::{BoneId, JointId, RagdollInstance};

/// Anatomical region of a wound on a humanoid ragdoll.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WoundRegion {
    None = 0,
    LeftArm = 1,
    RightArm = 2,
    Torso = 3,
    Head = 4,
    LeftLeg = 5,
    RightLeg = 6,
}

impl WoundRegion {
    /// Maps any of the 17 humanoid bones to its anatomical [`WoundRegion`].
    pub fn from_bone(bone: BoneId) -> WoundRegion {
        match bone {
            BoneId::LeftUpperArm | BoneId::LeftForearm | BoneId::LeftHand => WoundRegion::LeftArm,
            BoneId::RightUpperArm | BoneId::RightForearm | BoneId::RightHand => {
                WoundRegion::RightArm
            }
            BoneId::Pelvis | BoneId::LowerSpine | BoneId::Chest => WoundRegion::Torso,
            BoneId::Neck | BoneId::Head => WoundRegion::Head,
            BoneId::LeftThigh | BoneId::LeftShin | BoneId::LeftFoot => WoundRegion::LeftLeg,
            BoneId::RightThigh | BoneId::RightShin | BoneId::RightFoot => WoundRegion::RightLeg,
        }
    }
}

/// Active injury event tracked on a ragdoll for reach-for-wound and trauma reflexes.
#[derive(Debug, Clone, Copy)]
pub struct WoundEvent {
    pub bone: BoneId,
    pub region: WoundRegion,
    pub local_point: Vec3,
    pub severity: f32,
    pub time_since_hit: f32,
}

/// Per-ragdoll trauma, health, consciousness, and procedural reflex state.
#[derive(Debug, Clone)]
pub struct ReflexState {
    pub health: f32,
    pub active_wound: Option<WoundEvent>,
    pub bone_damage: [f32; 17],
    pub bracing: bool,
    pub unconscious: bool,
    pub writhe_time: f32,
}

impl Default for ReflexState {
    fn default() -> Self {
        Self {
            health: 100.0,
            active_wound: None,
            bone_damage: [0.0; 17],
            bracing: false,
            unconscious: false,
            writhe_time: 0.0,
        }
    }
}

impl RagdollInstance {
    /// Registers a ballistic or traumatic hit on `bone` at `local_point`, updating per-bone
    /// damage, health, consciousness, and the active wound target.
    pub fn register_hit(&mut self, bone: BoneId, local_point: Vec3, severity: f32, damage: f32) {
        let sev = if severity.is_finite() {
            severity.max(0.0)
        } else {
            0.0
        };
        let dmg = if damage.is_finite() {
            damage.max(0.0)
        } else {
            0.0
        };

        let idx = bone as usize;
        self.reflex.bone_damage[idx] = (self.reflex.bone_damage[idx] + sev * 0.85).min(1.0);
        self.reflex.health = (self.reflex.health - dmg).max(0.0);

        if (bone == BoneId::Head && sev >= 0.72) || self.reflex.health <= 10.0 {
            self.reflex.unconscious = true;
        }

        self.reflex.active_wound = Some(WoundEvent {
            bone,
            region: WoundRegion::from_bone(bone),
            local_point,
            severity: sev,
            time_since_hit: 0.0,
        });

        self.balance.downed_still_time = 0.0;
        if self.balance.state == BehaviorState::Recovering {
            self.balance.recovery_phase = 0.0;
            self.balance.recovery_stage = RecoveryStage::RollOrPrep;
            self.balance.recovery_stage_time = 0.0;
            self.balance.state = BehaviorState::Falling;
        }
    }
}

/// Applies procedural trauma, vestibular head stabilization, brace-for-impact, and
/// reach-for-wound reflexes after [`crate::balance::update_ragdoll_balance`] and before
/// [`Joint::apply_muscle_motor`].
pub fn apply_procedural_reflexes(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    dt: f32,
) {
    if dt <= 0.0 {
        return;
    }

    // 1. Unconscious / Passive check
    if !ragdoll.active_enabled {
        ragdoll.reflex.bracing = false;
        return;
    }

    if ragdoll.reflex.unconscious {
        ragdoll.balance.state = BehaviorState::Unconscious;
        ragdoll.reflex.bracing = false;
        for &j_idx in &ragdoll.joint_indices {
            joints[j_idx].motor_stiffness = 0.0;
            joints[j_idx].motor_damping = 0.0;
            joints[j_idx].max_torque = 0.0;
        }
        return;
    }

    ragdoll.reflex.writhe_time += dt;

    // 2. Per-limb damage scaling from child bone damage
    for &j_idx in &ragdoll.joint_indices {
        let child_idx = joints[j_idx].child;
        if let Some(bone_u8) = bodies[child_idx].bone_id {
            let dmg = ragdoll.reflex.bone_damage[bone_u8 as usize];
            joints[j_idx].damage_scale = (1.0 - 0.72 * dmg).clamp(0.18, 1.0);
        } else {
            joints[j_idx].damage_scale = 1.0;
        }
    }

    let profile = ragdoll.profile.sanitized();
    let s_h = profile.height_m / 1.78;
    let mass_scale = profile.mass_kg / 76.5;
    let grav_scale = (profile.mass_kg / 75.0) * s_h * profile.muscle_ratio;
    let damp_scale = ((profile.mass_kg / 75.0) * s_h * s_h * profile.muscle_ratio).sqrt();

    let state = ragdoll.balance.state;
    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let pelvis_idx = ragdoll.body_indices[BoneId::Pelvis as usize];
    let chest_rot = bodies[chest_idx].orientation;

    // 3. Vestibular head stabilization (conscious Balanced / Stumbling / Falling)
    if matches!(
        state,
        BehaviorState::Balanced | BehaviorState::Stumbling | BehaviorState::Falling
    ) {
        let up_in_chest = (chest_rot.conjugate() * Vec3::Y).normalize_or_zero();
        if up_in_chest.length_squared() > 0.5 {
            let q_counter = Quat::from_rotation_arc(Vec3::Y, up_in_chest);
            let chin_tuck = match state {
                BehaviorState::Falling => 0.18,
                BehaviorState::Stumbling => 0.08,
                _ => 0.03,
            };
            let q_neck = Quat::IDENTITY.slerp(q_counter, 0.45);
            let q_head = Quat::IDENTITY.slerp(q_counter, 0.55) * Quat::from_rotation_x(chin_tuck);
            set_motor_target(ragdoll, joints, JointId::Neck, q_neck);
            set_motor_target(ragdoll, joints, JointId::Head, q_head);

            // Keep neck & head active enough to resist whiplash during stumbles/falls
            for jid in [JointId::Neck, JointId::Head] {
                let j = &mut joints[ragdoll.joint_indices[jid as usize]];
                j.motor_stiffness = j.motor_stiffness.max(30.0 * grav_scale);
                j.max_torque = j.max_torque.max(26.0 * grav_scale);
            }
        }
    }

    // 4. Brace-for-Impact & Airborne Flail
    if state == BehaviorState::Falling && ragdoll.balance.com_vel.y < -0.5 {
        ragdoll.reflex.bracing = true;

        let fall_world = ragdoll.balance.com_vel.normalize_or_zero();
        let fall_local = (chest_rot.conjugate() * fall_world).normalize_or_zero();

        let mut dir_l = (fall_local + Vec3::new(0.28, 0.0, 0.22)).normalize_or_zero();
        let mut dir_r = (fall_local + Vec3::new(-0.28, 0.0, 0.22)).normalize_or_zero();

        if bodies[pelvis_idx].position.y > 1.45 * ragdoll.standing_pelvis_y {
            let wm = (ragdoll.reflex.writhe_time * 11.0).sin();
            dir_l =
                (Quat::from_euler(EulerRot::XYZ, wm * 0.55, 0.0, 0.20) * dir_l).normalize_or_zero();
            dir_r = (Quat::from_euler(EulerRot::XYZ, -wm * 0.55, 0.0, -0.20) * dir_r)
                .normalize_or_zero();
        }

        if dir_l.length_squared() > 0.5 {
            set_motor_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_rotation_arc(-Vec3::Y, dir_l),
            );
        }
        if dir_r.length_squared() > 0.5 {
            set_motor_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_rotation_arc(-Vec3::Y, dir_r),
            );
        }
        set_motor_target(
            ragdoll,
            joints,
            JointId::LeftElbow,
            Quat::from_rotation_x(-0.35),
        );
        set_motor_target(
            ragdoll,
            joints,
            JointId::RightElbow,
            Quat::from_rotation_x(-0.35),
        );

        for (jid, base_k, base_t) in [
            (JointId::LeftShoulder, 58.0, 65.0),
            (JointId::RightShoulder, 58.0, 65.0),
            (JointId::LeftElbow, 36.0, 40.0),
            (JointId::RightElbow, 36.0, 40.0),
        ] {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_stiffness = j.motor_stiffness.max(base_k * grav_scale);
            j.max_torque = j.max_torque.max(base_t * grav_scale);
        }
    } else {
        ragdoll.reflex.bracing = false;
    }

    // 5. Reach-for-Wound (Guard / Clutch Injury) & Downed Writhe
    let Some(mut wound) = ragdoll.reflex.active_wound else {
        return;
    };
    wound.time_since_hit += dt;
    ragdoll.reflex.active_wound = Some(wound);

    let wound_body_idx = ragdoll.body_indices[wound.bone as usize];
    let wound_world = bodies[wound_body_idx].world_point(wound.local_point);
    let w = (1.0 - (wound.time_since_hit / 12.0).clamp(0.0, 1.0)).max(0.25)
        * wound.severity.clamp(0.4, 1.0);

    // Injured leg buckling (drop hip/knee/ankle damage_scale to 0.25)
    match wound.region {
        WoundRegion::LeftLeg => {
            for jid in [JointId::LeftHip, JointId::LeftKnee, JointId::LeftAnkle] {
                let j = &mut joints[ragdoll.joint_indices[jid as usize]];
                j.damage_scale = j.damage_scale.min(0.25);
            }
        }
        WoundRegion::RightLeg => {
            for jid in [JointId::RightHip, JointId::RightKnee, JointId::RightAnkle] {
                let j = &mut joints[ragdoll.joint_indices[jid as usize]];
                j.damage_scale = j.damage_scale.min(0.25);
            }
        }
        _ => {}
    }

    // Do not hijack healthy arms or spine hunch while actively standing up (`Recovering`);
    // per-limb `damage_scale` weakening remains active above, and Reach-for-Wound resumes
    // once back in `Balanced` / `Stumbling`.
    if state == BehaviorState::Recovering {
        return;
    }

    // Spine hunch / twist / curl + Downed writhe
    let mut spine_pitch = match wound.region {
        WoundRegion::LeftArm | WoundRegion::RightArm => 0.18 * w,
        WoundRegion::Torso => 0.35 * w,
        WoundRegion::Head => 0.12 * w,
        WoundRegion::LeftLeg | WoundRegion::RightLeg => 0.22 * w,
        WoundRegion::None => 0.0,
    };
    let spine_yaw = match wound.region {
        WoundRegion::LeftArm => 0.20 * w,
        WoundRegion::RightArm => -0.20 * w,
        _ => 0.0,
    };
    let mut spine_roll = 0.0_f32;

    if state == BehaviorState::Downed {
        let writhe = (ragdoll.reflex.writhe_time * 2.2).sin();
        let writhe_c = (ragdoll.reflex.writhe_time * 1.5).cos();
        spine_pitch += 0.22 * w + 0.08 * writhe;
        spine_roll += 0.06 * writhe_c;
        for jid in [JointId::LowerSpine, JointId::ChestSpine] {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_stiffness = j.motor_stiffness.max(70.0 * grav_scale);
            j.motor_damping = j.motor_damping.max(4.5 * damp_scale);
            j.max_torque = j.max_torque.max(65.0 * grav_scale);
        }
    }

    let spine_offset = Quat::from_euler(EulerRot::XYZ, spine_pitch, spine_yaw, spine_roll);
    for jid in [JointId::LowerSpine, JointId::ChestSpine] {
        if matches!(state, BehaviorState::Downed | BehaviorState::Falling) {
            set_motor_target(ragdoll, joints, jid, spine_offset);
        } else {
            let j = &mut joints[ragdoll.joint_indices[jid as usize]];
            j.motor_target = (j.motor_target * spine_offset).normalize();
        }
    }

    // Arm reach-for-wound (skip while actively bracing mid-air)
    if ragdoll.reflex.bracing {
        return;
    }

    let mut reach = |is_left: bool, is_leg_reach: bool| {
        reach_arm_to_target(
            ragdoll,
            bodies,
            joints,
            is_left,
            wound_world,
            w,
            is_leg_reach,
            s_h,
            mass_scale,
            grav_scale,
            damp_scale,
            dt,
        );
    };

    match wound.region {
        WoundRegion::LeftArm => {
            reach(false, false);
            // Slightly tuck the injured left arm inward
            set_motor_target(
                ragdoll,
                joints,
                JointId::LeftShoulder,
                Quat::from_euler(EulerRot::XYZ, -0.25, 0.0, -0.10),
            );
            set_motor_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(-0.65),
            );
        }
        WoundRegion::RightArm => {
            reach(true, false);
            // Slightly tuck the injured right arm inward
            set_motor_target(
                ragdoll,
                joints,
                JointId::RightShoulder,
                Quat::from_euler(EulerRot::XYZ, -0.25, 0.0, 0.10),
            );
            set_motor_target(
                ragdoll,
                joints,
                JointId::RightElbow,
                Quat::from_rotation_x(-0.65),
            );
        }
        WoundRegion::Torso | WoundRegion::Head => {
            reach(true, false);
            reach(false, false);
        }
        WoundRegion::LeftLeg => {
            reach(true, true);
        }
        WoundRegion::RightLeg => {
            reach(false, true);
        }
        WoundRegion::None => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn reach_arm_to_target(
    ragdoll: &RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    is_left: bool,
    target_world: Vec3,
    w: f32,
    is_leg_reach: bool,
    s_h: f32,
    mass_scale: f32,
    grav_scale: f32,
    damp_scale: f32,
    dt: f32,
) {
    let side_sign = if is_left { 1.0_f32 } else { -1.0_f32 };
    let (shoulder_jid, elbow_jid, hand_bid) = if is_left {
        (JointId::LeftShoulder, JointId::LeftElbow, BoneId::LeftHand)
    } else {
        (
            JointId::RightShoulder,
            JointId::RightElbow,
            BoneId::RightHand,
        )
    };

    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let shoulder_j_idx = ragdoll.joint_indices[shoulder_jid as usize];
    let shoulder_world = bodies[chest_idx].world_point(joints[shoulder_j_idx].anchor_parent);

    let to_target_world = target_world - shoulder_world;
    let dist = to_target_world.length();
    let to_target_chest =
        (bodies[chest_idx].orientation.conjugate() * to_target_world).normalize_or_zero();

    if to_target_chest.length_squared() > 0.25 {
        let (q_shoulder, elbow_angle) = if is_leg_reach {
            let aim_dir =
                (to_target_chest + Vec3::new(side_sign * 0.08, -0.15, 0.25)).normalize_or_zero();
            (Quat::from_rotation_arc(-Vec3::Y, aim_dir), -0.55)
        } else {
            let aim_dir =
                (to_target_chest + Vec3::new(side_sign * 0.10, -0.22, 0.42)).normalize_or_zero();
            // Internal twist rotates upper-arm +X so negative elbow flexion folds forearm across chest
            let internal_twist = -side_sign * 0.85;
            let q_s = (Quat::from_rotation_arc(-Vec3::Y, aim_dir)
                * Quat::from_rotation_y(internal_twist))
            .normalize();
            let arm_reach = (0.56 * s_h).max(0.2);
            let t = (dist / arm_reach).clamp(0.0, 1.0);
            (q_s, -1.65 + 0.40 * t)
        };

        set_motor_target(ragdoll, joints, shoulder_jid, q_shoulder);
        set_motor_target(
            ragdoll,
            joints,
            elbow_jid,
            Quat::from_rotation_x(elbow_angle),
        );
    }

    // Ensure reaching arm has responsive motor authority (softer when Downed so he rests calmly)
    let arm_downed_scale = if ragdoll.balance.state == BehaviorState::Downed {
        0.35
    } else {
        1.0
    };
    let sh = &mut joints[ragdoll.joint_indices[shoulder_jid as usize]];
    sh.motor_stiffness = sh
        .motor_stiffness
        .max(85.0 * grav_scale * w * arm_downed_scale);
    sh.motor_damping = sh.motor_damping.max(1.4 * damp_scale);
    sh.max_torque = sh.max_torque.max(95.0 * grav_scale * w * arm_downed_scale);

    let el = &mut joints[ragdoll.joint_indices[elbow_jid as usize]];
    el.motor_stiffness = el
        .motor_stiffness
        .max(60.0 * grav_scale * w * arm_downed_scale);
    el.motor_damping = el.motor_damping.max(1.2 * damp_scale);
    el.max_torque = el.max_torque.max(65.0 * grav_scale * w * arm_downed_scale);

    // Strictly force-clamped physical reach force on hand with equal-and-opposite reaction on Chest.
    // Stop pulling once the hand reaches surface contact radius (~0.095 * s_h) so it rests calmly
    // on the wound without fighting Hand-vs-Chest self-collision (no downed jitter).
    let hand_idx = ragdoll.body_indices[hand_bid as usize];
    let raw_err = target_world - bodies[hand_idx].position;
    let arrival_radius = 0.095 * s_h;
    let err_len = raw_err.length();
    let eff_err = if err_len > arrival_radius {
        raw_err * ((err_len - arrival_radius) / err_len)
    } else {
        Vec3::ZERO
    };
    let downed_scale = if ragdoll.balance.state == BehaviorState::Downed {
        0.45
    } else {
        1.0
    };
    let rel_vel = bodies[hand_idx].linear_velocity - bodies[chest_idx].linear_velocity;
    let max_force = 45.0 * mass_scale * w * downed_scale;
    let mut force =
        eff_err * (160.0 * mass_scale * w * downed_scale) - rel_vel * (14.0 * mass_scale);
    let f_len = force.length();
    if f_len > max_force && f_len > 1e-6 {
        force *= max_force / f_len;
    }

    let impulse = force * dt;
    let inv_m_hand = bodies[hand_idx].inv_mass;
    let inv_m_chest = bodies[chest_idx].inv_mass;
    bodies[hand_idx].linear_velocity += impulse * inv_m_hand;
    bodies[chest_idx].linear_velocity -= impulse * inv_m_chest;
}

#[inline]
fn set_motor_target(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    joint_id: JointId,
    target: Quat,
) {
    let j_idx = ragdoll.joint_indices[joint_id as usize];
    let j = &mut joints[j_idx];
    j.motor_target = (j.frame_parent.conjugate() * target * j.frame_child).normalize();
}
