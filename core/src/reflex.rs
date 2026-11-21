//! Procedural reflexes, injury / reach-for-wound, vestibular head stabilization,
//! and brace-for-impact behaviors.

use glam::{EulerRot, Quat, Vec3};

use crate::balance::{BehaviorState, RecoveryStage};
use crate::body::RigidBody;
use crate::collision::StaticBox;
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
    pub hit_from_back: bool,
}

/// Active environmental hand grab latched onto a static ledge/edge point.
#[derive(Debug, Clone, Copy)]
pub struct HandGrab {
    pub world_point: Vec3,
    pub hold_time: f32,
    pub max_hold_time: f32,
}

/// Approaching dynamic threat detected by the spatial incoming-threat sensor.
#[derive(Debug, Clone, Copy)]
pub struct IncomingThreat {
    pub world_pos: Vec3,
    pub velocity: Vec3,
    pub time_to_impact: f32,
    pub urgency: f32, // 0.0..1.0 (1.0 = imminent impact)
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
    pub left_grab: Option<HandGrab>,
    pub right_grab: Option<HandGrab>,
    pub incoming_threat: Option<IncomingThreat>,
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
            left_grab: None,
            right_grab: None,
            incoming_threat: None,
        }
    }
}

impl RagdollInstance {
    /// Registers a ballistic or traumatic hit on `bone` at `local_point`, updating per-bone
    /// damage, health, consciousness, and the active wound target.
    pub fn register_hit(&mut self, bone: BoneId, local_point: Vec3, severity: f32, damage: f32) {
        self.register_hit_with_dir(bone, local_point, severity, damage, false);
    }

    /// Registers a directional hit on `bone` at `local_point`, recording whether the impact
    /// arrived from behind (`hit_from_back`) for directional spine/arm trauma reflexes.
    pub fn register_hit_with_dir(
        &mut self,
        bone: BoneId,
        local_point: Vec3,
        severity: f32,
        damage: f32,
        hit_from_back: bool,
    ) {
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
        if matches!(
            bone,
            BoneId::LeftUpperArm | BoneId::LeftForearm | BoneId::LeftHand
        ) {
            self.reflex.left_grab = None;
        } else if matches!(
            bone,
            BoneId::RightUpperArm | BoneId::RightForearm | BoneId::RightHand
        ) {
            self.reflex.right_grab = None;
        }

        self.reflex.active_wound = Some(WoundEvent {
            bone,
            region: WoundRegion::from_bone(bone),
            local_point,
            severity: sev,
            time_since_hit: 0.0,
            hit_from_back,
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

/// Applies procedural trauma, vestibular head stabilization, environmental hand grabbing,
/// brace-for-impact, and reach-for-wound reflexes after [`crate::balance::update_ragdoll_balance`]
/// and before [`Joint::apply_muscle_motor`].
pub fn apply_procedural_reflexes(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    static_boxes: &[StaticBox],
    dt: f32,
) {
    if dt <= 0.0 {
        return;
    }

    // 1. Unconscious / Passive check
    if !ragdoll.active_enabled {
        ragdoll.reflex.bracing = false;
        ragdoll.reflex.left_grab = None;
        ragdoll.reflex.right_grab = None;
        return;
    }

    if ragdoll.reflex.unconscious {
        ragdoll.balance.state = BehaviorState::Unconscious;
        ragdoll.reflex.bracing = false;
        ragdoll.reflex.left_grab = None;
        ragdoll.reflex.right_grab = None;
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

    // 4. Environmental Hand Grabbing (Ledge, Wall & Platform Edge Catch)
    let total_mass: f32 = ragdoll
        .body_indices
        .iter()
        .map(|&idx| bodies[idx].mass)
        .sum();
    let (left_grab_active, right_grab_active) = if ragdoll.reflex.health <= 18.0
        || matches!(
            state,
            BehaviorState::Balanced
                | BehaviorState::Recovering
                | BehaviorState::Downed
                | BehaviorState::Passive
                | BehaviorState::Unconscious
        ) {
        ragdoll.reflex.left_grab = None;
        ragdoll.reflex.right_grab = None;
        (false, false)
    } else if matches!(state, BehaviorState::Falling | BehaviorState::Stumbling)
        && ragdoll.reflex.health > 20.0
    {
        let left_active = update_hand_grab(
            ragdoll,
            bodies,
            joints,
            static_boxes,
            true,
            total_mass,
            s_h,
            grav_scale,
            damp_scale,
            dt,
        );
        let right_active = update_hand_grab(
            ragdoll,
            bodies,
            joints,
            static_boxes,
            false,
            total_mass,
            s_h,
            grav_scale,
            damp_scale,
            dt,
        );
        (left_active, right_active)
    } else {
        (
            ragdoll.reflex.left_grab.is_some(),
            ragdoll.reflex.right_grab.is_some(),
        )
    };

    // 5. Brace-for-Impact & Airborne Flail (suppressed per-hand while grabbing/reaching a ledge)
    if state == BehaviorState::Falling
        && ragdoll.balance.com_vel.y < -0.5
        && (!left_grab_active || !right_grab_active)
    {
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

        if !left_grab_active {
            if dir_l.length_squared() > 0.5 {
                set_motor_target(
                    ragdoll,
                    joints,
                    JointId::LeftShoulder,
                    Quat::from_rotation_arc(-Vec3::Y, dir_l),
                );
            }
            set_motor_target(
                ragdoll,
                joints,
                JointId::LeftElbow,
                Quat::from_rotation_x(-0.35),
            );
            let sh = &mut joints[ragdoll.joint_indices[JointId::LeftShoulder as usize]];
            sh.motor_stiffness = sh.motor_stiffness.max(58.0 * grav_scale);
            sh.max_torque = sh.max_torque.max(65.0 * grav_scale);
            let el = &mut joints[ragdoll.joint_indices[JointId::LeftElbow as usize]];
            el.motor_stiffness = el.motor_stiffness.max(36.0 * grav_scale);
            el.max_torque = el.max_torque.max(40.0 * grav_scale);
        }

        if !right_grab_active {
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
                JointId::RightElbow,
                Quat::from_rotation_x(-0.35),
            );
            let sh = &mut joints[ragdoll.joint_indices[JointId::RightShoulder as usize]];
            sh.motor_stiffness = sh.motor_stiffness.max(58.0 * grav_scale);
            sh.max_torque = sh.max_torque.max(65.0 * grav_scale);
            let el = &mut joints[ragdoll.joint_indices[JointId::RightElbow as usize]];
            el.motor_stiffness = el.motor_stiffness.max(36.0 * grav_scale);
            el.max_torque = el.max_torque.max(40.0 * grav_scale);
        }
    } else {
        ragdoll.reflex.bracing = false;
    }

    // 5b. Pre-Impact Defensive Reflex (Look-At -> Head Flinch, Defensive Guard Arms, Torso Lean-Away)
    if let Some(threat) = ragdoll.reflex.incoming_threat {
        if matches!(state, BehaviorState::Balanced | BehaviorState::Stumbling)
            && !ragdoll.reflex.bracing
        {
            apply_threat_anticipation(
                ragdoll,
                bodies,
                joints,
                threat,
                left_grab_active,
                right_grab_active,
                grav_scale,
            );
        }
    }

    // 6. Reach-for-Wound (Guard / Clutch Injury) & Downed Writhe
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
    let torso_back_arch = wound.region == WoundRegion::Torso && wound.hit_from_back;
    let back_arch_active = torso_back_arch && wound.time_since_hit < 1.6;
    let mut spine_pitch = match wound.region {
        WoundRegion::LeftArm | WoundRegion::RightArm => 0.18 * w,
        WoundRegion::Torso => {
            if back_arch_active {
                -0.38 * w
            } else if torso_back_arch {
                let hunch_blend = ((wound.time_since_hit - 1.6) / 0.6).clamp(0.0, 1.0);
                (-0.38 * (1.0 - hunch_blend) + 0.35 * hunch_blend) * w
            } else {
                0.35 * w
            }
        }
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

    if back_arch_active {
        let head_j = &mut joints[ragdoll.joint_indices[JointId::Head as usize]];
        let head_back = Quat::from_rotation_x(-0.25 * w);
        head_j.motor_target = (head_j.motor_target * head_back).normalize();
    }

    // Arm reach-for-wound (skip while actively bracing mid-air)
    if ragdoll.reflex.bracing {
        return;
    }

    let mut reach = |is_left: bool, is_leg_reach: bool| {
        if (is_left && left_grab_active) || (!is_left && right_grab_active) {
            return;
        }
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
        WoundRegion::Torso => {
            if back_arch_active {
                if !left_grab_active {
                    set_motor_target(
                        ragdoll,
                        joints,
                        JointId::LeftShoulder,
                        Quat::from_euler(EulerRot::XYZ, 0.35 * w, 0.0, 0.78 * w),
                    );
                    set_motor_target(
                        ragdoll,
                        joints,
                        JointId::LeftElbow,
                        Quat::from_rotation_x(-0.25),
                    );
                    let sh = &mut joints[ragdoll.joint_indices[JointId::LeftShoulder as usize]];
                    sh.motor_stiffness = sh.motor_stiffness.max(68.0 * grav_scale * w);
                    sh.max_torque = sh.max_torque.max(75.0 * grav_scale * w);
                }
                if !right_grab_active {
                    set_motor_target(
                        ragdoll,
                        joints,
                        JointId::RightShoulder,
                        Quat::from_euler(EulerRot::XYZ, 0.35 * w, 0.0, -0.78 * w),
                    );
                    set_motor_target(
                        ragdoll,
                        joints,
                        JointId::RightElbow,
                        Quat::from_rotation_x(-0.25),
                    );
                    let sh = &mut joints[ragdoll.joint_indices[JointId::RightShoulder as usize]];
                    sh.motor_stiffness = sh.motor_stiffness.max(68.0 * grav_scale * w);
                    sh.max_torque = sh.max_torque.max(75.0 * grav_scale * w);
                }
            } else {
                reach(true, false);
                reach(false, false);
            }
        }
        WoundRegion::Head => {
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

#[allow(clippy::too_many_arguments)]
fn update_hand_grab(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    static_boxes: &[StaticBox],
    is_left: bool,
    total_mass: f32,
    s_h: f32,
    grav_scale: f32,
    damp_scale: f32,
    dt: f32,
) -> bool {
    let (shoulder_jid, elbow_jid, forearm_bid, hand_bid) = if is_left {
        (
            JointId::LeftShoulder,
            JointId::LeftElbow,
            BoneId::LeftForearm,
            BoneId::LeftHand,
        )
    } else {
        (
            JointId::RightShoulder,
            JointId::RightElbow,
            BoneId::RightForearm,
            BoneId::RightHand,
        )
    };

    let healthy = ragdoll.reflex.bone_damage[forearm_bid as usize] < 0.5
        && ragdoll.reflex.bone_damage[hand_bid as usize] < 0.5;
    if !healthy {
        if is_left {
            ragdoll.reflex.left_grab = None;
        } else {
            ragdoll.reflex.right_grab = None;
        }
        return false;
    }

    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let pelvis_idx = ragdoll.body_indices[BoneId::Pelvis as usize];
    let hand_idx = ragdoll.body_indices[hand_bid as usize];
    let shoulder_j_idx = ragdoll.joint_indices[shoulder_jid as usize];
    let shoulder_world = bodies[chest_idx].world_point(joints[shoulder_j_idx].anchor_parent);
    let hand_pos = bodies[hand_idx].position;
    let pelvis_pos = bodies[pelvis_idx].position;
    let chest_pos = bodies[chest_idx].position;

    let profile = ragdoll.profile.sanitized();
    let current_grab = if is_left {
        ragdoll.reflex.left_grab
    } else {
        ragdoll.reflex.right_grab
    };

    if let Some(mut grab) = current_grab {
        grab.hold_time += dt;
        let err = grab.world_point - hand_pos;
        let dist = err.length();
        if grab.hold_time >= grab.max_hold_time || dist > 0.34 * s_h {
            if is_left {
                ragdoll.reflex.left_grab = None;
            } else {
                ragdoll.reflex.right_grab = None;
            }
            return false;
        }

        let max_grip_force = (total_mass
            * 9.81
            * 0.92
            * profile.muscle_ratio
            * (ragdoll.reflex.health / 100.0).clamp(0.4, 1.0))
        .max(200.0);
        let k = total_mass * 85.0;
        let c = total_mass * 10.0;
        let hand = &mut bodies[hand_idx];
        let c_eff = c / (1.0 + hand.inv_mass * dt * c);
        let mut force = err * k - hand.linear_velocity * c_eff;
        let f_len = force.length();
        if f_len > max_grip_force && f_len > 1e-6 {
            force *= max_grip_force / f_len;
        }
        hand.linear_velocity += force * (dt * hand.inv_mass);

        drive_arm_toward_ledge(
            ragdoll,
            bodies,
            joints,
            chest_idx,
            hand_idx,
            shoulder_world,
            shoulder_jid,
            elbow_jid,
            grab.world_point,
            true,
            total_mass,
            s_h,
            grav_scale,
            damp_scale,
            dt,
        );

        if is_left {
            ragdoll.reflex.left_grab = Some(grab);
        } else {
            ragdoll.reflex.right_grab = Some(grab);
        }
        return true;
    }

    let max_reach = 0.78 * s_h;
    let min_ledge_y = (pelvis_pos.y - 0.35 * s_h).max(0.35);
    let max_ledge_y = chest_pos.y + 0.65 * s_h;
    let mut best_point: Option<Vec3> = None;
    let mut best_dist_sq = f32::MAX;

    for sb in static_boxes {
        let top_y = sb.center.y + sb.half_extents.y;
        if top_y < min_ledge_y || top_y > max_ledge_y {
            continue;
        }
        let pt = closest_top_perimeter_point(sb, hand_pos);
        if shoulder_world.distance(pt) <= max_reach {
            let d_sq = hand_pos.distance_squared(pt);
            if d_sq < best_dist_sq {
                best_dist_sq = d_sq;
                best_point = Some(pt);
            }
        }
    }

    let Some(ledge_point) = best_point else {
        return false;
    };

    drive_arm_toward_ledge(
        ragdoll,
        bodies,
        joints,
        chest_idx,
        hand_idx,
        shoulder_world,
        shoulder_jid,
        elbow_jid,
        ledge_point,
        false,
        total_mass,
        s_h,
        grav_scale,
        damp_scale,
        dt,
    );

    if hand_pos.distance(ledge_point) <= 0.18 * s_h {
        let max_hold_time =
            (2.6 * profile.muscle_ratio / (profile.mass_kg / 75.0).sqrt()).clamp(1.0, 4.0);
        let grab = HandGrab {
            world_point: ledge_point,
            hold_time: 0.0,
            max_hold_time,
        };
        if is_left {
            ragdoll.reflex.left_grab = Some(grab);
        } else {
            ragdoll.reflex.right_grab = Some(grab);
        }
        return true;
    }

    true
}

#[inline]
fn closest_top_perimeter_point(sb: &StaticBox, p: Vec3) -> Vec3 {
    let min_x = sb.center.x - sb.half_extents.x;
    let max_x = sb.center.x + sb.half_extents.x;
    let min_z = sb.center.z - sb.half_extents.z;
    let max_z = sb.center.z + sb.half_extents.z;
    let top_y = sb.center.y + sb.half_extents.y;

    let mut cx = p.x.clamp(min_x, max_x);
    let mut cz = p.z.clamp(min_z, max_z);

    if p.x > min_x && p.x < max_x && p.z > min_z && p.z < max_z {
        let dx_min = p.x - min_x;
        let dx_max = max_x - p.x;
        let dz_min = p.z - min_z;
        let dz_max = max_z - p.z;
        let min_d = dx_min.min(dx_max).min(dz_min).min(dz_max);
        if min_d == dx_min {
            cx = min_x;
        } else if min_d == dx_max {
            cx = max_x;
        } else if min_d == dz_min {
            cz = min_z;
        } else {
            cz = max_z;
        }
    }

    Vec3::new(cx, top_y, cz)
}

#[allow(clippy::too_many_arguments)]
fn drive_arm_toward_ledge(
    ragdoll: &RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    chest_idx: usize,
    hand_idx: usize,
    shoulder_world: Vec3,
    shoulder_jid: JointId,
    elbow_jid: JointId,
    target_world: Vec3,
    is_holding: bool,
    total_mass: f32,
    s_h: f32,
    grav_scale: f32,
    damp_scale: f32,
    dt: f32,
) {
    let to_target_world = target_world - shoulder_world;
    let dist = to_target_world.length();
    let mut to_target_chest =
        (bodies[chest_idx].orientation.conjugate() * to_target_world).normalize_or_zero();
    if to_target_chest.length_squared() > 0.25 {
        if to_target_chest.y > 0.7 {
            to_target_chest = (to_target_chest + Vec3::new(0.0, 0.0, 0.2)).normalize_or_zero();
        }
        let q_shoulder = Quat::from_rotation_arc(-Vec3::Y, to_target_chest);
        let arm_len = (0.58 * s_h).max(0.2);
        let flex = if is_holding {
            -0.85 * (1.0 - (dist / arm_len).clamp(0.0, 1.0)).max(0.25)
        } else {
            -0.25
        };
        set_motor_target(ragdoll, joints, shoulder_jid, q_shoulder);
        set_motor_target(ragdoll, joints, elbow_jid, Quat::from_rotation_x(flex));
    }

    let sh = &mut joints[ragdoll.joint_indices[shoulder_jid as usize]];
    sh.motor_stiffness = sh.motor_stiffness.max(95.0 * grav_scale);
    sh.motor_damping = sh.motor_damping.max(1.6 * damp_scale);
    sh.max_torque = sh.max_torque.max(105.0 * grav_scale);

    let el = &mut joints[ragdoll.joint_indices[elbow_jid as usize]];
    el.motor_stiffness = el.motor_stiffness.max(65.0 * grav_scale);
    el.motor_damping = el.motor_damping.max(1.3 * damp_scale);
    el.max_torque = el.max_torque.max(70.0 * grav_scale);

    // Gentle internal reach assist (equal-and-opposite hand <-> chest) while reaching for a ledge
    if !is_holding {
        let mass_scale = total_mass / 76.5;
        let hand_err = target_world - bodies[hand_idx].position;
        let rel_vel = bodies[hand_idx].linear_velocity - bodies[chest_idx].linear_velocity;
        let max_force = 35.0 * mass_scale;
        let mut force = hand_err * (140.0 * mass_scale) - rel_vel * (10.0 * mass_scale);
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

#[inline]
fn blend_motor_target(
    ragdoll: &RagdollInstance,
    joints: &mut [Joint],
    joint_id: JointId,
    target: Quat,
    weight: f32,
) {
    let j_idx = ragdoll.joint_indices[joint_id as usize];
    let j = &mut joints[j_idx];
    let target_joint = (j.frame_parent.conjugate() * target * j.frame_child).normalize();
    j.motor_target = j
        .motor_target
        .slerp(target_joint, weight.clamp(0.0, 1.0))
        .normalize();
}

/// Applies pre-impact look-at/flinch, defensive guard arms, and torso lean-away
/// toward an approaching dynamic threat.
fn apply_threat_anticipation(
    ragdoll: &RagdollInstance,
    bodies: &[RigidBody],
    joints: &mut [Joint],
    threat: IncomingThreat,
    left_grab_active: bool,
    right_grab_active: bool,
    grav_scale: f32,
) {
    let chest_idx = ragdoll.body_indices[BoneId::Chest as usize];
    let chest_rot = bodies[chest_idx].orientation;
    let to_threat_local = (chest_rot.conjugate() * (threat.world_pos - bodies[chest_idx].position))
        .normalize_or_zero();
    if to_threat_local.length_squared() < 1e-6 {
        return;
    }

    let u = threat.urgency.clamp(0.0, 1.0);

    // 1. Look-At -> Head Flinch
    let yaw_to = to_threat_local
        .x
        .atan2(to_threat_local.z)
        .clamp(-1.05, 1.05);
    if u < 0.50 {
        let horiz = to_threat_local.x.hypot(to_threat_local.z).max(0.1);
        let pitch_to = (-to_threat_local.y.atan2(horiz)).clamp(-0.40, 0.40);
        set_motor_target(
            ragdoll,
            joints,
            JointId::Neck,
            (Quat::from_rotation_y(yaw_to * 0.45) * Quat::from_rotation_x(pitch_to * 0.45))
                .normalize(),
        );
        set_motor_target(
            ragdoll,
            joints,
            JointId::Head,
            (Quat::from_rotation_y(yaw_to * 0.55) * Quat::from_rotation_x(pitch_to * 0.55))
                .normalize(),
        );
    } else {
        // Instinctive flinch away from threat side + chin tuck to protect face
        let yaw_away = -to_threat_local.x.clamp(-1.0, 1.0) * 0.35 * u;
        let chin_tuck = 0.28 * u;
        set_motor_target(
            ragdoll,
            joints,
            JointId::Neck,
            (Quat::from_rotation_y(yaw_away * 0.45) * Quat::from_rotation_x(0.12 * u)).normalize(),
        );
        set_motor_target(
            ragdoll,
            joints,
            JointId::Head,
            (Quat::from_rotation_y(yaw_away * 0.55) * Quat::from_rotation_x(chin_tuck)).normalize(),
        );
    }
    for jid in [JointId::Neck, JointId::Head] {
        let j = &mut joints[ragdoll.joint_indices[jid as usize]];
        j.motor_stiffness = j.motor_stiffness.max(55.0 * grav_scale);
        j.max_torque = j.max_torque.max(45.0 * grav_scale);
    }

    // 2. Defensive Guard Arms (healthy, not grabbing a ledge, not clutching an active wound)
    let (left_clutching, right_clutching) = match ragdoll.reflex.active_wound {
        Some(wound) => match wound.region {
            WoundRegion::LeftArm => (false, true),
            WoundRegion::RightArm => (true, false),
            WoundRegion::Torso | WoundRegion::Head => (true, true),
            WoundRegion::LeftLeg => (true, false),
            WoundRegion::RightLeg => (false, true),
            WoundRegion::None => (false, false),
        },
        None => (false, false),
    };

    let blend = u.clamp(0.25, 1.0);
    let elbow_angle = (-1.45 * u - 0.30).clamp(-1.85, -0.35);
    let arms = [
        (
            1.0_f32,
            left_grab_active,
            left_clutching,
            BoneId::LeftUpperArm,
            BoneId::LeftForearm,
            BoneId::LeftHand,
            JointId::LeftShoulder,
            JointId::LeftElbow,
        ),
        (
            -1.0_f32,
            right_grab_active,
            right_clutching,
            BoneId::RightUpperArm,
            BoneId::RightForearm,
            BoneId::RightHand,
            JointId::RightShoulder,
            JointId::RightElbow,
        ),
    ];

    for (side_sign, grab_active, busy_wound, ua_bid, fa_bid, ha_bid, sh_jid, el_jid) in arms {
        let healthy = ragdoll.reflex.bone_damage[ua_bid as usize] < 0.5
            && ragdoll.reflex.bone_damage[fa_bid as usize] < 0.5
            && ragdoll.reflex.bone_damage[ha_bid as usize] < 0.5;
        if !healthy || grab_active || busy_wound {
            continue;
        }

        let aim_dir =
            (to_threat_local * 0.65 + Vec3::new(side_sign * 0.22, 0.55, 0.45)).normalize_or_zero();
        if aim_dir.length_squared() > 0.25 {
            let q_shoulder = Quat::from_rotation_arc(-Vec3::Y, aim_dir);
            blend_motor_target(ragdoll, joints, sh_jid, q_shoulder, blend);
        }
        blend_motor_target(
            ragdoll,
            joints,
            el_jid,
            Quat::from_rotation_x(elbow_angle),
            blend,
        );

        let sh = &mut joints[ragdoll.joint_indices[sh_jid as usize]];
        sh.motor_stiffness = sh.motor_stiffness.max(75.0 * grav_scale * u);
        sh.max_torque = sh.max_torque.max(85.0 * grav_scale * u);

        let el = &mut joints[ragdoll.joint_indices[el_jid as usize]];
        el.motor_stiffness = el.motor_stiffness.max(75.0 * grav_scale * u);
        el.max_torque = el.max_torque.max(85.0 * grav_scale * u);
    }

    // 3. Torso Lean-Away & Stiffen
    let spine_pitch = -to_threat_local.z * 0.22 * u;
    let spine_roll = to_threat_local.x * 0.20 * u;
    let spine_offset = Quat::from_euler(EulerRot::XYZ, spine_pitch, 0.0, spine_roll);
    for jid in [JointId::LowerSpine, JointId::ChestSpine] {
        let j = &mut joints[ragdoll.joint_indices[jid as usize]];
        j.motor_target = (j.motor_target * spine_offset).normalize();
        j.motor_stiffness *= 1.15;
    }
}
