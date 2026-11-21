//! Anthropometric 17-bone humanoid ragdoll skeleton builder.

use glam::{Quat, Vec3};

use crate::balance::BalanceState;
use crate::body::{ColliderShape, RigidBody};
use crate::joint::{Joint, JointKind};

/// Anthropometric body dimensions and muscular strength scaling profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyProfile {
    /// Standing height in meters (clamped to `1.40 .. 2.40`, default `1.78`).
    pub height_m: f32,
    /// Total body mass in kilograms (clamped to `40.0 .. 180.0`, default `75.0`).
    pub mass_kg: f32,
    /// Relative muscle strength multiplier (clamped to `0.25 .. 2.0`, default `1.0`).
    pub muscle_ratio: f32,
}

impl Default for BodyProfile {
    fn default() -> Self {
        Self {
            height_m: 1.78,
            mass_kg: 75.0,
            muscle_ratio: 1.0,
        }
    }
}

impl BodyProfile {
    /// Returns a copy with all fields sanitized against NaN/Inf and clamped to valid bounds.
    pub fn sanitized(self) -> Self {
        Self {
            height_m: if self.height_m.is_finite() {
                self.height_m.clamp(1.40, 2.40)
            } else {
                1.78
            },
            mass_kg: if self.mass_kg.is_finite() {
                self.mass_kg.clamp(40.0, 180.0)
            } else {
                75.0
            },
            muscle_ratio: if self.muscle_ratio.is_finite() {
                self.muscle_ratio.clamp(0.25, 2.0)
            } else {
                1.0
            },
        }
    }
}

/// 16-joint humanoid joint identifiers indexing into `RagdollInstance::joint_indices`.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointId {
    LowerSpine = 0,
    ChestSpine = 1,
    Neck = 2,
    Head = 3,
    LeftShoulder = 4,
    LeftElbow = 5,
    LeftWrist = 6,
    RightShoulder = 7,
    RightElbow = 8,
    RightWrist = 9,
    LeftHip = 10,
    LeftKnee = 11,
    LeftAnkle = 12,
    RightHip = 13,
    RightKnee = 14,
    RightAnkle = 15,
}

/// 17-bone humanoid bone identifiers.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoneId {
    Pelvis = 0,
    LowerSpine = 1,
    Chest = 2,
    Neck = 3,
    Head = 4,
    LeftUpperArm = 5,
    LeftForearm = 6,
    LeftHand = 7,
    RightUpperArm = 8,
    RightForearm = 9,
    RightHand = 10,
    LeftThigh = 11,
    LeftShin = 12,
    LeftFoot = 13,
    RightThigh = 14,
    RightShin = 15,
    RightFoot = 16,
}

/// Spawned humanoid ragdoll handle indexing into `ArsaWorld::bodies` and `ArsaWorld::joints`.
#[derive(Debug, Clone)]
pub struct RagdollInstance {
    pub id: usize,
    pub body_indices: [usize; 17],
    pub joint_indices: [usize; 16],
    pub profile: BodyProfile,
    pub active_enabled: bool,
    pub standing_pelvis_y: f32,
    pub balance: BalanceState,
}

/// Spawns a default 1.78m, 75.0 kg anthropometric humanoid ragdoll facing `+Z`
/// with standing foot bottoms resting at `origin.y`.
pub fn spawn_humanoid(
    bodies: &mut Vec<RigidBody>,
    joints: &mut Vec<Joint>,
    ragdoll_id: usize,
    origin: Vec3,
) -> RagdollInstance {
    spawn_humanoid_with_profile(bodies, joints, ragdoll_id, origin, BodyProfile::default())
}

/// Spawns a procedurally scaled 17-bone humanoid ragdoll matching `profile` facing `+Z`
/// with standing foot bottoms resting at `origin.y`.
pub fn spawn_humanoid_with_profile(
    bodies: &mut Vec<RigidBody>,
    joints: &mut Vec<Joint>,
    ragdoll_id: usize,
    origin: Vec3,
    profile: BodyProfile,
) -> RagdollInstance {
    let profile = profile.sanitized();
    let s_h = profile.height_m / 1.78;
    let s_r = ((profile.mass_kg / 75.0) / s_h).sqrt().clamp(0.72, 1.55);
    let s_w = 0.45 * s_h + 0.55 * s_r;
    let mass_scale = profile.mass_kg / 76.5;

    let base_body = bodies.len();
    let mut body_indices = [0usize; 17];

    let mut add_bone = |bone: BoneId, rel_pos: Vec3, mass: f32, collider: ColliderShape| -> usize {
        let idx = bodies.len();
        let mut rb = RigidBody::new(
            origin + rel_pos,
            Quat::IDENTITY,
            mass * mass_scale,
            collider,
        );
        rb.ragdoll_id = Some(ragdoll_id);
        rb.bone_id = Some(bone as u8);
        rb.friction = 0.75;
        rb.restitution = 0.02;
        bodies.push(rb);
        body_indices[bone as usize] = idx;
        idx
    };

    // Torso & Head
    let standing_pelvis_y = 0.940 * s_h;
    let pelvis_pos = Vec3::new(0.0, standing_pelvis_y, 0.0);
    let lower_spine_pos = Vec3::new(0.0, 1.120 * s_h, 0.0);
    let chest_pos = Vec3::new(0.0, 1.340 * s_h, 0.0);
    let neck_pos = Vec3::new(0.0, 1.515 * s_h, 0.0);
    let head_pos = Vec3::new(0.0, 1.670 * s_h, 0.0);

    add_bone(
        BoneId::Pelvis,
        pelvis_pos,
        11.0,
        ColliderShape::Box {
            half_extents: Vec3::new(0.135 * s_r, 0.085 * s_h, 0.095 * s_r),
        },
    );
    add_bone(
        BoneId::LowerSpine,
        lower_spine_pos,
        11.0,
        ColliderShape::Capsule {
            radius: 0.115 * s_r,
            half_length: 0.045 * s_h,
        },
    );
    add_bone(
        BoneId::Chest,
        chest_pos,
        16.5,
        ColliderShape::Box {
            half_extents: Vec3::new(0.165 * s_r, 0.130 * s_h, 0.105 * s_r),
        },
    );
    add_bone(
        BoneId::Neck,
        neck_pos,
        1.5,
        ColliderShape::Capsule {
            radius: 0.045 * s_r,
            half_length: 0.025 * s_h,
        },
    );
    add_bone(
        BoneId::Head,
        head_pos,
        5.0,
        ColliderShape::Capsule {
            radius: 0.095 * s_r,
            half_length: 0.015 * s_h,
        },
    );

    // Left Arm (+X side) & Right Arm (-X side)
    // Ensure zero chest/upper-arm overlap even for extreme low-height high-mass profiles.
    let arm_x = (0.215 * s_w).max(0.215 * s_r);
    let upper_arm_y = 1.280 * s_h;
    let forearm_y = 1.000 * s_h;
    let hand_y = 0.795 * s_h;

    for (side, ua, fa, ha) in [
        (
            1.0_f32,
            BoneId::LeftUpperArm,
            BoneId::LeftForearm,
            BoneId::LeftHand,
        ),
        (
            -1.0_f32,
            BoneId::RightUpperArm,
            BoneId::RightForearm,
            BoneId::RightHand,
        ),
    ] {
        add_bone(
            ua,
            Vec3::new(side * arm_x, upper_arm_y, 0.0),
            2.1,
            ColliderShape::Capsule {
                radius: 0.045 * s_r,
                half_length: 0.105 * s_h,
            },
        );
        add_bone(
            fa,
            Vec3::new(side * arm_x, forearm_y, 0.0),
            1.3,
            ColliderShape::Capsule {
                radius: 0.038 * s_r,
                half_length: 0.092 * s_h,
            },
        );
        add_bone(
            ha,
            Vec3::new(side * arm_x, hand_y, 0.0),
            0.45,
            ColliderShape::Box {
                half_extents: Vec3::new(0.022 * s_r, 0.075 * s_h, 0.045 * s_r),
            },
        );
    }

    // Left Leg (+X side) & Right Leg (-X side)
    let leg_x = (0.095 * s_w).max(0.078 * s_r);
    let thigh_y = 0.710 * s_h;
    let shin_y = 0.285 * s_h;
    let foot_y = 0.035 * s_h;
    let foot_z = 0.055 * s_h;

    for (side, th, sh, ft) in [
        (
            1.0_f32,
            BoneId::LeftThigh,
            BoneId::LeftShin,
            BoneId::LeftFoot,
        ),
        (
            -1.0_f32,
            BoneId::RightThigh,
            BoneId::RightShin,
            BoneId::RightFoot,
        ),
    ] {
        add_bone(
            th,
            Vec3::new(side * leg_x, thigh_y, 0.0),
            7.2,
            ColliderShape::Capsule {
                radius: 0.072 * s_r,
                half_length: 0.135 * s_h,
            },
        );
        add_bone(
            sh,
            Vec3::new(side * leg_x, shin_y, 0.0),
            3.6,
            ColliderShape::Capsule {
                radius: 0.055 * s_r,
                half_length: 0.155 * s_h,
            },
        );
        add_bone(
            ft,
            Vec3::new(side * leg_x, foot_y, foot_z),
            1.1,
            ColliderShape::Box {
                half_extents: Vec3::new(0.05 * s_r, 0.035 * s_h, 0.125 * s_h),
            },
        );
    }

    let b = |bone: BoneId| base_body + bone as usize;

    let mut joint_indices = [0usize; 16];
    let mut j_cursor = 0usize;
    let mut add_joint = |parent: BoneId,
                         child: BoneId,
                         world_anchor_rel: Vec3,
                         kind: JointKind,
                         damping: f32| {
        let p_idx = b(parent);
        let c_idx = b(child);
        let anchor_world = origin + world_anchor_rel;
        let anchor_parent = anchor_world - bodies[p_idx].position;
        let anchor_child = anchor_world - bodies[c_idx].position;

        let j_idx = joints.len();
        joints.push(
            Joint::new(p_idx, c_idx, anchor_parent, anchor_child, kind)
                .with_solver_params(0.0, damping),
        );
        joint_indices[j_cursor] = j_idx;
        j_cursor += 1;
    };

    // Spine & Head chain
    add_joint(
        BoneId::Pelvis,
        BoneId::LowerSpine,
        Vec3::new(0.0, 1.030 * s_h, 0.0),
        JointKind::Spherical {
            swing_x_limit: 0.45,
            swing_z_limit: 0.35,
            twist_min: -0.45,
            twist_max: 0.45,
        },
        10.0,
    );
    add_joint(
        BoneId::LowerSpine,
        BoneId::Chest,
        Vec3::new(0.0, 1.210 * s_h, 0.0),
        JointKind::Spherical {
            swing_x_limit: 0.45,
            swing_z_limit: 0.35,
            twist_min: -0.45,
            twist_max: 0.45,
        },
        10.0,
    );
    add_joint(
        BoneId::Chest,
        BoneId::Neck,
        Vec3::new(0.0, 1.470 * s_h, 0.0),
        JointKind::Spherical {
            swing_x_limit: 0.45,
            swing_z_limit: 0.40,
            twist_min: -0.55,
            twist_max: 0.55,
        },
        6.0,
    );
    add_joint(
        BoneId::Neck,
        BoneId::Head,
        Vec3::new(0.0, 1.560 * s_h, 0.0),
        JointKind::Spherical {
            swing_x_limit: 0.50,
            swing_z_limit: 0.40,
            twist_min: -0.65,
            twist_max: 0.65,
        },
        6.0,
    );

    // Arms (Elbows bend strictly forward toward +Z: [-2.50, 0.0] around local X)
    for (side, ua, fa, ha) in [
        (
            1.0_f32,
            BoneId::LeftUpperArm,
            BoneId::LeftForearm,
            BoneId::LeftHand,
        ),
        (
            -1.0_f32,
            BoneId::RightUpperArm,
            BoneId::RightForearm,
            BoneId::RightHand,
        ),
    ] {
        add_joint(
            BoneId::Chest,
            ua,
            Vec3::new(side * arm_x, 1.430 * s_h, 0.0),
            JointKind::Spherical {
                swing_x_limit: 1.55,
                swing_z_limit: 1.45,
                twist_min: -1.10,
                twist_max: 1.10,
            },
            5.0,
        );
        add_joint(
            ua,
            fa,
            Vec3::new(side * arm_x, 1.130 * s_h, 0.0),
            JointKind::Hinge {
                axis: Vec3::X,
                min_angle: -2.50,
                max_angle: 0.0,
            },
            4.5,
        );
        add_joint(
            fa,
            ha,
            Vec3::new(side * arm_x, 0.870 * s_h, 0.0),
            JointKind::Spherical {
                swing_x_limit: 0.85,
                swing_z_limit: 0.50,
                twist_min: -0.80,
                twist_max: 0.80,
            },
            3.5,
        );
    }

    // Legs (Knees bend strictly backward toward -Z: [0.0, 2.45] around local X)
    for (side, th, sh, ft) in [
        (
            1.0_f32,
            BoneId::LeftThigh,
            BoneId::LeftShin,
            BoneId::LeftFoot,
        ),
        (
            -1.0_f32,
            BoneId::RightThigh,
            BoneId::RightShin,
            BoneId::RightFoot,
        ),
    ] {
        add_joint(
            BoneId::Pelvis,
            th,
            Vec3::new(side * leg_x, 0.920 * s_h, 0.0),
            JointKind::Spherical {
                swing_x_limit: 1.25,
                swing_z_limit: 0.75,
                twist_min: -0.55,
                twist_max: 0.55,
            },
            7.0,
        );
        add_joint(
            th,
            sh,
            Vec3::new(side * leg_x, 0.500 * s_h, 0.0),
            JointKind::Hinge {
                axis: Vec3::X,
                min_angle: 0.0,
                max_angle: 2.45,
            },
            5.5,
        );
        add_joint(
            sh,
            ft,
            Vec3::new(side * leg_x, 0.070 * s_h, 0.0),
            JointKind::Spherical {
                swing_x_limit: 0.65,
                swing_z_limit: 0.35,
                twist_min: -0.35,
                twist_max: 0.35,
            },
            4.0,
        );
    }

    RagdollInstance {
        id: ragdoll_id,
        body_indices,
        joint_indices,
        profile,
        active_enabled: true,
        standing_pelvis_y,
        balance: BalanceState::default(),
    }
}
