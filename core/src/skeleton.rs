//! Anthropometric 17-bone humanoid ragdoll skeleton builder.

use glam::{Quat, Vec3};

use crate::body::{ColliderShape, RigidBody};
use crate::joint::{Joint, JointKind};

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
}

/// Spawns a 1.78m, ~76.5 kg anthropometric humanoid ragdoll facing `+Z`
/// with standing foot bottoms resting at `origin.y`.
pub fn spawn_humanoid(
    bodies: &mut Vec<RigidBody>,
    joints: &mut Vec<Joint>,
    ragdoll_id: usize,
    origin: Vec3,
) -> RagdollInstance {
    let base_body = bodies.len();
    let mut body_indices = [0usize; 17];

    let mut add_bone = |bone: BoneId, rel_pos: Vec3, mass: f32, collider: ColliderShape| -> usize {
        let idx = bodies.len();
        let mut rb = RigidBody::new(origin + rel_pos, Quat::IDENTITY, mass, collider);
        rb.ragdoll_id = Some(ragdoll_id);
        rb.bone_id = Some(bone as u8);
        rb.friction = 0.75;
        rb.restitution = 0.02;
        bodies.push(rb);
        body_indices[bone as usize] = idx;
        idx
    };

    // Torso & Head
    let pelvis_pos = Vec3::new(0.0, 0.940, 0.0);
    let lower_spine_pos = Vec3::new(0.0, 1.120, 0.0);
    let chest_pos = Vec3::new(0.0, 1.340, 0.0);
    let neck_pos = Vec3::new(0.0, 1.515, 0.0);
    let head_pos = Vec3::new(0.0, 1.670, 0.0);

    add_bone(
        BoneId::Pelvis,
        pelvis_pos,
        11.0,
        ColliderShape::Box {
            half_extents: Vec3::new(0.135, 0.085, 0.095),
        },
    );
    add_bone(
        BoneId::LowerSpine,
        lower_spine_pos,
        11.0,
        ColliderShape::Capsule {
            radius: 0.115,
            half_length: 0.045,
        },
    );
    add_bone(
        BoneId::Chest,
        chest_pos,
        16.5,
        ColliderShape::Box {
            half_extents: Vec3::new(0.165, 0.130, 0.105),
        },
    );
    add_bone(
        BoneId::Neck,
        neck_pos,
        1.5,
        ColliderShape::Capsule {
            radius: 0.045,
            half_length: 0.025,
        },
    );
    add_bone(
        BoneId::Head,
        head_pos,
        5.0,
        ColliderShape::Capsule {
            radius: 0.095,
            half_length: 0.015,
        },
    );

    // Left Arm (+X side) & Right Arm (-X side)
    let arm_x = 0.215;
    let upper_arm_y = 1.280;
    let forearm_y = 1.000;
    let hand_y = 0.795;

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
                radius: 0.045,
                half_length: 0.105,
            },
        );
        add_bone(
            fa,
            Vec3::new(side * arm_x, forearm_y, 0.0),
            1.3,
            ColliderShape::Capsule {
                radius: 0.038,
                half_length: 0.092,
            },
        );
        add_bone(
            ha,
            Vec3::new(side * arm_x, hand_y, 0.0),
            0.45,
            ColliderShape::Box {
                half_extents: Vec3::new(0.022, 0.075, 0.045),
            },
        );
    }

    // Left Leg (+X side) & Right Leg (-X side)
    let leg_x = 0.095;
    let thigh_y = 0.710;
    let shin_y = 0.285;
    let foot_y = 0.035;
    let foot_z = 0.055;

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
                radius: 0.072,
                half_length: 0.135,
            },
        );
        add_bone(
            sh,
            Vec3::new(side * leg_x, shin_y, 0.0),
            3.6,
            ColliderShape::Capsule {
                radius: 0.055,
                half_length: 0.155,
            },
        );
        add_bone(
            ft,
            Vec3::new(side * leg_x, foot_y, foot_z),
            1.1,
            ColliderShape::Box {
                half_extents: Vec3::new(0.05, 0.035, 0.125),
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
        Vec3::new(0.0, 1.030, 0.0),
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
        Vec3::new(0.0, 1.210, 0.0),
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
        Vec3::new(0.0, 1.470, 0.0),
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
        Vec3::new(0.0, 1.560, 0.0),
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
            Vec3::new(side * arm_x, 1.430, 0.0),
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
            Vec3::new(side * arm_x, 1.130, 0.0),
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
            Vec3::new(side * arm_x, 0.870, 0.0),
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
            Vec3::new(side * leg_x, 0.920, 0.0),
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
            Vec3::new(side * leg_x, 0.500, 0.0),
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
            Vec3::new(side * leg_x, 0.070, 0.0),
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
    }
}
