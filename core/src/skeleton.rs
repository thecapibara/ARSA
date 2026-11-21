//! Anthropometric 17-bone humanoid ragdoll skeleton builder.

use glam::{Quat, Vec3};

use crate::balance::BalanceState;
use crate::body::{ColliderShape, RigidBody};
use crate::joint::{Joint, JointKind};
use crate::math::{inertia_box, inertia_capsule, inertia_sphere};
use crate::reflex::ReflexState;

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

impl BoneId {
    /// Converts a raw `u8` bone index (`0..17`) into a [`BoneId`].
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Pelvis),
            1 => Some(Self::LowerSpine),
            2 => Some(Self::Chest),
            3 => Some(Self::Neck),
            4 => Some(Self::Head),
            5 => Some(Self::LeftUpperArm),
            6 => Some(Self::LeftForearm),
            7 => Some(Self::LeftHand),
            8 => Some(Self::RightUpperArm),
            9 => Some(Self::RightForearm),
            10 => Some(Self::RightHand),
            11 => Some(Self::LeftThigh),
            12 => Some(Self::LeftShin),
            13 => Some(Self::LeftFoot),
            14 => Some(Self::RightThigh),
            15 => Some(Self::RightShin),
            16 => Some(Self::RightFoot),
            _ => None,
        }
    }
}

/// Spawned humanoid ragdoll handle indexing into `ArsaWorld::bodies` and `ArsaWorld::joints`.
#[derive(Debug, Clone)]
pub struct RagdollInstance {
    pub id: usize,
    pub body_indices: [usize; 17],
    pub joint_indices: [usize; 16],
    pub profile: BodyProfile,
    pub active_enabled: bool,
    pub auto_get_up: bool,
    pub standing_pelvis_y: f32,
    pub balance: BalanceState,
    pub reflex: ReflexState,
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

#[derive(Clone, Copy)]
struct BoneSpec {
    bone: BoneId,
    rel_pos: Vec3,
    mass: f32,
    collider: ColliderShape,
}

#[derive(Clone, Copy)]
struct JointSpec {
    parent: BoneId,
    child: BoneId,
    world_anchor_rel: Vec3,
    frame_parent: Quat,
    kind: JointKind,
    damping: f32,
}

struct HumanoidSpec {
    profile: BodyProfile,
    standing_pelvis_y: f32,
    bones: [BoneSpec; 17],
    joints: [JointSpec; 16],
}

fn compute_humanoid_spec(profile: BodyProfile) -> HumanoidSpec {
    let profile = profile.sanitized();
    let s_h = profile.height_m / 1.78;
    let s_r = ((profile.mass_kg / 75.0) / s_h).sqrt().clamp(0.72, 1.55);
    let s_w = 0.45 * s_h + 0.55 * s_r;
    let mass_scale = profile.mass_kg / 76.5;

    let standing_pelvis_y = 0.940 * s_h;
    let arm_x = (0.215 * s_w).max(0.215 * s_r);
    let upper_arm_y = 1.280 * s_h;
    let forearm_y = 1.000 * s_h;
    let hand_y = 0.795 * s_h;

    let leg_x = (0.095 * s_w).max(0.078 * s_r);
    let thigh_y = 0.710 * s_h;
    let shin_y = 0.285 * s_h;
    let foot_y = 0.035 * s_h;
    let foot_z = 0.055 * s_h;

    let bone = |id: BoneId, rel_pos: Vec3, base_mass: f32, collider: ColliderShape| BoneSpec {
        bone: id,
        rel_pos,
        mass: base_mass * mass_scale,
        collider,
    };

    let bones = [
        bone(
            BoneId::Pelvis,
            Vec3::new(0.0, standing_pelvis_y, 0.0),
            11.0,
            ColliderShape::Box {
                half_extents: Vec3::new(0.135 * s_r, 0.085 * s_h, 0.095 * s_r),
            },
        ),
        bone(
            BoneId::LowerSpine,
            Vec3::new(0.0, 1.120 * s_h, 0.0),
            11.0,
            ColliderShape::Capsule {
                radius: 0.115 * s_r,
                half_length: 0.045 * s_h,
            },
        ),
        bone(
            BoneId::Chest,
            Vec3::new(0.0, 1.340 * s_h, 0.0),
            16.5,
            ColliderShape::Box {
                half_extents: Vec3::new(0.165 * s_r, 0.130 * s_h, 0.105 * s_r),
            },
        ),
        bone(
            BoneId::Neck,
            Vec3::new(0.0, 1.515 * s_h, 0.0),
            1.5,
            ColliderShape::Capsule {
                radius: 0.045 * s_r,
                half_length: 0.025 * s_h,
            },
        ),
        bone(
            BoneId::Head,
            Vec3::new(0.0, 1.670 * s_h, 0.0),
            5.0,
            ColliderShape::Capsule {
                radius: 0.095 * s_r,
                half_length: 0.015 * s_h,
            },
        ),
        // Left Arm (+X side)
        bone(
            BoneId::LeftUpperArm,
            Vec3::new(arm_x, upper_arm_y, 0.0),
            2.1,
            ColliderShape::Capsule {
                radius: 0.045 * s_r,
                half_length: 0.105 * s_h,
            },
        ),
        bone(
            BoneId::LeftForearm,
            Vec3::new(arm_x, forearm_y, 0.0),
            1.3,
            ColliderShape::Capsule {
                radius: 0.038 * s_r,
                half_length: 0.092 * s_h,
            },
        ),
        bone(
            BoneId::LeftHand,
            Vec3::new(arm_x, hand_y, 0.0),
            0.45,
            ColliderShape::Box {
                half_extents: Vec3::new(0.022 * s_r, 0.075 * s_h, 0.045 * s_r),
            },
        ),
        // Right Arm (-X side)
        bone(
            BoneId::RightUpperArm,
            Vec3::new(-arm_x, upper_arm_y, 0.0),
            2.1,
            ColliderShape::Capsule {
                radius: 0.045 * s_r,
                half_length: 0.105 * s_h,
            },
        ),
        bone(
            BoneId::RightForearm,
            Vec3::new(-arm_x, forearm_y, 0.0),
            1.3,
            ColliderShape::Capsule {
                radius: 0.038 * s_r,
                half_length: 0.092 * s_h,
            },
        ),
        bone(
            BoneId::RightHand,
            Vec3::new(-arm_x, hand_y, 0.0),
            0.45,
            ColliderShape::Box {
                half_extents: Vec3::new(0.022 * s_r, 0.075 * s_h, 0.045 * s_r),
            },
        ),
        // Left Leg (+X side)
        bone(
            BoneId::LeftThigh,
            Vec3::new(leg_x, thigh_y, 0.0),
            7.2,
            ColliderShape::Capsule {
                radius: 0.072 * s_r,
                half_length: 0.135 * s_h,
            },
        ),
        bone(
            BoneId::LeftShin,
            Vec3::new(leg_x, shin_y, 0.0),
            3.6,
            ColliderShape::Capsule {
                radius: 0.055 * s_r,
                half_length: 0.155 * s_h,
            },
        ),
        bone(
            BoneId::LeftFoot,
            Vec3::new(leg_x, foot_y, foot_z),
            1.1,
            ColliderShape::Box {
                half_extents: Vec3::new(0.05 * s_r, 0.035 * s_h, 0.125 * s_h),
            },
        ),
        // Right Leg (-X side)
        bone(
            BoneId::RightThigh,
            Vec3::new(-leg_x, thigh_y, 0.0),
            7.2,
            ColliderShape::Capsule {
                radius: 0.072 * s_r,
                half_length: 0.135 * s_h,
            },
        ),
        bone(
            BoneId::RightShin,
            Vec3::new(-leg_x, shin_y, 0.0),
            3.6,
            ColliderShape::Capsule {
                radius: 0.055 * s_r,
                half_length: 0.155 * s_h,
            },
        ),
        bone(
            BoneId::RightFoot,
            Vec3::new(-leg_x, foot_y, foot_z),
            1.1,
            ColliderShape::Box {
                half_extents: Vec3::new(0.05 * s_r, 0.035 * s_h, 0.125 * s_h),
            },
        ),
    ];

    let arm_joint = |side: f32, ua: BoneId, fa: BoneId, ha: BoneId| -> [JointSpec; 3] {
        let shoulder_bias = Quat::from_euler(glam::EulerRot::XYZ, -0.22, 0.0, side * 0.42);
        [
            JointSpec {
                parent: BoneId::Chest,
                child: ua,
                world_anchor_rel: Vec3::new(side * arm_x, 1.430 * s_h, 0.0),
                frame_parent: shoulder_bias,
                kind: JointKind::Spherical {
                    swing_x_limit: 1.35,
                    swing_z_limit: 0.95,
                    twist_min: -1.10,
                    twist_max: 1.10,
                },
                damping: 5.0,
            },
            JointSpec {
                parent: ua,
                child: fa,
                world_anchor_rel: Vec3::new(side * arm_x, 1.130 * s_h, 0.0),
                frame_parent: Quat::IDENTITY,
                kind: JointKind::Hinge {
                    axis: Vec3::X,
                    min_angle: -2.50,
                    max_angle: 0.0,
                },
                damping: 4.5,
            },
            JointSpec {
                parent: fa,
                child: ha,
                world_anchor_rel: Vec3::new(side * arm_x, 0.870 * s_h, 0.0),
                frame_parent: Quat::IDENTITY,
                kind: JointKind::Spherical {
                    swing_x_limit: 0.85,
                    swing_z_limit: 0.50,
                    twist_min: -0.80,
                    twist_max: 0.80,
                },
                damping: 3.5,
            },
        ]
    };

    let leg_joint = |side: f32, th: BoneId, sh: BoneId, ft: BoneId| -> [JointSpec; 3] {
        [
            JointSpec {
                parent: BoneId::Pelvis,
                child: th,
                world_anchor_rel: Vec3::new(side * leg_x, 0.920 * s_h, 0.0),
                frame_parent: Quat::IDENTITY,
                kind: JointKind::Spherical {
                    swing_x_limit: 1.25,
                    swing_z_limit: 0.75,
                    twist_min: -0.55,
                    twist_max: 0.55,
                },
                damping: 7.0,
            },
            JointSpec {
                parent: th,
                child: sh,
                world_anchor_rel: Vec3::new(side * leg_x, 0.500 * s_h, 0.0),
                frame_parent: Quat::IDENTITY,
                kind: JointKind::Hinge {
                    axis: Vec3::X,
                    min_angle: 0.0,
                    max_angle: 2.45,
                },
                damping: 5.5,
            },
            JointSpec {
                parent: sh,
                child: ft,
                world_anchor_rel: Vec3::new(side * leg_x, 0.070 * s_h, 0.0),
                frame_parent: Quat::IDENTITY,
                kind: JointKind::Spherical {
                    swing_x_limit: 0.65,
                    swing_z_limit: 0.35,
                    twist_min: -0.35,
                    twist_max: 0.35,
                },
                damping: 4.0,
            },
        ]
    };

    let [ls, le, lw] = arm_joint(
        1.0,
        BoneId::LeftUpperArm,
        BoneId::LeftForearm,
        BoneId::LeftHand,
    );
    let [rs, re, rw] = arm_joint(
        -1.0,
        BoneId::RightUpperArm,
        BoneId::RightForearm,
        BoneId::RightHand,
    );
    let [lh, lk, la] = leg_joint(1.0, BoneId::LeftThigh, BoneId::LeftShin, BoneId::LeftFoot);
    let [rh, rk, ra] = leg_joint(
        -1.0,
        BoneId::RightThigh,
        BoneId::RightShin,
        BoneId::RightFoot,
    );

    let joints = [
        JointSpec {
            parent: BoneId::Pelvis,
            child: BoneId::LowerSpine,
            world_anchor_rel: Vec3::new(0.0, 1.030 * s_h, 0.0),
            frame_parent: Quat::IDENTITY,
            kind: JointKind::Spherical {
                swing_x_limit: 0.45,
                swing_z_limit: 0.35,
                twist_min: -0.45,
                twist_max: 0.45,
            },
            damping: 10.0,
        },
        JointSpec {
            parent: BoneId::LowerSpine,
            child: BoneId::Chest,
            world_anchor_rel: Vec3::new(0.0, 1.210 * s_h, 0.0),
            frame_parent: Quat::IDENTITY,
            kind: JointKind::Spherical {
                swing_x_limit: 0.45,
                swing_z_limit: 0.35,
                twist_min: -0.45,
                twist_max: 0.45,
            },
            damping: 10.0,
        },
        JointSpec {
            parent: BoneId::Chest,
            child: BoneId::Neck,
            world_anchor_rel: Vec3::new(0.0, 1.470 * s_h, 0.0),
            frame_parent: Quat::IDENTITY,
            kind: JointKind::Spherical {
                swing_x_limit: 0.45,
                swing_z_limit: 0.40,
                twist_min: -0.55,
                twist_max: 0.55,
            },
            damping: 6.0,
        },
        JointSpec {
            parent: BoneId::Neck,
            child: BoneId::Head,
            world_anchor_rel: Vec3::new(0.0, 1.560 * s_h, 0.0),
            frame_parent: Quat::IDENTITY,
            kind: JointKind::Spherical {
                swing_x_limit: 0.50,
                swing_z_limit: 0.40,
                twist_min: -0.65,
                twist_max: 0.65,
            },
            damping: 6.0,
        },
        ls,
        le,
        lw,
        rs,
        re,
        rw,
        lh,
        lk,
        la,
        rh,
        rk,
        ra,
    ];

    HumanoidSpec {
        profile,
        standing_pelvis_y,
        bones,
        joints,
    }
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
    let spec = compute_humanoid_spec(profile);
    let mut body_indices = [0usize; 17];

    for b_spec in &spec.bones {
        let idx = bodies.len();
        let mut rb = RigidBody::new(
            origin + b_spec.rel_pos,
            Quat::IDENTITY,
            b_spec.mass,
            b_spec.collider,
        );
        rb.ragdoll_id = Some(ragdoll_id);
        rb.bone_id = Some(b_spec.bone as u8);
        rb.friction = match b_spec.bone {
            BoneId::LeftFoot | BoneId::RightFoot => 1.15,
            _ => 0.75,
        };
        rb.restitution = 0.02;
        bodies.push(rb);
        body_indices[b_spec.bone as usize] = idx;
    }

    let mut joint_indices = [0usize; 16];
    for (j_cursor, j_spec) in spec.joints.iter().enumerate() {
        let p_idx = body_indices[j_spec.parent as usize];
        let c_idx = body_indices[j_spec.child as usize];
        let anchor_parent = j_spec.world_anchor_rel - spec.bones[j_spec.parent as usize].rel_pos;
        let anchor_child = j_spec.world_anchor_rel - spec.bones[j_spec.child as usize].rel_pos;

        let j_idx = joints.len();
        joints.push(
            Joint::new(p_idx, c_idx, anchor_parent, anchor_child, j_spec.kind)
                .with_frames(j_spec.frame_parent, Quat::IDENTITY)
                .with_solver_params(0.0, j_spec.damping),
        );
        joint_indices[j_cursor] = j_idx;
    }

    RagdollInstance {
        id: ragdoll_id,
        body_indices,
        joint_indices,
        profile: spec.profile,
        active_enabled: true,
        auto_get_up: true,
        standing_pelvis_y: spec.standing_pelvis_y,
        balance: BalanceState::default(),
        reflex: ReflexState::default(),
    }
}

/// Updates an existing humanoid ragdoll's proportions, masses, inertia tensors,
/// colliders, and joint anchors in-place without resetting the simulation.
pub fn update_humanoid_profile(
    ragdoll: &mut RagdollInstance,
    bodies: &mut [RigidBody],
    joints: &mut [Joint],
    new_profile: BodyProfile,
) {
    let spec = compute_humanoid_spec(new_profile);

    let old_foot_hy = match bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]].collider {
        ColliderShape::Box { half_extents } => half_extents.y,
        _ => 0.0,
    };
    let old_lf_y = bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]]
        .position
        .y;
    let old_rf_y = bodies[ragdoll.body_indices[BoneId::RightFoot as usize]]
        .position
        .y;
    let old_min_foot_bottom = old_lf_y.min(old_rf_y) - old_foot_hy;

    // 1. Update each of the 17 bodies' mass, inertia, and collider in place
    for (bone_idx, b_spec) in spec.bones.iter().enumerate() {
        let b_idx = ragdoll.body_indices[bone_idx];
        let body = &mut bodies[b_idx];
        let mass = b_spec.mass;
        let inertia = match b_spec.collider {
            ColliderShape::Capsule {
                radius,
                half_length,
            } => inertia_capsule(mass, radius, half_length),
            ColliderShape::Box { half_extents } => inertia_box(mass, half_extents),
            ColliderShape::Sphere { radius } => inertia_sphere(mass, radius),
        }
        .max(Vec3::splat(1e-6));

        body.collider = b_spec.collider;
        body.mass = mass;
        body.inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        body.local_inertia = inertia;
        body.inv_local_inertia = Vec3::ONE / inertia;
        body.update_world_inertia();
    }

    // 2. Update each of the 16 joints' local anchors in place
    for (j_cursor, j_spec) in spec.joints.iter().enumerate() {
        let j_idx = ragdoll.joint_indices[j_cursor];
        let parent_rel = spec.bones[j_spec.parent as usize].rel_pos;
        let child_rel = spec.bones[j_spec.child as usize].rel_pos;
        joints[j_idx].anchor_parent = j_spec.world_anchor_rel - parent_rel;
        joints[j_idx].anchor_child = j_spec.world_anchor_rel - child_rel;
    }

    // 3. Propagate scaled bone positions top-down from Pelvis (preserving current pose/orientations
    //    with zero joint constraint error so XPBD does not inject a velocity spike).
    for &j_idx in &ragdoll.joint_indices {
        let (parent, child, anchor_parent, anchor_child) = {
            let j = &joints[j_idx];
            (j.parent, j.child, j.anchor_parent, j.anchor_child)
        };
        let anchor_world = bodies[parent].world_point(anchor_parent);
        let child_offset = bodies[child].orientation * anchor_child;
        bodies[child].position = anchor_world - child_offset;
    }

    // 4. Keep the lowest foot anchored at its pre-resize floor/support height (and never below y = 0.0)
    let new_foot_hy = match spec.bones[BoneId::LeftFoot as usize].collider {
        ColliderShape::Box { half_extents } => half_extents.y,
        _ => 0.0,
    };
    let new_lf_y = bodies[ragdoll.body_indices[BoneId::LeftFoot as usize]]
        .position
        .y;
    let new_rf_y = bodies[ragdoll.body_indices[BoneId::RightFoot as usize]]
        .position
        .y;
    let new_min_foot_bottom = new_lf_y.min(new_rf_y) - new_foot_hy;

    let mut delta_y = old_min_foot_bottom - new_min_foot_bottom;

    // Ensure no bone collider dips below y = 0.0 after vertical alignment
    let mut min_body_bottom = f32::INFINITY;
    for &b_idx in &ragdoll.body_indices {
        let b = &bodies[b_idx];
        let half_y = match b.collider {
            ColliderShape::Sphere { radius } => radius,
            ColliderShape::Capsule {
                radius,
                half_length,
            } => radius + (b.orientation * Vec3::Y).y.abs() * half_length,
            ColliderShape::Box { half_extents } => {
                let rot = glam::Mat3::from_quat(b.orientation);
                rot.x_axis.y.abs() * half_extents.x
                    + rot.y_axis.y.abs() * half_extents.y
                    + rot.z_axis.y.abs() * half_extents.z
            }
        };
        min_body_bottom = min_body_bottom.min(b.position.y + delta_y - half_y);
    }
    if min_body_bottom < 0.0 {
        delta_y += -min_body_bottom;
    }

    for &b_idx in &ragdoll.body_indices {
        let b = &mut bodies[b_idx];
        b.position.y += delta_y;
        b.prev_position = b.position;
    }

    ragdoll.profile = spec.profile;
    ragdoll.standing_pelvis_y = spec.standing_pelvis_y;
}
