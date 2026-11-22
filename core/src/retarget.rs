//! Universal cross-engine skeleton retargeting and coordinate-space conversion
//! for Mixamo, Unity Humanoid, and Unreal Engine 5 Manny rigs.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

use glam::{Quat, Vec3};

use crate::skeleton::BoneId;

/// Target engine coordinate system convention.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordinateSpace {
    /// ARSA native: right-handed, +X left, +Y up, +Z forward (meters).
    ArsaRightHandedYUp = 0,
    /// Unity: left-handed, +X right, +Y up, +Z forward (meters, X mirrored).
    UnityLeftHandedYUp = 1,
    /// Unreal Engine 5: left-handed, +X forward, +Y right, +Z up (centimeters).
    UnrealLeftHandedZUp = 2,
}

impl CoordinateSpace {
    /// Converts a raw `u32` coordinate space ID (`0..=2`) into a [`CoordinateSpace`].
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::ArsaRightHandedYUp),
            1 => Some(Self::UnityLeftHandedYUp),
            2 => Some(Self::UnrealLeftHandedZUp),
            _ => None,
        }
    }
}

/// Built-in skeleton retargeting presets aligned with standard humanoid rigs.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetargetPreset {
    /// Native ARSA 17-bone skeleton (arms-down bind pose, RH Y-up, meters).
    ArsaNative = 0,
    /// Mixamo standard rig (T-pose arms, RH Y-up, meters).
    MixamoTPose = 1,
    /// Unity Humanoid Avatar rig (T-pose arms, LH Y-up, meters).
    UnityHumanoidTPose = 2,
    /// Unreal Engine 5 Manny / Mannequin rig (A-pose arms, LH Z-up, centimeters).
    UnrealMannyAPose = 3,
}

impl RetargetPreset {
    /// Converts a raw `u32` preset ID (`0..=3`) into a [`RetargetPreset`].
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::ArsaNative),
            1 => Some(Self::MixamoTPose),
            2 => Some(Self::UnityHumanoidTPose),
            3 => Some(Self::UnrealMannyAPose),
            _ => None,
        }
    }
}

/// Standard Mixamo skeleton bone names indexed by [`BoneId`] (`0..17`).
pub const MIXAMO_BONE_NAMES: [&str; 17] = [
    "mixamorig:Hips",
    "mixamorig:Spine",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
];

/// Standard Unity `HumanBodyBones` names indexed by [`BoneId`] (`0..17`).
pub const UNITY_HUMANOID_BONE_NAMES: [&str; 17] = [
    "Hips",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
];

/// Standard Unreal Engine 5 `SK_Manny` bone names indexed by [`BoneId`] (`0..17`).
pub const UE5_MANNY_BONE_NAMES: [&str; 17] = [
    "pelvis",
    "spine_01",
    "spine_03",
    "neck_01",
    "head",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "thigh_l",
    "calf_l",
    "foot_l",
    "thigh_r",
    "calf_r",
    "foot_r",
];

/// Parent-to-child [`BoneId`] pairs matching [`crate::skeleton::JointId`] order (`0..16`).
pub const JOINT_BONE_PAIRS: [(BoneId, BoneId); 16] = [
    (BoneId::Pelvis, BoneId::LowerSpine),          // LowerSpine = 0
    (BoneId::LowerSpine, BoneId::Chest),           // ChestSpine = 1
    (BoneId::Chest, BoneId::Neck),                 // Neck = 2
    (BoneId::Neck, BoneId::Head),                  // Head = 3
    (BoneId::Chest, BoneId::LeftUpperArm),         // LeftShoulder = 4
    (BoneId::LeftUpperArm, BoneId::LeftForearm),   // LeftElbow = 5
    (BoneId::LeftForearm, BoneId::LeftHand),       // LeftWrist = 6
    (BoneId::Chest, BoneId::RightUpperArm),        // RightShoulder = 7
    (BoneId::RightUpperArm, BoneId::RightForearm), // RightElbow = 8
    (BoneId::RightForearm, BoneId::RightHand),     // RightWrist = 9
    (BoneId::Pelvis, BoneId::LeftThigh),           // LeftHip = 10
    (BoneId::LeftThigh, BoneId::LeftShin),         // LeftKnee = 11
    (BoneId::LeftShin, BoneId::LeftFoot),          // LeftAnkle = 12
    (BoneId::Pelvis, BoneId::RightThigh),          // RightHip = 13
    (BoneId::RightThigh, BoneId::RightShin),       // RightKnee = 14
    (BoneId::RightShin, BoneId::RightFoot),        // RightAnkle = 15
];

/// Per-bone bind-pose rotational offset (`arsa_bind_world.conjugate() * target_bind_world_in_arsa_space`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneRetargetConfig {
    /// Orientation offset mapping ARSA's arms-down bind pose to the target bone's bind pose in ARSA space.
    pub bind_offset: Quat,
}

impl Default for BoneRetargetConfig {
    fn default() -> Self {
        Self {
            bind_offset: Quat::IDENTITY,
        }
    }
}

/// Bidirectional pose retargeter between ARSA's 17-bone ragdoll and external game-engine skeletons.
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonRetargeter {
    pub space: CoordinateSpace,
    pub unit_scale: f32,
    pub bones: [BoneRetargetConfig; 17],
}

impl Default for SkeletonRetargeter {
    fn default() -> Self {
        Self::from_preset(RetargetPreset::ArsaNative)
    }
}

impl SkeletonRetargeter {
    /// Constructs a [`SkeletonRetargeter`] from one of the built-in engine/rig presets.
    pub fn from_preset(preset: RetargetPreset) -> Self {
        match preset {
            RetargetPreset::ArsaNative => Self {
                space: CoordinateSpace::ArsaRightHandedYUp,
                unit_scale: 1.0,
                bones: [BoneRetargetConfig::default(); 17],
            },
            RetargetPreset::MixamoTPose => Self {
                space: CoordinateSpace::ArsaRightHandedYUp,
                unit_scale: 1.0,
                bones: arm_angle_bind_offsets(FRAC_PI_2),
            },
            RetargetPreset::UnityHumanoidTPose => Self {
                space: CoordinateSpace::UnityLeftHandedYUp,
                unit_scale: 1.0,
                bones: arm_angle_bind_offsets(FRAC_PI_2),
            },
            RetargetPreset::UnrealMannyAPose => Self {
                space: CoordinateSpace::UnrealLeftHandedZUp,
                unit_scale: 100.0,
                bones: arm_angle_bind_offsets(FRAC_PI_4),
            },
        }
    }

    /// Constructs a custom retargeter from 17 world-space bind orientations expressed in `space`.
    pub fn from_custom_bind_poses(
        space: CoordinateSpace,
        unit_scale: f32,
        target_bind_world_in_target_space: &[Quat; 17],
    ) -> Self {
        let safe_scale = if unit_scale.is_finite() && unit_scale > 1e-6 {
            unit_scale
        } else {
            1.0
        };
        let arsa_bind_world = Quat::IDENTITY;
        let mut bones = [BoneRetargetConfig::default(); 17];
        for (dst, &target_q) in bones
            .iter_mut()
            .zip(target_bind_world_in_target_space.iter())
        {
            let q_arsa = Self::convert_quat_from_target(space, target_q);
            let normalized = if q_arsa.is_finite() && q_arsa.length_squared() >= 1e-12 {
                q_arsa.normalize()
            } else {
                Quat::IDENTITY
            };
            dst.bind_offset = (arsa_bind_world.conjugate() * normalized).normalize();
        }
        Self {
            space,
            unit_scale: safe_scale,
            bones,
        }
    }

    /// Converts a position vector from ARSA space (meters, RH Y-up) into `space` scaled by `scale`.
    #[inline]
    pub fn convert_pos_to_target(space: CoordinateSpace, scale: f32, p: Vec3) -> Vec3 {
        match space {
            CoordinateSpace::ArsaRightHandedYUp => p * scale,
            CoordinateSpace::UnityLeftHandedYUp => {
                Vec3::new(-p.x * scale, p.y * scale, p.z * scale)
            }
            CoordinateSpace::UnrealLeftHandedZUp => {
                Vec3::new(p.z * scale, -p.x * scale, p.y * scale)
            }
        }
    }

    /// Converts a position vector from `space` (scaled by `scale`) back into ARSA space (meters).
    #[inline]
    pub fn convert_pos_from_target(space: CoordinateSpace, scale: f32, p: Vec3) -> Vec3 {
        let inv_scale = if scale.is_finite() && scale.abs() > 1e-8 {
            1.0 / scale
        } else {
            1.0
        };
        match space {
            CoordinateSpace::ArsaRightHandedYUp => p * inv_scale,
            CoordinateSpace::UnityLeftHandedYUp => {
                Vec3::new(-p.x * inv_scale, p.y * inv_scale, p.z * inv_scale)
            }
            CoordinateSpace::UnrealLeftHandedZUp => {
                Vec3::new(-p.y * inv_scale, p.z * inv_scale, p.x * inv_scale)
            }
        }
    }

    /// Converts a quaternion from ARSA space (RH Y-up) into `space`.
    #[inline]
    pub fn convert_quat_to_target(space: CoordinateSpace, q: Quat) -> Quat {
        match space {
            CoordinateSpace::ArsaRightHandedYUp => q,
            CoordinateSpace::UnityLeftHandedYUp => Quat::from_xyzw(q.x, -q.y, -q.z, q.w),
            CoordinateSpace::UnrealLeftHandedZUp => Quat::from_xyzw(-q.z, q.x, -q.y, q.w),
        }
    }

    /// Converts a quaternion from `space` back into ARSA space (RH Y-up).
    #[inline]
    pub fn convert_quat_from_target(space: CoordinateSpace, q: Quat) -> Quat {
        match space {
            CoordinateSpace::ArsaRightHandedYUp => q,
            CoordinateSpace::UnityLeftHandedYUp => Quat::from_xyzw(q.x, -q.y, -q.z, q.w),
            CoordinateSpace::UnrealLeftHandedZUp => Quat::from_xyzw(q.y, -q.z, -q.x, q.w),
        }
    }

    /// Converts an axial vector (angular velocity or torque pseudovector) from ARSA space (RH Y-up)
    /// into `space` using the exact same axis permutation and sign rules as the `(x, y, z)`
    /// imaginary components of [`Self::convert_quat_to_target`].
    #[inline]
    pub fn convert_angular_vec_to_target(space: CoordinateSpace, v: Vec3) -> Vec3 {
        convert_angular_vec_to_target(space, v)
    }

    /// Converts an axial vector (angular velocity or torque pseudovector) from `space` back into
    /// ARSA space (RH Y-up).
    #[inline]
    pub fn convert_angular_vec_from_target(space: CoordinateSpace, v: Vec3) -> Vec3 {
        convert_angular_vec_from_target(space, v)
    }

    /// Retargets 17 ARSA world-space bone positions and orientations into the target engine's
    /// coordinate space, unit scale, and bind-pose frame.
    pub fn retarget_bones_to_target(
        &self,
        arsa_positions: &[Vec3; 17],
        arsa_orientations: &[Quat; 17],
    ) -> ([Vec3; 17], [Quat; 17]) {
        let mut out_pos = [Vec3::ZERO; 17];
        let mut out_rot = [Quat::IDENTITY; 17];
        for i in 0..17 {
            let world_arsa =
                (arsa_orientations[i] * self.bones[i].bind_offset.conjugate()).normalize();
            out_pos[i] =
                Self::convert_pos_to_target(self.space, self.unit_scale, arsa_positions[i]);
            out_rot[i] = Self::convert_quat_to_target(self.space, world_arsa);
        }
        (out_pos, out_rot)
    }

    /// Retargets 17 target-engine world-space bone orientations into 16 parent-relative ARSA joint
    /// targets ready for [`crate::ArsaWorld::set_ragdoll_target_pose`].
    pub fn retarget_world_quats_to_arsa_joint_targets(
        &self,
        target_world_quats: &[Quat; 17],
    ) -> [Quat; 16] {
        let mut arsa_world = [Quat::IDENTITY; 17];
        for i in 0..17 {
            let q_in_arsa = Self::convert_quat_from_target(self.space, target_world_quats[i]);
            arsa_world[i] = (q_in_arsa * self.bones[i].bind_offset).normalize();
        }
        let mut joint_targets = [Quat::IDENTITY; 16];
        for (j, &(parent, child)) in JOINT_BONE_PAIRS.iter().enumerate() {
            joint_targets[j] =
                (arsa_world[parent as usize].conjugate() * arsa_world[child as usize]).normalize();
        }
        joint_targets
    }
}

/// Computes bind offsets for T-pose (`FRAC_PI_2`) or A-pose (`FRAC_PI_4`) arms relative to ARSA's
/// arms-down bind pose (`-Y` along limb, `+X` character left, `+Z` forward).
///
/// In ARSA (+X left, -X right, -Y down), extending left arm outward to +X requires `+arm_angle_rad` around +Z,
/// and extending right arm outward to -X requires `-arm_angle_rad` around +Z.
fn arm_angle_bind_offsets(arm_angle_rad: f32) -> [BoneRetargetConfig; 17] {
    let mut bones = [BoneRetargetConfig::default(); 17];
    let left_offset = Quat::from_rotation_z(arm_angle_rad);
    let right_offset = Quat::from_rotation_z(-arm_angle_rad);
    for idx in [
        BoneId::LeftUpperArm as usize,
        BoneId::LeftForearm as usize,
        BoneId::LeftHand as usize,
    ] {
        bones[idx].bind_offset = left_offset;
    }
    for idx in [
        BoneId::RightUpperArm as usize,
        BoneId::RightForearm as usize,
        BoneId::RightHand as usize,
    ] {
        bones[idx].bind_offset = right_offset;
    }
    bones
}

/// Converts an axial vector (angular velocity or torque pseudovector) from ARSA space (RH Y-up)
/// into `space` using the exact same axis permutation and sign rules as the `(x, y, z)`
/// imaginary components of [`SkeletonRetargeter::convert_quat_to_target`].
#[inline]
pub fn convert_angular_vec_to_target(space: CoordinateSpace, v: Vec3) -> Vec3 {
    match space {
        CoordinateSpace::ArsaRightHandedYUp => v,
        CoordinateSpace::UnityLeftHandedYUp => Vec3::new(v.x, -v.y, -v.z),
        CoordinateSpace::UnrealLeftHandedZUp => Vec3::new(-v.z, v.x, -v.y),
    }
}

/// Converts an axial vector (angular velocity or torque pseudovector) from `space` back into
/// ARSA space (RH Y-up).
#[inline]
pub fn convert_angular_vec_from_target(space: CoordinateSpace, v: Vec3) -> Vec3 {
    match space {
        CoordinateSpace::ArsaRightHandedYUp => v,
        CoordinateSpace::UnityLeftHandedYUp => Vec3::new(v.x, -v.y, -v.z),
        CoordinateSpace::UnrealLeftHandedZUp => Vec3::new(v.y, -v.z, -v.x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_angle_bind_offsets() {
        let retargeter = SkeletonRetargeter::from_preset(RetargetPreset::MixamoTPose);
        let left_offset = retargeter.bones[BoneId::LeftUpperArm as usize].bind_offset;
        let right_offset = retargeter.bones[BoneId::RightUpperArm as usize].bind_offset;

        let arms_down = Vec3::new(0.0, -1.0, 0.0);
        let left_dir = left_offset * arms_down;
        let right_dir = right_offset * arms_down;

        assert!(
            (left_dir - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5,
            "expected left arm rotated by left_offset to point toward (+1, 0, 0), got {:?}",
            left_dir
        );
        assert!(
            (right_dir - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5,
            "expected right arm rotated by right_offset to point toward (-1, 0, 0), got {:?}",
            right_dir
        );
    }
}
