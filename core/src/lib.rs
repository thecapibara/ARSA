//! ARSA — Active Ragdoll Simulation Architecture core physics engine.

pub mod balance;
pub mod body;
pub mod collision;
pub mod joint;
pub mod math;
pub mod reflex;
pub mod retarget;
pub mod skeleton;
pub mod world;

pub use glam;

pub use balance::{
    update_ragdoll_balance, BalanceState, BehaviorState, RecoveryStage, RecoveryStrategy,
};
pub use body::{ColliderShape, RigidBody};
pub use collision::{Contact, Heightfield, StaticBox};
pub use joint::{Joint, JointKind};
pub use math::{inertia_box, inertia_capsule, inertia_sphere, swing_twist_decomposition};
pub use reflex::{
    apply_procedural_reflexes, HandGrab, IncomingThreat, ReflexState, WoundEvent, WoundRegion,
};
pub use retarget::{
    convert_angular_vec_from_target, convert_angular_vec_to_target, BoneRetargetConfig,
    CoordinateSpace, RetargetPreset, SkeletonRetargeter, JOINT_BONE_PAIRS, MIXAMO_BONE_NAMES,
    UE5_MANNY_BONE_NAMES, UNITY_HUMANOID_BONE_NAMES,
};
pub use skeleton::{
    spawn_humanoid, spawn_humanoid_with_profile, update_humanoid_profile, BodyProfile, BoneId,
    JointId, RagdollInstance, RagdollLod,
};
pub use world::{
    ArsaWorld, DynamiteTimer, ExplosionEvent, GrabConstraint, GroundHeightFn, RaycastHit,
    RaycastWorldFn, TetherConstraint, WeaponKind, WeaponShotResult, WorldQueryHooks,
};
