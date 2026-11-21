//! ARSA — Active Ragdoll Simulation Architecture core physics engine.

pub mod balance;
pub mod body;
pub mod collision;
pub mod joint;
pub mod math;
pub mod reflex;
pub mod skeleton;
pub mod world;

pub use glam;

pub use balance::{update_ragdoll_balance, BalanceState, BehaviorState};
pub use body::{ColliderShape, RigidBody};
pub use collision::{Contact, StaticBox};
pub use joint::{Joint, JointKind};
pub use math::{inertia_box, inertia_capsule, inertia_sphere, swing_twist_decomposition};
pub use reflex::{apply_procedural_reflexes, ReflexState, WoundEvent, WoundRegion};
pub use skeleton::{
    spawn_humanoid, spawn_humanoid_with_profile, update_humanoid_profile, BodyProfile, BoneId,
    JointId, RagdollInstance,
};
pub use world::{
    ArsaWorld, DynamiteTimer, ExplosionEvent, GrabConstraint, RaycastHit, TetherConstraint,
    WeaponKind, WeaponShotResult,
};
