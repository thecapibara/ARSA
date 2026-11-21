//! Native C ABI bindings (`arsa-c-abi`) for Unreal Engine 5, Unity, Godot, and C/C++.

use arsa_core::glam::{Quat, Vec3};
use arsa_core::{ArsaWorld, BodyProfile, WeaponKind};

/// Opaque handle owning an `ArsaWorld` simulation instance and a reusable scratch buffer.
pub struct ArsaWorldHandle {
    world: ArsaWorld,
    scratch: Vec<f32>,
}

/// C-compatible anthropometric body profile.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArsaBodyProfile {
    pub height_m: f32,
    pub mass_kg: f32,
    pub muscle_ratio: f32,
}

impl From<ArsaBodyProfile> for BodyProfile {
    fn from(p: ArsaBodyProfile) -> Self {
        Self {
            height_m: p.height_m,
            mass_kg: p.mass_kg,
            muscle_ratio: p.muscle_ratio,
        }
    }
}

/// C-compatible rigid body world transform (7 floats: position + XYZW quaternion).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArsaTransform {
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,
}

/// C-compatible joint muscle motor target for Host-Physics Control Mode (8 floats).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArsaJointMotorTarget {
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_torque: f32,
    pub damage_scale: f32,
}

/// Copies up to `max_floats` from `src` into `out_buf`, returning the number of floats written.
///
/// # Safety
/// If `out_buf` is non-null, it must be valid for writes of `max_floats` `f32` elements.
unsafe fn copy_floats_to_out(src: &[f32], out_buf: *mut f32, max_floats: usize) -> usize {
    if out_buf.is_null() || max_floats == 0 || src.is_empty() {
        return 0;
    }
    let count = src.len().min(max_floats);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), out_buf, count);
    }
    count
}

/// Allocates and initializes a new `ArsaWorldHandle`.
/// The caller owns the returned pointer and must free it with `arsa_world_destroy`.
#[no_mangle]
pub extern "C" fn arsa_world_create() -> *mut ArsaWorldHandle {
    Box::into_raw(Box::new(ArsaWorldHandle {
        world: ArsaWorld::new(),
        scratch: Vec::with_capacity(256),
    }))
}

/// Destroys an `ArsaWorldHandle` previously created by `arsa_world_create`.
///
/// # Safety
/// `handle` must be either null or a valid pointer returned by `arsa_world_create`
/// that has not yet been destroyed.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_destroy(handle: *mut ArsaWorldHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Clears all bodies, joints, tethers, static obstacles, ragdolls, and active grabs from the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_clear(handle: *mut ArsaWorldHandle) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.clear();
    }
}

/// Steps the simulation forward by `dt` seconds.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_step(handle: *mut ArsaWorldHandle, dt: f32) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.step(dt);
    }
}

/// Spawns a humanoid ragdoll at `(x, y, z)` and returns its ragdoll ID (`usize::MAX` on null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_spawn_ragdoll(
    handle: *mut ArsaWorldHandle,
    x: f32,
    y: f32,
    z: f32,
) -> usize {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.spawn_ragdoll(Vec3::new(x, y, z))
    } else {
        usize::MAX
    }
}

/// Spawns a procedurally scaled humanoid ragdoll at `(x, y, z)` and returns its ragdoll ID (`usize::MAX` on null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_spawn_ragdoll_profile(
    handle: *mut ArsaWorldHandle,
    x: f32,
    y: f32,
    z: f32,
    height_m: f32,
    mass_kg: f32,
    muscle_ratio: f32,
) -> usize {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.spawn_ragdoll_with_profile(
            Vec3::new(x, y, z),
            BodyProfile {
                height_m,
                mass_kg,
                muscle_ratio,
            },
        )
    } else {
        usize::MAX
    }
}

/// Updates the anthropometric profile of all spawned ragdolls in-place without resetting the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_update_all_profiles(
    handle: *mut ArsaWorldHandle,
    height_m: f32,
    mass_kg: f32,
    muscle_ratio: f32,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.update_all_ragdolls_profile(BodyProfile {
            height_m,
            mass_kg,
            muscle_ratio,
        });
    }
}

/// Enables or disables active balance & muscle motors for all ragdolls in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_active(handle: *mut ArsaWorldHandle, enabled: bool) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_all_ragdolls_active(enabled);
    }
}

/// Enables or disables automatic procedural get-up from `Downed` for all ragdolls in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_auto_get_up(handle: *mut ArsaWorldHandle, enabled: bool) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_all_ragdolls_auto_get_up(enabled);
    }
}

/// Sets an external animation target pose (`num_joints == 16`, i.e. 64 floats in XYZW order)
/// blended onto the ragdoll's joint motor targets by `blend_weight` (`0.0..=1.0`).
/// Returns `false` if `handle` or `quats_xyzw` is null, `num_joints != 16`, `ragdoll_id` is invalid,
/// or any quaternion / weight is non-finite or zero.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `quats_xyzw` is non-null, it must be valid for reads of `num_joints * 4` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_target_pose(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    quats_xyzw: *const f32,
    num_joints: usize,
    blend_weight: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    if quats_xyzw.is_null() || num_joints != 16 {
        return false;
    }
    let raw = unsafe { std::slice::from_raw_parts(quats_xyzw, 16 * 4) };
    let mut targets = [Quat::IDENTITY; 16];
    for (i, dst) in targets.iter_mut().enumerate() {
        let base = i * 4;
        *dst = Quat::from_xyzw(raw[base], raw[base + 1], raw[base + 2], raw[base + 3]);
    }
    h.world
        .set_ragdoll_target_pose(ragdoll_id, &targets, blend_weight)
}

/// Clears any external animation target pose on `ragdoll_id`.
/// Returns `false` if `handle` is null or `ragdoll_id` is out of range.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_clear_target_pose(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    h.world.clear_ragdoll_target_pose(ragdoll_id)
}

/// Sets the earthquake seismic magnitude (`0.0..=9.0`, `0.0` disables).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_earthquake(handle: *mut ArsaWorldHandle, magnitude: f32) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_earthquake(magnitude);
    }
}

/// Sets the procedural stagger/intoxication level (`0.0..=1.0`) for all ragdolls in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_intoxication(handle: *mut ArsaWorldHandle, level: f32) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_all_ragdolls_intoxication(level);
    }
}

/// Spawns a dynamic rigid-body box and returns its body index (`usize::MAX` on null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_spawn_box(
    handle: *mut ArsaWorldHandle,
    x: f32,
    y: f32,
    z: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    mass: f32,
    vx: f32,
    vy: f32,
    vz: f32,
) -> usize {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.spawn_box(
            Vec3::new(x, y, z),
            Vec3::new(hx, hy, hz),
            mass,
            Vec3::new(vx, vy, vz),
        )
    } else {
        usize::MAX
    }
}

/// Adds a static axis-aligned box obstacle and returns its index (`usize::MAX` on null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_add_static_box(
    handle: *mut ArsaWorldHandle,
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    friction: f32,
    restitution: f32,
) -> usize {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.add_static_box(
            Vec3::new(cx, cy, cz),
            Vec3::new(hx, hy, hz),
            friction,
            restitution,
        )
    } else {
        usize::MAX
    }
}

/// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and applies a broad two-handed shove
/// of `strength` N*s. Returns the hit body index, or `-1` if null or missed.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_apply_shove(
    handle: *mut ArsaWorldHandle,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    strength: f32,
) -> i32 {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return -1;
    };
    match h
        .world
        .apply_shove(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), strength)
    {
        Some(hit) => hit.body_idx as i32,
        None => -1,
    }
}

/// Raycasts from `(ox, oy, oz)` along `(dx, dy, dz)` and applies a sharp localized punch
/// of `strength` N*s. Returns the hit body index, or `-1` if null or missed.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_apply_punch(
    handle: *mut ArsaWorldHandle,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    strength: f32,
) -> i32 {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return -1;
    };
    match h
        .world
        .apply_punch(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), strength)
    {
        Some(hit) => hit.body_idx as i32,
        None => -1,
    }
}

/// Fires a ballistic weapon (`0 = Pistol`, `1 = Shotgun`, `2 = Rifle`) from `(ox, oy, oz)`
/// along `(dx, dy, dz)` and returns the number of ray hits (`0` if null or missed).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_fire_weapon(
    handle: *mut ArsaWorldHandle,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    weapon_kind: u32,
) -> i32 {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return 0;
    };
    let weapon = match weapon_kind {
        1 => WeaponKind::Shotgun,
        2 => WeaponKind::Rifle,
        _ => WeaponKind::Pistol,
    };
    let res = h
        .world
        .fire_weapon(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), weapon);
    res.hits.len() as i32
}

/// Fires a radial + directional kinetic force wave from `(ox, oy, oz)` along `(dx, dy, dz)`
/// affecting dynamic bodies within `radius` meters up to `max_impulse` N*s.
/// Returns the count of affected bodies (`0` if null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_apply_force_wave(
    handle: *mut ArsaWorldHandle,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    radius: f32,
    max_impulse: f32,
) -> usize {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return 0;
    };
    h.world.apply_force_wave(
        Vec3::new(ox, oy, oz),
        Vec3::new(dx, dy, dz),
        radius,
        max_impulse,
    )
}

/// Returns the number of dynamic rigid bodies in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_body_count(handle: *const ArsaWorldHandle) -> usize {
    if let Some(h) = unsafe { handle.as_ref() } {
        h.world.bodies.len()
    } else {
        0
    }
}

/// Returns the number of articulated joints in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_joint_count(handle: *const ArsaWorldHandle) -> usize {
    if let Some(h) = unsafe { handle.as_ref() } {
        h.world.joints.len()
    } else {
        0
    }
}

/// Synchronizes a body's simulated pose and velocities from an external host physics engine
/// (e.g. UE5 Chaos / Unity PhysX) so ARSA balance/reflex controllers can compute updated
/// `joint.motor_target` values. Returns `false` if `handle` is null, `body_idx` is invalid,
/// or any input is non-finite / degenerate.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_sync_bone_pose(
    handle: *mut ArsaWorldHandle,
    body_idx: usize,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    wx: f32,
    wy: f32,
    wz: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    h.world.sync_bone_pose(
        body_idx,
        Vec3::new(px, py, pz),
        Quat::from_xyzw(qx, qy, qz, qw),
        Vec3::new(vx, vy, vz),
        Vec3::new(wx, wy, wz),
    )
}

/// Copies packed transforms (`px, py, pz, qx, qy, qz, qw` per body) into `out_buf`,
/// writing at most `max_floats` floats, and returns the number of floats written.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_buf` is non-null, it must be valid for writes of `max_floats` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_transforms(
    handle: *mut ArsaWorldHandle,
    out_buf: *mut f32,
    max_floats: usize,
) -> usize {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return 0;
    };
    h.world.write_transforms(&mut h.scratch);
    unsafe { copy_floats_to_out(&h.scratch, out_buf, max_floats) }
}

/// Copies per-ragdoll balance & reflex telemetry (16 floats per ragdoll) into `out_buf`,
/// writing at most `max_floats` floats, and returns the number of floats written.
///
/// Layout per ragdoll:
/// `[0..3] com(x,y,z), [3..5] capture_point(x,z), [5] left_grounded, [6] right_grounded,`
/// `[7] behavior_state, [8..11] pelvis(x,y,z), [11] health, [12] wound_region, [13..16] wound_world(x,y,z)`.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_buf` is non-null, it must be valid for writes of `max_floats` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_ragdoll_telemetry(
    handle: *mut ArsaWorldHandle,
    out_buf: *mut f32,
    max_floats: usize,
) -> usize {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return 0;
    };
    h.world.write_ragdoll_telemetry(&mut h.scratch);
    unsafe { copy_floats_to_out(&h.scratch, out_buf, max_floats) }
}

/// Copies per-joint PD muscle motor targets (8 floats per joint:
/// `qx, qy, qz, qw, stiffness, damping, max_torque, damage_scale`) into `out_buf`
/// for Host-Physics Control Mode in external engines (e.g. UE5 Chaos / Unity PhysX),
/// writing at most `max_floats` floats, and returns the number of floats written.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_buf` is non-null, it must be valid for writes of `max_floats` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_joint_motor_targets(
    handle: *mut ArsaWorldHandle,
    out_buf: *mut f32,
    max_floats: usize,
) -> usize {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return 0;
    };
    h.scratch.clear();
    h.scratch.reserve(h.world.joints.len() * 8);
    for j in &h.world.joints {
        h.scratch.push(j.motor_target.x);
        h.scratch.push(j.motor_target.y);
        h.scratch.push(j.motor_target.z);
        h.scratch.push(j.motor_target.w);
        h.scratch.push(j.motor_stiffness);
        h.scratch.push(j.motor_damping);
        h.scratch.push(j.max_torque);
        h.scratch.push(j.damage_scale);
    }
    unsafe { copy_floats_to_out(&h.scratch, out_buf, max_floats) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_c_structs_layout() {
        assert_eq!(size_of::<ArsaBodyProfile>(), 12);
        assert_eq!(align_of::<ArsaBodyProfile>(), 4);
        assert_eq!(size_of::<ArsaTransform>(), 28);
        assert_eq!(align_of::<ArsaTransform>(), 4);
        assert_eq!(size_of::<ArsaJointMotorTarget>(), 32);
        assert_eq!(align_of::<ArsaJointMotorTarget>(), 4);
    }

    #[test]
    fn test_c_abi_lifecycle_and_queries() {
        unsafe {
            let world = arsa_world_create();
            assert!(!world.is_null());

            let rid = arsa_world_spawn_ragdoll_profile(world, 0.0, 0.0, 0.0, 2.17, 135.0, 1.0);
            assert_eq!(rid, 0);
            assert_eq!(arsa_world_body_count(world), 17);
            assert_eq!(arsa_world_joint_count(world), 16);

            for _ in 0..60 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            let mut telemetry = [0.0_f32; 16];
            let t_written = arsa_world_get_ragdoll_telemetry(world, telemetry.as_mut_ptr(), 16);
            assert_eq!(t_written, 16);
            assert_eq!(telemetry[7] as u32, 0); // Balanced
            assert_eq!(telemetry[11], 100.0);

            let mut motors = [0.0_f32; 16 * 8];
            let m_written =
                arsa_world_get_joint_motor_targets(world, motors.as_mut_ptr(), motors.len());
            assert_eq!(m_written, 16 * 8);

            assert!(arsa_world_sync_bone_pose(
                world, 0, 0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ));
            assert!(!arsa_world_sync_bone_pose(
                world, 9999, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ));

            let mut xforms = [0.0_f32; 17 * 7];
            let x_written = arsa_world_get_transforms(world, xforms.as_mut_ptr(), xforms.len());
            assert_eq!(x_written, 17 * 7);
            assert!((xforms[0] - 0.1).abs() < 1e-6);
            assert!((xforms[1] - 1.0).abs() < 1e-6);
            assert!((xforms[2] - 0.2).abs() < 1e-6);

            let mut quats = [0.0_f32; 16 * 4];
            for i in 0..16 {
                quats[i * 4 + 3] = 1.0;
            }
            assert!(arsa_world_set_target_pose(
                world,
                rid,
                quats.as_ptr(),
                16,
                0.75
            ));
            assert!(!arsa_world_set_target_pose(
                world,
                rid,
                quats.as_ptr(),
                15,
                0.75
            ));
            assert!(arsa_world_clear_target_pose(world, rid));
            assert!(!arsa_world_clear_target_pose(world, 999));

            arsa_world_clear(world);
            assert_eq!(arsa_world_body_count(world), 0);
            assert_eq!(arsa_world_joint_count(world), 0);

            arsa_world_destroy(world);
        }
    }
}
