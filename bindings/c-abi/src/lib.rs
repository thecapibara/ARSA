//! Native C ABI bindings (`arsa-c-abi`) for Unreal Engine 5, Unity, Godot, and C/C++.

use arsa_core::glam::{Quat, Vec3};
use arsa_core::{
    ArsaWorld, BodyProfile, CoordinateSpace, GroundHeightFn, Heightfield, RaycastWorldFn,
    RetargetPreset, SkeletonRetargeter, StatusEffect, WeaponKind, WorldQueryHooks,
};

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

/// C-compatible retargeted single-bone world transform (7 floats: position + XYZW quaternion).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArsaBoneTransform {
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

/// Runs a brain-only controller tick over currently synced bone transforms without integrating
/// positions/orientations or running XPBD solvers.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_step_brain_only(handle: *mut ArsaWorldHandle, dt: f32) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.step_brain_only(dt);
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

/// Configures the spatial crowd-scaling LOD focus point and distance thresholds (in meters).
/// Ignores non-finite focus coordinates; clamps `reduced_dist >= 1.0` and `sleep_dist >= reduced_dist + 1.0`.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_lod_focus(
    handle: *mut ArsaWorldHandle,
    x: f32,
    y: f32,
    z: f32,
    reduced_dist: f32,
    sleep_dist: f32,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world
            .set_lod_focus(Some(Vec3::new(x, y, z)), reduced_dist, sleep_dist);
    }
}

/// Clears the LOD focus point and wakes all sleeping ragdolls back to `ARSA_LOD_FULL`.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_clear_lod_focus(handle: *mut ArsaWorldHandle) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world
            .set_lod_focus(None, h.world.lod_reduced_dist, h.world.lod_sleep_dist);
    }
}

/// Returns the current `ArsaRagdollLod` tier (`0 = Full, 1 = Reduced, 2 = Sleep`) for `ragdoll_id`,
/// or `u32::MAX` (`UINT32_MAX`) if `handle` is null or `ragdoll_id` is invalid.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_ragdoll_lod(
    handle: *const ArsaWorldHandle,
    ragdoll_id: usize,
) -> u32 {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        return u32::MAX;
    };
    h.world
        .ragdoll_lod(ragdoll_id)
        .map(|l| l as u32)
        .unwrap_or(u32::MAX)
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

/// Adds a static oriented box obstacle (XYZW quaternion) and returns its index (`usize::MAX` on null).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_add_static_oriented_box(
    handle: *mut ArsaWorldHandle,
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    friction: f32,
    restitution: f32,
) -> usize {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.add_static_oriented_box(
            Vec3::new(cx, cy, cz),
            Vec3::new(hx, hy, hz),
            Quat::from_xyzw(qx, qy, qz, qw),
            friction,
            restitution,
        )
    } else {
        usize::MAX
    }
}

/// Sets a uniform-grid heightfield terrain collider from row-major `heights_ptr` (`cols * rows` floats).
/// Returns `false` if `handle` or `heights_ptr` is null, dimensions/cell_size are invalid, or any value is non-finite.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `heights_ptr` is non-null, it must be valid for reads of `cols * rows` `f32` elements.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_heightfield(
    handle: *mut ArsaWorldHandle,
    origin_x: f32,
    origin_z: f32,
    cell_size: f32,
    cols: usize,
    rows: usize,
    heights_ptr: *const f32,
    friction: f32,
    restitution: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    if heights_ptr.is_null()
        || cols < 2
        || rows < 2
        || !cell_size.is_finite()
        || cell_size <= 1e-4
        || !origin_x.is_finite()
        || !origin_z.is_finite()
        || !friction.is_finite()
        || !restitution.is_finite()
    {
        return false;
    }
    let Some(count) = cols.checked_mul(rows) else {
        return false;
    };
    let heights = unsafe { std::slice::from_raw_parts(heights_ptr, count) };
    if heights.iter().any(|v| !v.is_finite()) {
        return false;
    }
    h.world.set_heightfield(Some(Heightfield {
        origin_x,
        origin_z,
        cell_size,
        cols,
        rows,
        heights: heights.to_vec(),
        friction,
        restitution,
    }));
    true
}

/// Clears the world's heightfield terrain collider.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_clear_heightfield(handle: *mut ArsaWorldHandle) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_heightfield(None);
    }
}

/// Registers or clears host engine world-query callbacks (`ground_height_fn` and `raycast_fn`).
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// Any non-null callback pointers and `user_data` must remain valid for the duration of simulation calls.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_query_callbacks(
    handle: *mut ArsaWorldHandle,
    user_data: *mut std::ffi::c_void,
    ground_height_fn: Option<GroundHeightFn>,
    raycast_fn: Option<RaycastWorldFn>,
) {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return;
    };
    if ground_height_fn.is_none() && raycast_fn.is_none() {
        h.world.set_world_query_callbacks(None);
    } else {
        h.world.set_world_query_callbacks(Some(WorldQueryHooks {
            user_data,
            ground_height: ground_height_fn,
            raycast: raycast_fn,
        }));
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

/// Computes 16 world-space PD muscle motor torques (`48` floats: `16 * (tx, ty, tz)` in `N·m`)
/// for `ragdoll_id` converted into `coord_space` (`0 = ARSA RH Y-up`, `1 = Unity LH Y-up`,
/// `2 = Unreal LH Z-up`).
/// Returns `false` if `handle` or `out_torques_48` is null, `ragdoll_id` is invalid, or
/// `coord_space` is unknown.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_torques_48` is non-null, it must be valid for writes of `48` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_compute_host_joint_torques(
    handle: *const ArsaWorldHandle,
    ragdoll_id: usize,
    coord_space: u32,
    out_torques_48: *mut f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        return false;
    };
    if out_torques_48.is_null() {
        return false;
    }
    let Some(space) = CoordinateSpace::from_u32(coord_space) else {
        return false;
    };
    let Some(torques) = h.world.compute_host_joint_torques(ragdoll_id, space) else {
        return false;
    };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_torques_48, 48) };
    for (i, t) in torques.iter().enumerate() {
        out_slice[i * 3] = t.x;
        out_slice[i * 3 + 1] = t.y;
        out_slice[i * 3 + 2] = t.z;
    }
    true
}

/// Reads all 17 bone transforms of `ragdoll_id` retargeted into the target engine's coordinate space,
/// unit scale, and bind pose (`ArsaRetargetPreset`: `0=Native`, `1=MixamoTPose`, `2=UnityHumanoid`, `3=UnrealManny`).
/// Returns `false` if `handle` or `out_bones_17` is null, `ragdoll_id` is out of range, or `preset` is invalid.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_bones_17` is non-null, it must be valid for writes of 17 [`ArsaBoneTransform`] elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_retargeted_bones(
    handle: *const ArsaWorldHandle,
    ragdoll_id: usize,
    preset: u32,
    out_bones_17: *mut ArsaBoneTransform,
) -> bool {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        return false;
    };
    if out_bones_17.is_null() {
        return false;
    }
    let Some(preset_enum) = RetargetPreset::from_u32(preset) else {
        return false;
    };
    let Some(ragdoll) = h.world.ragdolls.get(ragdoll_id) else {
        return false;
    };

    let mut arsa_pos = [Vec3::ZERO; 17];
    let mut arsa_rot = [Quat::IDENTITY; 17];
    for (i, &b_idx) in ragdoll.body_indices.iter().enumerate() {
        let Some(body) = h.world.bodies.get(b_idx) else {
            return false;
        };
        arsa_pos[i] = body.position;
        arsa_rot[i] = body.orientation;
    }

    let retargeter = SkeletonRetargeter::from_preset(preset_enum);
    let (out_pos, out_rot) = retargeter.retarget_bones_to_target(&arsa_pos, &arsa_rot);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_bones_17, 17) };
    for i in 0..17 {
        out_slice[i] = ArsaBoneTransform {
            px: out_pos[i].x,
            py: out_pos[i].y,
            pz: out_pos[i].z,
            qx: out_rot[i].x,
            qy: out_rot[i].y,
            qz: out_rot[i].z,
            qw: out_rot[i].w,
        };
    }
    true
}

/// Converts 17 world-space bone orientations (`68` floats in XYZW order) from the target engine's
/// coordinate space and bind pose (`ArsaRetargetPreset`) into 16 ARSA joint targets and blends them
/// onto `ragdoll_id` by `blend_weight` (`0.0..=1.0`).
/// Returns `false` if `handle` or `target_world_quats_17x4` is null, `ragdoll_id` or `preset` is invalid,
/// or any quaternion / weight is non-finite or zero.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `target_world_quats_17x4` is non-null, it must be valid for reads of `17 * 4` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_retargeted_target_pose(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    preset: u32,
    target_world_quats_17x4: *const f32,
    blend_weight: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    if target_world_quats_17x4.is_null() || !blend_weight.is_finite() {
        return false;
    }
    let Some(preset_enum) = RetargetPreset::from_u32(preset) else {
        return false;
    };
    let raw = unsafe { std::slice::from_raw_parts(target_world_quats_17x4, 17 * 4) };
    let mut target_quats = [Quat::IDENTITY; 17];
    for (i, dst) in target_quats.iter_mut().enumerate() {
        let base = i * 4;
        let q = Quat::from_xyzw(raw[base], raw[base + 1], raw[base + 2], raw[base + 3]);
        if !q.is_finite() || q.length_squared() < 1e-12 {
            return false;
        }
        *dst = q.normalize();
    }

    let retargeter = SkeletonRetargeter::from_preset(preset_enum);
    let joint_targets = retargeter.retarget_world_quats_to_arsa_joint_targets(&target_quats);
    h.world
        .set_ragdoll_target_pose(ragdoll_id, &joint_targets, blend_weight)
}

/// Attaches a ragdoll's left or right hand to dynamic `body_idx` at `local_anchor (lx, ly, lz)`
/// with grip break threshold `break_force` (N). Returns `false` on null or invalid arguments.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn arsa_world_grab_body_with_hand(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    is_left: bool,
    body_idx: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    break_force: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    h.world.grab_body_with_hand(
        ragdoll_id,
        is_left,
        body_idx,
        Vec3::new(lx, ly, lz),
        break_force,
    )
}

/// Releases any held object on the specified hand (`is_left`) of `ragdoll_id`.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_release_hand_hold(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    is_left: bool,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.release_hand_hold(ragdoll_id, is_left);
    }
}

/// Spawns a dynamic box directly in front of `ragdoll_id`'s chest and attaches both hands.
/// Returns the spawned box body index, or `usize::MAX` (`SIZE_MAX`) on null/failure.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_give_ragdoll_held_box(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    hx: f32,
    hy: f32,
    hz: f32,
    mass: f32,
    break_force: f32,
) -> usize {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return usize::MAX;
    };
    h.world
        .give_ragdoll_held_box(ragdoll_id, Vec3::new(hx, hy, hz), mass, break_force)
        .unwrap_or(usize::MAX)
}

/// Returns `true` if the specified hand (`is_left`) of `ragdoll_id` is currently holding an object.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_is_hand_holding(
    handle: *const ArsaWorldHandle,
    ragdoll_id: usize,
    is_left: bool,
) -> bool {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        return false;
    };
    h.world.is_hand_holding(ragdoll_id, is_left)
}

/// Applies a procedural status effect (`ArsaStatusEffect`: `0 = None`, `1 = OnFire`, `2 = Electrocuted`)
/// with `intensity` (`0.0..=1.0`) and `duration_sec` (`>= 0.0`) to `ragdoll_id`.
/// Returns `false` if `handle` is null, `ragdoll_id` is invalid, or `effect` is unknown.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_ragdoll_status(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    effect: u32,
    intensity: f32,
    duration_sec: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    let Some(status) = StatusEffect::from_u32(effect) else {
        return false;
    };
    h.world
        .set_ragdoll_status(ragdoll_id, status, intensity, duration_sec)
}

/// Returns the active `ArsaStatusEffect` (`0 = None`, `1 = OnFire`, `2 = Electrocuted`)
/// for `ragdoll_id`, or `u32::MAX` (`UINT32_MAX`) on null handle or invalid `ragdoll_id`.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_ragdoll_status(
    handle: *const ArsaWorldHandle,
    ragdoll_id: usize,
) -> u32 {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        return u32::MAX;
    };
    h.world
        .ragdoll_status(ragdoll_id)
        .map(|(s, _)| s as u32)
        .unwrap_or(u32::MAX)
}

/// Sets the horizontal desired locomotion velocity `(vx, vz)` in m/s (clamped `<= 3.5` m/s)
/// for `ragdoll_id`. Returns `false` if `handle` is null or `ragdoll_id` is out of range.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_ragdoll_desired_velocity(
    handle: *mut ArsaWorldHandle,
    ragdoll_id: usize,
    vx: f32,
    vz: f32,
) -> bool {
    let Some(h) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    h.world.set_ragdoll_desired_velocity(ragdoll_id, vx, vz)
}

/// Sets the horizontal desired locomotion velocity `(vx, vz)` in m/s (clamped `<= 3.5` m/s)
/// for all ragdolls in the world.
///
/// # Safety
/// `handle` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_set_all_desired_velocity(
    handle: *mut ArsaWorldHandle,
    vx: f32,
    vz: f32,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.world.set_all_ragdolls_desired_velocity(vx, vz);
    }
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
        assert_eq!(size_of::<ArsaBoneTransform>(), 28);
        assert_eq!(align_of::<ArsaBoneTransform>(), 4);
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

    extern "C" fn test_ground_cb(
        _user_data: *mut std::ffi::c_void,
        _x: f32,
        _z: f32,
        max_y: f32,
        out_y: *mut f32,
        out_normal: *mut [f32; 3],
    ) -> bool {
        if 1.5 <= max_y {
            unsafe {
                *out_y = 1.5;
                *out_normal = [0.0, 1.0, 0.0];
            }
            true
        } else {
            false
        }
    }

    extern "C" fn test_ray_cb(
        _user_data: *mut std::ffi::c_void,
        _origin: *const [f32; 3],
        _dir: *const [f32; 3],
        max_dist: f32,
        out_dist: *mut f32,
        out_normal: *mut [f32; 3],
    ) -> bool {
        if 1.0 < max_dist {
            unsafe {
                *out_dist = 1.0;
                *out_normal = [0.0, 0.0, -1.0];
            }
            true
        } else {
            false
        }
    }

    #[test]
    fn test_c_abi_level_geometry_and_callbacks() {
        unsafe {
            let world = arsa_world_create();
            assert!(!world.is_null());

            let q = Quat::from_rotation_x(20.0_f32.to_radians());
            let sb_idx = arsa_world_add_static_oriented_box(
                world, 0.0, 0.5, 0.0, 1.0, 0.1, 1.0, q.x, q.y, q.z, q.w, 0.85, 0.05,
            );
            assert_eq!(sb_idx, 0);

            let heights = [0.2_f32; 4];
            assert!(!arsa_world_set_heightfield(
                world,
                0.0,
                0.0,
                1.0,
                1,
                2,
                heights.as_ptr(),
                0.85,
                0.05
            ));
            assert!(arsa_world_set_heightfield(
                world,
                -1.0,
                -1.0,
                1.0,
                2,
                2,
                heights.as_ptr(),
                0.85,
                0.05
            ));
            arsa_world_clear_heightfield(world);

            arsa_world_set_query_callbacks(
                world,
                std::ptr::null_mut(),
                Some(test_ground_cb),
                Some(test_ray_cb),
            );
            let box_idx =
                arsa_world_spawn_box(world, 0.0, 2.0, 3.0, 0.15, 0.15, 0.15, 5.0, 0.0, 0.0, 0.0);
            assert_eq!(box_idx, 0);

            for _ in 0..90 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            let mut xform = [0.0_f32; 7];
            assert_eq!(arsa_world_get_transforms(world, xform.as_mut_ptr(), 7), 7);
            assert!(
                (xform[1] - 1.65).abs() < 0.03,
                "expected box to settle on y=1.5 host ground (center ~1.65), got {}",
                xform[1]
            );

            // Raycast toward box at z=3.0 should be occluded by host raycast wall at dist=1.0
            let hit = arsa_world_apply_punch(world, 0.0, 1.65, 0.0, 0.0, 0.0, 1.0, 10.0);
            assert_eq!(hit, -1);

            arsa_world_set_query_callbacks(world, std::ptr::null_mut(), None, None);
            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_retargeting_roundtrip() {
        unsafe {
            let world = arsa_world_create();
            assert!(!world.is_null());

            let rid = arsa_world_spawn_ragdoll(world, 0.0, 0.0, 0.0);
            assert_eq!(rid, 0);

            // Reject invalid pointers, invalid ragdoll IDs, and invalid presets
            let mut bones = [ArsaBoneTransform {
                px: 0.0,
                py: 0.0,
                pz: 0.0,
                qx: 0.0,
                qy: 0.0,
                qz: 0.0,
                qw: 1.0,
            }; 17];
            assert!(!arsa_world_get_retargeted_bones(
                std::ptr::null(),
                rid,
                0,
                bones.as_mut_ptr()
            ));
            assert!(!arsa_world_get_retargeted_bones(
                world,
                rid,
                0,
                std::ptr::null_mut()
            ));
            assert!(!arsa_world_get_retargeted_bones(
                world,
                99,
                0,
                bones.as_mut_ptr()
            ));
            assert!(!arsa_world_get_retargeted_bones(
                world,
                rid,
                99,
                bones.as_mut_ptr()
            ));

            // Check Unreal Manny retargeting: unit scale 100.0 (cm) and Z-up coordinate swap
            assert!(arsa_world_get_retargeted_bones(
                world,
                rid,
                RetargetPreset::UnrealMannyAPose as u32,
                bones.as_mut_ptr()
            ));
            // Pelvis standing height ~0.94m -> Unreal Z ~ 94 cm
            assert!(
                bones[0].pz > 80.0 && bones[0].pz < 110.0,
                "expected Unreal Manny pelvis Z in cm (~94), got {}",
                bones[0].pz
            );

            // Round-trip neutral retargeted orientations for all 4 presets -> identity joint targets
            for preset in 0..=3 {
                assert!(arsa_world_get_retargeted_bones(
                    world,
                    rid,
                    preset,
                    bones.as_mut_ptr()
                ));
                let mut quats_17x4 = [0.0_f32; 17 * 4];
                for (i, b) in bones.iter().enumerate() {
                    quats_17x4[i * 4] = b.qx;
                    quats_17x4[i * 4 + 1] = b.qy;
                    quats_17x4[i * 4 + 2] = b.qz;
                    quats_17x4[i * 4 + 3] = b.qw;
                }
                assert!(arsa_world_set_retargeted_target_pose(
                    world,
                    rid,
                    preset,
                    quats_17x4.as_ptr(),
                    1.0
                ));
            }

            // Reject non-finite or zero quaternion inputs and invalid preset/ragdoll
            let mut bad_quats = [0.0_f32; 17 * 4];
            assert!(!arsa_world_set_retargeted_target_pose(
                world,
                rid,
                1,
                bad_quats.as_ptr(),
                1.0
            ));
            for i in 0..17 {
                bad_quats[i * 4 + 3] = 1.0;
            }
            assert!(!arsa_world_set_retargeted_target_pose(
                world,
                rid,
                1,
                bad_quats.as_ptr(),
                f32::NAN
            ));
            assert!(!arsa_world_set_retargeted_target_pose(
                world,
                rid,
                42,
                bad_quats.as_ptr(),
                1.0
            ));

            // Valid Mixamo T-pose input drives ragdoll smoothly without NaN
            assert!(arsa_world_set_retargeted_target_pose(
                world,
                rid,
                RetargetPreset::MixamoTPose as u32,
                bad_quats.as_ptr(),
                1.0
            ));
            for _ in 0..30 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            assert!(arsa_world_get_retargeted_bones(
                world,
                rid,
                RetargetPreset::UnityHumanoidTPose as u32,
                bones.as_mut_ptr()
            ));
            for (i, b) in bones.iter().enumerate() {
                assert!(
                    b.px.is_finite()
                        && b.py.is_finite()
                        && b.pz.is_finite()
                        && b.qx.is_finite()
                        && b.qy.is_finite()
                        && b.qz.is_finite()
                        && b.qw.is_finite(),
                    "retargeted bone {i} produced NaN/Inf"
                );
            }

            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_lod_controls() {
        unsafe {
            assert_eq!(arsa_world_get_ragdoll_lod(std::ptr::null(), 0), u32::MAX);

            let world = arsa_world_create();
            assert!(!world.is_null());
            assert_eq!(arsa_world_get_ragdoll_lod(world, 0), u32::MAX);

            let r0 = arsa_world_spawn_ragdoll(world, 0.0, 0.02, 0.0);
            let r1 = arsa_world_spawn_ragdoll(world, 20.0, 0.02, 0.0);
            let r2 = arsa_world_spawn_ragdoll(world, 45.0, 0.02, 0.0);

            arsa_world_set_lod_focus(world, 0.0, 0.0, 0.0, 15.0, 35.0);
            // Non-finite focus is ignored and preserves existing valid focus.
            arsa_world_set_lod_focus(world, f32::NAN, 0.0, 0.0, 15.0, 35.0);

            for _ in 0..90 {
                arsa_world_step(world, 1.0 / 60.0);
            }
            assert_eq!(arsa_world_get_ragdoll_lod(world, r0), 0); // Full
            assert_eq!(arsa_world_get_ragdoll_lod(world, r1), 1); // Reduced
            assert_eq!(arsa_world_get_ragdoll_lod(world, r2), 2); // Sleep

            // Shove sleeping ragdoll at x=45 -> wakes immediately to Full
            let hit = arsa_world_apply_shove(world, 45.0, 1.3, 2.0, 0.0, 0.0, -1.0, 80.0);
            assert!(hit >= 0);
            arsa_world_step(world, 1.0 / 60.0);
            assert_eq!(arsa_world_get_ragdoll_lod(world, r2), 0);

            arsa_world_clear_lod_focus(world);
            arsa_world_step(world, 1.0 / 60.0);
            assert_eq!(arsa_world_get_ragdoll_lod(world, r0), 0);
            assert_eq!(arsa_world_get_ragdoll_lod(world, r1), 0);
            assert_eq!(arsa_world_get_ragdoll_lod(world, r2), 0);

            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_host_physics_brain_loop() {
        unsafe {
            let world = arsa_world_create();
            assert!(!world.is_null());

            let rid = arsa_world_spawn_ragdoll(world, 0.0, 0.0, 0.0);
            assert_eq!(rid, 0);

            let mut xforms_before = [0.0_f32; 17 * 7];
            assert_eq!(
                arsa_world_get_transforms(world, xforms_before.as_mut_ptr(), xforms_before.len()),
                17 * 7
            );

            // Sync a tilted LowerSpine (body 1) orientation and angular velocity from host engine.
            let base = 7;
            let q_tilt = Quat::from_rotation_x(0.30);
            assert!(arsa_world_sync_bone_pose(
                world,
                1,
                xforms_before[base],
                xforms_before[base + 1],
                xforms_before[base + 2],
                q_tilt.x,
                q_tilt.y,
                q_tilt.z,
                q_tilt.w,
                0.0,
                0.0,
                0.0,
                0.4,
                0.0,
                0.0,
            ));

            let mut xforms_synced = [0.0_f32; 17 * 7];
            assert_eq!(
                arsa_world_get_transforms(world, xforms_synced.as_mut_ptr(), xforms_synced.len()),
                17 * 7
            );

            // Brain-only step must NOT move synced transforms.
            arsa_world_step_brain_only(world, 1.0 / 60.0);
            let mut xforms_after = [0.0_f32; 17 * 7];
            assert_eq!(
                arsa_world_get_transforms(world, xforms_after.as_mut_ptr(), xforms_after.len()),
                17 * 7
            );
            for i in 0..xforms_synced.len() {
                assert!(
                    (xforms_after[i] - xforms_synced[i]).abs() < 1e-6,
                    "step_brain_only modified transform float {i}"
                );
            }

            // Reject invalid pointers, ragdoll IDs, and coordinate spaces.
            let mut torques_arsa = [0.0_f32; 48];
            let mut torques_unreal = [0.0_f32; 48];
            assert!(!arsa_world_compute_host_joint_torques(
                std::ptr::null(),
                rid,
                0,
                torques_arsa.as_mut_ptr()
            ));
            assert!(!arsa_world_compute_host_joint_torques(
                world,
                rid,
                0,
                std::ptr::null_mut()
            ));
            assert!(!arsa_world_compute_host_joint_torques(
                world,
                99,
                0,
                torques_arsa.as_mut_ptr()
            ));
            assert!(!arsa_world_compute_host_joint_torques(
                world,
                rid,
                99,
                torques_arsa.as_mut_ptr()
            ));

            assert!(arsa_world_compute_host_joint_torques(
                world,
                rid,
                CoordinateSpace::ArsaRightHandedYUp as u32,
                torques_arsa.as_mut_ptr()
            ));
            assert!(arsa_world_compute_host_joint_torques(
                world,
                rid,
                CoordinateSpace::UnrealLeftHandedZUp as u32,
                torques_unreal.as_mut_ptr()
            ));

            for i in 0..48 {
                assert!(torques_arsa[i].is_finite());
                assert!(torques_unreal[i].is_finite());
            }

            // Joint 0 (LowerSpine) has a restoring torque in X -> maps to +Y in Unreal LH Z-up.
            let tau_arsa_spine = Vec3::new(torques_arsa[0], torques_arsa[1], torques_arsa[2]);
            let tau_unreal_spine =
                Vec3::new(torques_unreal[0], torques_unreal[1], torques_unreal[2]);
            assert!(tau_arsa_spine.length() > 1.0);
            assert!(
                (tau_unreal_spine
                    - SkeletonRetargeter::convert_angular_vec_to_target(
                        CoordinateSpace::UnrealLeftHandedZUp,
                        tau_arsa_spine
                    ))
                .length()
                    < 1e-4
            );

            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_held_object_lifecycle() {
        unsafe {
            assert!(!arsa_world_is_hand_holding(std::ptr::null(), 0, true));
            assert_eq!(
                arsa_world_give_ragdoll_held_box(
                    std::ptr::null_mut(),
                    0,
                    0.18,
                    0.18,
                    0.18,
                    6.0,
                    320.0
                ),
                usize::MAX
            );
            assert!(!arsa_world_grab_body_with_hand(
                std::ptr::null_mut(),
                0,
                true,
                0,
                0.0,
                0.0,
                0.0,
                200.0
            ));

            let world = arsa_world_create();
            assert!(!world.is_null());

            let rid = arsa_world_spawn_ragdoll(world, 0.0, 0.02, 0.0);
            assert_eq!(rid, 0);

            let box_idx =
                arsa_world_give_ragdoll_held_box(world, rid, 0.18, 0.18, 0.18, 6.0, 320.0);
            assert_ne!(box_idx, usize::MAX);
            assert_eq!(box_idx, 17);
            assert!(arsa_world_is_hand_holding(world, rid, true));
            assert!(arsa_world_is_hand_holding(world, rid, false));

            for _ in 0..60 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            assert!(arsa_world_is_hand_holding(world, rid, true));
            assert!(arsa_world_is_hand_holding(world, rid, false));

            let mut xforms = [0.0_f32; 18 * 7];
            assert_eq!(
                arsa_world_get_transforms(world, xforms.as_mut_ptr(), xforms.len()),
                18 * 7
            );
            let box_y = xforms[box_idx * 7 + 1];
            assert!(box_y.is_finite() && box_y > 0.75, "box_y={box_y}");

            arsa_world_release_hand_hold(world, rid, true);
            assert!(!arsa_world_is_hand_holding(world, rid, true));
            assert!(arsa_world_is_hand_holding(world, rid, false));

            let extra_box =
                arsa_world_spawn_box(world, 0.2, 1.1, 0.35, 0.1, 0.1, 0.1, 2.0, 0.0, 0.0, 0.0);
            assert!(arsa_world_grab_body_with_hand(
                world, rid, true, extra_box, 0.0, 0.0, 0.0, 200.0
            ));
            assert!(arsa_world_is_hand_holding(world, rid, true));

            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_status_effects() {
        unsafe {
            assert!(!arsa_world_set_ragdoll_status(
                std::ptr::null_mut(),
                0,
                1,
                1.0,
                1.0
            ));
            assert_eq!(arsa_world_get_ragdoll_status(std::ptr::null(), 0), u32::MAX);

            let world = arsa_world_create();
            assert!(!world.is_null());
            assert_eq!(arsa_world_get_ragdoll_status(world, 0), u32::MAX);
            assert!(!arsa_world_set_ragdoll_status(world, 0, 1, 1.0, 1.0));

            let rid = arsa_world_spawn_ragdoll(world, 0.0, 0.02, 0.0);
            assert_eq!(rid, 0);
            assert_eq!(arsa_world_get_ragdoll_status(world, rid), 0);
            assert!(!arsa_world_set_ragdoll_status(world, rid, 99, 1.0, 1.0));

            let box_idx =
                arsa_world_give_ragdoll_held_box(world, rid, 0.18, 0.18, 0.18, 4.0, 260.0);
            assert_ne!(box_idx, usize::MAX);
            assert!(arsa_world_is_hand_holding(world, rid, true));
            assert!(arsa_world_is_hand_holding(world, rid, false));

            assert!(arsa_world_set_ragdoll_status(world, rid, 2, 1.0, 1.5));
            assert_eq!(arsa_world_get_ragdoll_status(world, rid), 2);
            assert!(!arsa_world_is_hand_holding(world, rid, true));
            assert!(!arsa_world_is_hand_holding(world, rid, false));

            for _ in 0..45 {
                arsa_world_step(world, 1.0 / 60.0);
            }
            assert_eq!(arsa_world_get_ragdoll_status(world, rid), 2);

            assert!(arsa_world_set_ragdoll_status(world, rid, 1, 1.0, 0.5));
            assert_eq!(arsa_world_get_ragdoll_status(world, rid), 1);

            for _ in 0..40 {
                arsa_world_step(world, 1.0 / 60.0);
            }
            assert_eq!(arsa_world_get_ragdoll_status(world, rid), 0);

            let mut xforms = [0.0_f32; 18 * 7];
            let written = arsa_world_get_transforms(world, xforms.as_mut_ptr(), xforms.len());
            assert_eq!(written, 18 * 7);
            for &v in &xforms[..written] {
                assert!(v.is_finite());
            }

            arsa_world_destroy(world);
        }
    }

    #[test]
    fn test_c_abi_locomotion_velocity() {
        unsafe {
            assert!(!arsa_world_set_ragdoll_desired_velocity(
                std::ptr::null_mut(),
                0,
                0.0,
                1.4
            ));
            arsa_world_set_all_desired_velocity(std::ptr::null_mut(), 0.0, 1.4);

            let world = arsa_world_create();
            assert!(!world.is_null());
            assert!(!arsa_world_set_ragdoll_desired_velocity(world, 0, 0.0, 1.4));

            let rid = arsa_world_spawn_ragdoll(world, 0.0, 0.02, 0.0);
            assert_eq!(rid, 0);

            assert!(arsa_world_set_ragdoll_desired_velocity(
                world, rid, 0.0, 1.4
            ));
            for _ in 0..90 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            let mut telemetry = [0.0_f32; 16];
            assert_eq!(
                arsa_world_get_ragdoll_telemetry(world, telemetry.as_mut_ptr(), 16),
                16
            );
            let pelvis_y = telemetry[9];
            let pelvis_z = telemetry[10];
            let state = telemetry[7] as u32;
            assert!(
                pelvis_z > 0.8,
                "expected ragdoll to walk forward in +Z (> 0.8m), got z={pelvis_z:.3}"
            );
            assert!(
                pelvis_y > 0.65,
                "expected ragdoll to stay upright, got y={pelvis_y:.3}"
            );
            assert!(state <= 1, "expected Balanced or Stumbling, got {state}");

            arsa_world_set_all_desired_velocity(world, 0.0, 0.0);
            for _ in 0..60 {
                arsa_world_step(world, 1.0 / 60.0);
            }

            let mut xforms = [0.0_f32; 17 * 7];
            let written = arsa_world_get_transforms(world, xforms.as_mut_ptr(), xforms.len());
            assert_eq!(written, 17 * 7);
            for &v in &xforms[..written] {
                assert!(v.is_finite());
            }

            arsa_world_destroy(world);
        }
    }
}
