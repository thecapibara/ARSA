//! Native C ABI bindings (`arsa-c-abi`) for Unreal Engine, Unity, and Godot.

use arsa_core::glam::Vec3;
use arsa_core::ArsaWorld;

/// Opaque handle owning an `ArsaWorld` simulation instance.
pub struct ArsaWorldHandle {
    world: ArsaWorld,
}

/// Allocates and initializes a new `ArsaWorldHandle`.
/// The caller owns the returned pointer and must free it with `arsa_world_destroy`.
#[no_mangle]
pub extern "C" fn arsa_world_create() -> *mut ArsaWorldHandle {
    Box::into_raw(Box::new(ArsaWorldHandle {
        world: ArsaWorld::new(),
    }))
}

/// Destroys an `ArsaWorldHandle` previously created by `arsa_world_create`.
///
/// # Safety
/// `world` must be either null or a valid pointer returned by `arsa_world_create`
/// that has not yet been destroyed.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_destroy(world: *mut ArsaWorldHandle) {
    if !world.is_null() {
        drop(unsafe { Box::from_raw(world) });
    }
}

/// Steps the simulation forward by `dt` seconds.
///
/// # Safety
/// `world` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_step(world: *mut ArsaWorldHandle, dt: f32) {
    if let Some(handle) = unsafe { world.as_mut() } {
        handle.world.step(dt);
    }
}

/// Spawns a humanoid ragdoll at `(x, y, z)` and returns its ragdoll ID (`usize::MAX` on null).
///
/// # Safety
/// `world` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_spawn_ragdoll(
    world: *mut ArsaWorldHandle,
    x: f32,
    y: f32,
    z: f32,
) -> usize {
    if let Some(handle) = unsafe { world.as_mut() } {
        handle.world.spawn_ragdoll(Vec3::new(x, y, z))
    } else {
        usize::MAX
    }
}

/// Returns the number of dynamic rigid bodies in the world.
///
/// # Safety
/// `world` must be null or a valid pointer to an `ArsaWorldHandle`.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_body_count(world: *const ArsaWorldHandle) -> usize {
    if let Some(handle) = unsafe { world.as_ref() } {
        handle.world.bodies.len()
    } else {
        0
    }
}

/// Copies packed transforms (`px, py, pz, qx, qy, qz, qw` per body) into `out_ptr`,
/// writing at most `max_floats` floats, and returns the number of floats written.
///
/// # Safety
/// `world` must be null or a valid pointer to an `ArsaWorldHandle`.
/// If `out_ptr` is non-null, it must be valid for writes of `max_floats` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn arsa_world_get_transforms(
    world: *const ArsaWorldHandle,
    out_ptr: *mut f32,
    max_floats: usize,
) -> usize {
    let Some(handle) = (unsafe { world.as_ref() }) else {
        return 0;
    };
    if out_ptr.is_null() || max_floats == 0 {
        return 0;
    }

    let mut buf = Vec::new();
    handle.world.write_transforms(&mut buf);
    let count = buf.len().min(max_floats);
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), out_ptr, count);
    }
    count
}
