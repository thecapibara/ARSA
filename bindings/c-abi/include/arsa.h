/**
 * @file arsa.h
 * @brief Native C/C++ ABI for the ARSA Active Ragdoll Simulation Architecture.
 *
 * Coordinate System Convention (ARSA):
 *   - Right-handed, Y-up, meters:
 *     +X = character left, +Y = up, +Z = character forward
 *   - Quaternions are packed in XYZW order: (qx, qy, qz, qw)
 */

#ifndef ARSA_H
#define ARSA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Packed float counts for buffer queries. */
#define ARSA_TRANSFORM_FLOATS_PER_BODY 7
#define ARSA_TELEMETRY_FLOATS_PER_RAGDOLL 16
#define ARSA_MOTOR_TARGET_FLOATS_PER_JOINT 8

/** Opaque handle owning an ARSA simulation world. */
typedef struct ArsaWorldHandle ArsaWorldHandle;

/** Anthropometric body dimensions and muscular strength scaling profile. */
typedef struct ArsaBodyProfile {
    float height_m;     /**< Standing height in meters (clamped to 1.40 .. 2.40). */
    float mass_kg;      /**< Total mass in kilograms (clamped to 40.0 .. 180.0). */
    float muscle_ratio; /**< Relative muscle strength multiplier (0.25 .. 2.0). */
} ArsaBodyProfile;

/** Rigid-body world transform (7 floats: position + XYZW quaternion). */
typedef struct ArsaTransform {
    float px;
    float py;
    float pz;
    float qx;
    float qy;
    float qz;
    float qw;
} ArsaTransform;

/** Joint muscle motor target for Host-Physics Control Mode (8 floats). */
typedef struct ArsaJointMotorTarget {
    float qx;           /**< Target local orientation X. */
    float qy;           /**< Target local orientation Y. */
    float qz;           /**< Target local orientation Z. */
    float qw;           /**< Target local orientation W. */
    float stiffness;    /**< Proportional PD gain (N*m/rad). */
    float damping;      /**< Derivative PD gain (N*m*s/rad). */
    float max_torque;   /**< Torque saturation limit (N*m). */
    float damage_scale; /**< Injury attenuation factor (0.0 .. 1.0). */
} ArsaJointMotorTarget;

/** 17-bone humanoid bone identifiers (matching ArsaWorld body order per ragdoll). */
typedef enum ArsaBoneId {
    ARSA_BONE_PELVIS = 0,
    ARSA_BONE_LOWER_SPINE = 1,
    ARSA_BONE_CHEST = 2,
    ARSA_BONE_NECK = 3,
    ARSA_BONE_HEAD = 4,
    ARSA_BONE_LEFT_UPPER_ARM = 5,
    ARSA_BONE_LEFT_FOREARM = 6,
    ARSA_BONE_LEFT_HAND = 7,
    ARSA_BONE_RIGHT_UPPER_ARM = 8,
    ARSA_BONE_RIGHT_FOREARM = 9,
    ARSA_BONE_RIGHT_HAND = 10,
    ARSA_BONE_LEFT_THIGH = 11,
    ARSA_BONE_LEFT_SHIN = 12,
    ARSA_BONE_LEFT_FOOT = 13,
    ARSA_BONE_RIGHT_THIGH = 14,
    ARSA_BONE_RIGHT_SHIN = 15,
    ARSA_BONE_RIGHT_FOOT = 16,
    ARSA_BONE_COUNT = 17
} ArsaBoneId;

/** Behavioral balance state reported in telemetry[7]. */
typedef enum ArsaBehaviorState {
    ARSA_STATE_BALANCED = 0,
    ARSA_STATE_STUMBLING = 1,
    ARSA_STATE_FALLING = 2,
    ARSA_STATE_DOWNED = 3,
    ARSA_STATE_PASSIVE = 4,
    ARSA_STATE_UNCONSCIOUS = 5
} ArsaBehaviorState;

/** Anatomical wound region reported in telemetry[12]. */
typedef enum ArsaWoundRegion {
    ARSA_WOUND_NONE = 0,
    ARSA_WOUND_LEFT_ARM = 1,
    ARSA_WOUND_RIGHT_ARM = 2,
    ARSA_WOUND_TORSO = 3,
    ARSA_WOUND_HEAD = 4,
    ARSA_WOUND_LEFT_LEG = 5,
    ARSA_WOUND_RIGHT_LEG = 6
} ArsaWoundRegion;

/** Ballistic weapon presets for arsa_world_fire_weapon(). */
typedef enum ArsaWeaponKind {
    ARSA_WEAPON_PISTOL = 0,
    ARSA_WEAPON_SHOTGUN = 1,
    ARSA_WEAPON_RIFLE = 2
} ArsaWeaponKind;

/**
 * Allocates and initializes a new ARSA physics world.
 * Caller owns the returned handle and must free it with arsa_world_destroy().
 */
ArsaWorldHandle *arsa_world_create(void);

/** Destroys a world handle previously returned by arsa_world_create(). */
void arsa_world_destroy(ArsaWorldHandle *handle);

/** Clears all bodies, joints, tethers, static obstacles, ragdolls, and grabs. */
void arsa_world_clear(ArsaWorldHandle *handle);

/** Advances the simulation by dt seconds (clamped internally to <= 0.05s). */
void arsa_world_step(ArsaWorldHandle *handle, float dt);

/** Spawns a default 1.78m / 75kg ragdoll at (x, y, z) and returns its ragdoll ID (SIZE_MAX on null). */
size_t arsa_world_spawn_ragdoll(ArsaWorldHandle *handle, float x, float y, float z);

/** Spawns a procedurally scaled ragdoll at (x, y, z) and returns its ragdoll ID (SIZE_MAX on null). */
size_t arsa_world_spawn_ragdoll_profile(
    ArsaWorldHandle *handle,
    float x,
    float y,
    float z,
    float height_m,
    float mass_kg,
    float muscle_ratio);

/** Updates the anthropometric body profile of all spawned ragdolls in-place without resetting. */
void arsa_world_update_all_profiles(
    ArsaWorldHandle *handle,
    float height_m,
    float mass_kg,
    float muscle_ratio);

/** Enables or disables active balance and muscle motors for all ragdolls in the world. */
void arsa_world_set_active(ArsaWorldHandle *handle, bool enabled);

/** Spawns a dynamic rigid-body box and returns its body index (SIZE_MAX on null). */
size_t arsa_world_spawn_box(
    ArsaWorldHandle *handle,
    float x,
    float y,
    float z,
    float hx,
    float hy,
    float hz,
    float mass,
    float vx,
    float vy,
    float vz);

/** Adds a static axis-aligned box obstacle and returns its index (SIZE_MAX on null). */
size_t arsa_world_add_static_box(
    ArsaWorldHandle *handle,
    float cx,
    float cy,
    float cz,
    float hx,
    float hy,
    float hz,
    float friction,
    float restitution);

/**
 * Raycasts from (ox, oy, oz) along (dx, dy, dz) and applies a broad two-handed shove
 * of strength N*s. Returns the hit body index, or -1 if null or missed.
 */
int32_t arsa_world_apply_shove(
    ArsaWorldHandle *handle,
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float strength);

/**
 * Raycasts from (ox, oy, oz) along (dx, dy, dz) and applies a sharp localized punch
 * of strength N*s. Returns the hit body index, or -1 if null or missed.
 */
int32_t arsa_world_apply_punch(
    ArsaWorldHandle *handle,
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float strength);

/**
 * Fires a ballistic weapon (ArsaWeaponKind) from (ox, oy, oz) along (dx, dy, dz).
 * Returns the number of ray hits (0 if null or missed).
 */
int32_t arsa_world_fire_weapon(
    ArsaWorldHandle *handle,
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    uint32_t weapon_kind);

/**
 * Fires a kinetic force wave from (ox, oy, oz) along (dx, dy, dz) affecting dynamic
 * bodies within radius meters up to max_impulse N*s. Returns the number of affected bodies.
 */
size_t arsa_world_apply_force_wave(
    ArsaWorldHandle *handle,
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float radius,
    float max_impulse);

/** Returns the number of dynamic rigid bodies in the world. */
size_t arsa_world_body_count(const ArsaWorldHandle *handle);

/** Returns the number of articulated joints in the world. */
size_t arsa_world_joint_count(const ArsaWorldHandle *handle);

/**
 * Copies packed body transforms (7 floats per body: px, py, pz, qx, qy, qz, qw)
 * into out_buf (up to max_floats) and returns the number of floats written.
 */
size_t arsa_world_get_transforms(
    const ArsaWorldHandle *handle,
    float *out_buf,
    size_t max_floats);

/**
 * Copies per-ragdoll balance & reflex telemetry (16 floats per ragdoll) into out_buf:
 *   [0..2]  com_pos (x, y, z)
 *   [3..4]  capture_point (x, z)
 *   [5]     left_foot_grounded (0.0 or 1.0)
 *   [6]     right_foot_grounded (0.0 or 1.0)
 *   [7]     behavior_state (ArsaBehaviorState)
 *   [8..10] pelvis_pos (x, y, z)
 *   [11]    health (0.0 .. 100.0)
 *   [12]    wound_region (ArsaWoundRegion)
 *   [13..15] wound_world (x, y, z)
 * Returns the number of floats written.
 */
size_t arsa_world_get_ragdoll_telemetry(
    const ArsaWorldHandle *handle,
    float *out_buf,
    size_t max_floats);

/**
 * Copies per-joint PD muscle motor targets (8 floats per joint:
 * qx, qy, qz, qw, stiffness, damping, max_torque, damage_scale) into out_buf
 * for Host-Physics Control Mode. Returns the number of floats written.
 */
size_t arsa_world_get_joint_motor_targets(
    const ArsaWorldHandle *handle,
    float *out_buf,
    size_t max_floats);

#ifdef __cplusplus
}
#endif

#endif /* ARSA_H */
