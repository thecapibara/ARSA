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

/** Retargeted single-bone world transform (7 floats: position + XYZW quaternion). */
typedef struct ArsaBoneTransform {
    float px;
    float py;
    float pz;
    float qx;
    float qy;
    float qz;
    float qw;
} ArsaBoneTransform;

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
    ARSA_STATE_UNCONSCIOUS = 5,
    ARSA_STATE_RECOVERING = 6
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

/** Procedural status effects for arsa_world_set_ragdoll_status(). */
typedef enum ArsaStatusEffect {
    ARSA_STATUS_NONE = 0,
    ARSA_STATUS_ON_FIRE = 1,
    ARSA_STATUS_ELECTROCUTED = 2
} ArsaStatusEffect;

/** Universal skeleton retargeting presets for cross-engine bone conversion. */
typedef enum ArsaRetargetPreset {
    ARSA_RETARGET_NATIVE = 0,
    ARSA_RETARGET_MIXAMO_TPOSE = 1,
    ARSA_RETARGET_UNITY_HUMANOID = 2,
    ARSA_RETARGET_UNREAL_MANNY = 3
} ArsaRetargetPreset;

/** Crowd-scaling Level of Detail (LOD) tier for a humanoid ragdoll. */
typedef enum ArsaRagdollLod {
    ARSA_LOD_FULL = 0,
    ARSA_LOD_REDUCED = 1,
    ARSA_LOD_SLEEP = 2
} ArsaRagdollLod;

/** Target engine coordinate space convention for world-space vectors and torques. */
typedef enum ArsaCoordinateSpace {
    ARSA_SPACE_RH_Y_UP = 0,
    ARSA_SPACE_UNITY_LH_Y_UP = 1,
    ARSA_SPACE_UNREAL_LH_Z_UP = 2
} ArsaCoordinateSpace;

/**
 * Host engine ground-height query callback.
 * Given (user_data, x, z, max_y), writes surface height to *out_y and upward unit normal
 * to out_normal[3], returning true if a surface exists at y <= max_y.
 */
typedef bool (*ArsaGroundHeightFn)(
    void *user_data,
    float x,
    float z,
    float max_y,
    float *out_y,
    float out_normal[3]);

/**
 * Host engine static-world raycast occlusion callback.
 * Given (user_data, origin[3], dir[3], max_dist), writes hit distance to *out_dist and hit normal
 * to out_normal[3], returning true if static host geometry was hit closer than max_dist.
 */
typedef bool (*ArsaRaycastWorldFn)(
    void *user_data,
    const float origin[3],
    const float dir[3],
    float max_dist,
    float *out_dist,
    float out_normal[3]);

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

/**
 * Runs a brain-only controller tick (foot grounding, balance state machine, and procedural
 * reflexes) over currently synced bone transforms without integrating positions/orientations
 * or running XPBD solvers.
 */
void arsa_world_step_brain_only(ArsaWorldHandle *handle, float dt);

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

/** Enables or disables automatic procedural get-up from Downed for all ragdolls in the world. */
void arsa_world_set_auto_get_up(ArsaWorldHandle *handle, bool enabled);

/**
 * Sets an external animation target pose (num_joints == 16, i.e. 64 floats in XYZW order)
 * blended onto the ragdoll's joint motor targets by blend_weight (0.0 .. 1.0).
 * Returns false if handle or quats_xyzw is NULL, num_joints != 16, ragdoll_id is invalid,
 * or any quaternion/weight is non-finite or zero.
 */
bool arsa_world_set_target_pose(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    const float *quats_xyzw,
    size_t num_joints,
    float blend_weight);

/** Clears any external animation target pose on ragdoll_id. Returns false on NULL or invalid ID. */
bool arsa_world_clear_target_pose(ArsaWorldHandle *handle, size_t ragdoll_id);

/** Sets the earthquake seismic magnitude (0.0 .. 9.0, 0.0 disables). */
void arsa_world_set_earthquake(ArsaWorldHandle *handle, float magnitude);

/** Sets the procedural stagger/intoxication level (0.0 .. 1.0) for all ragdolls. */
void arsa_world_set_intoxication(ArsaWorldHandle *handle, float level);

/**
 * Configures the spatial crowd-scaling LOD focus point and distance thresholds (in meters).
 * Ignores non-finite focus coordinates; clamps reduced_dist >= 1.0 and sleep_dist >= reduced_dist + 1.0.
 */
void arsa_world_set_lod_focus(
    ArsaWorldHandle *handle,
    float x,
    float y,
    float z,
    float reduced_dist,
    float sleep_dist);

/** Clears the LOD focus point and wakes all sleeping ragdolls back to ARSA_LOD_FULL. */
void arsa_world_clear_lod_focus(ArsaWorldHandle *handle);

/** Returns the current ArsaRagdollLod tier for ragdoll_id, or UINT32_MAX on invalid handle/id. */
uint32_t arsa_world_get_ragdoll_lod(const ArsaWorldHandle *handle, size_t ragdoll_id);

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

/** Adds a static oriented box obstacle (XYZW quaternion) and returns its index (SIZE_MAX on null). */
size_t arsa_world_add_static_oriented_box(
    ArsaWorldHandle *handle,
    float cx,
    float cy,
    float cz,
    float hx,
    float hy,
    float hz,
    float qx,
    float qy,
    float qz,
    float qw,
    float friction,
    float restitution);

/**
 * Sets a uniform-grid heightfield terrain collider from row-major heights_ptr (cols * rows floats).
 * Returns false if handle or heights_ptr is NULL, dimensions/cell_size are invalid, or any value is non-finite.
 */
bool arsa_world_set_heightfield(
    ArsaWorldHandle *handle,
    float origin_x,
    float origin_z,
    float cell_size,
    size_t cols,
    size_t rows,
    const float *heights_ptr,
    float friction,
    float restitution);

/** Clears the world's heightfield terrain collider. */
void arsa_world_clear_heightfield(ArsaWorldHandle *handle);

/** Registers or clears host engine world-query callbacks (ground height & raycast occlusion). */
void arsa_world_set_query_callbacks(
    ArsaWorldHandle *handle,
    void *user_data,
    ArsaGroundHeightFn ground_height_fn,
    ArsaRaycastWorldFn raycast_fn);

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
 * Synchronizes a body's simulated pose and velocities from an external host physics engine
 * (e.g. UE5 Chaos / Unity PhysX) so ARSA balance/reflex controllers can compute updated
 * joint motor targets. Returns false if handle is NULL, body_idx is invalid, or any
 * input is non-finite / degenerate.
 */
bool arsa_world_sync_bone_pose(
    ArsaWorldHandle *handle,
    size_t body_idx,
    float px,
    float py,
    float pz,
    float qx,
    float qy,
    float qz,
    float qw,
    float vx,
    float vy,
    float vz,
    float wx,
    float wy,
    float wz);

/**
 * Copies packed body transforms (7 floats per body: px, py, pz, qx, qy, qz, qw)
 * into out_buf (up to max_floats) and returns the number of floats written.
 */
size_t arsa_world_get_transforms(
    ArsaWorldHandle *handle,
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
    ArsaWorldHandle *handle,
    float *out_buf,
    size_t max_floats);

/**
 * Copies per-joint PD muscle motor targets (8 floats per joint:
 * qx, qy, qz, qw, stiffness, damping, max_torque, damage_scale) into out_buf
 * for Host-Physics Control Mode. Returns the number of floats written.
 */
size_t arsa_world_get_joint_motor_targets(
    ArsaWorldHandle *handle,
    float *out_buf,
    size_t max_floats);

/**
 * Computes 16 world-space PD muscle motor torques (48 floats: 16 * (tx, ty, tz) in N*m)
 * for ragdoll_id, converted into the target engine's coordinate axes (ArsaCoordinateSpace).
 * Returns false if handle or out_torques_48 is NULL, ragdoll_id is invalid, or coord_space is unknown.
 */
bool arsa_world_compute_host_joint_torques(
    const ArsaWorldHandle *handle,
    size_t ragdoll_id,
    uint32_t coord_space,
    float *out_torques_48);

/**
 * Reads all 17 bone transforms of ragdoll_id retargeted into the target engine's coordinate space,
 * unit scale, and bind pose (ArsaRetargetPreset).
 * Returns false if handle or out_bones_17 is NULL, ragdoll_id is invalid, or preset is unknown.
 */
bool arsa_world_get_retargeted_bones(
    const ArsaWorldHandle *handle,
    size_t ragdoll_id,
    uint32_t preset,
    ArsaBoneTransform *out_bones_17);

/**
 * Converts 17 world-space bone orientations (68 floats in XYZW order) from the target engine's
 * coordinate space and bind pose (ArsaRetargetPreset) into 16 ARSA joint targets and blends them
 * onto ragdoll_id by blend_weight (0.0 .. 1.0).
 * Returns false if handle or target_world_quats_17x4 is NULL, ragdoll_id or preset is invalid,
 * or any quaternion/weight is non-finite or zero.
 */
bool arsa_world_set_retargeted_target_pose(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    uint32_t preset,
    const float *target_world_quats_17x4,
    float blend_weight);

/**
 * Attaches a ragdoll's left or right hand to dynamic body_idx at local_anchor (lx, ly, lz)
 * with grip break threshold break_force (N). Returns false on null/invalid arguments.
 */
bool arsa_world_grab_body_with_hand(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    bool is_left,
    size_t body_idx,
    float lx,
    float ly,
    float lz,
    float break_force);

/** Releases any held object on the specified hand (is_left) of ragdoll_id. */
void arsa_world_release_hand_hold(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    bool is_left);

/**
 * Spawns a dynamic box directly in front of ragdoll_id's chest and attaches both hands.
 * Returns the spawned box body index, or SIZE_MAX on failure.
 */
size_t arsa_world_give_ragdoll_held_box(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    float hx,
    float hy,
    float hz,
    float mass,
    float break_force);

/** Returns true if the specified hand (is_left) of ragdoll_id is currently holding an object. */
bool arsa_world_is_hand_holding(
    const ArsaWorldHandle *handle,
    size_t ragdoll_id,
    bool is_left);

/**
 * Applies a procedural status effect (ArsaStatusEffect) with intensity (0.0 .. 1.0)
 * and duration_sec (>= 0.0) to ragdoll_id. Returns false if handle is NULL, ragdoll_id
 * is invalid, or effect is unknown.
 */
bool arsa_world_set_ragdoll_status(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    uint32_t effect,
    float intensity,
    float duration_sec);

/**
 * Returns the active ArsaStatusEffect for ragdoll_id, or UINT32_MAX on invalid handle/id.
 */
uint32_t arsa_world_get_ragdoll_status(
    const ArsaWorldHandle *handle,
    size_t ragdoll_id);

/**
 * Sets the horizontal desired locomotion velocity (vx, vz in m/s, clamped <= 3.5 m/s)
 * for ragdoll_id. Returns false if handle is NULL or ragdoll_id is invalid.
 */
bool arsa_world_set_ragdoll_desired_velocity(
    ArsaWorldHandle *handle,
    size_t ragdoll_id,
    float vx,
    float vz);

/**
 * Sets the horizontal desired locomotion velocity (vx, vz in m/s, clamped <= 3.5 m/s)
 * for all ragdolls in the world.
 */
void arsa_world_set_all_desired_velocity(
    ArsaWorldHandle *handle,
    float vx,
    float vz);

#ifdef __cplusplus
}
#endif

#endif /* ARSA_H */
