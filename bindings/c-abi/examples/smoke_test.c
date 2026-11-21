#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "arsa.h"

int main(void) {
    ArsaWorldHandle *world = arsa_world_create();
    if (!world) {
        fprintf(stderr, "FAIL: arsa_world_create returned NULL\n");
        return 1;
    }

    /* Spawn a 2.17m / 135kg Heavy Brute active ragdoll at the origin. */
    size_t ragdoll_id =
        arsa_world_spawn_ragdoll_profile(world, 0.0f, 0.0f, 0.0f, 2.17f, 135.0f, 1.0f);
    if (ragdoll_id == (size_t)-1) {
        fprintf(stderr, "FAIL: spawn_ragdoll_profile failed\n");
        arsa_world_destroy(world);
        return 1;
    }

    size_t body_count = arsa_world_body_count(world);
    size_t joint_count = arsa_world_joint_count(world);
    if (body_count != ARSA_BONE_COUNT || joint_count != 16) {
        fprintf(stderr, "FAIL: unexpected counts bodies=%zu joints=%zu\n", body_count, joint_count);
        arsa_world_destroy(world);
        return 1;
    }

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        arsa_world_step(world, dt);
    }

    float telemetry[ARSA_TELEMETRY_FLOATS_PER_RAGDOLL];
    size_t t_written =
        arsa_world_get_ragdoll_telemetry(world, telemetry, ARSA_TELEMETRY_FLOATS_PER_RAGDOLL);
    if (t_written != ARSA_TELEMETRY_FLOATS_PER_RAGDOLL) {
        fprintf(stderr, "FAIL: expected 16 telemetry floats, got %zu\n", t_written);
        arsa_world_destroy(world);
        return 1;
    }

    int state = (int)(telemetry[7] + 0.5f);
    float pelvis_y = telemetry[9];
    float health = telemetry[11];
    if (state != ARSA_STATE_BALANCED) {
        fprintf(stderr, "FAIL: expected Balanced (%d), got %d (pelvis_y=%.3f)\n",
                ARSA_STATE_BALANCED, state, pelvis_y);
        arsa_world_destroy(world);
        return 1;
    }

    /* Locate the LeftForearm bone transform and fire a Pistol into it. */
    float transforms[ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY];
    size_t xform_written = arsa_world_get_transforms(
        world, transforms, ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY);
    if (xform_written != ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY) {
        fprintf(stderr, "FAIL: expected %d transform floats, got %zu\n",
                ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY, xform_written);
        arsa_world_destroy(world);
        return 1;
    }

    size_t forearm_base = (size_t)ARSA_BONE_LEFT_FOREARM * ARSA_TRANSFORM_FLOATS_PER_BODY;
    float fx = transforms[forearm_base + 0];
    float fy = transforms[forearm_base + 1];
    float fz = transforms[forearm_base + 2];

    int32_t hits = arsa_world_fire_weapon(
        world, fx, fy, fz + 2.0f, 0.0f, 0.0f, -1.0f, ARSA_WEAPON_PISTOL);
    if (hits <= 0) {
        fprintf(stderr, "FAIL: pistol shot missed left forearm at (%.3f, %.3f, %.3f)\n", fx, fy, fz);
        arsa_world_destroy(world);
        return 1;
    }

    for (int i = 0; i < 30; ++i) {
        arsa_world_step(world, dt);
    }

    t_written = arsa_world_get_ragdoll_telemetry(world, telemetry, ARSA_TELEMETRY_FLOATS_PER_RAGDOLL);
    if (t_written != ARSA_TELEMETRY_FLOATS_PER_RAGDOLL) {
        fprintf(stderr, "FAIL: telemetry read after shot failed\n");
        arsa_world_destroy(world);
        return 1;
    }

    int wound_region = (int)(telemetry[12] + 0.5f);
    float post_health = telemetry[11];
    if (wound_region != ARSA_WOUND_LEFT_ARM) {
        fprintf(stderr, "FAIL: expected WoundRegion == LeftArm (%d), got %d\n",
                ARSA_WOUND_LEFT_ARM, wound_region);
        arsa_world_destroy(world);
        return 1;
    }
    if (post_health >= health) {
        fprintf(stderr, "FAIL: expected health to decrease (before=%.1f, after=%.1f)\n",
                health, post_health);
        arsa_world_destroy(world);
        return 1;
    }

    /* Read joint motor targets for Host-Physics Control Mode verification. */
    float motors[16 * ARSA_MOTOR_TARGET_FLOATS_PER_JOINT];
    size_t m_written = arsa_world_get_joint_motor_targets(
        world, motors, 16 * ARSA_MOTOR_TARGET_FLOATS_PER_JOINT);
    if (m_written != joint_count * ARSA_MOTOR_TARGET_FLOATS_PER_JOINT) {
        fprintf(stderr, "FAIL: expected %zu motor floats, got %zu\n",
                joint_count * ARSA_MOTOR_TARGET_FLOATS_PER_JOINT, m_written);
        arsa_world_destroy(world);
        return 1;
    }

    for (size_t i = 0; i < m_written; ++i) {
        if (!isfinite(motors[i])) {
            fprintf(stderr, "FAIL: non-finite motor target value at index %zu\n", i);
            arsa_world_destroy(world);
            return 1;
        }
    }

    /* Verify Host-Physics pose synchronization input. */
    if (!arsa_world_sync_bone_pose(
            world,
            ARSA_BONE_PELVIS,
            transforms[0],
            transforms[1],
            transforms[2],
            transforms[3],
            transforms[4],
            transforms[5],
            transforms[6],
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f)) {
        fprintf(stderr, "FAIL: arsa_world_sync_bone_pose failed for valid pelvis bone\n");
        arsa_world_destroy(world);
        return 1;
    }
    if (arsa_world_sync_bone_pose(
            world, 9999, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)) {
        fprintf(stderr, "FAIL: arsa_world_sync_bone_pose should reject out-of-range body_idx\n");
        arsa_world_destroy(world);
        return 1;
    }

    /* Verify universal skeleton retargeting C ABI (Unreal Manny A-pose). */
    ArsaBoneTransform manny_bones[ARSA_BONE_COUNT];
    if (!arsa_world_get_retargeted_bones(world, ragdoll_id, ARSA_RETARGET_UNREAL_MANNY, manny_bones)) {
        fprintf(stderr, "FAIL: arsa_world_get_retargeted_bones failed\n");
        arsa_world_destroy(world);
        return 1;
    }
    float manny_quats[ARSA_BONE_COUNT * 4];
    for (int i = 0; i < ARSA_BONE_COUNT; ++i) {
        manny_quats[i * 4 + 0] = manny_bones[i].qx;
        manny_quats[i * 4 + 1] = manny_bones[i].qy;
        manny_quats[i * 4 + 2] = manny_bones[i].qz;
        manny_quats[i * 4 + 3] = manny_bones[i].qw;
    }
    if (!arsa_world_set_retargeted_target_pose(
            world, ragdoll_id, ARSA_RETARGET_UNREAL_MANNY, manny_quats, 0.5f)) {
        fprintf(stderr, "FAIL: arsa_world_set_retargeted_target_pose failed\n");
        arsa_world_destroy(world);
        return 1;
    }

    /* Verify Bidirectional Host-Physics Mode: brain-only tick + computed world-space joint torques. */
    arsa_world_step_brain_only(world, dt);
    float host_torques[48];
    if (!arsa_world_compute_host_joint_torques(
            world, ragdoll_id, ARSA_SPACE_UNREAL_LH_Z_UP, host_torques)) {
        fprintf(stderr, "FAIL: arsa_world_compute_host_joint_torques failed\n");
        arsa_world_destroy(world);
        return 1;
    }
    for (size_t i = 0; i < 48; ++i) {
        if (!isfinite(host_torques[i])) {
            fprintf(stderr, "FAIL: non-finite host joint torque at index %zu\n", i);
            arsa_world_destroy(world);
            return 1;
        }
    }
    if (arsa_world_compute_host_joint_torques(world, ragdoll_id, 99, host_torques)) {
        fprintf(stderr, "FAIL: arsa_world_compute_host_joint_torques should reject invalid coord_space\n");
        arsa_world_destroy(world);
        return 1;
    }

    printf("PASS: Heavy Brute (2.17m/135kg) Balanced (pelvis_y=%.3fm), "
           "LeftArm wound=%d, health=%.1f, motors=%zu floats\n",
           pelvis_y, wound_region, post_health, m_written);

    arsa_world_destroy(world);
    return 0;
}
