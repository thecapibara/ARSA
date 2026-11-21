use arsa_core::*;
use glam::{Quat, Vec3};

#[test]
fn test_swing_twist_round_trip() {
    let axes = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(1.0, 2.0, -1.0).normalize(),
    ];
    let rotations = [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.4, 0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -1.2, 2.1, -0.5),
        Quat::from_axis_angle(Vec3::X, std::f32::consts::PI),
        Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI * 0.95),
    ];

    for axis in axes {
        for q in rotations {
            let (swing, twist) = swing_twist_decomposition(q, axis);
            assert!(swing.is_normalized(), "swing must be normalized");
            assert!(twist.is_normalized(), "twist must be normalized");

            let reconstructed = (swing * twist).normalize();
            let dot = reconstructed.dot(q.normalize()).abs();
            assert!(
                (dot - 1.0).abs() < 1e-5,
                "round-trip mismatch for axis={axis:?}, q={q:?}: dot={dot}"
            );

            // Verify twist only rotates around `axis`
            let twist_vec = Vec3::new(twist.x, twist.y, twist.z);
            let cross = twist_vec.cross(axis.normalize());
            assert!(
                cross.length() < 1e-5,
                "twist axis deviates from target axis: cross={cross:?}"
            );
        }
    }
}

#[test]
fn test_humanoid_drop_stability_and_joint_limits() {
    let mut world = ArsaWorld::new();
    let ragdoll_id = world.spawn_ragdoll(Vec3::new(0.0, 0.2, 0.0));
    world.set_ragdoll_active(ragdoll_id, false);
    assert_eq!(world.bodies.len(), 17);
    assert_eq!(world.joints.len(), 16);

    // Step 3 seconds (180 frames at 1/60s)
    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf"
        );
        assert!(
            b.position.y > -0.25 && b.position.y < 3.0,
            "body {i} exploded or fell through floor: pos={:?}",
            b.position
        );
    }

    let pelvis = &world.bodies[world.ragdolls[ragdoll_id].body_indices[BoneId::Pelvis as usize]];
    assert!(
        pelvis.position.y >= 0.05 && pelvis.position.y < 1.2,
        "pelvis failed to settle near ground: y={}",
        pelvis.position.y
    );

    // Verify knee and elbow hinge joint angles stay within anatomical limits
    for (j_idx, joint) in world.joints.iter().enumerate() {
        if let JointKind::Hinge {
            min_angle,
            max_angle,
            ..
        } = joint.kind
        {
            let angle = joint.measure_hinge_angle(&world.bodies);
            let tol = 0.05;
            assert!(
                angle >= min_angle - tol && angle <= max_angle + tol,
                "hinge joint {j_idx} violated limits [{min_angle}, {max_angle}]: angle={angle}"
            );
        }
    }
}

#[test]
fn test_dynamic_box_rests_on_ground() {
    let mut world = ArsaWorld::new();
    let half = Vec3::new(0.25, 0.25, 0.25);
    let box_idx = world.spawn_box(Vec3::new(0.0, 1.5, 0.0), half, 2.0, Vec3::ZERO);

    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }

    let b = &world.bodies[box_idx];
    assert!(
        (b.position.y - half.y).abs() < 0.02,
        "box did not rest at half_extents.y ({}): actual y={}",
        half.y,
        b.position.y
    );
    assert!(
        b.linear_velocity.length() < 0.05,
        "box still moving: vel={:?}",
        b.linear_velocity
    );
}

#[test]
fn test_raycast_box_and_ragdoll_torso() {
    let mut world = ArsaWorld::new();
    let box_idx = world.spawn_box(Vec3::new(3.0, 0.5, 0.0), Vec3::splat(0.5), 1.0, Vec3::ZERO);
    let ragdoll_id = world.spawn_ragdoll(Vec3::new(0.0, 0.0, 0.0));
    let chest_idx = world.ragdolls[ragdoll_id].body_indices[BoneId::Chest as usize];

    let box_hit = world
        .raycast(Vec3::new(3.0, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0), 20.0)
        .expect("ray should hit box");
    assert_eq!(box_hit.body_idx, box_idx);
    assert!((box_hit.distance - 4.5).abs() < 1e-4);
    assert!((box_hit.normal - Vec3::Z).length() < 1e-4);

    let chest_pos = world.bodies[chest_idx].position;
    let torso_hit = world
        .raycast(
            chest_pos + Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, -1.0),
            20.0,
        )
        .expect("ray should hit ragdoll chest");
    assert_eq!(torso_hit.body_idx, chest_idx);
    assert!(torso_hit.distance > 3.5 && torso_hit.distance < 4.0);
}

#[test]
fn test_active_standing_balance_standard_and_heavy() {
    let mut world = ArsaWorld::new();
    let profiles = [
        BodyProfile {
            height_m: 1.78,
            mass_kg: 75.0,
            muscle_ratio: 1.0,
        },
        BodyProfile {
            height_m: 2.17,
            mass_kg: 135.0,
            muscle_ratio: 1.0,
        },
        BodyProfile {
            height_m: 1.65,
            mass_kg: 60.0,
            muscle_ratio: 1.0,
        },
    ];

    let ids: Vec<usize> = profiles
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let x = (i as f32 - 1.0) * 2.5;
            world.spawn_ragdoll_with_profile(Vec3::new(x, 0.0, 0.0), p)
        })
        .collect();

    // Step 2.0 seconds (120 frames at 1/60s)
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf"
        );
    }

    for &id in &ids {
        let r = &world.ragdolls[id];
        let pelvis = &world.bodies[r.body_indices[BoneId::Pelvis as usize]];
        assert!(
            pelvis.position.y > 0.72 * r.standing_pelvis_y,
            "ragdoll {id} ({:?}) slumped: pelvis.y={}, threshold={}",
            r.profile,
            pelvis.position.y,
            0.72 * r.standing_pelvis_y
        );
        assert_eq!(
            r.balance.state,
            BehaviorState::Balanced,
            "ragdoll {id} ({:?}) expected Balanced, got {:?}",
            r.profile,
            r.balance.state
        );
    }
}

#[test]
fn test_shove_triggers_stumble_or_fall() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.0, 0.0));

    // Let settle 0.5 seconds
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let initial_pos = world.bodies[pelvis_idx].position;

    // Strong backward shove to the chest
    let chest_pos = world.bodies[chest_idx].position;
    let hit = world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            320.0,
        )
        .expect("shove should hit chest");
    assert_eq!(hit.body_idx, chest_idx);

    let mut saw_reaction = false;
    for _ in 0..90 {
        world.step(1.0 / 60.0);
        let st = world.ragdolls[id].balance.state;
        if matches!(
            st,
            BehaviorState::Stumbling | BehaviorState::Falling | BehaviorState::Downed
        ) {
            saw_reaction = true;
        }
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf after shove"
        );
    }

    let final_pos = world.bodies[pelvis_idx].position;
    let horiz_disp = Vec3::new(
        final_pos.x - initial_pos.x,
        0.0,
        final_pos.z - initial_pos.z,
    )
    .length();

    assert!(
        saw_reaction && horiz_disp > 0.25,
        "shove did not trigger expected displacement/reaction: saw_reaction={saw_reaction}, horiz_disp={horiz_disp}"
    );
}

#[test]
fn test_pistol_arm_shot_triggers_reach_for_wound() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    for _ in 0..15 {
        world.step(1.0 / 60.0);
    }

    let left_forearm_idx = world.ragdolls[id].body_indices[BoneId::LeftForearm as usize];
    let right_hand_idx = world.ragdolls[id].body_indices[BoneId::RightHand as usize];
    let forearm_pos = world.bodies[left_forearm_idx].position;
    let resting_dist = (world.bodies[right_hand_idx].position - forearm_pos).length();

    let res = world.fire_weapon(
        forearm_pos + Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Pistol,
    );
    assert_eq!(res.hit_ragdoll_id, Some(id));
    assert_eq!(res.hit_region, WoundRegion::LeftArm);

    for _ in 0..48 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf after pistol arm shot"
        );
    }

    let wound = world.ragdolls[id]
        .reflex
        .active_wound
        .expect("active wound should be set");
    assert_eq!(wound.region, WoundRegion::LeftArm);
    let wound_world = world.bodies[world.ragdolls[id].body_indices[wound.bone as usize]]
        .world_point(wound.local_point);
    let reached_dist = (world.bodies[right_hand_idx].position - wound_world).length();
    assert!(
        reached_dist < resting_dist * 0.72,
        "RightHand did not reach toward LeftArm wound: resting_dist={resting_dist}, reached_dist={reached_dist}"
    );
}

#[test]
fn test_shotgun_knockback_and_brace() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    for _ in 0..15 {
        world.step(1.0 / 60.0);
    }

    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let initial_pos = world.bodies[pelvis_idx].position;
    let chest_pos = world.bodies[chest_idx].position;

    let res = world.fire_weapon(
        chest_pos + Vec3::new(0.0, 0.0, 2.2),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Shotgun,
    );
    assert!(!res.hits.is_empty());
    assert_eq!(res.hit_ragdoll_id, Some(id));

    for _ in 0..72 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf after shotgun blast"
        );
    }

    let final_pos = world.bodies[pelvis_idx].position;
    let backward_disp = initial_pos.z - final_pos.z;
    assert!(
        backward_disp > 0.35,
        "shotgun blast did not produce strong backward displacement: backward_disp={backward_disp}"
    );
    assert!(
        world.ragdolls[id].reflex.health < 100.0,
        "health should drop after shotgun blast"
    );
}

#[test]
fn test_headshot_rifle_unconscious() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    let head_idx = world.ragdolls[id].body_indices[BoneId::Head as usize];
    let head_pos = world.bodies[head_idx].position;

    let res = world.fire_weapon(
        head_pos + Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Rifle,
    );
    assert_eq!(res.hit_ragdoll_id, Some(id));
    assert_eq!(res.hit_region, WoundRegion::Head);

    for _ in 0..12 {
        world.step(1.0 / 60.0);
    }

    let r = &world.ragdolls[id];
    assert!(
        r.reflex.unconscious,
        "headshot rifle should cause unconsciousness"
    );
    assert_eq!(r.balance.state, BehaviorState::Unconscious);
}

#[test]
fn test_realtime_profile_resize_stability() {
    let mut world = ArsaWorld::new();
    let initial_profile = BodyProfile {
        height_m: 1.65,
        mass_kg: 60.0,
        muscle_ratio: 0.9,
    };
    let id = world.spawn_ragdoll_with_profile(Vec3::ZERO, initial_profile);

    // Step 0.5 seconds (30 frames at 1/60s)
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    // Resize in-place to 135 kg / 2.17 m
    let heavy_profile = BodyProfile {
        height_m: 2.17,
        mass_kg: 135.0,
        muscle_ratio: 1.0,
    };
    world.update_all_ragdolls_profile(heavy_profile);

    // Step 1.0 second (60 frames at 1/60s)
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf after realtime resize"
        );
    }

    let r = &world.ragdolls[id];
    assert_eq!(r.profile, heavy_profile);
    let pelvis = &world.bodies[r.body_indices[BoneId::Pelvis as usize]];
    assert!(
        pelvis.position.y > 0.72 * r.standing_pelvis_y,
        "resized ragdoll slumped: pelvis.y={}, threshold={}",
        pelvis.position.y,
        0.72 * r.standing_pelvis_y
    );
    assert_eq!(
        r.balance.state,
        BehaviorState::Balanced,
        "resized ragdoll expected Balanced, got {:?}",
        r.balance.state
    );
}

#[test]
fn test_wrecking_ball_tether_and_impact() {
    let mut world = ArsaWorld::new();
    let ragdoll_id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    world.add_static_box(
        Vec3::new(0.0, 4.35, 0.0),
        Vec3::new(1.4, 0.12, 0.18),
        0.75,
        0.05,
    );
    let anchor = Vec3::new(0.0, 4.25, 0.0);
    let tether_length = 3.15;
    let ball_idx = world.spawn_sphere(
        Vec3::new(-2.75, 2.35, 0.0),
        0.42,
        85.0,
        Vec3::new(1.8, 0.0, 0.0),
    );
    world.add_tether(anchor, ball_idx, tether_length);

    let pelvis_idx = world.ragdolls[ragdoll_id].body_indices[BoneId::Pelvis as usize];
    let initial_pelvis = world.bodies[pelvis_idx].position;

    // Step 1.5s (90 frames at 1/60s)
    for frame in 0..90 {
        world.step(1.0 / 60.0);
        let dist = (world.bodies[ball_idx].position - anchor).length();
        assert!(
            dist <= tether_length + 0.02,
            "tether length exceeded at frame {frame}: dist={dist}, limit={tether_length}"
        );
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf in wrecking ball test"
        );
    }

    let disp = (world.bodies[pelvis_idx].position - initial_pelvis).length();
    assert!(
        disp > 0.35,
        "mannequin was not significantly displaced by wrecking ball impact: disp={disp}"
    );
}

#[test]
fn test_force_wave_affects_crowd() {
    let mut world = ArsaWorld::new();
    let crowd = [
        (
            Vec3::new(0.0, 0.02, 0.75),
            BodyProfile {
                height_m: 1.78,
                mass_kg: 75.0,
                muscle_ratio: 1.0,
            },
        ),
        (
            Vec3::new(-0.75, 0.02, -0.05),
            BodyProfile {
                height_m: 1.65,
                mass_kg: 60.0,
                muscle_ratio: 0.9,
            },
        ),
        (
            Vec3::new(0.75, 0.02, -0.05),
            BodyProfile {
                height_m: 1.90,
                mass_kg: 70.0,
                muscle_ratio: 1.0,
            },
        ),
        (
            Vec3::new(0.0, 0.02, -0.95),
            BodyProfile {
                height_m: 2.17,
                mass_kg: 135.0,
                muscle_ratio: 1.35,
            },
        ),
    ];

    for (pos, profile) in crowd {
        world.spawn_ragdoll_with_profile(pos, profile);
    }

    let affected = world.apply_force_wave(
        Vec3::new(0.0, 1.8, 3.5),
        Vec3::new(0.0, -0.22, -1.0),
        3.2,
        42.0,
    );
    assert!(
        affected > 17,
        "expected force wave to affect multiple crowd bodies (>17), got {affected}"
    );

    // Step 0.5s (30 frames at 1/60s)
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            b.position.is_finite()
                && b.orientation.is_finite()
                && b.linear_velocity.is_finite()
                && b.angular_velocity.is_finite(),
            "body {i} produced NaN/Inf after force wave on crowd"
        );
    }
}

#[test]
fn test_extreme_dense_profile_no_helicopter_or_slide() {
    let extreme_profile = BodyProfile {
        height_m: 1.66,
        mass_kg: 160.0,
        muscle_ratio: 1.80,
    };

    // Part A: Standing — no helicopter spinning/flying explosion
    let mut world_stand = ArsaWorld::new();
    let id = world_stand.spawn_ragdoll_with_profile(Vec3::ZERO, extreme_profile);
    let pelvis_idx = world_stand.ragdolls[id].body_indices[BoneId::Pelvis as usize];

    let mut max_pelvis_y = 0.0_f32;
    let mut max_ang_vel = 0.0_f32;
    for _ in 0..180 {
        world_stand.step(1.0 / 60.0);
        max_pelvis_y = max_pelvis_y.max(world_stand.bodies[pelvis_idx].position.y);
        for (i, b) in world_stand.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf in extreme standing profile"
            );
            max_ang_vel = max_ang_vel.max(b.angular_velocity.length());
        }
    }

    assert!(
        max_pelvis_y < 1.35,
        "extreme dense profile pelvis flew too high: max_pelvis_y={max_pelvis_y}"
    );
    assert!(
        max_ang_vel < 15.0,
        "extreme dense profile exhibited excessive angular velocity: max_ang_vel={max_ang_vel}"
    );
    assert_eq!(
        world_stand.ragdolls[id].balance.state,
        BehaviorState::Balanced,
        "extreme dense profile failed to stay Balanced: {:?}",
        world_stand.ragdolls[id].balance.state
    );

    // Part B: Downed on ground — no floor sliding/drifting
    let mut world_down = ArsaWorld::new();
    let down_id = world_down.spawn_ragdoll_with_profile(Vec3::new(0.0, 0.2, 0.0), extreme_profile);
    world_down.set_ragdoll_active(down_id, false);
    let down_chest = world_down.ragdolls[down_id].body_indices[BoneId::Chest as usize];
    let down_pelvis = world_down.ragdolls[down_id].body_indices[BoneId::Pelvis as usize];
    // Tip the passive mannequin onto the floor
    let chest_pos = world_down.bodies[down_chest].position;
    world_down.apply_impulse_at_point(down_chest, Vec3::new(0.0, 0.0, -450.0), chest_pos);
    for _ in 0..90 {
        world_down.step(1.0 / 60.0);
    }

    // Re-enable active control while lying downed and let settle for 2.0s
    world_down.set_all_ragdolls_auto_get_up(false);
    world_down.set_ragdoll_active(down_id, true);
    for _ in 0..120 {
        world_down.step(1.0 / 60.0);
    }

    assert_eq!(
        world_down.ragdolls[down_id].balance.state,
        BehaviorState::Downed,
        "expected knocked mannequin to be in Downed state, got {:?}",
        world_down.ragdolls[down_id].balance.state
    );

    let start_pos = world_down.bodies[down_pelvis].position;
    // Measure horizontal drift over the final 1.0 second (60 frames)
    for _ in 0..60 {
        world_down.step(1.0 / 60.0);
    }
    let end_pos = world_down.bodies[down_pelvis].position;
    let horiz_drift = Vec3::new(end_pos.x - start_pos.x, 0.0, end_pos.z - start_pos.z).length();
    assert!(
        horiz_drift < 0.05,
        "downed extreme dense profile slid across floor: horiz_drift={horiz_drift}"
    );
}

#[test]
fn test_place_box_under_feet_no_launch() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));

    // Step 0.5 seconds (30 frames at 1/60s)
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    world.place_obstacle_box(0.0, 0.0, 0.35, 0.14, 0.18, 30.0);

    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let mut max_pelvis_y = 0.0_f32;

    // Step 1.5 seconds (90 frames at 1/60s)
    for _ in 0..90 {
        world.step(1.0 / 60.0);
        max_pelvis_y = max_pelvis_y.max(world.bodies[pelvis_idx].position.y);
        world.step(1.0 / 60.0);
        let py = world.bodies[pelvis_idx].position.y;
        if py > max_pelvis_y {
            max_pelvis_y = py;
        }
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf after placing box under feet"
            );
        }
    }

    let pelvis_y = world.bodies[pelvis_idx].position.y;
    assert!(
        max_pelvis_y < 1.85 && pelvis_y < 1.85,
        "placing box under feet launched ragdoll: pelvis_y={pelvis_y}, max_pelvis_y={max_pelvis_y}"
    );
}

#[test]
fn test_dynamite_detonates_and_applies_blast() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let initial_pelvis = world.bodies[pelvis_idx].position;

    let dyn_idx = world.spawn_dynamite(Vec3::new(0.0, 0.25, 0.8), Vec3::ZERO, 0.3);
    assert_eq!(world.bodies[dyn_idx].bone_id, Some(250));

    // Step 0.8 seconds (48 frames at 1/60s)
    for _ in 0..48 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf after dynamite blast"
            );
        }
    }

    let mut explosions = Vec::new();
    world.take_explosions(&mut explosions);
    assert_eq!(
        explosions.len(),
        4,
        "expected 1 explosion event (4 floats), got {} floats",
        explosions.len()
    );
    assert!(world.bodies[dyn_idx].position.y < -40.0);

    let disp = (world.bodies[pelvis_idx].position - initial_pelvis).length();
    assert!(
        disp > 0.5,
        "ragdoll was not significantly displaced by dynamite blast: disp={disp}"
    );
    assert!(
        world.ragdolls[id].reflex.health < 100.0,
        "ragdoll health should decrease from dynamite blast"
    );
}

#[test]
fn test_box_stacking_tower_stability() {
    let mut world = ArsaWorld::new();
    let half = Vec3::splat(0.24);
    let _b0 = world.spawn_box(Vec3::new(0.0, 0.24, 0.0), half, 22.0, Vec3::ZERO);
    let _b1 = world.spawn_box(Vec3::new(0.0, 0.722, 0.0), half, 22.0, Vec3::ZERO);
    let top_idx = world.spawn_box(Vec3::new(0.0, 1.204, 0.0), half, 22.0, Vec3::ZERO);

    // Step 2.0 seconds (120 frames at 1/60s)
    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during box stacking"
            );
        }
    }

    let top = &world.bodies[top_idx];
    let horiz_drift = Vec3::new(top.position.x, 0.0, top.position.z).length();
    assert!(
        top.position.y > 1.05,
        "top cube collapsed: y={}, horiz_drift={horiz_drift}",
        top.position.y
    );
    assert!(
        horiz_drift < 0.08,
        "top cube drifted horizontally: horiz_drift={horiz_drift}, pos={:?}",
        top.position
    );
}

#[test]
fn test_auto_get_up_after_delay_and_not_when_unconscious() {
    // Test A: Healthy active ragdoll knocked down recovers after delay
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    for _ in 0..20 {
        world.step(1.0 / 60.0);
    }

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            620.0,
        )
        .expect("shove should hit chest");

    // After 0.5s, must still be Falling or Downed (does NOT pop up instantly)
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }
    let state_after_half_sec = world.ragdolls[id].balance.state;
    assert!(
        matches!(
            state_after_half_sec,
            BehaviorState::Falling | BehaviorState::Downed
        ),
        "expected Falling or Downed after 0.5s, got {state_after_half_sec:?}"
    );

    // Step 5.0 more seconds; should enter Recovering and stand back up
    let mut saw_recovering = false;
    for _ in 0..300 {
        world.step(1.0 / 60.0);
        if world.ragdolls[id].balance.state == BehaviorState::Recovering {
            saw_recovering = true;
        }
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during recovery"
            );
        }
    }

    assert!(
        saw_recovering,
        "ragdoll never entered Recovering state; final state={:?}",
        world.ragdolls[id].balance.state
    );
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let pelvis_y = world.bodies[pelvis_idx].position.y;
    let threshold = 0.72 * world.ragdolls[id].standing_pelvis_y;
    assert!(
        pelvis_y > threshold,
        "ragdoll did not stand back up: pelvis_y={pelvis_y}, threshold={threshold}, state={:?}",
        world.ragdolls[id].balance.state
    );

    // Test B: Unconscious ragdoll (headshot with Rifle) NEVER recovers
    let mut world_unc = ArsaWorld::new();
    let unc_id = world_unc.spawn_ragdoll(Vec3::ZERO);
    let head_idx = world_unc.ragdolls[unc_id].body_indices[BoneId::Head as usize];
    let head_pos = world_unc.bodies[head_idx].position;
    let res = world_unc.fire_weapon(
        head_pos + Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Rifle,
    );
    assert_eq!(res.hit_region, WoundRegion::Head);

    for _ in 0..300 {
        world_unc.step(1.0 / 60.0);
        assert_ne!(
            world_unc.ragdolls[unc_id].balance.state,
            BehaviorState::Recovering,
            "unconscious ragdoll must never enter Recovering"
        );
    }
    assert_eq!(
        world_unc.ragdolls[unc_id].balance.state,
        BehaviorState::Unconscious
    );
}

#[test]
fn test_earthquake_shakes_ragdoll_and_boxes() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let half = Vec3::splat(0.24);
    let _b0 = world.spawn_box(Vec3::new(2.0, 0.24, 0.0), half, 20.0, Vec3::ZERO);
    let top_box_idx = world.spawn_box(Vec3::new(2.0, 0.722, 0.0), half, 20.0, Vec3::ZERO);

    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let initial_pelvis = world.bodies[pelvis_idx].position;
    let initial_box = world.bodies[top_box_idx].position;

    world.set_earthquake(8.0);

    let mut max_pelvis_disp = 0.0_f32;
    let mut max_box_disp = 0.0_f32;
    for _ in 0..90 {
        world.step(1.0 / 60.0);
        max_pelvis_disp =
            max_pelvis_disp.max((world.bodies[pelvis_idx].position - initial_pelvis).length());
        max_box_disp =
            max_box_disp.max((world.bodies[top_box_idx].position - initial_box).length());
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during earthquake"
            );
        }
    }

    assert!(
        max_pelvis_disp > 0.05,
        "earthquake did not shake ragdoll: max_pelvis_disp={max_pelvis_disp}"
    );
    assert!(
        max_box_disp > 0.05,
        "earthquake did not shake stacked box: max_box_disp={max_box_disp}"
    );
}

#[test]
fn test_arms_do_not_clip_through_chest() {
    // Verify zero spawn overlap between Forearm/Hand and Pelvis/LowerSpine/Chest even at 166cm / 160kg
    let extreme_profile = BodyProfile {
        height_m: 1.66,
        mass_kg: 160.0,
        muscle_ratio: 1.80,
    };
    let mut world_extreme = ArsaWorld::new();
    let ex_id = world_extreme.spawn_ragdoll_with_profile(Vec3::ZERO, extreme_profile);
    let ex_r = &world_extreme.ragdolls[ex_id];
    let ex_pelvis = &world_extreme.bodies[ex_r.body_indices[BoneId::Pelvis as usize]];
    let ex_pelvis_hx = match ex_pelvis.collider {
        ColliderShape::Box { half_extents } => half_extents.x,
        _ => panic!("expected Pelvis box collider"),
    };
    for arm_bone in [
        BoneId::LeftForearm,
        BoneId::LeftHand,
        BoneId::RightForearm,
        BoneId::RightHand,
    ] {
        let b = &world_extreme.bodies[ex_r.body_indices[arm_bone as usize]];
        let arm_hx = match b.collider {
            ColliderShape::Box { half_extents } => half_extents.x,
            ColliderShape::Capsule { radius, .. } => radius,
            ColliderShape::Sphere { radius } => radius,
        };
        assert!(
            b.position.x.abs() - arm_hx > ex_pelvis_hx,
            "{arm_bone:?} overlaps Pelvis horizontally at 166cm/160kg spawn"
        );
    }

    // Active ragdoll shotgun knockback + inward arm impulses
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    let res = world.fire_weapon(
        chest_pos + Vec3::new(0.0, 0.0, 2.2),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Shotgun,
    );
    assert_eq!(res.hit_ragdoll_id, Some(id));

    let chest_half_extents = match world.bodies[chest_idx].collider {
        ColliderShape::Box { half_extents } => half_extents,
        _ => panic!("expected Chest box collider"),
    };
    let inner_chest = chest_half_extents * 0.65;

    let arm_bones = [
        BoneId::LeftHand,
        BoneId::RightHand,
        BoneId::LeftForearm,
        BoneId::RightForearm,
    ];

    for frame in 0..90 {
        // Periodically push arms inward toward Chest to stress-test self-collision + shoulder cone
        if frame % 15 == 0 {
            let c_pos = world.bodies[chest_idx].position;
            for bone in arm_bones {
                let b_idx = world.ragdolls[id].body_indices[bone as usize];
                let b_pos = world.bodies[b_idx].position;
                let inward = (c_pos - b_pos).normalize_or_zero();
                world.apply_impulse_at_point(b_idx, inward * 12.0, b_pos);
            }
        }

        world.step(1.0 / 60.0);

        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf at frame {frame}"
            );
        }

        let c_pos = world.bodies[chest_idx].position;
        let c_inv_rot = world.bodies[chest_idx].orientation.conjugate();
        for bone in arm_bones {
            let b_idx = world.ragdolls[id].body_indices[bone as usize];
            let local_in_chest = c_inv_rot * (world.bodies[b_idx].position - c_pos);
            let deep_inside = local_in_chest.x.abs() < inner_chest.x
                && local_in_chest.y.abs() < inner_chest.y
                && local_in_chest.z.abs() < inner_chest.z;
            assert!(
                !deep_inside,
                "{bone:?} clipped deep inside Chest at frame {frame}: local_in_chest={local_in_chest:?}, inner_chest={inner_chest:?}"
            );
        }
    }
}

#[test]
fn test_raycast_occluded_by_static_wall() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;

    // Place a static wall between ray origin and ragdoll chest
    world.add_static_box(
        Vec3::new(0.0, 1.0, 1.5),
        Vec3::new(2.0, 2.0, 0.2),
        0.75,
        0.05,
    );

    let origin = chest_pos + Vec3::new(0.0, 0.0, 3.0);
    let dir = Vec3::new(0.0, 0.0, -1.0);

    // Raycast through static wall must be occluded and return None
    assert!(
        world.raycast(origin, dir, 20.0).is_none(),
        "raycast should not shoot through a StaticBox wall"
    );

    // Weapon shot through the wall must also record no hits
    let shot = world.fire_weapon(origin, dir, WeaponKind::Rifle);
    assert!(shot.hits.is_empty());
    assert_eq!(shot.hit_ragdoll_id, None);

    // Raycast starting in front of the wall (between wall and ragdoll) still hits Chest
    let clear_hit = world
        .raycast(chest_pos + Vec3::new(0.0, 0.0, 1.0), dir, 20.0)
        .expect("raycast in front of wall should hit ragdoll chest");
    assert_eq!(clear_hit.body_idx, chest_idx);
}

#[test]
fn test_shove_interrupts_recovering() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    world.ragdolls[id].balance.state = BehaviorState::Recovering;
    world.ragdolls[id].balance.recovery_phase = 0.55;
    world.ragdolls[id].balance.downed_still_time = 2.0;

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;

    let hit = world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            120.0,
        )
        .expect("shove should hit chest");
    assert_eq!(hit.body_idx, chest_idx);

    assert_eq!(world.ragdolls[id].balance.state, BehaviorState::Falling);
    assert_eq!(world.ragdolls[id].balance.downed_still_time, 0.0);
    assert_eq!(world.ragdolls[id].balance.recovery_phase, 0.0);
}

#[test]
fn test_sync_bone_pose_host_physics() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];

    let new_pos = Vec3::new(1.5, 2.25, -0.75);
    let unnorm_rot = Quat::from_xyzw(0.2, 0.4, -0.1, 1.2);
    let expected_rot = unnorm_rot.normalize();
    let lin_vel = Vec3::new(0.5, -1.2, 3.0);
    let ang_vel = Vec3::new(-0.3, 0.8, 0.1);

    assert!(world.sync_bone_pose(pelvis_idx, new_pos, unnorm_rot, lin_vel, ang_vel));

    let b = &world.bodies[pelvis_idx];
    assert_eq!(b.position, new_pos);
    assert_eq!(b.prev_position, new_pos);
    assert!((b.orientation.dot(expected_rot) - 1.0).abs() < 1e-6);
    assert!((b.prev_orientation.dot(expected_rot) - 1.0).abs() < 1e-6);
    assert_eq!(b.linear_velocity, lin_vel);
    assert_eq!(b.angular_velocity, ang_vel);

    // Out-of-range and non-finite / zero-quat inputs must fail without corrupting state
    assert!(!world.sync_bone_pose(9999, new_pos, expected_rot, lin_vel, ang_vel));
    assert!(!world.sync_bone_pose(pelvis_idx, Vec3::NAN, expected_rot, lin_vel, ang_vel));
    assert!(!world.sync_bone_pose(pelvis_idx, new_pos, Quat::NAN, lin_vel, ang_vel));
    assert!(!world.sync_bone_pose(
        pelvis_idx,
        new_pos,
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        lin_vel,
        ang_vel
    ));
    assert!(!world.sync_bone_pose(pelvis_idx, new_pos, expected_rot, Vec3::INFINITY, ang_vel));
    assert!(!world.sync_bone_pose(
        pelvis_idx,
        new_pos,
        expected_rot,
        lin_vel,
        Vec3::NEG_INFINITY
    ));

    let b_after = &world.bodies[pelvis_idx];
    assert_eq!(b_after.position, new_pos);
    assert_eq!(b_after.linear_velocity, lin_vel);
}

#[test]
fn test_obb_edge_edge_sat_separation() {
    let mut world = ArsaWorld::new();
    world.gravity = Vec3::ZERO;

    let half = Vec3::splat(0.5);
    let pos_b = Vec3::new(0.0, 2.0, 0.0);
    // Axis (1, -1, 1) / sqrt(3) is the cross product of A's X-axis (rot_z(45°)) and B's Y-axis (rot_x(45°)).
    // Combined half-projection along this cross axis is (1 + sqrt(2)) / sqrt(3) ≈ 1.39385.
    // At distance 1.42, all 6 face axes still overlap (>= 0.19m) while the edge-edge axis separates them.
    let sep_axis = Vec3::new(1.0, -1.0, 1.0).normalize();
    let pos_a = pos_b + sep_axis * 1.42;

    let idx_a = world.spawn_box(pos_a, half, 5.0, Vec3::ZERO);
    let idx_b = world.spawn_box(pos_b, half, 5.0, Vec3::ZERO);

    let rot_a = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let rot_b = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);
    world.bodies[idx_a].orientation = rot_a;
    world.bodies[idx_a].prev_orientation = rot_a;
    world.bodies[idx_a].update_world_inertia();
    world.bodies[idx_b].orientation = rot_b;
    world.bodies[idx_b].prev_orientation = rot_b;
    world.bodies[idx_b].update_world_inertia();

    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }

    let a = &world.bodies[idx_a];
    let b = &world.bodies[idx_b];
    assert!(
        a.linear_velocity.length() < 1e-5 && b.linear_velocity.length() < 1e-5,
        "separated angled boxes emitted phantom linear repulsion: va={:?}, vb={:?}",
        a.linear_velocity,
        b.linear_velocity
    );
    assert!(
        a.angular_velocity.length() < 1e-5 && b.angular_velocity.length() < 1e-5,
        "separated angled boxes emitted phantom angular repulsion: wa={:?}, wb={:?}",
        a.angular_velocity,
        b.angular_velocity
    );
    assert!(
        (a.position - pos_a).length() < 1e-5 && (b.position - pos_b).length() < 1e-5,
        "separated angled boxes moved unexpectedly: pa={:?}, pb={:?}",
        a.position,
        b.position
    );
}

#[test]
fn test_prop_sleeping_and_wakeup() {
    let mut world = ArsaWorld::new();
    let half = Vec3::splat(0.25);
    let box_a = world.spawn_box(Vec3::new(0.0, 0.25, 0.0), half, 2.0, Vec3::ZERO);
    let box_b = world.spawn_box(Vec3::new(2.0, 0.25, 0.0), half, 2.0, Vec3::ZERO);

    // Step 2.0s (120 frames at 1/60s): both props should settle and sleep
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }

    assert!(
        world.bodies[box_a].is_sleeping,
        "box_a did not enter sleep after 2.0s: vel={:?}, ang_vel={:?}, timer={}",
        world.bodies[box_a].linear_velocity,
        world.bodies[box_a].angular_velocity,
        world.bodies[box_a].sleep_timer
    );
    assert!(world.bodies[box_b].is_sleeping);

    // Wake box_a with an impulse and verify it wakes up and moves
    let pos_a_before = world.bodies[box_a].position;
    world.apply_impulse_at_point(box_a, Vec3::new(6.0, 0.0, 0.0), pos_a_before);
    assert!(!world.bodies[box_a].is_sleeping);
    assert!(world.bodies[box_b].is_sleeping);

    for _ in 0..12 {
        world.step(1.0 / 60.0);
    }
    assert!(!world.bodies[box_a].is_sleeping);
    assert!(
        world.bodies[box_a].position.x - pos_a_before.x > 0.05,
        "woken box_a did not move as expected: before={:?}, after={:?}",
        pos_a_before,
        world.bodies[box_a].position
    );

    // Wake remaining sleeping box_b with an earthquake
    world.set_earthquake(6.0);
    world.step(1.0 / 60.0);
    assert!(
        !world.bodies[box_b].is_sleeping,
        "earthquake failed to wake box_b"
    );
}

#[test]
fn test_right_arm_wounded_ragdoll_recovers_using_left_arm() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));

    for _ in 0..20 {
        world.step(1.0 / 60.0);
    }

    // Shoot RightForearm with Pistol -> WoundRegion::RightArm (~78% HP)
    let right_forearm_idx = world.ragdolls[id].body_indices[BoneId::RightForearm as usize];
    let forearm_pos = world.bodies[right_forearm_idx].position;
    let shot = world.fire_weapon(
        forearm_pos + Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        WeaponKind::Pistol,
    );
    assert_eq!(shot.hit_ragdoll_id, Some(id));
    assert_eq!(shot.hit_region, WoundRegion::RightArm);

    // Knock him down onto the floor
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            650.0,
        )
        .expect("shove should hit chest");

    let mut saw_recovering = false;
    let mut selected_strategy = None;
    // Step 6.5s (390 frames at 1/60s)
    for _ in 0..390 {
        world.step(1.0 / 60.0);
        if world.ragdolls[id].balance.state == BehaviorState::Recovering {
            saw_recovering = true;
            selected_strategy = Some(world.ragdolls[id].balance.recovery_strategy);
        }
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during right-arm-wounded recovery"
            );
        }
    }

    assert!(
        saw_recovering,
        "right-arm-wounded ragdoll never entered Recovering"
    );
    assert_eq!(
        selected_strategy,
        Some(RecoveryStrategy::OneArmSideRoll { healthy_left: true }),
        "expected OneArmSideRoll {{ healthy_left: true }}, got {selected_strategy:?}"
    );
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let pelvis_y = world.bodies[pelvis_idx].position.y;
    let threshold = 0.72 * world.ragdolls[id].standing_pelvis_y;
    assert!(
        pelvis_y > threshold,
        "right-arm-wounded ragdoll did not stand back up: pelvis_y={pelvis_y}, threshold={threshold}, state={:?}, stage={:?}",
        world.ragdolls[id].balance.state,
        world.ragdolls[id].balance.recovery_stage
    );
}

#[test]
fn test_heavy_140kg_ragdoll_recovers_via_quadruped() {
    let mut world = ArsaWorld::new();
    let heavy_profile = BodyProfile {
        height_m: 2.05,
        mass_kg: 140.0,
        muscle_ratio: 1.0,
    };
    let id = world.spawn_ragdoll_with_profile(Vec3::new(0.0, 0.02, 0.0), heavy_profile);

    for _ in 0..20 {
        world.step(1.0 / 60.0);
    }

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            1250.0,
        )
        .expect("shove should hit chest");

    let mut saw_recovering = false;
    let mut selected_strategy = None;
    // Step 7.5s (450 frames at 1/60s)
    for _ in 0..450 {
        world.step(1.0 / 60.0);
        if world.ragdolls[id].balance.state == BehaviorState::Recovering {
            saw_recovering = true;
            selected_strategy = Some(world.ragdolls[id].balance.recovery_strategy);
        }
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during heavy 140kg recovery"
            );
        }
    }

    assert!(
        saw_recovering,
        "heavy 140kg ragdoll never entered Recovering"
    );
    assert_eq!(
        selected_strategy,
        Some(RecoveryStrategy::HeavyQuadruped),
        "expected HeavyQuadruped, got {selected_strategy:?}"
    );
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let pelvis_y = world.bodies[pelvis_idx].position.y;
    let threshold = 0.72 * world.ragdolls[id].standing_pelvis_y;
    assert!(
        pelvis_y > threshold,
        "heavy 140kg ragdoll did not stand back up: pelvis_y={pelvis_y}, threshold={threshold}, state={:?}, stage={:?}",
        world.ragdolls[id].balance.state,
        world.ragdolls[id].balance.recovery_stage
    );
}

#[test]
fn test_external_target_pose_blending() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    // Step briefly to establish nominal standing motor targets
    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }

    // Invalid inputs must fail cleanly without mutating state
    let mut invalid_pose = [Quat::IDENTITY; 16];
    invalid_pose[0] = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    assert!(!world.set_ragdoll_target_pose(id, &invalid_pose, 1.0));
    assert!(!world.set_ragdoll_target_pose(99, &[Quat::IDENTITY; 16], 1.0));
    assert!(!world.set_ragdoll_target_pose(id, &[Quat::IDENTITY; 16], f32::NAN));
    assert!(!world.clear_ragdoll_target_pose(99));

    // Raise LeftShoulder forward & bend LeftElbow via external target pose at full weight
    let mut pose = [Quat::IDENTITY; 16];
    let shoulder_target = Quat::from_euler(glam::EulerRot::XYZ, -1.15, 0.0, 0.25);
    let elbow_target = Quat::from_rotation_x(-1.45);
    pose[JointId::LeftShoulder as usize] = shoulder_target;
    pose[JointId::LeftElbow as usize] = elbow_target;

    assert!(world.set_ragdoll_target_pose(id, &pose, 1.0));
    for _ in 0..24 {
        world.step(1.0 / 60.0);
    }

    let ls_j = &world.joints[world.ragdolls[id].joint_indices[JointId::LeftShoulder as usize]];
    let le_j = &world.joints[world.ragdolls[id].joint_indices[JointId::LeftElbow as usize]];
    let expected_ls =
        (ls_j.frame_parent.conjugate() * shoulder_target * ls_j.frame_child).normalize();
    let expected_le = (le_j.frame_parent.conjugate() * elbow_target * le_j.frame_child).normalize();

    assert!(
        ls_j.motor_target.dot(expected_ls).abs() > 0.999,
        "LeftShoulder motor_target did not track external pose"
    );
    assert!(
        le_j.motor_target.dot(expected_le).abs() > 0.999,
        "LeftElbow motor_target did not track external pose"
    );

    let elbow_angle = le_j.measure_hinge_angle(&world.bodies);
    assert!(
        elbow_angle < -0.9,
        "LeftElbow hinge did not flex toward external pose target: angle={elbow_angle}"
    );

    // Verify partial blend (0.5) interpolates between active balance target and external target
    assert!(world.set_ragdoll_target_pose(id, &pose, 0.5));
    world.step(1.0 / 60.0);
    let le_j_half = &world.joints[world.ragdolls[id].joint_indices[JointId::LeftElbow as usize]];
    let half_angle = 2.0 * le_j_half.motor_target.w.clamp(-1.0, 1.0).acos();
    assert!(
        half_angle > 0.4 && half_angle < 1.3,
        "partial blend_weight=0.5 did not interpolate elbow motor_target: half_angle={half_angle}"
    );

    // Clear external pose and verify LeftElbow returns toward relaxed standing (-0.25 rad)
    assert!(world.clear_ragdoll_target_pose(id));
    for _ in 0..45 {
        world.step(1.0 / 60.0);
    }
    let released_angle = world.joints
        [world.ragdolls[id].joint_indices[JointId::LeftElbow as usize]]
        .measure_hinge_angle(&world.bodies);
    assert!(
        released_angle > -0.55,
        "LeftElbow did not release back toward standing pose after clear: angle={released_angle}"
    );
}

#[test]
fn test_ledge_grab_catches_platform_edge() {
    let mut world = ArsaWorld::new();
    world.add_static_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(1.1, 0.1, 1.1),
        0.75,
        0.05,
    );
    // Spawn right off the platform brink (x=1.10) so the ragdoll drops past the ledge within arm's reach
    let id = world.spawn_ragdoll(Vec3::new(1.26, 2.12, 0.0));

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    world
        .apply_shove(
            chest_pos + Vec3::new(-2.0, 0.0, 0.10),
            Vec3::new(1.0, 0.0, 0.0),
            10.0,
        )
        .expect("shove should hit chest");

    let mut caught_ledge = false;
    for _ in 0..90 {
        world.step(1.0 / 60.0);
        if world.ragdolls[id].reflex.left_grab.is_some()
            || world.ragdolls[id].reflex.right_grab.is_some()
        {
            caught_ledge = true;
        }
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during ledge grab test"
            );
        }
    }

    assert!(
        caught_ledge,
        "ragdoll did not catch platform ledge with either hand during fall (state={:?})",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_terrain_aware_step_lifts_knee_for_stair() {
    let mut world = ArsaWorld::new();
    // Low stair right in front (+Z) of the standing ragdoll with top_y = 0.22
    world.add_static_box(
        Vec3::new(0.0, 0.11, 0.55),
        Vec3::new(1.2, 0.11, 0.35),
        0.75,
        0.05,
    );
    let id = world.spawn_ragdoll(Vec3::ZERO);

    assert!(
        world.sample_support_height(0.0, 0.55, 1.5) >= 0.20,
        "sample_support_height did not detect stair"
    );

    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_pos = world.bodies[chest_idx].position;
    world
        .apply_shove(
            chest_pos + Vec3::new(0.0, 0.0, -2.0),
            Vec3::new(0.0, 0.0, 1.0),
            260.0,
        )
        .expect("shove should hit chest");

    let mut max_step_ground_y = 0.0_f32;
    for _ in 0..24 {
        world.step(1.0 / 60.0);
        max_step_ground_y = max_step_ground_y.max(world.ragdolls[id].balance.step_target_ground_y);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during stair step test"
            );
        }
    }

    assert!(
        max_step_ground_y > 0.15,
        "expected step_target_ground_y > 0.15 on stair, got {max_step_ground_y}"
    );
}

#[test]
fn test_threat_anticipation_raises_guard_before_impact() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));

    // Settle for 0.2s
    for _ in 0..12 {
        world.step(1.0 / 60.0);
    }

    let left_hand_idx = world.ragdolls[id].body_indices[BoneId::LeftHand as usize];
    let right_hand_idx = world.ragdolls[id].body_indices[BoneId::RightHand as usize];
    let rest_left_hand_y = world.bodies[left_hand_idx].position.y;
    let rest_right_hand_y = world.bodies[right_hand_idx].position.y;

    // Fast heavy sphere approaching chest from +Z, arriving in ~0.49s
    world.spawn_sphere(
        Vec3::new(0.0, 1.35, 3.2),
        0.24,
        30.0,
        Vec3::new(0.0, 0.0, -6.5),
    );

    // Step 0.25s (15 frames) -- BEFORE the sphere hits
    for _ in 0..15 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during threat anticipation test"
            );
        }
    }

    assert!(
        world.ragdolls[id].reflex.incoming_threat.is_some(),
        "expected incoming_threat to be detected before impact"
    );

    let left_hand_y = world.bodies[left_hand_idx].position.y;
    let right_hand_y = world.bodies[right_hand_idx].position.y;
    assert!(
        left_hand_y > rest_left_hand_y + 0.18,
        "LeftHand did not rise into defensive guard before impact: rest={rest_left_hand_y}, now={left_hand_y}"
    );
    assert!(
        right_hand_y > rest_right_hand_y + 0.18,
        "RightHand did not rise into defensive guard before impact: rest={rest_right_hand_y}, now={right_hand_y}"
    );
}

#[test]
fn test_teeter_on_edge_detects_dropoff_and_leans_safe() {
    let mut world = ArsaWorld::new();
    // Elevated platform with top_y = 2.1 and +X edge at x = 1.1
    world.add_static_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(1.1, 0.1, 1.1),
        0.75,
        0.05,
    );
    let id = world.spawn_ragdoll(Vec3::new(0.95, 2.12, 0.0));

    for _ in 0..18 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during teeter on edge test"
            );
        }
    }

    assert!(
        world.ragdolls[id].balance.teetering,
        "expected ragdoll near +X platform brink to detect teetering (state={:?})",
        world.ragdolls[id].balance.state
    );
    assert!(
        world.ragdolls[id].balance.teeter_escape_dir.x < -0.5,
        "expected teeter_escape_dir.x < -0.5 pointing away from +X cliff, got {:?}",
        world.ragdolls[id].balance.teeter_escape_dir
    );
}

#[test]
fn test_shot_from_behind_sets_back_flag_and_arches() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::ZERO);

    for _ in 0..12 {
        world.step(1.0 / 60.0);
    }

    let res = world.fire_weapon(
        Vec3::new(0.0, 1.32, -2.5),
        Vec3::new(0.0, 0.0, 1.0),
        WeaponKind::Pistol,
    );
    assert_eq!(res.hit_ragdoll_id, Some(id));
    assert_eq!(res.hit_region, WoundRegion::Torso);

    let wound = world.ragdolls[id]
        .reflex
        .active_wound
        .expect("active_wound should be set after shot from behind");
    assert!(
        wound.hit_from_back,
        "expected hit_from_back == true when shot into back of Chest"
    );

    for _ in 0..24 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during back shot test"
            );
        }
    }
}

#[test]
fn test_high_fall_activates_then_transitions_to_brace() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 5.5, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];

    let mut saw_high_fall = false;
    let mut saw_brace_near_ground = false;

    // Step up to 2.0 seconds (120 frames at 1/60s)
    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during high fall test"
            );
        }

        let r = &world.ragdolls[id];
        let pelvis_y = world.bodies[pelvis_idx].position.y;
        if r.balance.state == BehaviorState::Falling && r.balance.com_vel.y < -2.2 && pelvis_y > 1.8
        {
            assert!(
                r.reflex.high_falling,
                "expected high_falling == true at pelvis_y={pelvis_y}, com_vel.y={}",
                r.balance.com_vel.y
            );
            saw_high_fall = true;
        }
        if saw_high_fall && pelvis_y < 1.3 && r.reflex.bracing {
            saw_brace_near_ground = true;
        }
    }

    assert!(
        saw_high_fall,
        "high fall phase never activated during drop from y=5.5"
    );
    assert!(
        saw_brace_near_ground,
        "high fall did not transition to pre-impact bracing near ground (y < 1.3)"
    );
}

#[test]
fn test_fast_spin_tumble_triggers_fetal_protect() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 1.5, 0.0));
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let pivot = world.bodies[pelvis_idx].position;
    let tumble_tilt = Quat::from_rotation_x(1.25);

    // Pitch ragdoll into a mid-tumble orientation and apply 5.5 rad/s rigid-body tumble spin
    let spin = Vec3::new(5.5, 0.0, 0.0);
    for &b_idx in &world.ragdolls[id].body_indices {
        let b = &mut world.bodies[b_idx];
        b.position = pivot + tumble_tilt * (b.position - pivot);
        b.orientation = (tumble_tilt * b.orientation).normalize();
        b.angular_velocity = spin;
        b.linear_velocity = spin.cross(b.position - pivot);
        b.update_world_inertia();
    }

    // Step 0.15s (9 frames at 1/60s)
    for _ in 0..9 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during fast spin tumble test"
            );
        }
    }

    let r = &world.ragdolls[id];
    assert!(
        r.reflex.tumbling,
        "expected tumbling == true after fast spin, got state={:?}, chest_w={:?}, pelvis_w={:?}, high_falling={}",
        r.balance.state,
        world.bodies[chest_idx].angular_velocity,
        world.bodies[pelvis_idx].angular_velocity,
        r.reflex.high_falling
    );
}

#[test]
fn test_standing_5s_zero_drift_and_zero_yaw_spin() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];

    // Step 5.0 seconds (300 frames at 1/60s)
    for _ in 0..300 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during 5s standing test"
            );
        }
    }

    let pelvis = &world.bodies[pelvis_idx];
    let chest = &world.bodies[chest_idx];
    let horiz_drift = Vec3::new(pelvis.position.x, 0.0, pelvis.position.z).length();
    let pelvis_fwd = pelvis.orientation * Vec3::Z;
    let fwd_xz = Vec3::new(pelvis_fwd.x, 0.0, pelvis_fwd.z).normalize_or_zero();
    let yaw_angle = fwd_xz.x.atan2(fwd_xz.z).abs();
    let pelvis_up_dot = (pelvis.orientation * Vec3::Y).y;
    let chest_up_dot = (chest.orientation * Vec3::Y).y;

    assert!(
        horiz_drift < 0.04,
        "standing ragdoll drifted horizontally over 5s: horiz_drift={horiz_drift}, pos={:?}",
        pelvis.position
    );
    assert!(
        yaw_angle < 0.06,
        "standing ragdoll spun in yaw over 5s: yaw_angle={yaw_angle} rad, fwd={fwd_xz:?}"
    );
    assert!(
        pelvis_up_dot > 0.96 && chest_up_dot > 0.96,
        "standing ragdoll leaned off-vertical over 5s: pelvis_up_dot={pelvis_up_dot}, chest_up_dot={chest_up_dot}"
    );
    assert_eq!(
        world.ragdolls[id].balance.state,
        BehaviorState::Balanced,
        "expected Balanced after 5s standing, got {:?}",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_intoxication_causes_visible_sway_and_stagger() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];

    world.set_all_ragdolls_intoxication(0.85);
    assert!((world.ragdolls[id].balance.intoxication - 0.85).abs() < 1e-6);

    let mut max_pelvis_horiz_disp = 0.0_f32;
    let mut max_chest_tilt = 0.0_f32;
    let mut max_com_speed = 0.0_f32;
    let mut saw_stumble = false;

    // Step 2.5 seconds (150 frames at 1/60s)
    for _ in 0..150 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during intoxication test"
            );
        }
        let pelvis_pos = world.bodies[pelvis_idx].position;
        max_pelvis_horiz_disp =
            max_pelvis_horiz_disp.max(Vec3::new(pelvis_pos.x, 0.0, pelvis_pos.z).length());
        let chest_up_y = (world.bodies[chest_idx].orientation * Vec3::Y)
            .y
            .clamp(-1.0, 1.0);
        max_chest_tilt = max_chest_tilt.max(chest_up_y.acos());
        let com_v = world.ragdolls[id].balance.com_vel;
        max_com_speed = max_com_speed.max(Vec3::new(com_v.x, 0.0, com_v.z).length());
        if matches!(
            world.ragdolls[id].balance.state,
            BehaviorState::Stumbling | BehaviorState::Falling
        ) {
            saw_stumble = true;
        }
    }

    assert!(
        max_chest_tilt > 0.12 && (max_pelvis_horiz_disp > 0.06 || saw_stumble || max_com_speed > 0.18),
        "intoxication did not produce visible sway/stagger: max_chest_tilt={max_chest_tilt}, max_pelvis_horiz_disp={max_pelvis_horiz_disp}, max_com_speed={max_com_speed}, saw_stumble={saw_stumble}"
    );
}
