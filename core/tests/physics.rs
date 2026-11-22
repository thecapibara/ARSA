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

#[test]
fn test_oriented_static_ramp_collision_and_raycast() {
    let mut world = ArsaWorld::new();
    let tilt = Quat::from_rotation_x(20.0_f32.to_radians());
    // Tilted 20-degree ramp centered at y = 0.85
    world.add_static_oriented_box(
        Vec3::new(0.0, 0.85, 0.0),
        Vec3::new(1.5, 0.1, 2.0),
        tilt,
        0.90,
        0.05,
    );

    // Support height directly above center of ramp should match 0.85 + 0.1 / cos(20 deg)
    let expected_top = 0.85 + 0.1 / 20.0_f32.to_radians().cos();
    let h_center = world.sample_support_height(0.0, 0.0, 2.5);
    assert!(
        (h_center - expected_top).abs() < 1e-3,
        "expected oriented ramp support height ~{expected_top}, got {h_center}"
    );
    let h_upslope = world.sample_support_height(0.0, -1.0, 2.5);
    let h_downslope = world.sample_support_height(0.0, 1.0, 2.5);
    assert!(
        (h_upslope - h_downslope).abs() > 0.5,
        "expected sloped ramp height difference along Z, got upslope={h_upslope}, downslope={h_downslope}"
    );

    // Dynamic box dropped onto the tilted ramp
    let box_idx = world.spawn_box(
        Vec3::new(0.0, 1.35, 0.0),
        Vec3::splat(0.15),
        5.0,
        Vec3::ZERO,
    );
    // Dynamic sphere placed underneath the ramp (offset in X from the box) for raycast occlusion check
    let sphere_idx = world.spawn_sphere(Vec3::new(0.8, 0.25, 0.0), 0.15, 2.0, Vec3::ZERO);

    for _ in 0..60 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf on oriented ramp"
            );
        }
    }

    let box_pos = world.bodies[box_idx].position;
    let ramp_h_at_box = world.sample_support_height(box_pos.x, box_pos.z, 2.5);
    assert!(
        box_pos.y > ramp_h_at_box && box_pos.y > 0.5,
        "dynamic box fell through tilted ramp: box_pos={box_pos:?}, ramp_h={ramp_h_at_box}"
    );

    // Raycast from above the ramp toward the sphere underneath must be occluded by the tilted ramp
    let occluded = world.raycast(Vec3::new(0.8, 3.0, 0.0), Vec3::NEG_Y, 10.0);
    assert!(
        occluded.is_none(),
        "raycast aimed through tilted ramp should be occluded, got {occluded:?}"
    );

    // Raycast from below the ramp (y = 0.6) hits the sphere
    let hit = world
        .raycast(Vec3::new(0.8, 0.6, 0.0), Vec3::NEG_Y, 2.0)
        .expect("raycast below ramp should hit sphere");
    assert_eq!(hit.body_idx, sphere_idx);
}

#[test]
fn test_heightfield_slope_collision_and_standing_balance() {
    use arsa_core::Heightfield;

    let cols = 9usize;
    let rows = 9usize;
    let cell_size = 1.0_f32;
    let origin_x = -4.0_f32;
    let origin_z = -4.0_f32;
    let mut heights = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        let z = origin_z + (r as f32) * cell_size;
        let h = 1.0 + 0.12 * z;
        for _c in 0..cols {
            heights.push(h);
        }
    }

    let hf = Heightfield {
        origin_x,
        origin_z,
        cell_size,
        cols,
        rows,
        heights,
        friction: 0.85,
        restitution: 0.05,
    };

    let (h0, n0) = hf.sample(0.0, 0.0).expect("center sample should succeed");
    assert!((h0 - 1.0).abs() < 1e-5);
    assert!(n0.y > 0.9 && n0.z < -0.05);
    assert!(hf.sample(5.0, 0.0).is_none());

    let mut world = ArsaWorld::new();
    world.set_heightfield(Some(hf));
    let id = world.spawn_ragdoll(Vec3::new(0.0, 1.02, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];

    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf on heightfield slope"
            );
        }
    }

    let pelvis_pos = world.bodies[pelvis_idx].position;
    let terrain_h = world.sample_support_height(pelvis_pos.x, pelvis_pos.z, pelvis_pos.y);
    assert!(
        pelvis_pos.y > terrain_h + 0.55,
        "ragdoll fell through sloped heightfield: pelvis_pos={pelvis_pos:?}, terrain_h={terrain_h}"
    );
    assert!(
        matches!(
            world.ragdolls[id].balance.state,
            BehaviorState::Balanced | BehaviorState::Stumbling
        ),
        "expected ragdoll to balance on gentle heightfield slope, got {:?}",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_host_world_query_callbacks_ground_and_occlusion() {
    use arsa_core::WorldQueryHooks;

    extern "C" fn host_ground_height(
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

    extern "C" fn host_raycast(
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

    let mut world = ArsaWorld::new();
    world.set_world_query_callbacks(Some(WorldQueryHooks {
        user_data: std::ptr::null_mut(),
        ground_height: Some(host_ground_height),
        raycast: Some(host_raycast),
    }));

    let box_idx = world.spawn_box(Vec3::new(0.0, 2.0, 3.0), Vec3::splat(0.15), 5.0, Vec3::ZERO);

    for _ in 0..90 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf in host query callback test"
            );
        }
    }

    let settled_y = world.bodies[box_idx].position.y;
    assert!(
        (settled_y - 1.65).abs() < 0.03,
        "expected box to settle on host y=1.5 ground (center ~1.65), got {settled_y}"
    );

    // Raycast from (0, 1.65, 0) toward +Z is occluded by host wall at dist = 1.0
    let occluded = world.raycast(Vec3::new(0.0, 1.65, 0.0), Vec3::Z, 10.0);
    assert!(
        occluded.is_none(),
        "expected raycast toward z=3.0 box to be occluded by host wall at dist=1.0, got {occluded:?}"
    );

    // Removing the raycast hook allows the ray to hit the dynamic box at z = 3.0
    world.set_world_query_callbacks(Some(WorldQueryHooks {
        user_data: std::ptr::null_mut(),
        ground_height: Some(host_ground_height),
        raycast: None,
    }));
    let hit = world
        .raycast(Vec3::new(0.0, 1.65, 0.0), Vec3::Z, 10.0)
        .expect("raycast should hit dynamic box when host occlusion is disabled");
    assert_eq!(hit.body_idx, box_idx);
}

#[test]
fn test_retarget_coordinate_and_preset_roundtrip() {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    let sample_pos = Vec3::new(0.25, 1.42, -0.37);
    let sample_quat = Quat::from_euler(glam::EulerRot::YXZ, 0.65, -0.42, 0.88).normalize();
    let scales = [
        (CoordinateSpace::ArsaRightHandedYUp, 1.0_f32),
        (CoordinateSpace::UnityLeftHandedYUp, 1.0_f32),
        (CoordinateSpace::UnrealLeftHandedZUp, 100.0_f32),
    ];

    for (space, scale) in scales {
        let p_target = SkeletonRetargeter::convert_pos_to_target(space, scale, sample_pos);
        let p_back = SkeletonRetargeter::convert_pos_from_target(space, scale, p_target);
        assert!(
            (p_back - sample_pos).length() < 1e-5,
            "position roundtrip mismatch for {space:?}: expected {sample_pos:?}, got {p_back:?}"
        );

        let q_target = SkeletonRetargeter::convert_quat_to_target(space, sample_quat);
        let q_back = SkeletonRetargeter::convert_quat_from_target(space, q_target);
        assert!(
            q_back.dot(sample_quat).abs() > 1.0 - 1e-5,
            "quat roundtrip mismatch for {space:?}: expected {sample_quat:?}, got {q_back:?}"
        );
    }

    // Verify neutral standing pose -> Quat::IDENTITY joint targets across all 4 presets
    let mut world = ArsaWorld::new();
    let rid = world.spawn_ragdoll(Vec3::ZERO);
    let mut neutral_pos = [Vec3::ZERO; 17];
    let mut neutral_rot = [Quat::IDENTITY; 17];
    for (i, &b_idx) in world.ragdolls[rid].body_indices.iter().enumerate() {
        neutral_pos[i] = world.bodies[b_idx].position;
        neutral_rot[i] = world.bodies[b_idx].orientation;
    }

    let presets = [
        RetargetPreset::ArsaNative,
        RetargetPreset::MixamoTPose,
        RetargetPreset::UnityHumanoidTPose,
        RetargetPreset::UnrealMannyAPose,
    ];

    for preset in presets {
        let retargeter = SkeletonRetargeter::from_preset(preset);
        let (target_pos, target_rot) =
            retargeter.retarget_bones_to_target(&neutral_pos, &neutral_rot);

        // Verify expected position conversion on Pelvis
        let expected_pelvis_pos = SkeletonRetargeter::convert_pos_to_target(
            retargeter.space,
            retargeter.unit_scale,
            neutral_pos[BoneId::Pelvis as usize],
        );
        assert!(
            (target_pos[BoneId::Pelvis as usize] - expected_pelvis_pos).length() < 1e-4,
            "pelvis position mismatch for {preset:?}"
        );

        let joint_targets = retargeter.retarget_world_quats_to_arsa_joint_targets(&target_rot);
        for (j_idx, &jt) in joint_targets.iter().enumerate() {
            assert!(
                jt.dot(Quat::IDENTITY).abs() > 1.0 - 1e-4,
                "neutral pose joint {j_idx} not identity for {preset:?}: {jt:?}"
            );
        }
    }

    // Verify non-trivial posed skeleton (bent spine, shoulders, elbows, hips, knees) round-trips
    let mut posed_rot = [Quat::IDENTITY; 17];
    posed_rot[BoneId::Pelvis as usize] = Quat::from_rotation_y(0.35);
    posed_rot[BoneId::LowerSpine as usize] =
        posed_rot[BoneId::Pelvis as usize] * Quat::from_rotation_x(0.18);
    posed_rot[BoneId::Chest as usize] = posed_rot[BoneId::LowerSpine as usize]
        * Quat::from_euler(glam::EulerRot::XYZ, 0.12, -0.2, 0.08);
    posed_rot[BoneId::Neck as usize] =
        posed_rot[BoneId::Chest as usize] * Quat::from_rotation_x(-0.1);
    posed_rot[BoneId::Head as usize] =
        posed_rot[BoneId::Neck as usize] * Quat::from_rotation_y(0.25);

    // Raise & flex arms
    posed_rot[BoneId::LeftUpperArm as usize] =
        posed_rot[BoneId::Chest as usize] * Quat::from_rotation_z(FRAC_PI_2);
    posed_rot[BoneId::LeftForearm as usize] =
        posed_rot[BoneId::LeftUpperArm as usize] * Quat::from_rotation_x(-1.1);
    posed_rot[BoneId::LeftHand as usize] =
        posed_rot[BoneId::LeftForearm as usize] * Quat::from_rotation_z(0.2);

    posed_rot[BoneId::RightUpperArm as usize] =
        posed_rot[BoneId::Chest as usize] * Quat::from_rotation_z(-FRAC_PI_4);
    posed_rot[BoneId::RightForearm as usize] =
        posed_rot[BoneId::RightUpperArm as usize] * Quat::from_rotation_x(-0.85);
    posed_rot[BoneId::RightHand as usize] =
        posed_rot[BoneId::RightForearm as usize] * Quat::from_rotation_y(-0.15);

    // Bend legs
    posed_rot[BoneId::LeftThigh as usize] =
        posed_rot[BoneId::Pelvis as usize] * Quat::from_rotation_x(-0.55);
    posed_rot[BoneId::LeftShin as usize] =
        posed_rot[BoneId::LeftThigh as usize] * Quat::from_rotation_x(1.05);
    posed_rot[BoneId::LeftFoot as usize] =
        posed_rot[BoneId::LeftShin as usize] * Quat::from_rotation_x(-0.35);

    posed_rot[BoneId::RightThigh as usize] =
        posed_rot[BoneId::Pelvis as usize] * Quat::from_rotation_x(0.25);
    posed_rot[BoneId::RightShin as usize] =
        posed_rot[BoneId::RightThigh as usize] * Quat::from_rotation_x(0.45);
    posed_rot[BoneId::RightFoot as usize] =
        posed_rot[BoneId::RightShin as usize] * Quat::from_rotation_x(-0.15);

    let expected_joint_targets: [Quat; 16] = std::array::from_fn(|j| {
        let (p, c) = JOINT_BONE_PAIRS[j];
        (posed_rot[p as usize].conjugate() * posed_rot[c as usize]).normalize()
    });

    for preset in presets {
        let retargeter = SkeletonRetargeter::from_preset(preset);
        let (_, target_rot) = retargeter.retarget_bones_to_target(&neutral_pos, &posed_rot);
        let roundtrip_joints = retargeter.retarget_world_quats_to_arsa_joint_targets(&target_rot);
        for (j_idx, (&got, &expected)) in roundtrip_joints
            .iter()
            .zip(expected_joint_targets.iter())
            .enumerate()
        {
            let dot = got.dot(expected).abs();
            assert!(
                dot > 1.0 - 1e-4,
                "posed joint {j_idx} roundtrip mismatch for {preset:?}: dot={dot}, got={got:?}, expected={expected:?}"
            );
        }
    }

    // Verify custom bind pose constructor matches MixamoTPose when given equivalent bind poses
    let mixamo_bind_quats: [Quat; 17] = std::array::from_fn(|i| {
        SkeletonRetargeter::from_preset(RetargetPreset::MixamoTPose).bones[i].bind_offset
    });
    let custom = SkeletonRetargeter::from_custom_bind_poses(
        CoordinateSpace::ArsaRightHandedYUp,
        1.0,
        &mixamo_bind_quats,
    );
    let mixamo_preset = SkeletonRetargeter::from_preset(RetargetPreset::MixamoTPose);
    assert_eq!(custom.space, mixamo_preset.space);
    assert_eq!(custom.unit_scale, mixamo_preset.unit_scale);
    for i in 0..17 {
        assert!(
            custom.bones[i]
                .bind_offset
                .dot(mixamo_preset.bones[i].bind_offset)
                .abs()
                > 1.0 - 1e-6,
            "custom bind pose mismatch at bone {i}"
        );
    }
}

#[test]
fn test_retarget_drives_ragdoll_via_target_pose() {
    // Mixamo T-pose animation: target_world_quats = [Quat::IDENTITY; 17] (arms horizontal in Mixamo)
    {
        let mut world = ArsaWorld::new();
        let rid = world.spawn_ragdoll(Vec3::ZERO);
        let mixamo = SkeletonRetargeter::from_preset(RetargetPreset::MixamoTPose);
        let mixamo_tpose_quats = [Quat::IDENTITY; 17];
        let joint_targets = mixamo.retarget_world_quats_to_arsa_joint_targets(&mixamo_tpose_quats);

        assert!(world.set_ragdoll_target_pose(rid, &joint_targets, 1.0));
        for _ in 0..45 {
            world.step(1.0 / 60.0);
            for (i, b) in world.bodies.iter().enumerate() {
                assert!(
                    b.position.is_finite()
                        && b.orientation.is_finite()
                        && b.linear_velocity.is_finite()
                        && b.angular_velocity.is_finite(),
                    "Mixamo T-pose retargeting caused NaN/Inf on body {i}"
                );
            }
        }

        // Arms should abduct upward toward horizontal T-pose (+X for Left, -X for Right)
        let chest_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::Chest as usize]].position;
        let left_hand_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::LeftHand as usize]].position;
        let right_hand_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::RightHand as usize]].position;
        assert!(
            left_hand_pos.x > chest_pos.x + 0.40,
            "LeftHand did not raise toward Mixamo T-pose (+X): left_hand={left_hand_pos:?}, chest={chest_pos:?}"
        );
        assert!(
            right_hand_pos.x < chest_pos.x - 0.40,
            "RightHand did not raise toward Mixamo T-pose (-X): right_hand={right_hand_pos:?}, chest={chest_pos:?}"
        );
    }

    // Unreal Manny A-pose animation: target_world_quats = [Quat::IDENTITY; 17] (arms angled 45 deg in UE5)
    {
        let mut world = ArsaWorld::new();
        let rid = world.spawn_ragdoll(Vec3::ZERO);
        let manny = SkeletonRetargeter::from_preset(RetargetPreset::UnrealMannyAPose);
        let manny_apose_quats = [Quat::IDENTITY; 17];
        let joint_targets = manny.retarget_world_quats_to_arsa_joint_targets(&manny_apose_quats);

        assert!(world.set_ragdoll_target_pose(rid, &joint_targets, 1.0));
        for _ in 0..45 {
            world.step(1.0 / 60.0);
            for (i, b) in world.bodies.iter().enumerate() {
                assert!(
                    b.position.is_finite()
                        && b.orientation.is_finite()
                        && b.linear_velocity.is_finite()
                        && b.angular_velocity.is_finite(),
                    "Unreal Manny A-pose retargeting caused NaN/Inf on body {i}"
                );
            }
        }

        let chest_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::Chest as usize]].position;
        let left_hand_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::LeftHand as usize]].position;
        let right_hand_pos =
            world.bodies[world.ragdolls[rid].body_indices[BoneId::RightHand as usize]].position;
        assert!(
            left_hand_pos.x > chest_pos.x + 0.30,
            "LeftHand did not abduct toward Manny A-pose: left_hand={left_hand_pos:?}, chest={chest_pos:?}"
        );
        assert!(
            right_hand_pos.x < chest_pos.x - 0.30,
            "RightHand did not abduct toward Manny A-pose: right_hand={right_hand_pos:?}, chest={chest_pos:?}"
        );
    }
}

#[test]
fn test_ragdoll_lod_transitions_and_wakeup_on_impulse() {
    use arsa_core::RagdollLod;

    let mut world = ArsaWorld::new();
    let r_full = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let r_reduced = world.spawn_ragdoll(Vec3::new(20.0, 0.02, 0.0));
    let r_sleep = world.spawn_ragdoll(Vec3::new(45.0, 0.02, 0.0));

    world.set_lod_focus(Some(Vec3::ZERO), 15.0, 35.0);

    for _ in 0..90 {
        world.step(1.0 / 60.0);
    }

    assert_eq!(world.ragdoll_lod(r_full), Some(RagdollLod::Full));
    assert_eq!(world.ragdoll_lod(r_reduced), Some(RagdollLod::Reduced));
    assert_eq!(world.ragdoll_lod(r_sleep), Some(RagdollLod::Sleep));

    for (label, r_id) in [("full", r_full), ("reduced", r_reduced), ("sleep", r_sleep)] {
        let pelvis_idx = world.ragdolls[r_id].body_indices[BoneId::Pelvis as usize];
        let pelvis_y = world.bodies[pelvis_idx].position.y;
        assert!(
            pelvis_y > 0.75,
            "ragdoll '{label}' collapsed unexpectedly (pelvis_y={pelvis_y:.3})"
        );
        for &b_idx in &world.ragdolls[r_id].body_indices {
            let b = &world.bodies[b_idx];
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "ragdoll '{label}' produced NaN/Inf on bone {b_idx}"
            );
        }
    }

    let sleep_pelvis_idx = world.ragdolls[r_sleep].body_indices[BoneId::Pelvis as usize];
    let pelvis_before = world.bodies[sleep_pelvis_idx].position;

    let hit = world.apply_shove(Vec3::new(45.0, 1.30, 2.0), Vec3::new(0.0, 0.0, -1.0), 90.0);
    assert!(hit.is_some(), "expected shove ray to hit distant ragdoll");
    assert_eq!(world.ragdoll_lod(r_sleep), Some(RagdollLod::Full));

    for _ in 0..2 {
        world.step(1.0 / 60.0);
    }

    assert_eq!(world.ragdoll_lod(r_sleep), Some(RagdollLod::Full));
    let pelvis_after = world.bodies[sleep_pelvis_idx].position;
    let disp = (pelvis_after - pelvis_before).length();
    assert!(
        disp > 0.005,
        "expected woken ragdoll pelvis to move after shove, disp={disp:.5}"
    );
}

#[test]
fn test_angular_vec_conversion_matches_small_quat_rotation() {
    let spaces = [
        CoordinateSpace::ArsaRightHandedYUp,
        CoordinateSpace::UnityLeftHandedYUp,
        CoordinateSpace::UnrealLeftHandedZUp,
    ];
    let samples = [
        Vec3::new(1.2, -0.7, 2.3),
        Vec3::new(-3.1, 0.4, -1.8),
        Vec3::new(0.0, 2.5, -0.9),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
    ];
    let eps = 1.0e-3_f32;

    for space in spaces {
        for v in samples {
            let q = Quat::from_scaled_axis(v * eps);
            let q_target = SkeletonRetargeter::convert_quat_to_target(space, q);
            let axis_from_quat = q_target.to_scaled_axis();
            let vec_from_ang = convert_angular_vec_to_target(space, v) * eps;
            let err = (axis_from_quat - vec_from_ang).length();
            assert!(
                err < 1e-4,
                "angular vector mismatch for {space:?}, v={v:?}: quat_axis={axis_from_quat:?}, ang_vec={vec_from_ang:?}, err={err}"
            );

            let v_target = SkeletonRetargeter::convert_angular_vec_to_target(space, v);
            let v_back = SkeletonRetargeter::convert_angular_vec_from_target(space, v_target);
            assert!(
                (v_back - v).length() < 1e-5,
                "angular vector roundtrip mismatch for {space:?}: expected {v:?}, got {v_back:?}"
            );
        }
    }
}

#[test]
fn test_host_physics_brain_only_and_joint_torques() {
    let mut world = ArsaWorld::new();
    let rid = world.spawn_ragdoll(Vec3::ZERO);

    assert!(world
        .compute_host_joint_torques(99, CoordinateSpace::ArsaRightHandedYUp)
        .is_none());

    // Sync a bent/tilted upper-body and hip pose with angular velocity as if driven by PhysX/Chaos.
    let spine_idx = world.ragdolls[rid].body_indices[BoneId::LowerSpine as usize];
    let chest_idx = world.ragdolls[rid].body_indices[BoneId::Chest as usize];
    let l_thigh_idx = world.ragdolls[rid].body_indices[BoneId::LeftThigh as usize];

    let spine_pos = world.bodies[spine_idx].position;
    let chest_pos = world.bodies[chest_idx].position;
    let l_thigh_pos = world.bodies[l_thigh_idx].position;

    assert!(world.sync_bone_pose(
        spine_idx,
        spine_pos,
        Quat::from_euler(glam::EulerRot::XYZ, 0.28, -0.10, 0.12),
        Vec3::ZERO,
        Vec3::new(0.4, -0.2, 0.1),
    ));
    assert!(world.sync_bone_pose(
        chest_idx,
        chest_pos,
        Quat::from_euler(glam::EulerRot::XYZ, 0.42, -0.15, 0.18),
        Vec3::ZERO,
        Vec3::new(0.5, 0.1, -0.2),
    ));
    assert!(world.sync_bone_pose(
        l_thigh_idx,
        l_thigh_pos,
        Quat::from_rotation_x(-0.35),
        Vec3::ZERO,
        Vec3::new(-0.3, 0.0, 0.0),
    ));

    let before_positions: Vec<Vec3> = world.bodies.iter().map(|b| b.position).collect();
    let before_orientations: Vec<Quat> = world.bodies.iter().map(|b| b.orientation).collect();

    // Run multiple brain-only ticks; host-synced bone transforms must remain completely untouched.
    for _ in 0..5 {
        world.step_brain_only(1.0 / 60.0);
    }

    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            (b.position - before_positions[i]).length() < 1e-6,
            "step_brain_only moved body {i} position from {:?} to {:?}",
            before_positions[i],
            b.position
        );
        assert!(
            b.orientation.dot(before_orientations[i]).abs() > 1.0 - 1e-6,
            "step_brain_only rotated body {i} orientation"
        );
    }

    // Both feet remain resting on y = 0.0 floor so foot grounding should be detected.
    assert!(world.ragdolls[rid].balance.left_foot_grounded);
    assert!(world.ragdolls[rid].balance.right_foot_grounded);

    let torques_arsa = world
        .compute_host_joint_torques(rid, CoordinateSpace::ArsaRightHandedYUp)
        .expect("valid ragdoll id");
    let torques_unity = world
        .compute_host_joint_torques(rid, CoordinateSpace::UnityLeftHandedYUp)
        .expect("valid ragdoll id");
    let torques_unreal = world
        .compute_host_joint_torques(rid, CoordinateSpace::UnrealLeftHandedZUp)
        .expect("valid ragdoll id");

    for i in 0..16 {
        let max_t = world.joints[world.ragdolls[rid].joint_indices[i]].max_torque;
        assert!(torques_arsa[i].is_finite());
        assert!(torques_unity[i].is_finite());
        assert!(torques_unreal[i].is_finite());
        assert!(
            torques_arsa[i].length() <= max_t + 1e-3,
            "joint {i} torque exceeded max_torque: {} > {}",
            torques_arsa[i].length(),
            max_t
        );
        assert!(
            (torques_unity[i]
                - convert_angular_vec_to_target(
                    CoordinateSpace::UnityLeftHandedYUp,
                    torques_arsa[i]
                ))
            .length()
                < 1e-5
        );
        assert!(
            (torques_unreal[i]
                - convert_angular_vec_to_target(
                    CoordinateSpace::UnrealLeftHandedZUp,
                    torques_arsa[i]
                ))
            .length()
                < 1e-5
        );
    }

    // Tilted spine, chest, and left hip joints must produce strong non-zero restoring torques.
    assert!(
        torques_arsa[JointId::LowerSpine as usize].length() > 1.0,
        "expected non-zero LowerSpine restoring torque, got {:?}",
        torques_arsa[JointId::LowerSpine as usize]
    );
    assert!(
        torques_arsa[JointId::ChestSpine as usize].length() > 1.0,
        "expected non-zero ChestSpine restoring torque, got {:?}",
        torques_arsa[JointId::ChestSpine as usize]
    );
    assert!(
        torques_arsa[JointId::LeftHip as usize].length() > 1.0,
        "expected non-zero LeftHip restoring torque, got {:?}",
        torques_arsa[JointId::LeftHip as usize]
    );

    // Disabling active mode must zero all computed host joint torques after a brain tick.
    world.set_all_ragdolls_active(false);
    world.step_brain_only(1.0 / 60.0);
    let passive_torques = world
        .compute_host_joint_torques(rid, CoordinateSpace::ArsaRightHandedYUp)
        .expect("valid ragdoll id");
    for (i, tau) in passive_torques.iter().enumerate() {
        assert_eq!(
            *tau,
            Vec3::ZERO,
            "passive joint {i} should output zero torque"
        );
    }
}

#[test]
fn test_dynamic_box_resting_on_static_ledge_edge_no_sinking() {
    let mut world = ArsaWorld::new();
    // Large static platform with top surface at y = 2.0 and front edge at z = 1.5
    world.add_static_box(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(2.0, 1.0, 1.5),
        0.75,
        0.05,
    );
    // Spawn a dynamic crate straddling the platform's front edge (z = 1.5), tilted slightly
    // so its bottom corners are outside the platform while the platform edge intersects its bottom face.
    let box_idx = world.spawn_box(
        Vec3::new(0.0, 2.24, 1.62),
        Vec3::splat(0.22),
        10.0,
        Vec3::ZERO,
    );

    // Step 1.5 seconds: the box must tip off the edge via edge-vs-edge SAT without sinking into the platform
    let mut max_edge_penetration = 0.0_f32;
    for _ in 0..90 {
        world.step(1.0 / 60.0);
        let b = &world.bodies[box_idx];
        assert!(b.position.is_finite() && b.orientation.is_finite());
        // If the box center is near the edge (z < 1.72 && y > 1.78), verify it never sinks deep inside (z < 1.5, y < 2.0)
        if b.position.z < 1.5 && b.position.y < 2.0 {
            let depth = (1.5_f32 - b.position.z).min(2.0_f32 - b.position.y);
            max_edge_penetration = max_edge_penetration.max(depth);
        }
    }

    assert!(
        max_edge_penetration < 0.02,
        "dynamic box sank into static platform edge: max_edge_penetration={max_edge_penetration}"
    );
    assert!(
        world.bodies[box_idx].position.y < 1.5,
        "crate straddling past the edge should have tipped off cleanly, got pos={:?}",
        world.bodies[box_idx].position
    );
}

#[test]
fn test_held_box_two_handed_carry_and_drop_on_arm_shot() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let box_idx = world
        .give_ragdoll_held_box(id, Vec3::splat(0.18), 6.0, 320.0)
        .expect("give_ragdoll_held_box should succeed");

    assert!(world.is_hand_holding(id, true));
    assert!(world.is_hand_holding(id, false));

    for _ in 0..60 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during held box carry"
            );
        }
    }

    assert_eq!(
        world.ragdolls[id].balance.state,
        BehaviorState::Balanced,
        "expected ragdoll carrying 6kg box to remain Balanced, got {:?}",
        world.ragdolls[id].balance.state
    );
    assert!(
        world.is_hand_holding(id, true) && world.is_hand_holding(id, false),
        "expected both hands to remain holding after 1.0s carry"
    );
    let box_y = world.bodies[box_idx].position.y;
    assert!(
        box_y > 0.75,
        "expected held box to stay lifted (> 0.75m), got box_y={box_y}"
    );

    // Shoot the right forearm from the right (-X) side to avoid box occlusion.
    let r_forearm_idx = world.ragdolls[id].body_indices[BoneId::RightForearm as usize];
    let r_forearm_pos = world.bodies[r_forearm_idx].position;
    let res = world.fire_weapon(
        r_forearm_pos + Vec3::new(-2.0, 0.0, 0.0),
        Vec3::X,
        WeaponKind::Pistol,
    );
    assert_eq!(res.hit_ragdoll_id, Some(id));
    assert_eq!(res.hit_region, WoundRegion::RightArm);
    assert!(
        !world.is_hand_holding(id, false),
        "expected right hand to immediately drop the box when shot in right forearm"
    );

    for _ in 0..30 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf after right forearm shot"
            );
        }
    }
}

#[test]
fn test_held_box_breaks_free_on_heavy_impact() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let box_idx = world
        .give_ragdoll_held_box(id, Vec3::splat(0.18), 6.0, 240.0)
        .expect("give_ragdoll_held_box should succeed");

    for _ in 0..18 {
        world.step(1.0 / 60.0);
    }
    assert!(world.is_hand_holding(id, true) && world.is_hand_holding(id, false));

    let box_before = world.bodies[box_idx].position;
    let affected = world.apply_explosion(box_before + Vec3::new(0.0, 0.0, 0.45), 3.2, 85.0, 0.0);
    assert!(affected > 0);

    for _ in 0..30 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during heavy impact held box break"
            );
        }
    }

    assert!(
        !world.is_hand_holding(id, true) && !world.is_hand_holding(id, false),
        "expected both hands to release held box on heavy impact (left={}, right={})",
        world.is_hand_holding(id, true),
        world.is_hand_holding(id, false)
    );
    let box_after = world.bodies[box_idx].position;
    assert!(
        (box_after - box_before).length() > 0.30,
        "expected released box to fly freely after explosion, moved only {:?}",
        box_after - box_before
    );
}

#[test]
fn test_taser_electrocution_drops_item_locks_muscles_and_topples() {
    use arsa_core::StatusEffect;

    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let _box_idx = world
        .give_ragdoll_held_box(id, Vec3::splat(0.18), 4.0, 260.0)
        .expect("give_ragdoll_held_box should succeed");

    for _ in 0..20 {
        world.step(1.0 / 60.0);
    }
    assert!(world.is_hand_holding(id, true) && world.is_hand_holding(id, false));

    assert!(world.set_ragdoll_status(id, StatusEffect::Electrocuted, 1.0, 2.5));
    assert!(
        !world.is_hand_holding(id, true) && !world.is_hand_holding(id, false),
        "expected tased ragdoll to immediately drop held box"
    );

    let mut min_left_elbow_cos = 1.0_f32;
    let mut min_right_elbow_cos = 1.0_f32;
    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during electrocution"
            );
        }

        let r = &world.ragdolls[id];
        let l_ua_y =
            world.bodies[r.body_indices[BoneId::LeftUpperArm as usize]].orientation * Vec3::Y;
        let l_fa_y =
            world.bodies[r.body_indices[BoneId::LeftForearm as usize]].orientation * Vec3::Y;
        min_left_elbow_cos = min_left_elbow_cos.min(l_ua_y.dot(l_fa_y));

        let r_ua_y =
            world.bodies[r.body_indices[BoneId::RightUpperArm as usize]].orientation * Vec3::Y;
        let r_fa_y =
            world.bodies[r.body_indices[BoneId::RightForearm as usize]].orientation * Vec3::Y;
        min_right_elbow_cos = min_right_elbow_cos.min(r_ua_y.dot(r_fa_y));
    }

    assert!(
        min_left_elbow_cos < 0.72 && min_right_elbow_cos < 0.72,
        "expected elbows to clench tightly (cos < 0.72), got left={min_left_elbow_cos:.3}, right={min_right_elbow_cos:.3}"
    );
    assert!(
        matches!(
            world.ragdolls[id].balance.state,
            BehaviorState::Falling | BehaviorState::Downed
        ),
        "expected tased ragdoll to topple into Falling or Downed, got {:?}",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_on_fire_panic_flail_and_stagger() {
    use arsa_core::StatusEffect;

    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    assert!(world.set_ragdoll_status(id, StatusEffect::OnFire, 1.0, 1.2));
    assert_eq!(world.ragdoll_status(id), Some((StatusEffect::OnFire, 1.0)));

    let l_hand_idx = world.ragdolls[id].body_indices[BoneId::LeftHand as usize];
    let r_hand_idx = world.ragdolls[id].body_indices[BoneId::RightHand as usize];
    let mut max_hand_speed = 0.0_f32;

    for step in 0..90 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during OnFire panic"
            );
        }
        if step < 68 {
            let l_spd = world.bodies[l_hand_idx].linear_velocity.length();
            let r_spd = world.bodies[r_hand_idx].linear_velocity.length();
            max_hand_speed = max_hand_speed.max(l_spd.max(r_spd));
        }
    }

    assert!(
        max_hand_speed > 1.2,
        "expected vigorous panic arm flail (max_hand_speed > 1.2), got {max_hand_speed:.3}"
    );
    assert_eq!(
        world.ragdoll_status(id),
        Some((StatusEffect::None, 0.0)),
        "expected OnFire status to expire cleanly after 1.2s"
    );
    let health = world.ragdolls[id].reflex.health;
    assert!(
        (25.0..100.0).contains(&health),
        "expected gentle health drain with floor 25.0, got {health}"
    );
}

#[test]
fn test_procedural_locomotion_walks_forward_upright() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let standing_pelvis_y = world.ragdolls[id].standing_pelvis_y;
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    let start_z = world.bodies[pelvis_idx].position.z;

    assert!(world.set_ragdoll_desired_velocity(id, 0.0, 1.4));

    for _ in 0..150 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during procedural locomotion"
            );
        }
    }

    let pelvis_pos = world.bodies[pelvis_idx].position;
    let traveled_z = pelvis_pos.z - start_z;
    assert!(
        traveled_z > 1.5,
        "expected pelvis to travel > 1.5m in +Z over 2.5s, got {traveled_z:.3}m"
    );
    assert!(
        pelvis_pos.y > 0.72 * standing_pelvis_y,
        "expected pelvis to stay upright (>{:.3}m), got {:.3}m",
        0.72 * standing_pelvis_y,
        pelvis_pos.y
    );
    assert!(
        matches!(
            world.ragdolls[id].balance.state,
            BehaviorState::Balanced | BehaviorState::Stumbling
        ),
        "expected Balanced or Stumbling while walking, got {:?}",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_locomotion_with_leg_wound_limps_slower() {
    let mut world = ArsaWorld::new();
    let healthy_id = world.spawn_ragdoll(Vec3::new(-1.5, 0.02, 0.0));
    let wounded_id = world.spawn_ragdoll(Vec3::new(1.5, 0.02, 0.0));

    world.ragdolls[wounded_id].reflex.bone_damage[BoneId::LeftThigh as usize] = 0.65;

    assert!(world.set_ragdoll_desired_velocity(healthy_id, 0.0, 1.4));
    assert!(world.set_ragdoll_desired_velocity(wounded_id, 0.0, 1.4));

    let healthy_pelvis = world.ragdolls[healthy_id].body_indices[BoneId::Pelvis as usize];
    let wounded_pelvis = world.ragdolls[wounded_id].body_indices[BoneId::Pelvis as usize];
    let healthy_start_z = world.bodies[healthy_pelvis].position.z;
    let wounded_start_z = world.bodies[wounded_pelvis].position.z;

    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during wounded locomotion"
            );
        }
    }

    let healthy_dist = world.bodies[healthy_pelvis].position.z - healthy_start_z;
    let wounded_dist = world.bodies[wounded_pelvis].position.z - wounded_start_z;
    assert!(
        healthy_dist > 1.0 && wounded_dist > 0.5,
        "expected both ragdolls to move forward cleanly, got healthy={healthy_dist:.3}m, wounded={wounded_dist:.3}m"
    );
    assert!(
        wounded_dist < healthy_dist * 0.88,
        "expected wounded ragdoll to limp noticeably slower than healthy (healthy={healthy_dist:.3}m, wounded={wounded_dist:.3}m)"
    );
}

#[test]
fn test_water_buoyancy_floats_ragdoll_and_extinguishes_fire() {
    use arsa_core::StatusEffect;

    let mut world = ArsaWorld::new();
    world.set_water_level(Some(1.6));

    let id = world.spawn_ragdoll(Vec3::new(0.0, 1.8, 0.0));
    assert!(world.set_ragdoll_status(id, StatusEffect::OnFire, 1.0, 5.0));
    assert_eq!(world.ragdoll_status(id), Some((StatusEffect::OnFire, 1.0)));

    let box_idx = world.spawn_box(Vec3::new(2.0, 1.8, 0.0), Vec3::splat(0.22), 6.0, Vec3::ZERO);

    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];
    for _ in 0..120 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf in water buoyancy test"
            );
        }
    }

    assert_eq!(
        world.ragdoll_status(id),
        Some((StatusEffect::None, 0.0)),
        "expected water immersion to extinguish OnFire status"
    );

    let pelvis_y = world.bodies[pelvis_idx].position.y;
    assert!(
        pelvis_y > 0.65,
        "expected ragdoll pelvis to float near water surface (> 0.65m), got {pelvis_y:.3}m"
    );

    let box_y = world.bodies[box_idx].position.y;
    assert!(
        (box_y - 1.6).abs() < 0.35,
        "expected 6kg wooden box to float near water level y=1.6, got {box_y:.3}m"
    );
}

#[test]
fn test_taser_while_walking_topples_forward_along_momentum() {
    use arsa_core::StatusEffect;

    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    let pelvis_idx = world.ragdolls[id].body_indices[BoneId::Pelvis as usize];

    assert!(world.set_ragdoll_desired_velocity(id, 0.0, 1.8));
    for _ in 0..48 {
        world.step(1.0 / 60.0);
    }

    let pre_taser_z = world.bodies[pelvis_idx].position.z;
    assert!(
        pre_taser_z > 0.3,
        "expected ragdoll to build forward walking momentum before taser, got z={pre_taser_z:.3}"
    );

    assert!(world.set_ragdoll_status(id, StatusEffect::Electrocuted, 1.0, 2.5));
    for _ in 0..72 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during walking taser fall"
            );
        }
    }

    let post_taser_z = world.bodies[pelvis_idx].position.z;
    assert!(
        post_taser_z > pre_taser_z + 0.25,
        "expected walking ragdoll to fall forward along +Z momentum (pre={pre_taser_z:.3}, post={post_taser_z:.3})"
    );
    assert!(
        matches!(
            world.ragdolls[id].balance.state,
            BehaviorState::Falling | BehaviorState::Downed
        ),
        "expected tased walking ragdoll to enter Falling or Downed, got {:?}",
        world.ragdolls[id].balance.state
    );
}

#[test]
fn test_fast_projectile_box_anti_tunneling_hits_ragdoll() {
    let mut world = ArsaWorld::new();
    let id = world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
    for _ in 0..15 {
        world.step(1.0 / 60.0);
    }

    let chest_idx = world.ragdolls[id].body_indices[BoneId::Chest as usize];
    let chest_start = world.bodies[chest_idx].position;

    let proj_idx = world.spawn_box(
        Vec3::new(chest_start.x, chest_start.y, chest_start.z + 2.0),
        Vec3::splat(0.14),
        6.0,
        Vec3::new(0.0, 0.0, -45.0),
    );

    let mut min_chest_vz = 0.0_f32;
    for _ in 0..18 {
        world.step(1.0 / 60.0);
        for (i, b) in world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "body {i} produced NaN/Inf during fast projectile test"
            );
        }
        min_chest_vz = min_chest_vz.min(world.bodies[chest_idx].linear_velocity.z);
    }

    let proj_vel_z = world.bodies[proj_idx].linear_velocity.z;
    let chest_disp_z = world.bodies[chest_idx].position.z - chest_start.z;
    assert!(
        min_chest_vz < -1.5 && chest_disp_z < -0.10,
        "expected 45 m/s box to collide with ragdoll chest without tunneling: min_chest_vz={min_chest_vz:.3}, chest_disp_z={chest_disp_z:.3}, proj_vel_z={proj_vel_z:.3}"
    );
    assert!(
        proj_vel_z > -42.0,
        "expected projectile box to lose forward speed upon impacting chest, got proj_vel_z={proj_vel_z:.3}"
    );
}

#[test]
fn test_low_friction_ice_allows_sliding() {
    let mut normal_world = ArsaWorld::new();
    normal_world.set_floor_friction(0.75);
    let normal_box = normal_world.spawn_box(
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::splat(0.2),
        8.0,
        Vec3::new(4.0, 0.0, 0.0),
    );

    let mut ice_world = ArsaWorld::new();
    ice_world.set_floor_friction(0.04);
    let ice_box = ice_world.spawn_box(
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::splat(0.2),
        8.0,
        Vec3::new(4.0, 0.0, 0.0),
    );
    let ice_ragdoll = ice_world.spawn_ragdoll(Vec3::new(0.0, 0.02, 2.0));
    let ice_pelvis = ice_world.ragdolls[ice_ragdoll].body_indices[BoneId::Pelvis as usize];
    for _ in 0..10 {
        ice_world.step(1.0 / 60.0);
    }
    let _hit = ice_world.apply_shove(Vec3::new(-2.0, 1.15, 2.0), Vec3::new(1.0, 0.0, 0.0), 55.0);

    for _ in 0..90 {
        normal_world.step(1.0 / 60.0);
        ice_world.step(1.0 / 60.0);
        for (i, b) in ice_world.bodies.iter().enumerate() {
            assert!(
                b.position.is_finite()
                    && b.orientation.is_finite()
                    && b.linear_velocity.is_finite()
                    && b.angular_velocity.is_finite(),
                "ice world body {i} produced NaN/Inf"
            );
        }
    }

    let normal_x = normal_world.bodies[normal_box].position.x;
    let ice_x = ice_world.bodies[ice_box].position.x;
    let ice_ragdoll_x = ice_world.bodies[ice_pelvis].position.x;
    assert!(
        ice_x > normal_x * 2.5 && ice_x > 3.0,
        "expected box on 0.04 ice friction to slide much farther than 0.75 friction: normal_x={normal_x:.3}, ice_x={ice_x:.3}"
    );
    assert!(
        ice_ragdoll_x > 0.2,
        "expected shoved ragdoll on ice to slide in +X, got ice_ragdoll_x={ice_ragdoll_x:.3}"
    );
}
