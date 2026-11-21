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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn test_swing_twist_round_trip() {
        let axes = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -1.0).normalize()];
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
        let box_idx = world.spawn_box(
            Vec3::new(3.0, 0.5, 0.0),
            Vec3::splat(0.5),
            1.0,
            Vec3::ZERO,
        );
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
        assert!(r.reflex.unconscious, "headshot rifle should cause unconsciousness");
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
}
