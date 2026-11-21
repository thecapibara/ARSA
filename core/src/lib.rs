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
pub use world::{ArsaWorld, GrabConstraint, RaycastHit, WeaponKind, WeaponShotResult};

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
}
