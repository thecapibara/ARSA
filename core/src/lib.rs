//! ARSA — Active Ragdoll Simulation Architecture core physics engine.

pub mod body;
pub mod collision;
pub mod joint;
pub mod math;
pub mod skeleton;
pub mod world;

pub use glam;

pub use body::{ColliderShape, RigidBody};
pub use collision::{Contact, StaticBox};
pub use joint::{Joint, JointKind};
pub use math::{inertia_box, inertia_capsule, inertia_sphere, swing_twist_decomposition};
pub use skeleton::{spawn_humanoid, BoneId, RagdollInstance};
pub use world::{ArsaWorld, GrabConstraint, RaycastHit};

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
}
