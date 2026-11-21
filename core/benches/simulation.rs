use std::hint::black_box;
use std::time::Instant;

use arsa_core::glam::Vec3;
use arsa_core::{ArsaWorld, BodyProfile, WeaponKind};

const WARMUP_FRAMES: usize = 60;
const TIMED_FRAMES: usize = 300;
const DT: f32 = 1.0 / 60.0;

fn run_bench<F, S>(name: &str, mut setup: F, mut on_frame: S)
where
    F: FnMut() -> ArsaWorld,
    S: FnMut(&mut ArsaWorld, usize),
{
    let mut world = setup();

    for frame in 0..WARMUP_FRAMES {
        on_frame(&mut world, frame);
        world.step(black_box(DT));
    }

    let mut samples_us = Vec::with_capacity(TIMED_FRAMES);
    for frame in 0..TIMED_FRAMES {
        on_frame(&mut world, WARMUP_FRAMES + frame);
        let start = Instant::now();
        world.step(black_box(DT));
        samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    black_box(&world.bodies);

    samples_us.sort_by(|a, b| a.total_cmp(b));
    let mean_us = samples_us.iter().sum::<f64>() / (samples_us.len() as f64);
    let median_us = percentile(&samples_us, 0.50);
    let p95_us = percentile(&samples_us, 0.95);
    let p99_us = percentile(&samples_us, 0.99);
    let max_fps = 1_000_000.0 / p95_us.max(1e-6);

    println!(
        "{name:<28} | median: {median_us:7.1} µs | mean: {mean_us:7.1} µs | p95: {p95_us:7.1} µs | p99: {p99_us:7.1} µs | p95 FPS: {max_fps:7.1}"
    );
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let default_substeps = ArsaWorld::new().substeps;
    println!(
        "ARSA Core Benchmarks ({WARMUP_FRAMES} warmup + {TIMED_FRAMES} timed frames @ 60Hz x {default_substeps} substeps)"
    );
    println!("{}", "-".repeat(118));

    // 1. Single active balanced ragdoll (baseline)
    run_bench(
        "single_active_ragdoll",
        || {
            let mut world = ArsaWorld::new();
            world.spawn_ragdoll(Vec3::new(0.0, 0.02, 0.0));
            world
        },
        |_, _| {},
    );

    // 2. Crowd 4x diverse active ragdolls (calm standing baseline)
    run_bench(
        "crowd_4x_standing",
        || {
            let mut world = ArsaWorld::new();
            let profiles = [
                (
                    Vec3::new(-1.5, 0.02, 0.0),
                    BodyProfile {
                        height_m: 1.65,
                        mass_kg: 60.0,
                        muscle_ratio: 0.9,
                    },
                ),
                (Vec3::new(-0.5, 0.02, 0.0), BodyProfile::default()),
                (
                    Vec3::new(0.5, 0.02, 0.0),
                    BodyProfile {
                        height_m: 1.90,
                        mass_kg: 70.0,
                        muscle_ratio: 1.0,
                    },
                ),
                (
                    Vec3::new(1.5, 0.02, 0.0),
                    BodyProfile {
                        height_m: 2.17,
                        mass_kg: 135.0,
                        muscle_ratio: 1.35,
                    },
                ),
            ];
            for (pos, profile) in profiles {
                world.spawn_ragdoll_with_profile(pos, profile);
            }
            world
        },
        |_, _| {},
    );

    // 3. Crowd 4x CHAOS: tight domino formation + crates + shotgun + explosion + ongoing earthquake
    //    Heavily stresses narrowphase, inter-ragdoll collisions, reflexes, bracing & recovery.
    run_bench(
        "crowd_4x_chaos_impact",
        || {
            let mut world = ArsaWorld::new();
            let formation = [
                (Vec3::new(0.0, 0.02, 0.55), BodyProfile::default()),
                (
                    Vec3::new(-0.55, 0.02, 0.0),
                    BodyProfile {
                        height_m: 1.65,
                        mass_kg: 60.0,
                        muscle_ratio: 0.9,
                    },
                ),
                (
                    Vec3::new(0.55, 0.02, 0.0),
                    BodyProfile {
                        height_m: 1.90,
                        mass_kg: 70.0,
                        muscle_ratio: 1.0,
                    },
                ),
                (
                    Vec3::new(0.0, 0.02, -0.55),
                    BodyProfile {
                        height_m: 2.17,
                        mass_kg: 135.0,
                        muscle_ratio: 1.35,
                    },
                ),
            ];
            for (pos, profile) in formation {
                world.spawn_ragdoll_with_profile(pos, profile);
            }
            // Add obstacle crates right around the crowd
            world.place_obstacle_box(0.0, -1.1, 0.35, 0.14, 0.18, 30.0);
            world.spawn_box(
                Vec3::new(-0.9, 0.24, 0.3),
                Vec3::splat(0.24),
                22.0,
                Vec3::ZERO,
            );
            world.set_earthquake(6.5);
            world
        },
        |world, frame| match frame {
            65 => {
                // Shotgun blast into front ragdoll -> domino collision into crowd
                world.fire_weapon(
                    Vec3::new(0.0, 1.3, 3.0),
                    Vec3::new(0.0, 0.0, -1.0),
                    WeaponKind::Shotgun,
                );
            }
            140 => {
                // Radial blast wave inside the crowd
                world.apply_explosion(Vec3::new(0.0, 0.4, 0.0), 4.0, 55.0, 35.0);
            }
            220 => {
                // Kinetic force wave across recovering/tumbled crowd
                world.apply_force_wave(
                    Vec3::new(2.5, 1.4, 2.0),
                    Vec3::new(-0.8, -0.15, -0.6),
                    3.5,
                    40.0,
                );
            }
            _ => {}
        },
    );

    // 4. Stacked 5-cube SAT tower under mild seismic perturbation (keeps solver active)
    run_bench(
        "stacked_box_tower_5x",
        || {
            let mut world = ArsaWorld::new();
            let half = Vec3::splat(0.24);
            for i in 0..5 {
                let y = half.y + (i as f32) * (half.y * 2.0 + 0.002);
                world.spawn_box(Vec3::new(0.0, y, 0.0), half, 22.0, Vec3::ZERO);
            }
            world
        },
        |_, _| {},
    );

    // 5. 20-NPC mixed-LOD crowd (4 Full <15m, 6 Reduced 18..28m, 10 Sleep 40..60m)
    run_bench(
        "crowd_20x_lod_mixed",
        || {
            let mut world = ArsaWorld::new();
            let full_positions = [
                Vec3::new(-3.0, 0.02, 0.0),
                Vec3::new(-1.0, 0.02, 0.0),
                Vec3::new(1.0, 0.02, 0.0),
                Vec3::new(3.0, 0.02, 0.0),
            ];
            for pos in full_positions {
                world.spawn_ragdoll(pos);
            }
            for i in 0..6 {
                let radius = 18.0 + (i as f32) * 2.0;
                let angle = (i as f32) * (std::f32::consts::TAU / 6.0);
                let pos = Vec3::new(radius * angle.cos(), 0.02, radius * angle.sin());
                world.spawn_ragdoll(pos);
            }
            for i in 0..10 {
                let radius = 40.0 + (i as f32) * 2.0;
                let angle = (i as f32) * (std::f32::consts::TAU / 10.0);
                let pos = Vec3::new(radius * angle.cos(), 0.02, radius * angle.sin());
                world.spawn_ragdoll(pos);
            }
            world.set_lod_focus(Some(Vec3::ZERO), 15.0, 35.0);
            world
        },
        |_, _| {},
    );
}
