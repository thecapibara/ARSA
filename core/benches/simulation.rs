use std::hint::black_box;
use std::time::Instant;

use arsa_core::{ArsaWorld, BodyProfile};
use glam::Vec3;

const DT: f32 = 1.0 / 60.0;
const WARMUP_FRAMES: usize = 60;
const BENCH_FRAMES: usize = 300;

fn bench_scenario(name: &str, mut world: ArsaWorld) {
    for _ in 0..WARMUP_FRAMES {
        world.step(DT);
    }

    let start = Instant::now();
    for _ in 0..BENCH_FRAMES {
        world.step(DT);
    }
    black_box(&world.bodies);
    let elapsed = start.elapsed();

    let us_per_step = (elapsed.as_secs_f64() * 1_000_000.0) / BENCH_FRAMES as f64;
    let ms_per_step = us_per_step / 1_000.0;
    let max_fps = 1_000_000.0 / us_per_step;

    println!(
        "{name:<28} | {us_per_step:8.2} µs/step | {ms_per_step:6.3} ms/step | {max_fps:8.1} FPS max"
    );
}

fn single_active_ragdoll() -> ArsaWorld {
    let mut world = ArsaWorld::new();
    world.spawn_ragdoll(Vec3::ZERO);
    world
}

fn crowd_4x_active_ragdolls() -> ArsaWorld {
    let mut world = ArsaWorld::new();
    let profiles = [
        (
            Vec3::new(-1.5, 0.0, 0.0),
            BodyProfile {
                height_m: 1.65,
                mass_kg: 60.0,
                muscle_ratio: 0.9,
            },
        ),
        (
            Vec3::new(-0.5, 0.0, 0.0),
            BodyProfile {
                height_m: 1.78,
                mass_kg: 75.0,
                muscle_ratio: 1.0,
            },
        ),
        (
            Vec3::new(0.5, 0.0, 0.0),
            BodyProfile {
                height_m: 1.90,
                mass_kg: 70.0,
                muscle_ratio: 1.0,
            },
        ),
        (
            Vec3::new(1.5, 0.0, 0.0),
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
}

fn stacked_box_tower_5x() -> ArsaWorld {
    let mut world = ArsaWorld::new();
    let half = Vec3::splat(0.24);
    for i in 0..5 {
        let y = 0.24 + (i as f32) * 0.482;
        world.spawn_box(Vec3::new(0.0, y, 0.0), half, 22.0, Vec3::ZERO);
    }
    world
}

fn main() {
    println!(
        "ARSA Core Physics Benchmarks (60Hz x 10 substeps, {WARMUP_FRAMES} warmup + {BENCH_FRAMES} timed frames)"
    );
    println!("{}", "-".repeat(78));
    bench_scenario("single_active_ragdoll", single_active_ragdoll());
    bench_scenario("crowd_4x_active_ragdolls", crowd_4x_active_ragdolls());
    bench_scenario("stacked_box_tower_5x", stacked_box_tower_5x());
}
