//! Scaling benchmark for [`propagate_perception`].
//!
//! Broadcast propagation is the crate's only many-to-many hot path: every
//! [`PerceptionMessage`] fired in a frame has to be resolved against every
//! entity holding a [`Memory`]. Message count itself scales with actor count in
//! a shipping game (footsteps, casts, impacts), so the pair count grows
//! quadratically and this is the loop worth measuring.
//!
//! Scenarios (`perceivers` × `messages`, both in {100, 1000}):
//!
//! - `propagate/{N}x{M}` — perceivers and message origins scattered over the same
//!   square, each message with a radius covering roughly one percent of it. This is
//!   the steady state: most pairs are out of range and are rejected by the distance
//!   cull.
//! - `burst/1000x1000` — every message shares one origin, source and percept, the
//!   shape a single explosion or a rapid-fire impact stream produces.
//!
//! Run: `cargo bench --bench perception_scaling`

use bevy::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use msg_perception::prelude::*;
use msg_perception::propagate_perception;
use std::hint::black_box;

/// Side length of the square perceivers and message origins are scattered over.
const WORLD_SIZE: f32 = 4000.0;
/// Propagation radius of a scattered message.
const MESSAGE_RANGE: f32 = 250.0;
/// Base intensity of every message, high enough to survive attenuation at the
/// edge of `MESSAGE_RANGE`.
const BASE_VALUE: f32 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
enum BenchPercept {
    Sound,
}

/// Deterministic value source, so a bench run is reproducible without a
/// dependency on an RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn next_point(&mut self) -> Vec2 {
        Vec2::new(self.next_unit() * WORLD_SIZE, self.next_unit() * WORLD_SIZE)
    }
}

/// The messages one bench iteration fires, rewritten into the message buffer on
/// every run so the measured work is identical each time.
#[derive(Resource)]
struct FrameMessages(Vec<PerceptionMessage<BenchPercept>>);

fn refresh_message_buffer(mut messages: ResMut<Messages<PerceptionMessage<BenchPercept>>>) {
    messages.update();
}

fn emit_frame_messages(
    frame: Res<FrameMessages>,
    mut writer: MessageWriter<PerceptionMessage<BenchPercept>>,
) {
    for message in &frame.0 {
        writer.write(message.clone());
    }
}

/// Builds a world with `perceivers` memory holders and a schedule that fires
/// `messages` and propagates them once per run.
fn setup(perceivers: usize, messages: Vec<PerceptionMessage<BenchPercept>>) -> (World, Schedule) {
    let mut world = World::new();
    world.insert_resource(PerceptionConfig::default());
    world.insert_resource(Messages::<PerceptionMessage<BenchPercept>>::default());
    world.insert_resource(FrameMessages(messages));

    let mut rng = Lcg(0x5eed);
    for _ in 0..perceivers {
        let position = rng.next_point();
        world.spawn((
            Transform::from_translation(position.extend(0.0)),
            Memory::<BenchPercept>::default(),
        ));
    }

    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            refresh_message_buffer,
            emit_frame_messages,
            propagate_perception::<BenchPercept>,
        )
            .chain(),
    );
    schedule.initialize(&mut world).unwrap();

    (world, schedule)
}

/// Messages scattered over the world, each from a distinct source.
fn scattered_messages(count: usize) -> Vec<PerceptionMessage<BenchPercept>> {
    let mut rng = Lcg(0xf00d);
    (0..count)
        .map(|i| PerceptionMessage {
            percept: BenchPercept::Sound,
            base_value: BASE_VALUE,
            origin: rng.next_point(),
            range: MESSAGE_RANGE,
            source: Some(Entity::from_raw_u32(i as u32 + 1).unwrap()),
        })
        .collect()
}

/// Messages that all share one origin, source and percept.
fn burst_messages(count: usize) -> Vec<PerceptionMessage<BenchPercept>> {
    let origin = Vec2::splat(WORLD_SIZE * 0.5);
    let source = Entity::from_raw_u32(1).unwrap();
    (0..count)
        .map(|i| PerceptionMessage {
            percept: BenchPercept::Sound,
            base_value: BASE_VALUE - (i % 10) as f32,
            origin,
            range: MESSAGE_RANGE * 4.0,
            source: Some(source),
        })
        .collect()
}

fn bench_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagate");
    for perceivers in [100usize, 1000] {
        for messages in [100usize, 1000] {
            let (mut world, mut schedule) = setup(perceivers, scattered_messages(messages));
            group.bench_function(format!("{perceivers}x{messages}"), |b| {
                b.iter(|| schedule.run(black_box(&mut world)));
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("burst");
    let (mut world, mut schedule) = setup(1000, burst_messages(1000));
    group.bench_function("1000x1000", |b| {
        b.iter(|| schedule.run(black_box(&mut world)));
    });
    group.finish();
}

criterion_group!(benches, bench_propagation);
criterion_main!(benches);
