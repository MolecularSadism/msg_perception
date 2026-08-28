//! Benchmark for [`Memory::digest`] / [`Memory::digest_into`].
//!
//! The digest rebuild runs once per frame for every entity carrying a
//! [`MemoryDigest`] alongside a non-empty [`Memory`], so its cost is the
//! per-actor overhead the summary feature adds. The memory under test holds
//! 100 entries spread across 5 percept types — a well-populated actor near a
//! typical retention cap.
//!
//! Scenarios:
//!
//! - `digest/into_100x5` — [`Memory::digest_into`] reusing one map across
//!   iterations, the steady state [`update_memory_digest`] hits each frame.
//! - `digest/alloc_100x5` — [`Memory::digest`] allocating a fresh map per
//!   call, the cost of one-off summaries outside the plugin's scratch reuse.
//!
//! Run: `cargo bench --bench memory_digest`
//!
//! [`MemoryDigest`]: msg_perception::MemoryDigest
//! [`update_memory_digest`]: msg_perception::update_memory_digest

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use msg_perception::prelude::*;
use std::hint::black_box;

/// Number of entries in the benched memory.
const ENTRY_COUNT: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
enum BenchPercept {
    Sound,
    Sight,
    Smell,
    Pain,
    Threat,
}

const PERCEPTS: [BenchPercept; 5] = [
    BenchPercept::Sound,
    BenchPercept::Sight,
    BenchPercept::Smell,
    BenchPercept::Pain,
    BenchPercept::Threat,
];

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
}

/// Builds a memory holding [`ENTRY_COUNT`] entries cycling through all five
/// percept types, with a mix of present and absent locations and sources.
fn populated_memory() -> Memory<BenchPercept> {
    let mut rng = Lcg(0x5eed);
    let mut memory = Memory::default();
    for i in 0..ENTRY_COUNT {
        let value = rng.next_unit() * 100.0;
        let location = (i % 3 != 0).then(|| {
            LocationKnowledge::Origin(Vec2::new(rng.next_unit() * 500.0, rng.next_unit() * 500.0))
        });
        let source = (i % 4 != 0).then(|| Entity::from_raw_u32(i as u32 + 1).unwrap());
        memory.push(
            MemoryEntry {
                percept: PERCEPTS[i % PERCEPTS.len()],
                value,
                location,
                source,
            },
            ENTRY_COUNT,
        );
    }
    memory
}

fn bench_digest(c: &mut Criterion) {
    let memory = populated_memory();
    let mut group = c.benchmark_group("digest");

    let mut digests = HashMap::new();
    group.bench_function("into_100x5", |b| {
        b.iter(|| {
            black_box(&memory).digest_into(&mut digests);
            black_box(&digests);
        });
    });

    group.bench_function("alloc_100x5", |b| {
        b.iter(|| black_box(&memory).digest());
    });

    group.finish();
}

criterion_group!(benches, bench_digest);
criterion_main!(benches);
