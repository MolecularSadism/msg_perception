//! Broadcast propagation of [`PerceptionMessage`] into [`Memory`] components.
//!
//! Propagation is a many-to-many pass: every message fired in a frame is
//! resolved against every entity holding a [`Memory<P>`]. Because message count
//! scales with actor count in a busy scene (footsteps, casts, impacts), the pair
//! count grows quadratically, so the pass is built around three reductions:
//!
//! 1. **Squared-distance culling.** A pair is rejected by comparing squared
//!    distance against squared range, before any square root or logarithm.
//! 2. **Same-frame merging.** Messages that share a percept, a source, a range
//!    and an origin cell collapse into one carrying the strongest value, so a
//!    burst of impacts writes one memory instead of filling the ring buffer.
//! 3. **Spatial indexing.** Above a pair-count threshold the merged messages are
//!    bucketed into a uniform grid, and each perceiver only tests the messages
//!    whose radius reaches its own cell.
//!
//! Culling and indexing are conservative: they only skip pairs the exact
//! per-pair check would have rejected anyway, so the attenuation formula, the
//! [`LocationKnowledge`] written, and the push order per perceiver match what an
//! exhaustive loop produces. Merging is the one deliberate behavior change and
//! is tunable through [`PerceptionConfig::merge_radius`].
//!
//! ## Extension points
//!
//! The index is deliberately self-contained: it is rebuilt from the frame's
//! messages and needs no persistent spatial structure from the host app. A
//! caller that already maintains a broadphase can bypass it by scheduling
//! [`propagate_perception`] with a narrower [`Memory`] query — the system reads
//! whatever perceivers the query yields.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasherDefault, Hasher};

use bevy::prelude::*;

use crate::{LocationKnowledge, Memory, MemoryEntry, Percept, PerceptionConfig, PerceptionMessage};

/// Fewest messages in a frame that make the spatial index worth building.
const INDEX_MIN_MESSAGES: usize = 8;

/// Fewest perceivers that make the spatial index worth building. Below this
/// there are too few lookups to amortize bucketing the messages.
const INDEX_MIN_PERCEIVERS: usize = 32;

/// Fewest message × perceiver pairs that make the spatial index worth building.
/// Below this the exhaustive inner loop is cheaper than the bucketing pass.
const INDEX_MIN_PAIRS: usize = 4096;

/// Largest number of grid cells a single message may be bucketed into. Messages
/// reaching further are tested against every perceiver instead, so one
/// world-spanning explosion cannot blow up the index.
const MAX_CELLS_PER_MESSAGE: usize = 64;

/// Sort key used to group same-frame messages that are candidates for merging:
/// origin cell, range bits, source, and the message's position in the batch.
type MergeKey = (u32, u32, u32, Option<u64>, u32);

/// Reusable buffers for [`propagate_perception`], held in a `Local` so a frame's
/// batch and spatial index do not reallocate every run.
pub struct PropagationScratch<P: Percept> {
    /// Messages of the current frame, after merging.
    batch: Vec<PerceptionMessage<P>>,
    /// Merge grouping keys, sorted in place.
    merge_keys: Vec<MergeKey>,
    /// Per-batch flags marking messages folded into an earlier one.
    merged: Vec<bool>,
    /// Message ranges, used to pick the index cell size.
    ranges: Vec<f32>,
    /// Grid of the current batch.
    index: MessageIndex,
    /// Message indices to test for the perceiver being processed, ascending.
    candidates: Vec<u32>,
}

impl<P: Percept> Default for PropagationScratch<P> {
    fn default() -> Self {
        Self {
            batch: Vec::new(),
            merge_keys: Vec::new(),
            merged: Vec::new(),
            ranges: Vec::new(),
            index: MessageIndex::default(),
            candidates: Vec::new(),
        }
    }
}

/// Propagate perception messages to all [`Memory`] holders within range.
///
/// Intensity is attenuated using a saturating logarithmic drop-off:
/// `V = V_base - 20 * log10(distance)`, clamped to zero.
///
/// Self-caused perceptions (where `source == perceiver`) are skipped.
///
/// Messages sharing a percept, source, range and origin cell within
/// [`PerceptionConfig::merge_radius`] are merged into one memory carrying the
/// strongest of their base values.
pub fn propagate_perception<P: Percept>(
    mut messages: MessageReader<PerceptionMessage<P>>,
    mut q_perceivers: Query<(Entity, &Transform, &mut Memory<P>)>,
    config: Res<PerceptionConfig>,
    mut scratch: Local<PropagationScratch<P>>,
) {
    let PropagationScratch {
        batch,
        merge_keys,
        merged,
        ranges,
        index,
        candidates,
    } = &mut *scratch;

    // A message whose base value cannot clear the threshold at zero distance
    // never produces a memory, and a negative range never matches anything.
    batch.clear();
    batch.extend(
        messages
            .read()
            .filter(|message| {
                message.range >= 0.0 && message.base_value.max(0.0) > config.min_threshold
            })
            .cloned(),
    );
    if batch.is_empty() {
        return;
    }

    merge_batch(batch, merge_keys, merged, config.merge_radius);

    let perceiver_count = q_perceivers.count();
    if perceiver_count == 0 {
        return;
    }

    let indexed = batch.len() >= INDEX_MIN_MESSAGES
        && perceiver_count >= INDEX_MIN_PERCEIVERS
        && perceiver_count.saturating_mul(batch.len()) >= INDEX_MIN_PAIRS
        && index.build(batch, ranges);

    if !indexed {
        candidates.clear();
        candidates.extend(0..batch.len() as u32);
    }

    for (perceiver, transform, mut memory) in &mut q_perceivers {
        let perceiver_pos = transform.translation.truncate();

        if indexed {
            index.candidates_at(perceiver_pos, candidates);
        }

        for &message_index in candidates.iter() {
            let message = &batch[message_index as usize];

            // Skip self-caused perceptions
            if message.source == Some(perceiver) {
                continue;
            }

            let distance_sq = perceiver_pos.distance_squared(message.origin);
            if distance_sq > message.range * message.range {
                continue;
            }

            let distance = distance_sq.sqrt();
            let attenuation = if distance <= 1.0 {
                0.0
            } else {
                20.0 * distance.log10()
            };
            let initial_value = (message.base_value - attenuation).max(0.0);

            if initial_value <= config.min_threshold {
                continue;
            }

            memory.push(
                MemoryEntry {
                    percept: message.percept.clone(),
                    value: initial_value,
                    location: Some(LocationKnowledge::Origin(message.origin)),
                    source: message.source,
                },
                config.max_count,
            );
        }
    }
}

// ─── Merging ─────────────────────────────────────────────────────────────────

/// Fold messages that share a percept, source, range and origin cell into the
/// earliest of them, keeping the strongest base value.
///
/// Order is preserved: a merged group stays at the position of its first
/// message, so per-perceiver memory ordering matches an unmerged run.
fn merge_batch<P: Percept>(
    batch: &mut Vec<PerceptionMessage<P>>,
    keys: &mut Vec<MergeKey>,
    merged: &mut Vec<bool>,
    merge_radius: f32,
) {
    if batch.len() < 2 || merge_radius < 0.0 {
        return;
    }

    keys.clear();
    keys.extend(batch.iter().enumerate().map(|(index, message)| {
        let (x, y) = origin_key(message.origin, merge_radius);
        (
            x,
            y,
            message.range.to_bits(),
            message.source.map(Entity::to_bits),
            index as u32,
        )
    }));
    keys.sort_unstable();

    merged.clear();
    merged.resize(batch.len(), false);
    let mut any_merged = false;

    let mut group_start = 0;
    while group_start < keys.len() {
        let mut group_end = group_start + 1;
        while group_end < keys.len() && same_group(&keys[group_start], &keys[group_end]) {
            group_end += 1;
        }

        // Keys sort by index last, so a group's members are already ascending
        // and the first survivor of a run is the earliest message in the batch.
        let group = &keys[group_start..group_end];
        for (position, key) in group.iter().enumerate() {
            let keeper = key.4 as usize;
            if merged[keeper] {
                continue;
            }
            for later in &group[position + 1..] {
                let candidate = later.4 as usize;
                if merged[candidate] || !percepts_match(&batch[keeper], &batch[candidate]) {
                    continue;
                }
                let value = batch[candidate].base_value;
                batch[keeper].base_value = batch[keeper].base_value.max(value);
                merged[candidate] = true;
                any_merged = true;
            }
        }

        group_start = group_end;
    }

    if !any_merged {
        return;
    }

    let mut index = 0;
    batch.retain(|_| {
        let keep = !merged[index];
        index += 1;
        keep
    });
}

/// Whether two merge keys agree on everything but the message index.
fn same_group(a: &MergeKey, b: &MergeKey) -> bool {
    a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3
}

/// Quantize an origin for merge grouping.
///
/// A positive `merge_radius` snaps origins to a grid of that size, so nearby
/// stimuli share a key. At `0.0` the raw bit patterns are used, restricting
/// merging to bit-identical origins.
fn origin_key(origin: Vec2, merge_radius: f32) -> (u32, u32) {
    if merge_radius > 0.0 {
        let cell = (origin / merge_radius).floor();
        (cell.x as i32 as u32, cell.y as i32 as u32)
    } else {
        (origin.x.to_bits(), origin.y.to_bits())
    }
}

/// Whether two messages carry the same percept value.
///
/// [`Percept`] only guarantees reflection, so equality goes through
/// `PartialReflect::reflect_partial_eq`. Types that cannot answer report no
/// match, which leaves their messages unmerged.
fn percepts_match<P: Percept>(a: &PerceptionMessage<P>, b: &PerceptionMessage<P>) -> bool {
    a.percept
        .reflect_partial_eq(b.percept.as_partial_reflect())
        .unwrap_or(false)
}

// ─── Spatial index ───────────────────────────────────────────────────────────

/// Uniform grid over one frame's messages.
///
/// Each message is bucketed into every cell its radius reaches, so a perceiver
/// only has to look at the bucket for the cell it stands in. Buckets are kept
/// across frames and cleared in place, so a steady message load stops
/// allocating after the first few frames.
#[derive(Default)]
struct MessageIndex {
    /// Edge length of a grid cell.
    cell_size: f32,
    /// Cell key to bucket slot.
    cells: HashMap<u64, u32, BuildHasherDefault<CellHasher>>,
    /// Message indices per occupied cell, ascending. Retained for reuse.
    buckets: Vec<Vec<u32>>,
    /// Number of `buckets` entries in use this frame.
    used_buckets: usize,
    /// Messages too wide to bucket; tested against every perceiver.
    wide: Vec<u32>,
}

impl MessageIndex {
    /// Bucket every message of `batch`, reusing the previous frame's storage.
    ///
    /// Returns `false` when the batch has no usable extent to index, in which
    /// case the caller falls back to testing every message against every
    /// perceiver.
    fn build<P: Percept>(&mut self, batch: &[PerceptionMessage<P>], ranges: &mut Vec<f32>) -> bool {
        // Sizing cells by the median range keeps at least half the batch inside
        // a 3x3 bucketing while leaving outliers to the wide list.
        ranges.clear();
        ranges.extend(batch.iter().map(|message| message.range));
        let middle = ranges.len() / 2;
        ranges.select_nth_unstable_by(middle, f32::total_cmp);
        let cell_size = ranges[middle];
        if cell_size <= 0.0 || !cell_size.is_finite() {
            return false;
        }

        self.reset(cell_size);
        let inverse = 1.0 / cell_size;

        for (index, message) in batch.iter().enumerate() {
            let radius = Vec2::splat(message.range);
            let min = ((message.origin - radius) * inverse).floor();
            let max = ((message.origin + radius) * inverse).floor();
            let cells = (max.x - min.x + 1.0) * (max.y - min.y + 1.0);

            if !cells.is_finite() || cells > MAX_CELLS_PER_MESSAGE as f32 {
                self.wide.push(index as u32);
                continue;
            }

            let (min, max) = (min.as_ivec2(), max.as_ivec2());
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    self.insert(cell_key(IVec2::new(x, y)), index as u32);
                }
            }
        }

        true
    }

    /// Drop the previous frame's contents while keeping their allocations.
    fn reset(&mut self, cell_size: f32) {
        self.cell_size = cell_size;
        self.cells.clear();
        for bucket in &mut self.buckets[..self.used_buckets] {
            bucket.clear();
        }
        self.used_buckets = 0;
        self.wide.clear();
    }

    /// Record `message` as reaching `key`'s cell.
    fn insert(&mut self, key: u64, message: u32) {
        let slot = match self.cells.entry(key) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let slot = self.used_buckets as u32;
                self.used_buckets += 1;
                if self.buckets.len() < self.used_buckets {
                    self.buckets.push(Vec::new());
                }
                vacant.insert(slot);
                slot
            }
        };
        self.buckets[slot as usize].push(message);
    }

    /// Messages bucketed into the cell containing `position`.
    fn bucket_at(&self, position: Vec2) -> &[u32] {
        let key = cell_key((position / self.cell_size).floor().as_ivec2());
        match self.cells.get(&key) {
            Some(&slot) => &self.buckets[slot as usize],
            None => &[],
        }
    }

    /// Gather the messages that can reach `position`, in batch order.
    fn candidates_at(&self, position: Vec2, candidates: &mut Vec<u32>) {
        candidates.clear();
        let local = self.bucket_at(position);

        if self.wide.is_empty() {
            candidates.extend_from_slice(local);
            return;
        }

        // Both runs are ascending; merge them so the per-perceiver test order
        // still matches the order the messages were written in.
        let (mut i, mut j) = (0, 0);
        while i < local.len() && j < self.wide.len() {
            if local[i] <= self.wide[j] {
                candidates.push(local[i]);
                i += 1;
            } else {
                candidates.push(self.wide[j]);
                j += 1;
            }
        }
        candidates.extend_from_slice(&local[i..]);
        candidates.extend_from_slice(&self.wide[j..]);
    }
}

/// Pack a cell coordinate into a hash key.
fn cell_key(cell: IVec2) -> u64 {
    ((cell.x as u32 as u64) << 32) | (cell.y as u32 as u64)
}

/// Multiply-shift hasher for the packed cell keys.
///
/// The keys are already two dense integers side by side, so a single
/// multiplication spreads them across the table; the default `SipHash` costs
/// more than the lookups it protects at this call volume.
#[derive(Default)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        let mixed = (self.0 ^ value).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = mixed ^ (mixed >> 32);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
    enum TestPercept {
        Sound,
        Pain,
    }

    fn message(
        percept: TestPercept,
        base_value: f32,
        origin: Vec2,
        source: Option<Entity>,
    ) -> PerceptionMessage<TestPercept> {
        PerceptionMessage {
            percept,
            base_value,
            origin,
            range: 100.0,
            source,
        }
    }

    fn merge(
        mut batch: Vec<PerceptionMessage<TestPercept>>,
        merge_radius: f32,
    ) -> Vec<PerceptionMessage<TestPercept>> {
        let mut keys = Vec::new();
        let mut merged = Vec::new();
        merge_batch(&mut batch, &mut keys, &mut merged, merge_radius);
        batch
    }

    /// A spread of small-radius messages, large enough to exercise the index.
    fn scattered_batch() -> Vec<PerceptionMessage<TestPercept>> {
        (0..16)
            .map(|i| PerceptionMessage {
                percept: TestPercept::Sound,
                base_value: 50.0,
                origin: Vec2::new(i as f32 * 40.0, 0.0),
                range: 30.0,
                source: None,
            })
            .collect()
    }

    fn build(batch: &[PerceptionMessage<TestPercept>]) -> MessageIndex {
        let mut index = MessageIndex::default();
        let mut ranges = Vec::new();
        assert!(index.build(batch, &mut ranges));
        index
    }

    #[test]
    fn merges_identical_origins_keeping_strongest() {
        let origin = Vec2::new(3.0, 4.0);
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, origin, None),
                message(TestPercept::Sound, 25.0, origin, None),
                message(TestPercept::Sound, 15.0, origin, None),
            ],
            0.0,
        );

        assert_eq!(batch.len(), 1);
        assert!((batch[0].base_value - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn keeps_messages_with_different_percepts_apart() {
        let origin = Vec2::new(1.0, 1.0);
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, origin, None),
                message(TestPercept::Pain, 20.0, origin, None),
            ],
            0.0,
        );

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn keeps_messages_from_different_sources_apart() {
        let origin = Vec2::ZERO;
        let a = Entity::from_raw_u32(1).unwrap();
        let b = Entity::from_raw_u32(2).unwrap();
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, origin, Some(a)),
                message(TestPercept::Sound, 20.0, origin, Some(b)),
            ],
            0.0,
        );

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn merge_radius_groups_nearby_origins() {
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, Vec2::new(0.1, 0.1), None),
                message(TestPercept::Sound, 30.0, Vec2::new(0.9, 0.9), None),
            ],
            4.0,
        );

        assert_eq!(batch.len(), 1);
        assert!((batch[0].base_value - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn negative_merge_radius_disables_merging() {
        let origin = Vec2::ZERO;
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, origin, None),
                message(TestPercept::Sound, 20.0, origin, None),
            ],
            -1.0,
        );

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn merging_preserves_batch_order() {
        let repeated = Vec2::new(5.0, 5.0);
        let batch = merge(
            vec![
                message(TestPercept::Sound, 10.0, repeated, None),
                message(TestPercept::Sound, 40.0, Vec2::new(80.0, 0.0), None),
                message(TestPercept::Sound, 30.0, repeated, None),
            ],
            0.0,
        );

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].origin, repeated);
        assert!((batch[0].base_value - 30.0).abs() < f32::EPSILON);
        assert_eq!(batch[1].origin, Vec2::new(80.0, 0.0));
    }

    #[test]
    fn index_reports_every_message_reaching_a_cell() {
        let batch = scattered_batch();
        let index = build(&batch);

        let mut candidates = Vec::new();
        for probe in [Vec2::ZERO, Vec2::new(41.0, 5.0), Vec2::new(600.0, -10.0)] {
            index.candidates_at(probe, &mut candidates);
            let expected = batch.iter().enumerate().filter(|(_, message)| {
                probe.distance_squared(message.origin) <= message.range * message.range
            });
            for (message_index, _) in expected {
                assert!(
                    candidates.contains(&(message_index as u32)),
                    "message {message_index} must be a candidate at {probe}"
                );
            }
        }
    }

    #[test]
    fn wide_messages_are_candidates_everywhere() {
        let mut batch = scattered_batch();
        batch.push(PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 50.0,
            origin: Vec2::ZERO,
            range: 100_000.0,
            source: None,
        });
        let wide_index = batch.len() as u32 - 1;

        let index = build(&batch);
        assert_eq!(index.wide, vec![wide_index]);

        let mut candidates = Vec::new();
        index.candidates_at(Vec2::new(9_000.0, 9_000.0), &mut candidates);
        assert_eq!(candidates, vec![wide_index]);
    }

    #[test]
    fn candidates_stay_in_batch_order() {
        let mut batch = scattered_batch();
        // Cluster the origins so one cell holds several messages, and make one
        // of them wide so both candidate sources contribute.
        for (message_index, message) in batch.iter_mut().enumerate() {
            message.origin = Vec2::new(message_index as f32, 0.0);
        }
        batch[3].range = 100_000.0;

        let index = build(&batch);
        let mut candidates = Vec::new();
        index.candidates_at(Vec2::ZERO, &mut candidates);

        assert!(candidates.len() > 1);
        assert!(
            candidates.windows(2).all(|pair| pair[0] < pair[1]),
            "candidates must be ascending: {candidates:?}"
        );
    }

    #[test]
    fn rebuilding_reuses_buckets_without_stale_entries() {
        let batch = scattered_batch();
        let mut index = MessageIndex::default();
        let mut ranges = Vec::new();
        assert!(index.build(&batch, &mut ranges));
        let bucket_capacity = index.buckets.len();

        let smaller = &batch[..8];
        assert!(index.build(smaller, &mut ranges));
        assert!(
            index.buckets.len() >= bucket_capacity,
            "buckets are retained"
        );

        let mut candidates = Vec::new();
        index.candidates_at(Vec2::new(600.0, 0.0), &mut candidates);
        assert!(
            candidates.iter().all(|&i| (i as usize) < smaller.len()),
            "no indices from the previous build survive: {candidates:?}"
        );
    }
}
