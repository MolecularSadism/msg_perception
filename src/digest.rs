//! Per-type memory digest that summarizes the [`Memory`] component.
//!
//! The [`MemoryDigest<P>`] component provides a pre-computed summary of an
//! actor's memories grouped by percept type. For each type it tracks the
//! strongest memory's value, the strongest memory's location and source, and
//! the total accumulated value across all memories of that type.
//!
//! Host systems read this instead of iterating raw memory entries. Add
//! [`MemoryDigestPlugin<P>`] alongside [`PerceptionPlugin<P>`] and spawn a
//! [`MemoryDigest<P>`] next to each [`Memory<P>`] that should be summarized.

use std::hash::Hash;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

#[cfg(doc)]
use crate::PerceptionPlugin;
use crate::{LocationKnowledge, Memory, Percept, PerceptionSystems};

/// Summary of all memories for a single percept type.
#[derive(Debug, Clone, Default, Reflect)]
pub struct PerceptDigest {
    /// The value of the strongest (highest-value) memory of this type.
    pub strongest_value: f32,
    /// Location knowledge from the strongest memory, if any.
    pub strongest_location: Option<LocationKnowledge>,
    /// The entity that caused the strongest memory, if known.
    pub strongest_source: Option<Entity>,
    /// Sum of all memory values for this type.
    pub total_value: f32,
}

/// Component that holds a per-type summary of an actor's [`Memory<P>`].
///
/// Updated each frame by [`update_memory_digest`], which
/// [`MemoryDigestPlugin<P>`] schedules after decay and propagation.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
pub struct MemoryDigest<P: Percept + Eq + Hash + Copy> {
    /// Per-type summaries, keyed by percept type.
    pub digests: HashMap<P, PerceptDigest>,
}

impl<P: Percept + Eq + Hash + Copy> Default for MemoryDigest<P> {
    fn default() -> Self {
        Self {
            digests: HashMap::new(),
        }
    }
}

impl<P: Percept + Eq + Hash + Copy> MemoryDigest<P> {
    /// Get the digest for a specific percept type, if any memories of that type exist.
    pub fn get(&self, percept: &P) -> Option<&PerceptDigest> {
        self.digests.get(percept)
    }

    /// The strongest value across all percept types.
    pub fn strongest_value(&self) -> f32 {
        self.digests
            .values()
            .map(|d| d.strongest_value)
            .fold(0.0_f32, f32::max)
    }

    /// The total value across all percept types.
    pub fn total_value(&self) -> f32 {
        self.digests.values().map(|d| d.total_value).sum()
    }
}

impl<P: Percept + Eq + Hash + Copy> Memory<P> {
    /// Summarize the current entries grouped by percept type.
    ///
    /// For each type present, the result tracks the strongest entry's value,
    /// location and source, plus the summed value across all entries of that
    /// type.
    ///
    /// ```
    /// use msg_perception::prelude::*;
    /// use bevy::prelude::*;
    ///
    /// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
    /// struct Sound;
    ///
    /// let mut memory = Memory::<Sound>::default();
    /// memory.push(
    ///     MemoryEntry { percept: Sound, value: 10.0, location: None, source: None },
    ///     16,
    /// );
    /// memory.push(
    ///     MemoryEntry { percept: Sound, value: 3.0, location: None, source: None },
    ///     16,
    /// );
    ///
    /// let digests = memory.digest();
    /// assert_eq!(digests[&Sound].strongest_value, 10.0);
    /// assert_eq!(digests[&Sound].total_value, 13.0);
    /// ```
    pub fn digest(&self) -> HashMap<P, PerceptDigest> {
        let mut digests = HashMap::new();
        self.digest_into(&mut digests);
        digests
    }

    /// Clear `digests` and refill it with the summary of the current entries.
    ///
    /// Like [`Memory::digest`], but reuses the map's allocation.
    pub fn digest_into(&self, digests: &mut HashMap<P, PerceptDigest>) {
        digests.clear();

        for entry in self.iter() {
            let d = digests.entry(entry.percept).or_default();
            d.total_value += entry.value;

            if entry.value > d.strongest_value {
                d.strongest_value = entry.value;
                d.strongest_location = entry.location.clone();
                d.strongest_source = entry.source;
            }
        }
    }
}

/// Rebuilds each [`MemoryDigest<P>`] from the raw [`Memory<P>`] entries.
pub fn update_memory_digest<P: Percept + Eq + Hash + Copy>(
    mut q: Query<(&mut MemoryDigest<P>, &Memory<P>)>,
) {
    for (mut digest, memory) in &mut q {
        // Idle actors with no memories and an already-empty digest need no work.
        if digest.digests.is_empty() && memory.is_empty() {
            continue;
        }
        let digest = &mut *digest;
        memory.digest_into(&mut digest.digests);
    }
}

/// Registers the per-type memory digest for percept type `P`.
///
/// Adds type registrations and [`update_memory_digest`] under
/// [`PerceptionSystems::Digest`], scheduled after decay and propagation so
/// the digest summarizes the frame's final memory state.
pub struct MemoryDigestPlugin<P: Percept + Eq + Hash + Copy> {
    _marker: std::marker::PhantomData<P>,
}

impl<P: Percept + Eq + Hash + Copy> Default for MemoryDigestPlugin<P> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<P: Percept + Eq + Hash + Copy> Plugin for MemoryDigestPlugin<P> {
    fn build(&self, app: &mut App) {
        app.register_type::<PerceptDigest>();
        app.register_type::<MemoryDigest<P>>();

        app.configure_sets(
            Update,
            PerceptionSystems::Digest
                .after(PerceptionSystems::Decay)
                .after(PerceptionSystems::Propagate),
        );
        app.add_systems(
            Update,
            update_memory_digest::<P>.in_set(PerceptionSystems::Digest),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryEntry;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
    enum TestPercept {
        Sound,
        Pain,
    }

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(MemoryDigestPlugin::<TestPercept>::default());
        app.register_type::<Memory<TestPercept>>();
        app
    }

    #[test]
    fn digest_groups_by_percept_type() {
        let mut app = create_test_app();

        let attacker = app.world_mut().spawn_empty().id();

        let mut memory = Memory::default();
        memory.push(
            MemoryEntry {
                percept: TestPercept::Sound,
                value: 10.0,
                location: Some(LocationKnowledge::Origin(Vec2::new(5.0, 0.0))),
                source: None,
            },
            16,
        );
        memory.push(
            MemoryEntry {
                percept: TestPercept::Sound,
                value: 3.0,
                location: None,
                source: None,
            },
            16,
        );
        memory.push(
            MemoryEntry {
                percept: TestPercept::Pain,
                value: 7.0,
                location: Some(LocationKnowledge::Direction(Dir2::X)),
                source: Some(attacker),
            },
            16,
        );

        let entity = app
            .world_mut()
            .spawn((MemoryDigest::<TestPercept>::default(), memory))
            .id();

        app.update();

        let digest = app
            .world()
            .entity(entity)
            .get::<MemoryDigest<TestPercept>>()
            .unwrap();

        let sound = digest.get(&TestPercept::Sound).unwrap();
        assert!((sound.strongest_value - 10.0).abs() < f32::EPSILON);
        assert!((sound.total_value - 13.0).abs() < f32::EPSILON);
        assert_eq!(
            sound.strongest_location,
            Some(LocationKnowledge::Origin(Vec2::new(5.0, 0.0)))
        );
        assert!(sound.strongest_source.is_none());

        let pain = digest.get(&TestPercept::Pain).unwrap();
        assert!((pain.strongest_value - 7.0).abs() < f32::EPSILON);
        assert!((pain.total_value - 7.0).abs() < f32::EPSILON);
        assert_eq!(pain.strongest_source, Some(attacker));
    }

    #[test]
    fn digest_empty_with_no_memories() {
        let mut app = create_test_app();

        let entity = app
            .world_mut()
            .spawn((
                MemoryDigest::<TestPercept>::default(),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.update();

        let digest = app
            .world()
            .entity(entity)
            .get::<MemoryDigest<TestPercept>>()
            .unwrap();
        assert!(digest.digests.is_empty());
        assert!((digest.strongest_value()).abs() < f32::EPSILON);
        assert!((digest.total_value()).abs() < f32::EPSILON);
    }

    #[test]
    fn digest_updates_when_memories_change() {
        let mut app = create_test_app();

        let entity = app
            .world_mut()
            .spawn((
                MemoryDigest::<TestPercept>::default(),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.update();

        // Add a memory after initial update
        app.world_mut()
            .entity_mut(entity)
            .get_mut::<Memory<TestPercept>>()
            .unwrap()
            .push(
                MemoryEntry {
                    percept: TestPercept::Pain,
                    value: 5.0,
                    location: None,
                    source: None,
                },
                16,
            );
        app.update();

        let digest = app
            .world()
            .entity(entity)
            .get::<MemoryDigest<TestPercept>>()
            .unwrap();
        let pain = digest.get(&TestPercept::Pain).unwrap();
        assert!((pain.strongest_value - 5.0).abs() < f32::EPSILON);
    }
}
