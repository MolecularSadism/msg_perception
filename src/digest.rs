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
use bevy::platform::collections::hash_map::Entry;
use bevy::prelude::*;

#[cfg(doc)]
use crate::PerceptionPlugin;
use crate::{LocationKnowledge, Memory, Percept, PerceptionSystems};

/// Summary of all memories for a single percept type.
///
/// The `strongest_*` fields always describe one real entry — the one with the
/// highest value — regardless of sign, matching [`Memory::strongest`].
#[derive(Debug, Clone, Default, PartialEq, Reflect)]
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
/// Requires [`Memory<P>`], so spawning a digest on its own inserts an empty
/// memory alongside it. Rebuilt by [`update_memory_digest`], which
/// [`MemoryDigestPlugin<P>`] schedules after decay and propagation; the
/// component is only written when the summary actually differs, so
/// `Changed<MemoryDigest<P>>` stays a meaningful filter for host systems.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[require(Memory::<P>)]
pub struct MemoryDigest<P: Percept + Eq + Hash> {
    /// Per-type summaries, keyed by percept type.
    pub digests: HashMap<P, PerceptDigest>,
}

impl<P: Percept + Eq + Hash> Default for MemoryDigest<P> {
    fn default() -> Self {
        Self {
            digests: HashMap::new(),
        }
    }
}

impl<P: Percept + Eq + Hash> MemoryDigest<P> {
    /// Get the digest for a specific percept type, if any memories of that type exist.
    pub fn get(&self, percept: &P) -> Option<&PerceptDigest> {
        self.digests.get(percept)
    }

    /// The strongest value across all percept types, or `None` when the
    /// digest is empty.
    pub fn strongest_value(&self) -> Option<f32> {
        self.digests
            .values()
            .map(|d| d.strongest_value)
            .max_by(f32::total_cmp)
    }

    /// The total value across all percept types.
    pub fn total_value(&self) -> f32 {
        self.digests.values().map(|d| d.total_value).sum()
    }
}

impl<P: Percept + Eq + Hash> Memory<P> {
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
            match digests.entry(entry.percept.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(PerceptDigest {
                        strongest_value: entry.value,
                        strongest_location: entry.location.clone(),
                        strongest_source: entry.source,
                        total_value: entry.value,
                    });
                }
                Entry::Occupied(slot) => {
                    let d = slot.into_mut();
                    d.total_value += entry.value;

                    if entry.value > d.strongest_value {
                        d.strongest_value = entry.value;
                        d.strongest_location = entry.location.clone();
                        d.strongest_source = entry.source;
                    }
                }
            }
        }
    }
}

/// Rebuilds each [`MemoryDigest<P>`] from the raw [`Memory<P>`] entries.
///
/// The rebuilt summary is compared against the stored one and the component
/// is only written when they differ, so change detection on
/// [`MemoryDigest<P>`] only fires when the summary actually changed.
pub fn update_memory_digest<P: Percept + Eq + Hash>(
    mut q: Query<(&mut MemoryDigest<P>, &Memory<P>)>,
    mut scratch: Local<HashMap<P, PerceptDigest>>,
) {
    for (mut digest, memory) in &mut q {
        // Idle actors with no memories and an already-empty digest need no work.
        if digest.digests.is_empty() && memory.is_empty() {
            continue;
        }
        memory.digest_into(&mut scratch);
        if *scratch != digest.digests {
            std::mem::swap(&mut digest.digests, &mut *scratch);
        }
    }
}

/// Registers the per-type memory digest for percept type `P`.
///
/// Adds type registrations and [`update_memory_digest`] under
/// [`PerceptionSystems::Digest`], scheduled after decay and propagation.
/// The digest summarizes memory as of [`PerceptionSystems::Digest`]; host
/// systems should schedule `.after(PerceptionSystems::Digest)` to read the
/// current frame's summary. [`PerceptionEvent`] observers that mutate
/// [`Memory<P>`] later in the frame appear in the next frame's digest.
///
/// [`PerceptionEvent`]: crate::PerceptionEvent
pub struct MemoryDigestPlugin<P: Percept + Eq + Hash> {
    _marker: std::marker::PhantomData<P>,
}

impl<P: Percept + Eq + Hash> Default for MemoryDigestPlugin<P> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<P: Percept + Eq + Hash> Plugin for MemoryDigestPlugin<P> {
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
        assert!(digest.strongest_value().is_none());
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

    #[test]
    fn digest_all_negative_values_tracks_real_entry() {
        let mut world = World::new();
        let attacker = world.spawn_empty().id();

        let mut memory = Memory::default();
        memory.push(
            MemoryEntry {
                percept: TestPercept::Pain,
                value: -3.0,
                location: None,
                source: None,
            },
            16,
        );
        memory.push(
            MemoryEntry {
                percept: TestPercept::Pain,
                value: -1.0,
                location: Some(LocationKnowledge::Origin(Vec2::new(2.0, 0.0))),
                source: Some(attacker),
            },
            16,
        );

        let digests = memory.digest();
        let pain = &digests[&TestPercept::Pain];
        assert!((pain.strongest_value - -1.0).abs() < f32::EPSILON);
        assert_eq!(
            pain.strongest_location,
            Some(LocationKnowledge::Origin(Vec2::new(2.0, 0.0)))
        );
        assert_eq!(pain.strongest_source, Some(attacker));
        assert!((pain.total_value - -4.0).abs() < f32::EPSILON);

        let digest = MemoryDigest { digests };
        assert_eq!(digest.strongest_value(), Some(-1.0));
    }

    #[test]
    fn digest_works_with_non_copy_percepts() {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect)]
        struct Named(String);

        let mut memory = Memory::default();
        memory.push(
            MemoryEntry {
                percept: Named("footsteps".to_string()),
                value: 4.0,
                location: None,
                source: None,
            },
            16,
        );

        let digests = memory.digest();
        let footsteps = &digests[&Named("footsteps".to_string())];
        assert!((footsteps.strongest_value - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn digest_requires_memory() {
        let mut app = create_test_app();

        let entity = app
            .world_mut()
            .spawn(MemoryDigest::<TestPercept>::default())
            .id();

        assert!(app.world().entity(entity).contains::<Memory<TestPercept>>());
    }

    #[test]
    fn digest_cleared_when_memory_becomes_empty() {
        let mut app = create_test_app();

        let mut memory = Memory::default();
        memory.push(
            MemoryEntry {
                percept: TestPercept::Sound,
                value: 5.0,
                location: None,
                source: None,
            },
            16,
        );

        let entity = app
            .world_mut()
            .spawn((MemoryDigest::<TestPercept>::default(), memory))
            .id();

        app.update();

        assert!(
            !app.world()
                .entity(entity)
                .get::<MemoryDigest<TestPercept>>()
                .unwrap()
                .digests
                .is_empty()
        );

        app.world_mut()
            .entity_mut(entity)
            .get_mut::<Memory<TestPercept>>()
            .unwrap()
            .decay(10.0, 0.5);
        app.update();

        let digest = app
            .world()
            .entity(entity)
            .get::<MemoryDigest<TestPercept>>()
            .unwrap();
        assert!(digest.digests.is_empty());
    }

    #[test]
    fn digest_not_dirtied_when_memory_static() {
        #[derive(Resource, Default)]
        struct ChangeCount(usize);

        fn count_changes(
            q: Query<(), Changed<MemoryDigest<TestPercept>>>,
            mut count: ResMut<ChangeCount>,
        ) {
            count.0 += q.iter().count();
        }

        let mut app = create_test_app();
        app.init_resource::<ChangeCount>();
        app.add_systems(Update, count_changes.after(PerceptionSystems::Digest));

        let mut memory = Memory::default();
        memory.push(
            MemoryEntry {
                percept: TestPercept::Sound,
                value: 5.0,
                location: None,
                source: None,
            },
            16,
        );
        app.world_mut()
            .spawn((MemoryDigest::<TestPercept>::default(), memory));

        // First update sees the fresh spawn; the memory never changes after
        // that, so the digest must not be rewritten on later updates.
        app.update();
        app.update();
        app.update();

        assert_eq!(app.world().resource::<ChangeCount>().0, 1);
    }
}
