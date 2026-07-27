//! Generic perception and memory system for Bevy.
//!
//! Environmental stimuli are stored as [`MemoryEntry<P>`] values inside a
//! [`Memory<P>`] component. The generic parameter `P` defines the percept
//! type (e.g. Sound, Pain). Each entry decays over time and is removed once
//! it falls below a configurable threshold.
//!
//! ## Two stimulus paths
//!
//! - **[`PerceptionMessage<P>`]** (broadcast): Spatial stimulus that propagates to all [`Memory`]
//!   holders within range, attenuated by distance (20·log10).
//! - **[`PerceptionEvent<P>`]** (entity-targeted): Direct stimulus fired via `trigger` on a
//!   specific entity, no distance attenuation.
//!
//! ## Source tracking
//!
//! Both stimulus paths carry an optional `source` entity that identifies who
//! caused the stimulus. The propagation system automatically filters out
//! self-caused perceptions (where `source == perceiver`).
//!
//! ## Propagation cost
//!
//! [`propagate_perception`] resolves every broadcast of a frame against every
//! [`Memory`] holder, so its cost grows with the product of the two. It culls
//! pairs on squared distance, merges same-frame broadcasts that share a percept,
//! source, range and origin cell (see [`PerceptionConfig::merge_radius`]), and
//! past a pair-count threshold buckets the batch into a uniform grid so each
//! perceiver only tests the messages that can reach it.
//!
//! ## Usage
//!
//! ```
//! use bevy::prelude::*;
//! use msg_perception::prelude::*;
//!
//! #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
//! enum MyPercept { Sound, Pain }
//!
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! app.add_plugins(PerceptionPlugin::<MyPercept>::default());
//! app.update();
//! ```

use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;

use bevy::prelude::*;

mod propagation;

pub use propagation::{PropagationScratch, propagate_perception};

// ─── Percept trait ───────────────────────────────────────────────────────────

/// Marker trait for percept types used as the generic parameter in the
/// perception system. Automatically implemented for any qualifying type.
///
/// Any type that derives `Reflect` (and `Debug + Clone`) satisfies this trait.
pub trait Percept:
    Debug
    + Clone
    + Reflect
    + FromReflect
    + TypePath
    + bevy::reflect::GetTypeRegistration
    + bevy::reflect::Typed
{
}

impl<
    T: Debug
        + Clone
        + Reflect
        + FromReflect
        + TypePath
        + bevy::reflect::GetTypeRegistration
        + bevy::reflect::Typed,
> Percept for T
{
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Resource controlling memory decay and capacity.
#[derive(Resource, Clone, Debug)]
pub struct PerceptionConfig {
    /// How fast memory values decay per second.
    pub decay_rate: f32,
    /// Entries with value at or below this threshold are removed.
    pub min_threshold: f32,
    /// Maximum number of memory entries per entity.
    pub max_count: usize,
    /// Origin grid size used to collapse same-frame [`PerceptionMessage`]s.
    ///
    /// Messages of one frame that share a percept, a source, a range, and an
    /// origin cell of this size become a single memory carrying the strongest
    /// of their base values, so a burst of impacts from one spot does not flush
    /// an actor's whole memory ring. `0.0` merges only bit-identical origins;
    /// a negative value disables merging.
    pub merge_radius: f32,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            decay_rate: 5.0,
            min_threshold: 0.5,
            max_count: 16,
            merge_radius: 0.0,
        }
    }
}

// ─── LocationKnowledge ───────────────────────────────────────────────────────

/// Spatial knowledge about a perceived stimulus.
#[derive(Debug, Clone, PartialEq, Reflect)]
pub enum LocationKnowledge {
    /// Exact world position of the stimulus source.
    Origin(Vec2),
    /// Normalized direction toward the stimulus source (position unknown).
    Direction(Dir2),
}

impl LocationKnowledge {
    /// Get the direction from `pos` toward this knowledge source.
    pub fn direction_from(&self, pos: Vec2) -> Option<Dir2> {
        match self {
            LocationKnowledge::Origin(origin) => Dir2::new(*origin - pos).ok(),
            LocationKnowledge::Direction(dir) => Some(*dir),
        }
    }

    /// Get the origin position, if known.
    pub fn origin(&self) -> Option<Vec2> {
        match self {
            LocationKnowledge::Origin(pos) => Some(*pos),
            LocationKnowledge::Direction(_) => None,
        }
    }
}

// ─── MemoryEntry ─────────────────────────────────────────────────────────────

/// A single memory of a perceived stimulus.
#[derive(Debug, Clone, Reflect)]
pub struct MemoryEntry<P: Percept> {
    /// What kind of stimulus created this memory.
    pub percept: P,
    /// Current intensity/importance (decays over time).
    pub value: f32,
    /// Spatial knowledge about the stimulus source.
    pub location: Option<LocationKnowledge>,
    /// The entity that caused this stimulus, if known.
    pub source: Option<Entity>,
}

// ─── Memory ──────────────────────────────────────────────────────────────────

/// Component that stores an actor's short-term memory of perceived stimuli.
///
/// Entries are ordered newest-first. The system automatically decays values
/// and removes entries that fall below the configured threshold.
///
/// ```
/// use bevy::prelude::*;
/// use msg_perception::prelude::*;
///
/// #[derive(Debug, Clone, Copy, Reflect)]
/// struct Pain;
///
/// let mut memory = Memory::<Pain>::default();
/// memory.push(
///     MemoryEntry { percept: Pain, value: 10.0, location: None, source: None },
///     16,
/// );
/// assert_eq!(memory.len(), 1);
/// ```
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
pub struct Memory<P: Percept> {
    entries: VecDeque<MemoryEntry<P>>,
}

impl<P: Percept> Default for Memory<P> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }
}

impl<P: Percept> Memory<P> {
    /// Push a new memory entry, evicting the oldest if at capacity.
    pub fn push(&mut self, entry: MemoryEntry<P>, max_count: usize) {
        if max_count > 0 && self.entries.len() >= max_count {
            self.entries.pop_back();
        }
        self.entries.push_front(entry);
    }

    /// Sum of all current memory values.
    pub fn sum(&self) -> f32 {
        self.entries.iter().map(|e| e.value).sum()
    }

    /// The entry with the highest current value, if any.
    pub fn strongest(&self) -> Option<&MemoryEntry<P>> {
        self.entries.iter().max_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &MemoryEntry<P>> {
        self.entries.iter()
    }

    /// Number of active memories.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no active memories.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Decay all entries by `amount`, removing those that fall below `threshold`.
    pub fn decay(&mut self, amount: f32, threshold: f32) {
        for entry in &mut self.entries {
            entry.value -= amount;
        }
        self.entries.retain(|e| e.value > threshold);
    }
}

// ─── Events ──────────────────────────────────────────────────────────────────

/// A broadcast message fired when a spatial stimulus occurs.
///
/// The propagation system finds all [`Memory`] holders within `range`
/// and inserts a [`MemoryEntry`] with intensity attenuated by distance.
/// When `source` matches the perceiving entity, the stimulus is ignored.
#[derive(Message, Debug, Clone)]
pub struct PerceptionMessage<P: Percept> {
    /// The perceived stimulus.
    pub percept: P,
    /// Base intensity before distance attenuation.
    pub base_value: f32,
    /// World position where the stimulus originated.
    pub origin: Vec2,
    /// Maximum propagation radius.
    pub range: f32,
    /// The entity that caused this stimulus, if known.
    /// Self-caused perceptions (source == perceiver) are automatically filtered.
    pub source: Option<Entity>,
}

/// An entity-targeted perception event fired via `commands.trigger()`.
///
/// Used for direct stimuli (e.g. pain) where only the targeted entity should
/// receive the memory, with no spatial propagation or distance attenuation.
#[derive(EntityEvent, Debug, Clone)]
pub struct PerceptionEvent<P: Percept> {
    /// The entity receiving the perception.
    pub entity: Entity,
    /// The perceived stimulus.
    pub percept: P,
    /// Intensity of the stimulus (stored directly as memory value).
    pub value: f32,
    /// Spatial knowledge about the stimulus source.
    pub location: Option<LocationKnowledge>,
    /// The entity that caused this stimulus, if known.
    pub source: Option<Entity>,
}

// ─── System sets ─────────────────────────────────────────────────────────────

/// System sets for scheduling perception systems within the host application.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PerceptionSystems {
    /// Memory decay (runs per frame).
    Decay,
    /// Broadcast message propagation.
    Propagate,
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// Observer that handles entity-targeted [`PerceptionEvent`] triggers.
///
/// Inserts a [`MemoryEntry`] directly on the targeted entity's [`Memory`]
/// with no distance attenuation.
pub fn on_perception_event<P: Percept>(
    trigger: On<PerceptionEvent<P>>,
    mut q_memory: Query<&mut Memory<P>>,
    config: Res<PerceptionConfig>,
) {
    let Ok(mut memory) = q_memory.get_mut(trigger.entity) else {
        return;
    };

    let event = trigger.event();

    if event.value <= config.min_threshold {
        return;
    }

    memory.push(
        MemoryEntry {
            percept: event.percept.clone(),
            value: event.value,
            location: event.location.clone(),
            source: event.source,
        },
        config.max_count,
    );
}

/// Decay all memory entries every frame, removing those below threshold.
pub fn decay_memories<P: Percept>(
    time: Res<Time>,
    mut q_memory: Query<&mut Memory<P>>,
    config: Res<PerceptionConfig>,
) {
    let dt = time.delta_secs();
    let decay_amount = config.decay_rate * dt;
    let threshold = config.min_threshold;

    for mut memory in &mut q_memory {
        memory.decay(decay_amount, threshold);
    }
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

/// Registers the generic perception system for percept type `P`.
///
/// Adds type registrations, the config resource, message/event handling,
/// and the decay + propagation systems under [`PerceptionSystems`] sets.
pub struct PerceptionPlugin<P: Percept> {
    _marker: PhantomData<P>,
}

impl<P: Percept> Default for PerceptionPlugin<P> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<P: Percept> Plugin for PerceptionPlugin<P> {
    fn build(&self, app: &mut App) {
        app.register_type::<LocationKnowledge>();
        app.register_type::<MemoryEntry<P>>();
        app.register_type::<Memory<P>>();
        app.init_resource::<PerceptionConfig>();
        app.add_message::<PerceptionMessage<P>>();

        app.add_observer(on_perception_event::<P>);

        app.add_systems(Update, decay_memories::<P>.in_set(PerceptionSystems::Decay));
        app.add_systems(
            Update,
            propagate_perception::<P>.in_set(PerceptionSystems::Propagate),
        );
    }
}

// ─── Prelude ─────────────────────────────────────────────────────────────────

pub mod prelude {
    pub use crate::{
        LocationKnowledge, Memory, MemoryEntry, Percept, PerceptionConfig, PerceptionEvent,
        PerceptionMessage, PerceptionPlugin, PerceptionSystems,
    };
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

    fn make_entry(percept: TestPercept, value: f32) -> MemoryEntry<TestPercept> {
        MemoryEntry {
            percept,
            value,
            location: None,
            source: None,
        }
    }

    // ── Memory unit tests ────────────────────────────────────────────────

    #[test]
    fn push_and_sum() {
        let mut memory = Memory::<TestPercept>::default();
        memory.push(make_entry(TestPercept::Sound, 10.0), 16);
        memory.push(make_entry(TestPercept::Sound, 5.0), 16);
        assert_eq!(memory.len(), 2);
        assert!((memory.sum() - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn push_evicts_oldest_at_capacity() {
        let mut memory = Memory::<TestPercept>::default();
        memory.push(make_entry(TestPercept::Sound, 1.0), 2);
        memory.push(make_entry(TestPercept::Sound, 2.0), 2);
        memory.push(make_entry(TestPercept::Sound, 3.0), 2);
        assert_eq!(memory.len(), 2);
        assert!((memory.sum() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decay_removes_below_threshold() {
        let mut memory = Memory::<TestPercept>::default();
        memory.push(make_entry(TestPercept::Sound, 5.0), 16);
        memory.push(make_entry(TestPercept::Sound, 1.0), 16);
        memory.decay(2.0, 1.0);
        assert_eq!(memory.len(), 1);
        assert!((memory.sum() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn strongest_returns_highest() {
        let mut memory = Memory::<TestPercept>::default();
        memory.push(make_entry(TestPercept::Sound, 3.0), 16);
        memory.push(make_entry(TestPercept::Sound, 7.0), 16);
        memory.push(make_entry(TestPercept::Sound, 5.0), 16);
        let strongest = memory.strongest().unwrap();
        assert!((strongest.value - 7.0).abs() < f32::EPSILON);
    }

    // ── LocationKnowledge tests ──────────────────────────────────────────

    #[test]
    fn location_knowledge_origin_direction() {
        let loc = LocationKnowledge::Origin(Vec2::new(10.0, 0.0));
        let dir = loc.direction_from(Vec2::ZERO).unwrap();
        assert!((dir.x - 1.0).abs() < f32::EPSILON);
        assert!(dir.y.abs() < f32::EPSILON);
        assert_eq!(loc.origin(), Some(Vec2::new(10.0, 0.0)));
    }

    #[test]
    fn location_knowledge_direction_only() {
        let loc = LocationKnowledge::Direction(Dir2::Y);
        let dir = loc.direction_from(Vec2::ZERO).unwrap();
        assert!((dir.y - 1.0).abs() < f32::EPSILON);
        assert_eq!(loc.origin(), None);
    }

    // ── Integration tests ────────────────────────────────────────────────

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(PerceptionPlugin::<TestPercept>::default());
        app
    }

    #[derive(Resource)]
    struct PendingMessages(Vec<PerceptionMessage<TestPercept>>);

    fn fire_pending_messages(
        mut pending: ResMut<PendingMessages>,
        mut writer: MessageWriter<PerceptionMessage<TestPercept>>,
    ) {
        for msg in pending.0.drain(..) {
            writer.write(msg);
        }
    }

    #[test]
    fn perception_message_creates_memory() {
        let mut app = create_test_app();

        let perceiver = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::ZERO),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.insert_resource(PendingMessages(vec![PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 40.0,
            origin: Vec2::new(10.0, 0.0),
            range: 100.0,
            source: None,
        }]));

        app.add_systems(
            Update,
            fire_pending_messages.before(PerceptionSystems::Propagate),
        );
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        assert_eq!(memory.len(), 1);
        let entry = memory.strongest().unwrap();
        assert_eq!(entry.percept, TestPercept::Sound);
        // 40 - 20*log10(10) = 40 - 20 = 20
        assert!((entry.value - 20.0).abs() < 0.1);
        assert!(entry.source.is_none());
    }

    #[test]
    fn perception_message_out_of_range_ignored() {
        let mut app = create_test_app();

        let perceiver = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::ZERO),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.insert_resource(PendingMessages(vec![PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 40.0,
            origin: Vec2::new(200.0, 0.0),
            range: 100.0,
            source: None,
        }]));

        app.add_systems(
            Update,
            fire_pending_messages.before(PerceptionSystems::Propagate),
        );
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        assert!(memory.is_empty());
    }

    #[test]
    fn logarithmic_attenuation_at_close_range() {
        let mut app = create_test_app();

        let perceiver = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.insert_resource(PendingMessages(vec![PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 30.0,
            origin: Vec2::ZERO,
            range: 100.0,
            source: None,
        }]));

        app.add_systems(
            Update,
            fire_pending_messages.before(PerceptionSystems::Propagate),
        );
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        let entry = memory.strongest().unwrap();
        assert!((entry.value - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn multiple_messages_accumulate() {
        let mut app = create_test_app();

        let perceiver = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::ZERO),
                Memory::<TestPercept>::default(),
            ))
            .id();

        app.insert_resource(PendingMessages(vec![
            PerceptionMessage {
                percept: TestPercept::Sound,
                base_value: 30.0,
                origin: Vec2::new(1.0, 0.0),
                range: 100.0,
                source: None,
            },
            PerceptionMessage {
                percept: TestPercept::Sound,
                base_value: 20.0,
                origin: Vec2::new(0.0, 1.0),
                range: 100.0,
                source: None,
            },
        ]));

        app.add_systems(
            Update,
            fire_pending_messages.before(PerceptionSystems::Propagate),
        );
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        assert_eq!(memory.len(), 2);
        assert!(memory.sum() > 0.0);
    }

    #[test]
    fn perception_event_inserts_memory_on_target() {
        let mut app = create_test_app();

        let perceiver = app.world_mut().spawn(Memory::<TestPercept>::default()).id();

        app.update();

        let origin = Vec2::new(50.0, 0.0);
        app.world_mut().commands().trigger(PerceptionEvent {
            entity: perceiver,
            percept: TestPercept::Pain,
            value: 15.0,
            location: Some(LocationKnowledge::Origin(origin)),
            source: None,
        });
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        assert_eq!(memory.len(), 1);
        let entry = memory.strongest().unwrap();
        assert_eq!(entry.percept, TestPercept::Pain);
        // Value may be slightly reduced by decay running in the same update tick
        assert!((entry.value - 15.0).abs() < 1.0);
        assert_eq!(entry.location, Some(LocationKnowledge::Origin(origin)));
    }

    #[test]
    fn perception_event_ignored_without_memory() {
        let mut app = create_test_app();

        let entity = app.world_mut().spawn_empty().id();

        app.update();

        app.world_mut().commands().trigger(PerceptionEvent {
            entity,
            percept: TestPercept::Pain,
            value: 10.0,
            location: None,
            source: None,
        });
        // Should not panic
        app.update();
    }

    #[test]
    fn decay_removes_weak_memories() {
        let mut memory = Memory::<TestPercept>::default();
        memory.push(make_entry(TestPercept::Pain, 2.0), 16);
        memory.decay(5.0, 0.5);
        assert!(memory.is_empty());
    }

    #[test]
    fn self_caused_message_filtered() {
        let mut app = create_test_app();

        let actor = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::ZERO),
                Memory::<TestPercept>::default(),
            ))
            .id();

        // Actor emits a sound at its own position — should not perceive it
        app.insert_resource(PendingMessages(vec![PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 40.0,
            origin: Vec2::ZERO,
            range: 100.0,
            source: Some(actor),
        }]));

        app.add_systems(
            Update,
            fire_pending_messages.before(PerceptionSystems::Propagate),
        );
        app.update();

        let memory = app
            .world()
            .entity(actor)
            .get::<Memory<TestPercept>>()
            .unwrap();
        assert!(
            memory.is_empty(),
            "Actor should not perceive its own sounds"
        );
    }

    // ── Propagation regression tests ─────────────────────────────────────

    /// Builds a world running propagation on its own, so memory values can be
    /// pinned without the decay system trimming them in the same run.
    fn propagation_world(config: PerceptionConfig) -> (World, Schedule) {
        let mut world = World::new();
        world.insert_resource(config);
        world.insert_resource(Messages::<PerceptionMessage<TestPercept>>::default());

        let mut schedule = Schedule::default();
        schedule.add_systems(propagate_perception::<TestPercept>);
        schedule.initialize(&mut world).unwrap();

        (world, schedule)
    }

    fn spawn_perceiver(world: &mut World, position: Vec2) -> Entity {
        world
            .spawn((
                Transform::from_translation(position.extend(0.0)),
                Memory::<TestPercept>::default(),
            ))
            .id()
    }

    fn write_message(world: &mut World, message: PerceptionMessage<TestPercept>) {
        world
            .resource_mut::<Messages<PerceptionMessage<TestPercept>>>()
            .write(message);
    }

    fn memory_values(world: &World, entity: Entity) -> Vec<f32> {
        world
            .entity(entity)
            .get::<Memory<TestPercept>>()
            .unwrap()
            .iter()
            .map(|entry| entry.value)
            .collect()
    }

    #[test]
    fn attenuation_matches_reference_values() {
        // `V = V_base - 20 * log10(distance)`, flat inside one unit.
        let cases = [
            (0.5, 30.0, 30.0),
            (1.0, 30.0, 30.0),
            (2.0, 30.0, 30.0 - 20.0 * 2.0f32.log10()),
            (10.0, 40.0, 20.0),
            (100.0, 60.0, 20.0),
            (1000.0, 90.0, 30.0),
        ];

        for (distance, base_value, expected) in cases {
            let (mut world, mut schedule) = propagation_world(PerceptionConfig::default());
            let perceiver = spawn_perceiver(&mut world, Vec2::new(distance, 0.0));
            write_message(
                &mut world,
                PerceptionMessage {
                    percept: TestPercept::Sound,
                    base_value,
                    origin: Vec2::ZERO,
                    range: 5000.0,
                    source: None,
                },
            );
            schedule.run(&mut world);

            let values = memory_values(&world, perceiver);
            assert_eq!(values.len(), 1, "distance {distance} should yield a memory");
            assert!(
                (values[0] - expected).abs() < 1e-3,
                "distance {distance}: expected {expected}, got {}",
                values[0]
            );
        }
    }

    #[test]
    fn range_boundary_is_exclusive_beyond_the_radius() {
        let (mut world, mut schedule) = propagation_world(PerceptionConfig::default());
        let inside = spawn_perceiver(&mut world, Vec2::new(99.0, 0.0));
        let outside = spawn_perceiver(&mut world, Vec2::new(101.0, 0.0));
        write_message(
            &mut world,
            PerceptionMessage {
                percept: TestPercept::Sound,
                base_value: 80.0,
                origin: Vec2::ZERO,
                range: 100.0,
                source: None,
            },
        );
        schedule.run(&mut world);

        assert_eq!(memory_values(&world, inside).len(), 1);
        assert!(memory_values(&world, outside).is_empty());
    }

    #[test]
    fn identical_origins_collapse_into_one_memory() {
        let (mut world, mut schedule) = propagation_world(PerceptionConfig::default());
        let perceiver = spawn_perceiver(&mut world, Vec2::new(10.0, 0.0));
        for base_value in [30.0, 45.0, 35.0] {
            write_message(
                &mut world,
                PerceptionMessage {
                    percept: TestPercept::Sound,
                    base_value,
                    origin: Vec2::ZERO,
                    range: 100.0,
                    source: None,
                },
            );
        }
        schedule.run(&mut world);

        let values = memory_values(&world, perceiver);
        assert_eq!(values.len(), 1, "a burst from one origin is one memory");
        // Strongest of the three, attenuated by 20*log10(10).
        assert!((values[0] - 25.0).abs() < 1e-3, "got {}", values[0]);
    }

    #[test]
    fn distinct_origins_stay_separate_memories() {
        let (mut world, mut schedule) = propagation_world(PerceptionConfig::default());
        let perceiver = spawn_perceiver(&mut world, Vec2::ZERO);
        for offset in [10.0, 20.0, 30.0] {
            write_message(
                &mut world,
                PerceptionMessage {
                    percept: TestPercept::Sound,
                    base_value: 60.0,
                    origin: Vec2::new(offset, 0.0),
                    range: 100.0,
                    source: None,
                },
            );
        }
        schedule.run(&mut world);

        assert_eq!(memory_values(&world, perceiver).len(), 3);
    }

    #[test]
    fn indexed_and_exhaustive_paths_agree() {
        // Enough messages and perceivers to cross the spatial-index threshold,
        // compared against a run kept below it by splitting the batch.
        let origins: Vec<Vec2> = (0..64)
            .map(|i| Vec2::new((i % 8) as f32 * 60.0, (i / 8) as f32 * 60.0))
            .collect();
        let positions: Vec<Vec2> = (0..96)
            .map(|i| Vec2::new((i % 12) as f32 * 40.0, (i / 12) as f32 * 40.0))
            .collect();
        let message_at = |origin: Vec2| PerceptionMessage {
            percept: TestPercept::Sound,
            base_value: 70.0,
            origin,
            range: 150.0,
            source: None,
        };

        let (mut indexed_world, mut schedule) = propagation_world(PerceptionConfig::default());
        let indexed: Vec<Entity> = positions
            .iter()
            .map(|&position| spawn_perceiver(&mut indexed_world, position))
            .collect();
        for &origin in &origins {
            write_message(&mut indexed_world, message_at(origin));
        }
        schedule.run(&mut indexed_world);

        // One message per run keeps the batch under INDEX_MIN_MESSAGES.
        let (mut exhaustive_world, mut schedule) = propagation_world(PerceptionConfig {
            max_count: 0,
            ..PerceptionConfig::default()
        });
        let exhaustive: Vec<Entity> = positions
            .iter()
            .map(|&position| spawn_perceiver(&mut exhaustive_world, position))
            .collect();
        for &origin in &origins {
            write_message(&mut exhaustive_world, message_at(origin));
            schedule.run(&mut exhaustive_world);
        }

        for (a, b) in indexed.iter().zip(&exhaustive) {
            let mut expected = memory_values(&exhaustive_world, *b);
            // The indexed run keeps the default 16-entry cap; compare the same
            // newest-first window.
            expected.truncate(16);
            let got = memory_values(&indexed_world, *a);
            assert_eq!(
                got.len(),
                expected.len(),
                "memory count must match the exhaustive run"
            );
            for (got, expected) in got.iter().zip(&expected) {
                assert!(
                    (got - expected).abs() < 1e-4,
                    "indexed {got} vs exhaustive {expected}"
                );
            }
        }
    }

    #[test]
    fn source_entity_preserved_in_memory() {
        let mut app = create_test_app();

        let perceiver = app.world_mut().spawn(Memory::<TestPercept>::default()).id();
        let attacker = app.world_mut().spawn_empty().id();

        app.update();

        app.world_mut().commands().trigger(PerceptionEvent {
            entity: perceiver,
            percept: TestPercept::Pain,
            value: 10.0,
            location: None,
            source: Some(attacker),
        });
        app.update();

        let memory = app
            .world()
            .entity(perceiver)
            .get::<Memory<TestPercept>>()
            .unwrap();
        let entry = memory.strongest().unwrap();
        assert_eq!(entry.source, Some(attacker));
    }
}
