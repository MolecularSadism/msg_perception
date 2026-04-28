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
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            decay_rate: 5.0,
            min_threshold: 0.5,
            max_count: 16,
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

/// Propagate perception messages to all [`Memory`] holders within range.
///
/// Intensity is attenuated using a saturating logarithmic drop-off:
/// `V = V_base - 20 * log10(distance)`, clamped to zero.
///
/// Self-caused perceptions (where `source == perceiver`) are skipped.
pub fn propagate_perception<P: Percept>(
    mut events: MessageReader<PerceptionMessage<P>>,
    mut q_perceivers: Query<(Entity, &Transform, &mut Memory<P>)>,
    config: Res<PerceptionConfig>,
) {
    for event in events.read() {
        for (perceiver_entity, transform, mut memory) in &mut q_perceivers {
            // Skip self-caused perceptions
            if event.source == Some(perceiver_entity) {
                continue;
            }

            let perceiver_pos = transform.translation.truncate();
            let distance = perceiver_pos.distance(event.origin);

            if distance > event.range {
                continue;
            }

            let attenuation = if distance <= 1.0 {
                0.0
            } else {
                20.0 * distance.log10()
            };
            let initial_value = (event.base_value - attenuation).max(0.0);

            if initial_value <= config.min_threshold {
                continue;
            }

            memory.push(
                MemoryEntry {
                    percept: event.percept.clone(),
                    value: initial_value,
                    location: Some(LocationKnowledge::Origin(event.origin)),
                    source: event.source,
                },
                config.max_count,
            );
        }
    }
}

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
