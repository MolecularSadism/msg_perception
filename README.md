# msg_perception

A generic perception and memory system for [Bevy](https://bevyengine.org).

Environmental stimuli are stored as `MemoryEntry<P>` values inside a `Memory<P>` component on actor entities. The generic parameter `P` defines the percept type (e.g. `Sound`, `Pain`). Each entry decays over time and is removed once it falls below a configurable threshold.

## Features

- **Broadcast stimuli** via `PerceptionMessage<P>` — spatial propagation with logarithmic distance attenuation (`V = V_base - 20·log10(distance)`)
- **Targeted stimuli** via `PerceptionEvent<P>` — direct trigger on a specific entity, no attenuation
- **Self-filtering** — broadcast stimuli where `source == perceiver` are automatically ignored
- **Configurable decay** — per-frame value reduction with a minimum threshold for removal
- **Source tracking** — both stimulus paths carry an optional source entity

## Usage

```rust
use bevy::prelude::*;
use msg_perception::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
enum MyPercept { Sound, Pain }

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PerceptionPlugin::<MyPercept>::default())
        .run();
}
```

Spawn actors with a `Memory<MyPercept>` component and fire stimuli with `MessageWriter<PerceptionMessage<MyPercept>>` (broadcast) or `commands.trigger(PerceptionEvent { ... })` (targeted).

## Configuration

Insert or modify `PerceptionConfig` to tune the system at runtime:

```rust
app.insert_resource(PerceptionConfig {
    decay_rate: 5.0,     // value lost per second
    min_threshold: 0.5,  // entries below this are removed
    max_count: 16,       // max memories per entity
});
```

## System scheduling

The plugin exposes `PerceptionSystems::Decay` and `PerceptionSystems::Propagate` set labels so the host app can order its own systems relative to perception processing.

## Bevy compatibility

| `msg_perception` | Bevy   |
|-----------------|--------|
| 0.1             | 0.18   |

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
