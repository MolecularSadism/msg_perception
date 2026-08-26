# Benchmark baselines

Criterion baseline `base`, captured 2026-08-26.

| | |
|---|---|
| Commit | `c05a5f5a5878` |
| Branch | `claude/memory-digest` |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18) |
| Host | Linux x86_64 container (shared/virtualised) |

## Results

Mean with 95% confidence interval.

| Benchmark | Mean | 95% CI |
|---|---:|---|
| `burst/1000x1000` | 23.23 µs | 23.13 µs – 23.33 µs |
| `digest/alloc_100x5` | 530.8 ns | 529.6 ns – 532.2 ns |
| `digest/into_100x5` | 488.8 ns | 487.5 ns – 490.3 ns |
| `propagate/1000x100` | 55.82 µs | 55.24 µs – 56.45 µs |
| `propagate/1000x1000` | 464.7 µs | 460.2 µs – 469.8 µs |
| `propagate/100x100` | 12.87 µs | 12.62 µs – 13.11 µs |
| `propagate/100x1000` | 109.8 µs | 108.6 µs – 111.7 µs |

## Reproducing

The bench target must be named. A bare `cargo bench` also runs the lib test
harness, which rejects criterion's flags with
`error: Unrecognized option: 'baseline'`.

```sh
# capture
cargo bench --bench perception_scaling -- --save-baseline base
cargo bench --bench memory_digest -- --save-baseline base

# compare against it
cargo bench --bench perception_scaling -- --baseline base
cargo bench --bench memory_digest -- --baseline base
```

## How much to trust these

Taken in a shared virtualised container. Treat them as an order-of-magnitude
record, not a regression gate.

Re-running `take_batch/3x1000_pending` on byte-identical code about an hour
later on the *same* host reported `+31%` with `p = 0.00`. Criterion's
significance test measures sampling noise within a run; it cannot see the
host's load drifting between runs. So a reported change of this size here is
not evidence of a real regression.

To draw a conclusion from a comparison, capture the baseline and the
comparison back to back in one sitting, and treat anything under roughly
1.5x as inconclusive. Comparing these absolute numbers against a different
machine is meaningless.
