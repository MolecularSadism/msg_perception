# Benchmark baselines

Criterion baseline `base`, captured 2026-08-26.

| | |
|---|---|
| Commit | `429d3789405d` |
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

```sh
cargo bench -- --save-baseline base   # capture
cargo bench -- --baseline base        # compare against it
```

These were taken in a shared virtualised container, so absolute figures carry
more run-to-run noise than a dedicated machine. Comparisons made with
`--baseline base` on the same host are meaningful; comparing these absolute
numbers against a different machine is not.
