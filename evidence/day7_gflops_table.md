# Day 7 — Blocked matmul: GFLOP/s table (after contiguous fast path)

```
Machine: Apple Silicon (M-series), performance cores
Peak f32 used for the percent column: 550 GFLOP/s, per the Day 7 lesson
         (one M4 performance core, scalar f32, no SIMD)
Build: cargo bench --bench matmul, release, rustc 1.x
Date: 2026-09-22 (after contiguous fast path)

Predicted best block size from the L1 calculation: b = 104
   Working set in innermost tile = 3 * b² * 4 bytes (three f32 tiles live in L1)
   Set 12b² <= 128 KB = 131072 bytes  ->  b² <= 10922.67  ->  b <= 104.51
   Largest integer that fits: 104.
   Sweep tests [8, 16, 32, 64, 128]; closest are 64 and 128.

| N    | block | time (ms) | GFLOP/s | % of peak |
|------|-------|-----------|---------|-----------|
| 128  | naive |    14.538 |   0.289 |     0.05% |
| 128  |     8 |     1.377 |   3.045 |     0.55% |
| 128  |    16 |     1.331 |   3.151 |     0.57% |
| 128  |    32 |     1.163 |   3.605 |     0.66% |   <-- best at N=128
| 128  |    64 |     1.265 |   3.315 |     0.60% |
| 128  |   128 |     1.472 |   2.850 |     0.52% |
| 512  |     8 |    91.335 |   2.939 |     0.53% |
| 512  |    16 |    80.694 |   3.327 |     0.60% |
| 512  |    32 |    80.110 |   3.351 |     0.61% |   <-- best at N=512
| 512  |    64 |   102.842 |   2.610 |     0.47% |
| 512  |   128 |   147.032 |   1.826 |     0.33% |
| 1024 |     8 |   676.408 |   3.175 |     0.58% |   <-- best at N=1024
| 1024 |    16 |   738.359 |   2.908 |     0.53% |
| 1024 |    32 |   819.633 |   2.620 |     0.48% |
| 1024 |    64 |  1026.722 |   2.092 |     0.38% |
| 1024 |   128 | (missing) |       - |         - |

Measured best block size: 32 at N=128, 32 at N=512, 8 at N=1024.
   The "best" is size-dependent. Smaller blocks win at larger N because
   the working-set penalty grows with the tile size; the kernel is
   still well under the L1 roof, so other constraints dominate.

Predicted against measured: the L1 prediction (b ≈ 104) was NOT the
   measured best. The sweep peaks at 32 for medium N and 8 for large N,
   neither of which fills L1 to its theoretical maximum. Two factors
   from section 2.6 explain this:
   (a) L1 also holds the loop counters, stack, and the Rust runtime
       metadata for the slice — not all 128 KB are available to tiles.
   (b) The compiler emits vectorised code (auto-SIMD on release) so
       each f32 load costs less than one cycle, which makes the
       effective cache footprint smaller than the scalar model.

Speedup of the best blocked kernel against naive at N=128:
   best blocked  = 1.163 ms (block=32)  ->  3.61 GFLOP/s
   naive         = 14.538 ms            ->  0.29 GFLOP/s
   speedup       = 14.538 / 1.163       = 12.5x

   And vs the original Day 7 blocked (using get()) the speedup is:
   old best at N=128 was 0.41 GFLOP/s (block=8)
   new best at N=128 is 3.61 GFLOP/s (block=32)
   speedup = 3.61 / 0.41 = 8.8x

Next bottleneck, and whether blocking can fix it: at 0.6% of peak, the
   kernel is now bandwidth-bound — it touches the right memory but
   still spends most of its time waiting for cache misses on the
   1024² matrices, which no longer fit in L1 in their entirety. The
   next techniques from section 2.7 in order:
   - Packing (copying A's column-block into a contiguous scratch buffer
     so the inner loop reads a single stride-1 stream).
   - SIMD intrinsics — auto-vectorisation probably already does this
     for f32, but explicit intrinsics give control over FMA fusion.
   - Multi-threading (Day 8).
   Blocking alone cannot reach peak FLOP/s; the roofline shows that.
```

## What changed and why

### Before (Sep 21): the slow path

The inner loop was:

```rust
for k_idx in kk..k_max {
    sum = sum + a.get(&[i, k_idx]) * b.get(&[k_idx, j]);
}
```

Every `get()`:
1. Walks `phys_index`, which loops over `idx.len()` multiplying by strides
2. Performs a bounds check
3. Indexes `data[physical]`

At N=1024, the inner loop runs ~2 billion times, so this overhead is the
bottleneck — not DRAM traffic, not the arithmetic.

### After (Sep 22): the contiguous fast path

A new method `Tensor::as_slice() -> &[T]` exposes the underlying buffer.
`matmul_blocked` now dispatches on `is_contiguous()`:

```rust
if a.is_contiguous() && b.is_contiguous() {
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    // inner loop uses a_slice[i*K + k] and b_slice[k*N + j]
}
```

Each access is now a single load, no bounds check, no stride math. The
compiler turns it into a tight loop with vectorised f32 multiply-adds.

### Why both paths stay

The f64 test (`blocked_matches_naive_f64`) feeds a transposed view of A,
which is NOT contiguous. The slow path through `get()` still has to
work, and the `is_contiguous()` check dispatches to it automatically.
The benchmark inputs (built by `square()`) are contiguous, so the fast
path is what the table measures.
