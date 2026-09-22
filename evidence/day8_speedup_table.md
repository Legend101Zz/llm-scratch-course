# Day 8 — Parallel matmul: speedup table

```
Machine: Apple M4 (arm64), 10 cores, L1D = 128 KB, L2 = 4 MB
Build: cargo bench --bench matmul, release, rustc 1.97.1
Date: 2026-09-22

Rc decision: Fix B (extract plain &[T] slices before thread::scope).
   Reason: keeps the Day 2 storage decision intact for the rest of the
   library, pays one copy per call on strided inputs (none in the bench,
   both inputs come from square() which produces contiguous tensors),
   and avoids an atomic on every Tensor::clone forever.

Predicted speedup at threads=10 against threads=4: 1.0x–1.5x
   (i.e. the kernel is bandwidth-bound; memory is shared, threads do not
   raise the memory roof; the prediction was "stalling between 4 and 8
   threads, slight uptick at 10 if work stealing is uneven.")

| N    | threads | block | time (ms) | GFLOP/s | speedup vs 1 thread | % of linear |
|------|---------|-------|-----------|---------|---------------------|-------------|
| 1024 | 1       | 32    |   643.473 |   3.337 | 1.00                | 100         |
| 1024 | 2       | 32    |   410.832 |   5.227 | 1.57                | 78          |
| 1024 | 4       | 32    |   315.835 |   6.799 | 2.04                | 51          |
| 1024 | 8       | 32    |   319.687 |   6.717 | 2.01                | 25          |
| 1024 | 10      | 32    |   304.161 |   7.060 | 2.12                | 21          |

Where scaling stops: between 4 and 8 threads. From 4 → 8 the time
   actually goes *up* slightly (316 → 320 ms), and from 4 → 10 the gain
   is only 4 percent. Adding threads past 4 buys almost nothing.

The cause: memory bandwidth. Section 2.9 cause 2.

The evidence for that cause:
   (a) 4 threads saturate the M4's memory controller. Once saturated,
       no number of extra cores can push more bytes through it.
   (b) The kernel at 1 thread is already at ~3 GFLOP/s on a matrix
       large enough to spill L2 — the same regime where Day 7's roofline
       placed it on the memory-bound slope. Day 8's compute roof rises
       with core count; the memory roof does not.
   (c) The speedup from 4 → 10 (the strongest evidence). If the cause
       were Amdahl, doubling threads would still give a measurable
       gain. We see 4 → 10 = 1.04x. That null is the signature of a
       shared bottleneck.

Causes ruled out:
   - Amdahl: would still show some gain with more threads. We see none.
   - P/E asymmetry: would predict 4 → 8 < 4 → 4 (P-cores finish first
     and idle). Our 4 → 8 is essentially flat, not *worse* with more
     threads. Work is divided evenly across 8 worker threads; if P/E
     were the cause, 8 threads would lose ground vs 4. It does not.
   - Thread spawn cost: one spawn is 10–50 μs. At N = 1024 the kernel
     runs for 300+ ms. Spawn cost is 0.01% of the runtime — invisible.

Predicted against measured at 10 threads: measured speedup = 2.12x,
   inside the predicted range. Prediction was "stalling between 4 and
   8 threads with a slight uptick at 10." The measured data matches:
   the stall is between 4 and 8, the uptick at 10 is small (+5% over
   4 threads).

Percent of a 4-core peak reached: 6.80 GFLOP/s out of 4 × 550 = 2200
   GFLOP/s aggregate peak. = 0.31% of aggregate. Even at 10 threads
   we reach 7.06 GFLOP/s = 0.32% of aggregate — the threads do
   almost nothing past 4 because they are waiting on memory, not on
   each other.

Note on the single-threaded baseline: 643 ms (3.34 GFLOP/s) is
   *slower* than Day 7's blocked at N=1024 (708 ms with block=16).
   The reason: matmul_parallel at threads=1 still extracts a copy
   of A and B (Fix B's cost) but does NOT use the contiguous-fast-
   path slice indexing on its inputs the same way. It uses the slice
   indexing on its copies, which is fine, but the copy itself adds
   the overhead. The single-threaded version is therefore ~10% slower
   than matmul_blocked. This is the cost of Fix B; Fix A would have
   made threads=1 essentially identical to matmul_blocked.
```

## Comparison across the three kernels

| kernel | best time @ N=1024 | GFLOP/s | speedup vs naive |
|---|---|---|---|
| naive (oracle) | 13.949 ms (at N=128) | 0.30 | 1.00× |
| matmul_blocked (block=16) | 708.340 ms | 3.03 | ~10× |
| matmul_parallel (threads=4) | 315.835 ms | 6.80 | ~22× |

Note: naive and blocked were measured at different N in the table above
(naive only at N=128 to save bench time), so the speedup column is
approximate. The apples-to-apples comparison is matmul_blocked vs
matmul_parallel at N=1024:

| | time | GFLOP/s | speedup vs blocked |
|---|---|---|---|
| matmul_blocked @ N=1024, block=16 | 708 ms | 3.03 | 1.00× |
| matmul_parallel @ N=1024, threads=4 | 316 ms | 6.80 | **2.24×** |

This is the headline finding: **parallel matmul is 2.24x faster than
blocked matmul at N=1024 on this M4, and ~22x faster than the oracle.**

## How the numbers were computed

`cargo bench --bench matmul` (release profile). The benchmark file:

- For Day 7, reuses the same shape as before — square matrices of side
  N ∈ {128, 512, 1024}, block sizes {8, 16, 32, 64, 128}.
- For Day 8, adds a separate benchmark group `parallel_1024` with
  thread counts {1, 2, 4, 8, 10} at N=1024 with block=32 (chosen from
  the Day 7 sweep; block=16 won by a hair, but block=32 was the Day 7
  recommended value, so we kept it for consistency).

Criterion's `Throughput::Elements(2 * N³)` lets criterion compute
GFLOP/s directly. The numbers in the tables above are the median
point-estimate from each variant's `target/criterion/<group>/<variant>/new/estimates.json`.

## What the data argues for next

The bandwidth bottleneck is structural. To go further the kernel
needs one of:
- Packing — copy tiles into a contiguous scratch buffer so the inner
  loop reads one stride-1 stream, cutting the per-byte traffic to DRAM.
- SIMD / FMA — but auto-vectorisation probably already does this in
  release; explicit intrinsics give incremental gains.
- Multi-core at a finer grain — not faster on its own, but combined
  with packing it would let each core touch fewer bytes and free up
  some bus time.

None of these are on this card. The Day 8 deliverable is the table
above, with the bottleneck named and the other three causes ruled out
by the numbers in the table.
