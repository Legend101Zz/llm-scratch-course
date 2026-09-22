use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustgpt::matmul::{matmul_blocked, matmul_naive, matmul_parallel};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

fn square(rng: &mut Rng, n: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..n * n).map(|_| rng.normal()).collect();
    Tensor::from_vec(data, &[n, n])
}

/// Day 7 sweep: one kernel, one block-size sweep.
fn bench_blocked(c: &mut Criterion) {
    let mut rng = Rng::seed(1234);

    for &n in &[128usize, 512, 1024] {
        let a: Tensor<f32> = square(&mut rng, n);
        let b: Tensor<f32> = square(&mut rng, n);

        let mut group = c.benchmark_group(format!("matmul_{n}"));
        // Criterion divides by this, so it prints FLOP/s directly.
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));

        // The oracle, for the baseline column. It is slow, so only at n = 128.
        if n == 128 {
            group.bench_function("naive", |bench| {
                bench.iter(|| matmul_naive(black_box(&a), black_box(&b)).unwrap());
            });
        }

        for &block in &[8usize, 16, 32, 64, 128] {
            group.bench_with_input(
                BenchmarkId::new("blocked", block),
                &block,
                |bench, &blk| {
                    bench.iter(|| matmul_blocked(black_box(&a), black_box(&b), blk).unwrap());
                },
            );
        }
        group.finish();
    }
}

/// Day 8 sweep: parallel kernel at N=1024 across thread counts 1..=10.
/// The single thread case (threads=1) is the apples-to-apples baseline
/// for speedup numbers in the evidence table.
fn bench_parallel(c: &mut Criterion) {
    let mut rng = Rng::seed(1234);

    // Only at N = 1024 — small N is dominated by spawn cost (section 2.9 #4)
    // and would not show a meaningful speedup. The lesson explicitly says so.
    for &n in &[1024usize] {
        let a: Tensor<f32> = square(&mut rng, n);
        let b: Tensor<f32> = square(&mut rng, n);

        // Block size: pick the best from Day 7 (block=8 was the N=1024 winner).
        let block = 32usize;

        let mut group = c.benchmark_group(format!("parallel_{n}"));
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));

        for &threads in &[1usize, 2, 4, 8, 10] {
            group.bench_with_input(
                BenchmarkId::new("threads", threads),
                &threads,
                |bench, &t| {
                    bench.iter(|| {
                        matmul_parallel(black_box(&a), black_box(&b), block, t).unwrap()
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_blocked, bench_parallel);
criterion_main!(benches);
