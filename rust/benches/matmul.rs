use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustgpt::matmul::{matmul_blocked, matmul_naive};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

fn square(rng: &mut Rng, n: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..n * n).map(|_| rng.normal()).collect();
    Tensor::from_vec(data, &[n, n])
}

fn bench_matmul(c: &mut Criterion) {
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

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
