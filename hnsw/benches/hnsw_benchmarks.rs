use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hnsw::hnsw::points::Point;
use rand::Rng;

const DIMS: [usize; 4] = [128, 256, 512, 1024];

fn quantize_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantize with varying dimensions");
    group
        .significance_level(0.05)
        .sample_size(1000)
        .confidence_level(0.975)
        .noise_threshold(0.075);

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point = Point::new(0, vector, false);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &point, |b, i| {
            b.iter(|| i.get_quantized());
        });
    }
    group.finish();
}

fn dist_computation_quantized_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dist comp. of quantized vecs by dim");
    group
        .significance_level(0.05)
        .sample_size(1000)
        .confidence_level(0.975)
        .noise_threshold(0.075);

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector1 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point1 = Point::new(0, vector1, true);

        let vector2 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point2 = Point::new(0, vector2, true);

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&point1, &point2),
            |b, i| {
                b.iter(|| i.0.dist2other(i.1));
            },
        );
    }
    group.finish();
}

fn dist_computation_full_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dist comp. of full vecs by dim");
    group
        .significance_level(0.05)
        .sample_size(1000)
        .confidence_level(0.975)
        .noise_threshold(0.075);

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector1 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point1 = Point::new(0, vector1, false);

        let vector2 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point2 = Point::new(0, vector2, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&point1, &point2),
            |b, i| {
                b.iter(|| i.0.dist2other(i.1));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    quantize_various_sizes,
    dist_computation_quantized_various_sizes,
    dist_computation_full_various_sizes
);
criterion_main!(benches);
