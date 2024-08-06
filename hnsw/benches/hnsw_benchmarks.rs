#![allow(unused)]

fn main() {
    use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
    use hnsw::hnsw::points::Point;
    use rand::Rng;

    pub fn criterion_benchmark(c: &mut Criterion) {
        let mut rng = rand::thread_rng();
        let vector = (0..100).map(|_| rng.gen::<f32>()).collect();
        let point = Point::new(0, vector, false);
        c.bench_with_input(
            BenchmarkId::new("Quantize Point", "Random 100d f32 vector"),
            &point,
            |b, i| b.iter(|| i.get_quantized()),
        );
    }

    criterion_group!(benches, criterion_benchmark);
    criterion_main!(benches);
}
