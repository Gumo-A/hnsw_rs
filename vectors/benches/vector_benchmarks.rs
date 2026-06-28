use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vectors::{FullVec, QuantVec, VecBase, gen_rand_vecs};

const DIMS: [usize; 6] = [8, 32, 128, 512, 1024, 2048];

fn full_dist(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 distance of two full vecs, by dim");
    for dim in DIMS.iter() {
        let vectors = gen_rand_vecs(*dim, 2);
        let v0 = FullVec::new(&vectors[0]);
        let v1 = FullVec::new(&vectors[1]);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&v0, &v1), |b, i| {
            b.iter(|| i.0.distance(i.1));
        });
    }
    group.finish();
}

fn full_dist_to_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 distance of one full vec to many, by dim");
    for dim in DIMS.iter() {
        let v = FullVec::new(&gen_rand_vecs(*dim, 1)[0]);
        let others: Vec<FullVec> = gen_rand_vecs(*dim, 128)
            .iter()
            .map(|v| FullVec::new(v))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&v, &others), |b, i| {
            b.iter(|| i.0.dist2many(i.1.iter()).collect::<Vec<f32>>());
        });
    }
    group.finish();
}

fn quant_dist(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 distance of two quantized vecs, by dim");
    for dim in DIMS.iter() {
        let vectors = gen_rand_vecs(*dim, 2);
        let v0 = QuantVec::new(&vectors[0]);
        let v1 = QuantVec::new(&vectors[1]);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&v0, &v1), |b, i| {
            b.iter(|| i.0.distance(i.1));
        });
    }
    group.finish();
}

fn quant_dist_to_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 distance of one quantized vec to many, by dim");
    for dim in DIMS.iter() {
        let v = QuantVec::new(&gen_rand_vecs(*dim, 1)[0]);
        let others: Vec<QuantVec> = gen_rand_vecs(*dim, 128)
            .iter()
            .map(|v| QuantVec::new(v))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&v, &others), |b, i| {
            b.iter(|| i.0.dist2many(i.1.iter()).collect::<Vec<f32>>());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    quant_dist,
    quant_dist_to_many,
    full_dist,
    full_dist_to_many
);
criterion_main!(benches);
