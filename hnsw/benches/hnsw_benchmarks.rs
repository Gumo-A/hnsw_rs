// TODO
// write benchmarks for HNSW insertion algorithm,
// both for the whole process and for the individual
// functions that are called during insertion.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::{build_index, HNSW};
use hnsw::hnsw::params::get_default_ml;
use points::point::Point;
use points::point_collection::Points;
use rand::Rng;
use vectors::{LVQVec, VecTrait};

const DIMS: [usize; 4] = [128, 256, 512, 1024];
const GLOVE_DIMS: [usize; 3] = [50, 100, 300];
const M: u8 = 12;

fn insert_at_10000_m12(c: &mut Criterion) {
    let ml = get_default_ml(M);
    let mut group = c.benchmark_group("Insert @ 10_000 nodes, m12");
    // group
    //     .sample_size(1000)
    //     .measurement_time(Duration::from_secs(1000));

    for dim in GLOVE_DIMS.iter() {
        let (_, embeddings) = load_glove_array(10_000, format!("glove.6B.{dim}d"), false).unwrap();
        let index = build_index(
            M,
            None,
            Points::new_quant(embeddings[..9_999].to_vec(), ml),
            false,
        )
        .unwrap();
        let vector = embeddings[9_999].clone();
        group.bench_function(BenchmarkId::from_parameter(dim), |b| {
            b.iter_batched(
                || (index.clone(), vector.clone()),
                move |(mut i, vect): (HNSW<LVQVec>, Vec<f32>)| {
                    i.insert_point(Point::new_quant(0, 0, &vect)).unwrap();
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn build_10000_m12(c: &mut Criterion) {
    let ml = get_default_ml(M);
    let mut group = c.benchmark_group("Build GloVe 10k m12");
    // group
    //     .sample_size(1000)
    //     .measurement_time(Duration::from_secs(1000));

    for dim in GLOVE_DIMS.iter() {
        let (_, embeddings) = load_glove_array(10_000, format!("glove.6B.{dim}d"), false).unwrap();
        group.bench_function(BenchmarkId::from_parameter(dim), |b| {
            b.iter_batched(
                || embeddings.clone(),
                move |embs| {
                    let _ = build_index(M, None, Points::new_quant(embs, ml), false);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn quantize_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantize with varying dimensions");
    // group
    //     .sample_size(1000)
    //     .measurement_time(Duration::from_secs(1000));

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point = Point::new_full(0, 0, vector);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &point, |b, i| {
            b.iter(|| i.quantized());
        });
    }
    group.finish();
}

fn dist_computation_quantized_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dist comp. of quantized vecs by dim");
    // group
    //     .sample_size(1000)
    //     .measurement_time(Duration::from_secs(1000));

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector1 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point1 = Point::new_quant(0, 0, &vector1);

        let vector2 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point2 = Point::new_quant(0, 0, &vector2);

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&point1, &point2),
            |b, i| {
                b.iter(|| i.0.distance(i.1));
            },
        );
    }
    group.finish();
}

fn dist_computation_full_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dist comp. of full vecs by dim");
    // group
    //     .sample_size(1000)
    //     .measurement_time(Duration::from_secs(1000));

    for dim in DIMS.iter() {
        let mut rng = rand::thread_rng();
        let vector1 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point1 = Point::new_full(0, 0, vector1);

        let vector2 = (0..*dim).map(|_| rng.gen::<f32>()).collect();
        let point2 = Point::new_full(0, 0, vector2);

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&point1, &point2),
            |b, i| {
                b.iter(|| i.0.distance(i.1));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    insert_at_10000_m12,
    build_10000_m12,
    quantize_various_sizes,
    dist_computation_quantized_various_sizes,
    dist_computation_full_various_sizes
);
criterion_main!(benches);
