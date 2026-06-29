use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hnsw::helpers::glove::load_glove_array;
use hnsw::template::HNSW;
use points::point::Point;
use std::fs::File;
use std::time::Duration;

const M: [usize; 3] = [32, 64, 128];
const FILE_NAME: &str = "/home/gamal/glove_dataset/glove.6B/glove.6B.300d.txt";

fn insert_at_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Insert @ 10k nodes"));
    for m in M {
        let file = File::open(FILE_NAME).unwrap();
        let (_, embeddings) = load_glove_array(10_000, file, false).unwrap();
        let index = HNSW::new(m, None, embeddings[0].len());
        let index = index
            .insert_bulk(embeddings[..9_999].to_vec(), 1, false)
            .unwrap();
        let vector = embeddings[9_999].clone();
        group.bench_function(BenchmarkId::from_parameter(m), |b| {
            b.iter_batched(
                || (index.clone(), vector.clone()),
                move |(mut i, vect): (HNSW, Vec<f32>)| {
                    i.insert_vec(&vect).unwrap();
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn build_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Build GloVe 10k"));
    group.sample_size(10);
    for m in M {
        let file = File::open(FILE_NAME).unwrap();
        let (_, embeddings) = load_glove_array(10_000, file, false).unwrap();
        group.bench_function(BenchmarkId::from_parameter(m), |b| {
            b.iter_batched(
                || embeddings.clone(),
                move |embs| {
                    let index = HNSW::new(m, None, embs[0].len());
                    let _ = index.insert_bulk(embs, 1, false);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, insert_at_10k, build_10k,);
criterion_main!(benches);
