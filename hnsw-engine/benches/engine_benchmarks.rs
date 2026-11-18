use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hnsw::helpers::glove::load_glove_array;
use hnsw_engine::engine::Engine;
use rand::{Rng, thread_rng};
use std::fs::File;

const M: [usize; 3] = [32, 64, 128];
const FILE_NAME: &str = "/home/gamal/glove/300d.txt";

fn init_index(m: usize) -> Engine {
    let tmp_dir = std::env::temp_dir().join("hnsw_tests");
    let rand = thread_rng().r#gen::<usize>();
    Engine::new(tmp_dir.join(format!("{rand}")).to_path_buf(), m).unwrap()
}

fn insert_at_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Insert @ 10k nodes"));
    for m in M {
        let file = File::open(FILE_NAME).unwrap();
        let (_, embeddings) = load_glove_array(10_000, file).unwrap();
        let index = init_index(m);
        index.insert(&embeddings[..9_999]).unwrap();
        let vector = embeddings[9_999].clone();
        group.bench_function(BenchmarkId::from_parameter(m), |b| {
            b.iter_batched(
                || (index.clone(), vector.clone()),
                move |(i, vect): (Engine, Vec<f32>)| {
                    i.insert(&vec![vect]).unwrap();
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// fn build_10k(c: &mut Criterion) {
//     let mut group = c.benchmark_group(format!("Build GloVe 10k"));
//     group.sample_size(10);
//     for m in M {
//         let file = File::open(FILE_NAME).unwrap();
//         let (_, embeddings) = load_glove_array(10_000, file).unwrap();
//         group.bench_function(BenchmarkId::from_parameter(m), |b| {
//             b.iter_batched(
//                 || embeddings.clone(),
//                 move |embs| {
//                     let index = init_index(m);
//                     let _ = index.insert(&embs);
//                 },
//                 criterion::BatchSize::LargeInput,
//             )
//         });
//     }
//     group.finish();
// }

criterion_group!(benches, insert_at_10k);
criterion_main!(benches);
