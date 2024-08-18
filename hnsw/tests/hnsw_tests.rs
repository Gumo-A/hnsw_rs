extern crate hnsw;

use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;
use hnsw::hnsw::params::Params;
use rand::Rng;

const DIM: usize = 100;
const N: usize = 100;

#[test]
fn hnsw_init() {
    let params = Params::from_m(12, DIM);
    let _index: HNSW = HNSW::new(params.m, None, params.dim);
    let _index: HNSW = HNSW::from_params(params);
}

#[test]
fn hnsw_build() {
    let vectors = make_rand_vectors(N);
    let mut index = HNSW::build_index(12, None, vectors, false).unwrap();
    index.insert(0, false).unwrap();

    assert_eq!(index.points.len(), N);
}

#[test]
fn hnsw_build_small_loop() {
    let (_, vectors) = load_glove_array(DIM, 10, true, false).expect("Could not load glove");
    for _ in 0..100_000 {
        let _ = HNSW::build_index(12, None, vectors.clone(), false).unwrap();
    }
}

#[test]
fn hnsw_build_big() {
    let (_, vectors) = load_glove_array(DIM, 40_000, true, false).expect("Could not load glove");
    let _ = HNSW::build_index(24, None, vectors, false).unwrap();
}

#[test]
fn hnsw_serialize() {
    let index = HNSW::build_index(24, None, make_rand_vectors(N), false).unwrap();

    let index_path = "./hnsw_index.json";
    index.save(index_path).unwrap();
    let loaded_index = HNSW::from_path(index_path).unwrap();

    assert_eq!(N, loaded_index.points.len());
    std::fs::remove_file(index_path).unwrap();
}

fn make_rand_vectors(n: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::new();
    for _ in 0..n {
        let vector = (0..DIM).map(|_| rng.gen::<f32>()).collect();
        vectors.push(vector)
    }
    vectors
}
