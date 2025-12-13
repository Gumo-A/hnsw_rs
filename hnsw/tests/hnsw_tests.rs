extern crate hnsw;

use std::path::Path;

use hnsw::{helpers::glove::load_glove_array, params::get_default_ml, template::HNSW};
use points::point_collection::Points;
use rand::Rng;
use vectors::{FullVec, LVQVec};

const DIM: u32 = 10;
const N: usize = 100;

#[test]
fn hnsw_init() {
    let _index: HNSW<FullVec> = HNSW::new(12, None, 128);
}

#[test]
fn hnsw_build() {
    let vectors = make_rand_vectors(N);
    let index: HNSW<FullVec> = HNSW::new(12, None, 128);
    let index = index
        .insert_bulk(Points::new_full(vectors, get_default_ml(12)), 1)
        .unwrap();
    assert_eq!(index.len(), N);
}

#[test]
fn hnsw_build_glove() {
    let (_, vectors) =
        load_glove_array(10000, format!("glove.50d"), false).expect("Could not load glove");
    let index: HNSW<FullVec> = HNSW::new(12, None, 128);
    let index = index
        .insert_bulk(Points::new_full(vectors, get_default_ml(12)), 1)
        .unwrap();
}

#[test]
fn hnsw_serialize() {
    let vectors = make_rand_vectors(N);
    let index: HNSW<FullVec> = HNSW::new(12, None, 128);
    let index = index
        .insert_bulk(Points::new_full(vectors, get_default_ml(12)), 1)
        .unwrap();

    let index_path = Path::new("./index_ser_test");
    index.save(index_path);
    let loaded_index: HNSW<LVQVec> = HNSW::load(index_path).unwrap();

    std::fs::remove_dir_all(index_path).unwrap();

    assert_eq!(N, loaded_index.len());
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
