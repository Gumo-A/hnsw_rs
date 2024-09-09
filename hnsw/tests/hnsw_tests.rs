extern crate hnsw;

use std::collections::BinaryHeap;

use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::dist::Dist;
use hnsw::hnsw::graph::Graph;
use hnsw::hnsw::index::Searcher;
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
    let mut searcher = Searcher::new();
    index.insert(0, &mut searcher).unwrap();

    assert_eq!(index.points.len(), N);
}

#[test]
fn hnsw_build_small_loop() {
    let (_, vectors) =
        load_glove_array(10, format!("glove.6B.{DIM}d"), false).expect("Could not load glove");
    for _ in 0..100 {
        let _ = HNSW::build_index(12, None, vectors.clone(), false).unwrap();
    }
}

#[test]
fn hnsw_build_big() {
    let (_, vectors) =
        load_glove_array(40_000, format!("glove.6B.{DIM}d"), false).expect("Could not load glove");
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

#[test]
fn dist_binaryheap() {
    let dist1 = Dist::new(0.5, 0);
    let dist2 = Dist::new(0.2, 1);
    let dist3 = Dist::new(0.7, 2);
    let dist4 = Dist::new(0.1, 3);
    let mut bh = BinaryHeap::from([dist1, dist2, dist3, dist4]);

    assert_eq!(bh.pop().unwrap().dist, 0.7);
    assert_eq!(bh.pop().unwrap().dist, 0.5);
    assert_eq!(bh.pop().unwrap().dist, 0.2);
    assert_eq!(bh.pop().unwrap().dist, 0.1);
}

#[test]
fn set_dist() {
    let mut set = nohash_hasher::IntSet::default();
    let dist1 = Dist::new(0.5, 0);
    set.insert(dist1);
    let dist2 = Dist::new(0.2, 1);
    set.insert(dist2);
    let dist3 = Dist::new(0.7, 2);
    set.insert(dist3);
    let dist4 = Dist::new(0.1, 3);
    set.insert(dist4);

    assert!(!set.insert(Dist::new(0.0, 0)));
    assert!(set.remove(&Dist::new(0.0, 3)));
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
