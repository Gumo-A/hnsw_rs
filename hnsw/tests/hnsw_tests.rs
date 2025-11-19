extern crate hnsw;

use std::collections::BinaryHeap;

use graph::nodes::Node;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::Searcher;
use hnsw::hnsw::index::{build_index, HNSW};
use hnsw::hnsw::params::{get_default_ml, Params};
use points::point_collection::Points;
use rand::Rng;
use vectors::FullVec;

const DIM: u32 = 100;
const N: usize = 1000;

#[test]
fn hnsw_init() {
    let params = Params::from_m(12, DIM);
    let _index: HNSW<FullVec> = HNSW::new(params.m, None, params.dim, false);
}

#[test]
fn hnsw_build() {
    let vectors = make_rand_vectors(N);
    let mut index = build_index(
        12,
        None,
        Points::new_full(vectors, get_default_ml(12)),
        false,
    )
    .unwrap();
    assert_eq!(index.len(), N);
}

#[test]
fn hnsw_build_small_loop() {
    let (_, vectors) =
        load_glove_array(10, format!("glove.6B.{DIM}d"), false).expect("Could not load glove");
    for _ in 0..100 {
        let _ = build_index(12, None, Points::new_full(vectors.clone(), 0.5), false).unwrap();
    }
}

// #[test]
// fn hnsw_serialize() {
//     let index = build_index(24, None, Points::new_full(make_rand_vectors(N), 0.5), false).unwrap();
//     index.print_index();
//     println!("");

//     let index_path = "./hnsw_index.ann";
//     index.save(index_path).unwrap();
//     let loaded_index = HNSW::from_path(index_path).unwrap();
//     loaded_index.print_index();

//     std::fs::remove_file(index_path).unwrap();

//     assert_eq!(N, loaded_index.points.len());
// }

#[test]
fn dist_binaryheap() {
    let dist1 = Node::new_with_dist(0.5, 0);
    let dist2 = Node::new_with_dist(0.2, 1);
    let dist3 = Node::new_with_dist(0.7, 2);
    let dist4 = Node::new_with_dist(0.1, 3);
    let mut bh = BinaryHeap::from([dist1, dist2, dist3, dist4]);

    assert_eq!(bh.pop().unwrap().dist.unwrap(), 0.7);
    assert_eq!(bh.pop().unwrap().dist.unwrap(), 0.5);
    assert_eq!(bh.pop().unwrap().dist.unwrap(), 0.2);
    assert_eq!(bh.pop().unwrap().dist.unwrap(), 0.1);
}

#[test]
fn set_dist() {
    let mut set = nohash_hasher::IntSet::default();
    let dist1 = Node::new_with_dist(0.5, 0);
    set.insert(dist1);
    let dist2 = Node::new_with_dist(0.2, 1);
    set.insert(dist2);
    let dist3 = Node::new_with_dist(0.7, 2);
    set.insert(dist3);
    let dist4 = Node::new_with_dist(0.1, 3);
    set.insert(dist4);

    assert!(!set.insert(Node::new_with_dist(0.5, 0)));
    assert!(set.remove(&Node::new_with_dist(0.1, 3)));
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
