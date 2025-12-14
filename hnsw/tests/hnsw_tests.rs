extern crate hnsw;

use std::{collections::HashSet, path::Path};

use graph::nodes::{Dist, Node};
use hnsw::{helpers::glove::load_glove_array, params::get_default_ml, template::HNSW};
use itertools::Itertools;
use points::{point::Point, point_collection::Points};
use rand::Rng;
use vectors::{FullVec, LVQVec, VecBase};

const DIM: usize = 10;
const N: usize = 100;
const M: usize = 12;
const NB_STORED: usize = 1_000;
const NB_QUERIES: usize = 100;

#[test]
fn hnsw_init() {
    let _index: HNSW<FullVec> = HNSW::new(12, None, 128);
}

#[test]
fn hnsw_build() {
    let vectors = make_rand_vectors(N, DIM);
    let index: HNSW<FullVec> = HNSW::new(12, None, DIM);
    let index = index
        .insert_bulk(Points::new_full(vectors, get_default_ml(12)), 1, false)
        .unwrap();
    assert_eq!(index.len(), N);
}

#[test]
fn hnsw_ann_accuracy() {
    let dim = 128;
    let vectors = make_rand_vectors(10, dim);
    let index: HNSW<FullVec> = HNSW::new(12, None, dim);
    let points = Points::new_full(vectors, get_default_ml(12));

    let index = index.insert_bulk(points.clone(), 1, false).unwrap();

    let query = Point::new_full(999_999, 0, vec![1.0]);

    let ann = index.ann_by_vector(&query, 8, 100).unwrap();
    let closest = points.get_point(ann[0]).unwrap().distance(&query);
    for i in 1..ann.len() {
        let next = points.get_point(ann[i]).unwrap().distance(&query);
        println!("closest is {closest}, i={i} is {next}");
        assert!(next >= closest);
    }
}

#[test]
#[should_panic]
fn can_not_add_different_dim() {
    let index: HNSW<FullVec> = HNSW::new(12, None, 128);

    let vectors = make_rand_vectors(10, 128);
    let points = Points::new_full(vectors, get_default_ml(12));
    let index = index.insert_bulk(points.clone(), 1, false).unwrap();

    let vectors = make_rand_vectors(10, 128);
    let points = Points::new_full(vectors, get_default_ml(12));
    let _index = index.insert_bulk(points.clone(), 1, false).unwrap();
}

#[test]
fn hnsw_full_glove_build_eval() {
    let (_, vectors) = load_glove_array(NB_STORED + NB_QUERIES, format!("glove.50d"), false)
        .expect("Could not load glove");

    let stored = vectors[NB_QUERIES..].iter().cloned().collect();
    let queries: Vec<Vec<f32>> = vectors[..NB_QUERIES].iter().cloned().collect();

    let points = Points::new_full(stored, get_default_ml(M));

    let mut queries_nn: Vec<HashSet<Node>> = Vec::new();
    for query in queries.iter() {
        let query_point = Point::new_full(0, 0, query.clone());
        let query_true_nn = (0..NB_STORED as Node)
            .map(|idx| Dist::new(idx, points.distance2point(&query_point, idx).unwrap()))
            .sorted()
            .map(|dist| dist.id)
            .take(10)
            .collect();
        queries_nn.push(query_true_nn);
    }

    let index: HNSW<FullVec> = HNSW::new(M, Some(512), 50);
    let index = index.insert_bulk(points.clone(), 1, false).unwrap();

    let mut total_hits = 0;
    for (idx, query) in queries.iter().enumerate() {
        let query_point = Point::new_full(0, 0, query.clone());
        let query_ann = index.ann_by_vector(&query_point, 10, 100).unwrap();
        let query_ann: HashSet<Node> = HashSet::from_iter(query_ann.iter().copied());

        let query_true_nn = queries_nn.get(idx).unwrap();
        let hits = query_true_nn.intersection(&query_ann).count();
        total_hits += hits;
    }
    let final_acc = total_hits as f32 / (NB_QUERIES * 10) as f32;
    println!("Final accuracy was {final_acc}");
    assert!(final_acc > 0.8);
}

#[test]
fn hnsw_quant_glove_build_eval() {
    let (_, vectors) = load_glove_array(NB_STORED + NB_QUERIES, format!("glove.50d"), false)
        .expect("Could not load glove");

    let stored = vectors[NB_QUERIES..].iter().cloned().collect();
    let queries: Vec<Vec<f32>> = vectors[..NB_QUERIES].iter().cloned().collect();

    let points = Points::new_quant(stored, get_default_ml(M));

    let mut queries_nn: Vec<HashSet<Node>> = Vec::new();
    for query in queries.iter() {
        let query_point = Point::new_quant(0, 0, query);
        let query_true_nn = (0..NB_STORED as Node)
            .map(|idx| Dist::new(idx, points.distance2point(&query_point, idx).unwrap()))
            .sorted()
            .map(|dist| dist.id)
            .take(10)
            .collect();
        queries_nn.push(query_true_nn);
    }

    let index: HNSW<LVQVec> = HNSW::new(M, Some(512), 50);
    let index = index.insert_bulk(points.clone(), 1, false).unwrap();

    let mut total_hits = 0;
    for (idx, query) in queries.iter().enumerate() {
        let query_point = Point::new_quant(0, 0, query);
        let query_ann = index.ann_by_vector(&query_point, 10, 100).unwrap();
        let query_ann: HashSet<Node> = HashSet::from_iter(query_ann.iter().copied());

        let query_true_nn = queries_nn.get(idx).unwrap();
        let hits = query_true_nn.intersection(&query_ann).count();
        total_hits += hits;
    }
    let final_acc = total_hits as f32 / (NB_QUERIES * 10) as f32;
    println!("Final accuracy was {final_acc}");
    assert!(final_acc > 0.8);
}

#[test]
fn hnsw_serialize() {
    let vectors = make_rand_vectors(N, DIM);
    let index: HNSW<FullVec> = HNSW::new(12, None, DIM);
    let index = index
        .insert_bulk(Points::new_full(vectors, get_default_ml(12)), 1, false)
        .unwrap();

    let index_path = Path::new("./index_ser_test");
    index.save(index_path);
    let loaded_index: HNSW<FullVec> = HNSW::load(index_path).unwrap();

    std::fs::remove_dir_all(index_path).unwrap();

    assert_eq!(N, loaded_index.len());
}

fn make_rand_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::new();
    for _ in 0..n {
        let vector = (0..dim).map(|_| rng.gen::<f32>()).collect();
        vectors.push(vector)
    }
    vectors
}
