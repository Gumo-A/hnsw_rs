use crate::helpers::distance::norm_vector;
use ndarray::{Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Payload {
    pub data: HashMap<String, PayloadType>,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub id: usize,
    pub vector: Array<f32, Dim<[usize; 1]>>,
    pub neighbors: HashSet<usize, BuildNoHashHasher<usize>>,
    pub payload: Option<Payload>,
}

impl Point {
    pub fn new(
        id: usize,
        vector: ArrayView<f32, Dim<[usize; 1]>>,
        neighbors: Option<HashSet<usize, BuildNoHashHasher<usize>>>,
        payload: Option<Payload>,
    ) -> Point {
        Point {
            id,
            vector: norm_vector(&vector),
            neighbors: neighbors.unwrap_or(HashSet::with_hasher(BuildNoHashHasher::default())),
            payload,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PayloadType {
    StringPayload(String),
    BoolPayload(bool),
    NumericPayload(f32),
}
