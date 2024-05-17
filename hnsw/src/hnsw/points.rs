use crate::helpers::distance::norm_vector;
use ndarray::{Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

use super::lvq::CompressedVec;

#[derive(Debug, Clone)]
pub struct Payload {
    pub data: HashMap<String, PayloadType>,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub id: usize,
    pub vector: CompressedVec,
    pub neighbors: HashSet<usize, BuildNoHashHasher<usize>>,
    pub payload: Option<Payload>,
}

impl Point {
    pub fn new(
        id: usize,
        vector: CompressedVec,
        neighbors: Option<HashSet<usize, BuildNoHashHasher<usize>>>,
        payload: Option<Payload>,
    ) -> Point {
        Point {
            id,
            vector,
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
