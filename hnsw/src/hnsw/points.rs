use crate::helpers::distance::norm_vector;
use ndarray::{Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

use super::lvq::LVQVec;

#[derive(Debug, Clone)]
pub enum Vector {
    Compressed(LVQVec),
    Full(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct Payload {
    pub data: HashMap<String, PayloadType>,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub id: usize,
    pub vector: Vector,
    pub neighbors: HashSet<usize, BuildNoHashHasher<usize>>,
    pub payload: Option<Payload>,
}

impl Point {
    pub fn new(
        id: usize,
        vector: Vec<f32>,
        neighbors: Option<HashSet<usize, BuildNoHashHasher<usize>>>,
        payload: Option<Payload>,
        quantize: bool,
    ) -> Point {
        let vector_stored = if quantize {
            Vector::Compressed(LVQVec::new(&vector, 8))
        } else {
            Vector::Full(vector)
        };
        Point {
            id,
            vector: vector_stored,
            neighbors: neighbors.unwrap_or(HashSet::with_hasher(BuildNoHashHasher::default())),
            payload,
        }
    }

    pub fn dist(&self, other: &Vector) -> f32 {
        match &self.vector {
            Vector::Compressed(compressed_self) => match other {
                Vector::Compressed(compressed_other) => {
                    compressed_self.dist2other(&compressed_other)
                }
                Vector::Full(full_other) => compressed_self.dist2vec(&full_other),
            },
            Vector::Full(full_self) => match other {
                Vector::Compressed(compressed_other) => compressed_other.dist2vec(&full_self),
                Vector::Full(full_other) => {
                    let mut result = 0.0;
                    for (x, y) in full_self.iter().zip(full_other) {
                        result += (x - y).powi(2);
                    }
                    result
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum PayloadType {
    StringPayload(String),
    BoolPayload(bool),
    NumericPayload(f32),
}
