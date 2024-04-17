use crate::helpers::distance::norm_vector;
use ndarray::{Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

pub trait Filtering {
    fn apply_filter<F>(&self, closure: F) -> bool
    where
        F: Fn(&Payload) -> bool;
}

#[derive(Debug, Clone)]
pub struct Payload {
    pub data: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub id: usize,
    pub vector: Array<f32, Dim<[usize; 1]>>,
    pub neighbors: HashSet<usize, BuildNoHashHasher<usize>>,
    pub payload: Option<Payload>,
}

impl Filtering for Point {
    fn apply_filter<F>(&self, closure: F) -> bool
    where
        F: Fn(&Payload) -> bool,
    {
        match &self.payload {
            None => false,
            Some(load) => closure(load),
        }
    }
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
            vector: norm_vector(&vector.view()),
            neighbors: neighbors.unwrap_or(HashSet::with_hasher(BuildNoHashHasher::default())),
            payload,
        }
    }
}

// impl<F> FilterTrait for F
// where
//     F: Fn(Payload) -> bool,
// {
//     fn apply_filter(&self, payload: Option<Payload>) -> bool {
//         match payload {
//             None => false,
//             Some(val) => self(val),
//         }
//     }
// }
