use crate::helpers::distance::norm_vector;
use ndarray::{Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

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
            vector: norm_vector(&vector.view()),
            neighbors: neighbors.unwrap_or(HashSet::with_hasher(BuildNoHashHasher::default())),
            payload,
        }
    }

    pub fn filter_closure<F>(&self, f: &Option<Filter<F>>) -> bool
    where
        F: Fn(Payload) -> bool,
    {
        match self.payload {
            None => false,
            Some(val) => match f {
                None => true,
                Some(filter) => match filter {
                    Filter::NoFilter => true,
                    Filter::ClosureFilter(f) => f(val)
                },
            }
        }
    }
}

pub enum Filter<F>
where
    F: Fn(Payload) -> bool,
{
    NoFilter,
    ClosureFilter(F),
}

// Too complicated to implement, so I'll just
// make strings the only allowed value for payloads.
// I'd like to make something similar to what I tried
// here though.
// #[derive(Debug, Clone)]
// pub enum PayloadType {
//     StringPayload(String),
//     BoolPayload(bool),
//     NumericPayload(f32),
// }

#[derive(Debug, Clone)]
pub struct Payload {
    pub data: HashMap<String, String>,
    // pub data: HashMap<String, PayloadType>,
}
