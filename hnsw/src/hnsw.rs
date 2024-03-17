use crate::graph;
use ndarray::{Array, Dim};

struct HNSW {
    pub layers: Vec<graph::Graph>,
}

impl HNSW {
    pub fn insert(&mut self, node_id: i32, vector: Array<f32, Dim<[usize; 1]>>) {}

    fn search_layer(&self) {}
}
