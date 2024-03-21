use crate::helpers::distance::norm_vector;
use ndarray::{Array, Dim};
use std::collections::{HashMap, HashSet};

pub struct Graph {
    pub nodes: HashMap<i32, (HashSet<i32>, Array<f32, Dim<[usize; 1]>>)>,
    // pub node_vectors: HashMap<i32, Array1<f32>>,
    pub self_connexions: bool,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            self_connexions: false,
        }
    }
    pub fn from_layer_data(dim: u32, data: HashMap<i32, (HashSet<i32>, Vec<f32>)>) -> Self {
        let mut nodes = HashMap::new();
        for (node_id, node_data) in data.iter() {
            let vector = Array::from_shape_vec((dim as usize,), node_data.1.clone())
                .expect("Could not load a vector.");
            let neighbors = node_data.0.clone();
            nodes.insert(*node_id, (neighbors, vector));
        }

        Self {
            nodes,
            self_connexions: false,
        }
    }
    pub fn add_node(&mut self, node_id: i32, vector: Array<f32, Dim<[usize; 1]>>) {
        if !self.nodes.contains_key(&node_id) {
            self.nodes
                .insert(node_id, (HashSet::new(), norm_vector(vector)));
            // self.node_vectors.insert(node_id, vector);
        }
    }

    pub fn add_edge(&mut self, node_a: i32, node_b: i32) {
        if (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
            | ((node_a == node_b) & !self.self_connexions)
        {
            println!("didnt add edge");
            return ();
        }
        let a_neighbors = self
            .nodes
            .get_mut(&node_a)
            .expect("Could not get the value of node {node_a}");
        a_neighbors.0.insert(node_b);

        let b_neighbors = self
            .nodes
            .get_mut(&node_b)
            .expect("Could not get the value of node {node_b}");
        b_neighbors.0.insert(node_a);
    }

    pub fn remove_edge(&mut self, node_a: i32, node_b: i32) {
        let a_neighbors = self
            .nodes
            .get_mut(&node_a)
            .expect("Could not get neighbors of {node_a}");
        a_neighbors.0.remove(&node_b);

        let b_neighbors = self
            .nodes
            .get_mut(&node_b)
            .expect("Could not get neighbors of {node_b}");
        b_neighbors.0.remove(&node_a);
    }

    pub fn neighbors(&self, node_id: i32) -> &HashSet<i32> {
        &self
            .nodes
            .get(&node_id)
            .expect("Could not get the neighbors of {node_id}")
            .0
    }

    pub fn node(&self, node_id: i32) -> &(HashSet<i32>, Array<f32, Dim<[usize; 1]>>) {
        self.nodes
            .get(&node_id)
            .expect("Could not fetch node {node_id}")
        // .clone()
        // .to_owned()
    }

    pub fn degree(&self, node_id: i32) -> i32 {
        self.nodes
            .get(&node_id)
            .expect("Could not get the neighbors of {node_id}")
            .0
            .len() as i32
    }

    pub fn order(&self) -> usize {
        self.nodes.len()
    }
}
