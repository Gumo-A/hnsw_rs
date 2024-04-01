use crate::helpers::distance::norm_vector;
use ndarray::{Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

pub struct Graph {
    pub nodes: HashMap<
        usize,
        (
            HashSet<usize, BuildNoHashHasher<usize>>,
            Array<f32, Dim<[usize; 1]>>,
        ),
    >,
    pub self_connexions: bool,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            self_connexions: false,
        }
    }
    pub fn from_layer_data(
        dim: usize,
        data: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)>,
    ) -> Self {
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
    pub fn add_node(&mut self, node_id: usize, vector: Array<f32, Dim<[usize; 1]>>) {
        if !self.nodes.contains_key(&node_id) {
            self.nodes.insert(
                node_id,
                (
                    HashSet::with_hasher(BuildNoHashHasher::default()),
                    norm_vector(vector),
                ),
            );
        }
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize) {
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

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) {
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

    pub fn neighbors(&self, node_id: usize) -> &HashSet<usize, BuildNoHashHasher<usize>> {
        &self
            .nodes
            .get(&node_id)
            .expect("Could not get the neighbors of {node_id}")
            .0
    }

    pub fn node(
        &self,
        node_id: usize,
    ) -> &(
        HashSet<usize, BuildNoHashHasher<usize>>,
        Array<f32, Dim<[usize; 1]>>,
    ) {
        self.nodes
            .get(&node_id)
            .expect("Could not fetch node {node_id}")
    }

    pub fn degree(&self, node_id: usize) -> usize {
        self.nodes
            .get(&node_id)
            .expect("Could not get the neighbors of {node_id}")
            .0
            .len() as usize
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }
}
