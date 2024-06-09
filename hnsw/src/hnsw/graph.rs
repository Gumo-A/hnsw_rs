use crate::hnsw::points::Point;
use core::panic;
use nohash_hasher::BuildNoHashHasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>, BuildNoHashHasher<usize>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }
    pub fn from_layer_data(
        data: HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>,
    ) -> Graph {
        let mut nodes = HashMap::with_hasher(BuildNoHashHasher::default());
        for (node_id, neighbors) in data.iter() {
            nodes.insert(*node_id, neighbors.clone());
        }
        Graph { nodes }
    }

    pub fn add_node(&mut self, point: &Point) {
        if self.nodes.contains_key(&point.id) {
            return ();
        } else {
            self.nodes
                .insert(point.id, HashSet::with_hasher(BuildNoHashHasher::default()));
        };
    }

    pub fn add_node_by_id(&mut self, id: usize) {
        if self.nodes.contains_key(&id) {
            return ();
        } else {
            self.nodes
                .insert(id, HashSet::with_hasher(BuildNoHashHasher::default()));
        };
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize) {
        if (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
            | (node_a == node_b)
        {
            return ();
        }
        let a_neighbors = self.nodes.get_mut(&node_a).unwrap();
        a_neighbors.insert(node_b);
        let b_neighbors = self.nodes.get_mut(&node_b).unwrap();
        b_neighbors.insert(node_a);
    }

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) {
        let a_neighbors = self.nodes.get_mut(&node_a).unwrap();
        a_neighbors.remove(&node_b);
        let b_neighbors = self.nodes.get_mut(&node_b).unwrap();
        b_neighbors.remove(&node_a);
    }

    pub fn neighbors(&self, node_id: usize) -> &HashSet<usize, BuildNoHashHasher<usize>> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => neighbors,
            None => {
                println!("Size of edges set: {}", self.nodes.len());
                println!(
                    "Set contains {node_id}: {}",
                    self.nodes.contains_key(&node_id)
                );
                panic!("Could not get the neighbors of {node_id}");
            }
        }
        // .expect(format!("Could not get the neighbors of {node_id}").as_str())
    }

    pub fn degree(&self, node_id: usize) -> usize {
        self.nodes
            .get(&node_id)
            .expect("Could not get the degree of {node_id}")
            .len() as usize
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }
}
