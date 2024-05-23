use crate::hnsw::points::Point;
use ndarray::Array;
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

use super::lvq::LVQVec;

#[derive(Debug)]
pub struct Graph {
    pub nodes: HashMap<usize, Point>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: HashMap::new(),
        }
    }
    pub fn from_layer_data(
        dim: usize,
        data: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)>,
    ) -> Graph {
        let mut nodes = HashMap::new();
        for (node_id, node_data) in data.iter() {
            let neighbors = node_data.0.clone();
            let point = Point::new(*node_id, node_data.1.clone(), Some(neighbors), None, true);
            nodes.insert(*node_id, point);
        }

        Graph { nodes }
    }
    pub fn add_node(&mut self, point: &Point) {
        // if !self.nodes.contains_key(&point.id) {
        let point = point.clone();
        self.nodes.insert(point.id, point);
        // }
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize) {
        if (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
            | (node_a == node_b)
        {
            return ();
        }
        let a_neighbors = self
            .nodes
            .get_mut(&node_a)
            .expect(format!("Could not get point {node_a}").as_str());
        a_neighbors.neighbors.insert(node_b);

        let b_neighbors = self
            .nodes
            .get_mut(&node_b)
            .expect(format!("Could not get point {node_b}").as_str());
        b_neighbors.neighbors.insert(node_a);
    }

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) {
        let a_neighbors = self
            .nodes
            .get_mut(&node_a)
            .expect(format!("Could not get neighbors of {node_a}").as_str());
        a_neighbors.neighbors.remove(&node_b);

        let b_neighbors = self
            .nodes
            .get_mut(&node_b)
            .expect(format!("Could not get neighbors of {node_b}").as_str());
        b_neighbors.neighbors.remove(&node_a);
    }

    pub fn neighbors(&self, node_id: usize) -> &HashSet<usize, BuildNoHashHasher<usize>> {
        &self
            .nodes
            .get(&node_id)
            .expect(format!("Could not get the neighbors of {node_id}").as_str())
            .neighbors
    }

    pub fn node(&self, node_id: usize) -> &Point {
        match self.nodes.get(&node_id) {
            Some(node) => node,
            None => {
                println!(
                    "{node_id} not found in graph. This graph has {} nodes",
                    self.nb_nodes()
                );
                panic!();
            }
        }
    }

    pub fn degree(&self, node_id: usize) -> usize {
        self.nodes
            .get(&node_id)
            .expect("Could not get the neighbors of {node_id}")
            .neighbors
            .len() as usize
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }
}
