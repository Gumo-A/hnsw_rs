use crate::hnsw::points::Point;
use core::panic;
use nohash_hasher::BuildNoHashHasher;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Graph {
    nodes: HashSet<usize>,
    edges: HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
        }
    }
    // TODO: change so that the "data" param is a reference to the stored data in the index
    // pub fn from_layer_data(
    //     data: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)>,
    // ) -> Graph<'a> {
    //     let mut nodes = HashMap::new();
    //     for (node_id, node_data) in data.iter() {
    //         let neighbors = node_data.0.clone();
    //         let point = Point::new(*node_id, node_data.1.clone(), Some(neighbors), true);
    //         nodes.insert(*node_id, point);
    //     }
    //     Graph { nodes }
    // }

    pub fn add_node(&mut self, point: &Point) {
        if self.nodes.insert(point.id) {
            self.edges
                .insert(point.id, HashSet::with_hasher(BuildNoHashHasher::default()));
        };
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize) {
        if (!self.nodes.contains(&node_a) | !self.nodes.contains(&node_b)) | (node_a == node_b) {
            return ();
        }
        let a_neighbors = self.edges.get_mut(&node_a).unwrap();
        a_neighbors.insert(node_b);
        let b_neighbors = self.edges.get_mut(&node_b).unwrap();
        b_neighbors.insert(node_a);
    }

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) {
        let a_neighbors = self.edges.get_mut(&node_a).unwrap();
        a_neighbors.remove(&node_b);
        let b_neighbors = self.edges.get_mut(&node_b).unwrap();
        b_neighbors.remove(&node_a);
    }

    pub fn neighbors(&self, node_id: usize) -> &HashSet<usize, BuildNoHashHasher<usize>> {
        match self.edges.get(&node_id) {
            Some(neighbors) => neighbors,
            None => {
                println!("Size of edges set: {}", self.edges.len());
                println!(
                    "Set contains {node_id}: {}",
                    self.edges.contains_key(&node_id)
                );
                panic!("Could not get the neighbors of {node_id}");
            }
        }
        // .expect(format!("Could not get the neighbors of {node_id}").as_str())
    }

    pub fn node(&self, node_id: usize) -> &usize {
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
        self.edges
            .get(&node_id)
            .expect("Could not get the degree of {node_id}")
            .len() as usize
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }
}
