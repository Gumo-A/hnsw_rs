use core::panic;
use nohash_hasher::IntMap;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Drain;

use super::dist::Dist;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Graph {
    pub nodes: IntMap<usize, IntMap<usize, Dist>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: IntMap::default(),
        }
    }

    pub fn from_layer_data(data: IntMap<usize, IntMap<usize, Dist>>) -> Graph {
        let mut nodes = IntMap::default();
        for (node_id, neighbors) in data.iter() {
            nodes.insert(*node_id, neighbors.clone());
        }
        Graph { nodes }
    }

    pub fn add_node(&mut self, point_id: usize) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.nodes.entry(point_id) {
            e.insert(IntMap::default());
        } else {
            
        }
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize, dist: Dist) -> Result<(), String> {
        if (node_a == node_b)
            | (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
        {
            return Ok(());
        }

        match self.nodes.get_mut(&node_a) {
            Some(a_n) => a_n.insert(node_b, dist),
            None => {
                let msg = format!("Error adding edge: {node_a} is not in the graph.");
                return Err(msg);
            }
        };

        match self.nodes.get_mut(&node_b) {
            Some(b_n) => b_n.insert(node_a, dist),
            None => {
                let msg = format!("Error adding edge: {node_b} is not in the graph.");
                return Err(msg);
            }
        };

        Ok(())
    }

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) -> Result<(), String> {
        if (node_a == node_b)
            | (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
        {
            return Ok(());
        }

        match self.nodes.get_mut(&node_a) {
            Some(a_n) => a_n.remove(&node_b),
            None => {
                return Err(format!("Error adding edge: {node_a} is not in the graph."));
            }
        };

        match self.nodes.get_mut(&node_b) {
            Some(b_n) => b_n.remove(&node_a),
            None => {
                return Err(format!("Error adding edge: {node_b} is not in the graph."));
            }
        };

        Ok(())
    }

    pub fn neighbors(&self, node_id: usize) -> Result<&IntMap<usize, Dist>, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn replace_neighbors(
        &mut self,
        node_id: usize,
        new_neighbors: &IntMap<usize, Dist>,
    ) -> Result<(), String> {
        let olds = self.nodes.get(&node_id).unwrap().clone();
        for (old, _) in olds {
            self.remove_edge(node_id, old)?;
        }
        for (node, dist) in new_neighbors {
            self.add_edge(node_id, *node, *dist)?;
        }
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: usize) {
        for (node, _dist) in self
            .remove_neighbors(node_id)
            .collect::<Vec<(usize, Dist)>>()
        {
            self.nodes.get_mut(&node).unwrap().remove(&node_id);
        }
    }

    fn remove_neighbors(&mut self, node_id: usize) -> Drain<'_, usize, Dist> {
        match self.nodes.get_mut(&node_id) {
            Some(neighbors) => neighbors.drain(),
            None => {
                panic!("Could not get the neighbors of {node_id}. The graph does not contain this node");
            }
        }
    }

    pub fn degree(&self, node_id: usize) -> Result<usize, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.len()),
            None => Err(format!(
                "Error getting neighbors of {node_id}, (function 'degree') it is not in the graph."
            )),
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node_id: &usize) -> bool {
        self.nodes.contains_key(node_id)
    }
}
