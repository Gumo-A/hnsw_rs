use core::panic;
use nohash_hasher::{IntMap, IntSet};
use serde::{Deserialize, Serialize};
use std::{collections::hash_set::Drain, time::Instant};

use super::dist::Dist;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Graph {
    pub nodes: IntMap<usize, IntSet<Dist>>,
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
            nodes.insert(*node_id, IntSet::from_iter(neighbors.values().copied()));
        }
        Graph { nodes }
    }

    pub fn add_node(&mut self, point_id: usize) {
        self.nodes.entry(point_id).or_insert(IntSet::default());
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize, dist: Dist) -> Result<(), String> {
        // This if statatements garantee the unwraps() below won't fail.
        if node_a == node_b {
            return Ok(());
        }

        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b) {
            println!("node_a is in graph {}", self.nodes.contains_key(&node_a));
            println!("node_b is in graph {}", self.nodes.contains_key(&node_b));
            return Err(format!(
                "Error adding edge, one of the nodes is not in the graph."
            ));
        }

        let dist_to_a = Dist::new(dist.dist, node_a);
        let dist_to_b = Dist::new(dist.dist, node_b);

        self.nodes.get_mut(&node_a).unwrap().insert(dist_to_b);
        self.nodes.get_mut(&node_b).unwrap().insert(dist_to_a);

        Ok(())
    }

    /// Removes an edge from the Graph.
    /// Since the add_edge method won't allow for self-connecting nodes, we don't check that here.
    /// Returns whether the edge was removed.
    pub fn remove_edge(&mut self, node_a: usize, dist: Dist) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&dist.id) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        if (self.degree(node_a)? == 1) | (self.degree(dist.id)? == 1) {
            return Ok(false);
        }

        let a_rem = self.nodes.get_mut(&node_a).unwrap().remove(&dist);
        let b_rem = self
            .nodes
            .get_mut(&dist.id)
            .unwrap()
            .remove(&Dist::new(dist.dist, node_a));

        Ok(a_rem & b_rem)
    }

    pub fn neighbors(&self, node_id: usize) -> Result<&IntSet<Dist>, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn replace_or_add_neighbors<I>(
        &mut self,
        node_id: usize,
        new_neighbors: I,
    ) -> Result<(), String>
    where
        I: Iterator<Item = Dist>,
    {
        if self.degree(node_id)? == 0 {
            for dist in new_neighbors {
                self.add_edge(node_id, dist.id, dist)?;
            }
            return Ok(());
        }
        let news = IntSet::from_iter(new_neighbors);
        let olds = self.neighbors(node_id)?;

        let to_remove: Vec<Dist> = olds.difference(&news).copied().collect();
        let to_add: Vec<Dist> = news.difference(olds).copied().collect();

        for dist in to_add {
            self.add_edge(node_id, dist.id, dist)?;
        }

        for dist in to_remove {
            self.remove_edge(node_id, dist)?;
        }

        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: usize) {
        for dist in self.remove_neighbors(node_id).collect::<Vec<Dist>>() {
            self.nodes
                .get_mut(&dist.id)
                .unwrap()
                .remove(&Dist::new(0.0, node_id));
        }
    }

    fn remove_neighbors(&mut self, node_id: usize) -> Drain<'_, Dist> {
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
                "Error getting degree of {node_id}, it is not in the graph."
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
