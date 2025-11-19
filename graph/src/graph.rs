use core::panic;
use nohash_hasher::{IntMap, IntSet};
use std::sync::{Arc, Mutex};

use crate::nodes::Node;

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: IntMap<u32, Arc<Mutex<IntSet<Node>>>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: IntMap::default(),
        }
    }

    pub fn from_layer_data(data: IntMap<u32, IntMap<u32, Node>>) -> Graph {
        let mut nodes = IntMap::default();
        for (node_id, neighbors) in data.iter() {
            nodes.insert(
                *node_id,
                Arc::new(Mutex::new(IntSet::from_iter(neighbors.values().copied()))),
            );
        }
        Self { nodes }
    }

    pub fn add_node(&mut self, point_id: u32) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    pub fn add_edge(&self, node_a: u32, node_b: u32, dist: Node) -> Result<(), String> {
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

        let dist_to_a = Node::new_with_dist(dist.dist.unwrap(), node_a);
        let dist_to_b = Node::new_with_dist(dist.dist.unwrap(), node_b);

        {
            self.nodes
                .get(&node_a)
                .unwrap()
                .lock()
                .unwrap()
                .insert(dist_to_b);
        }
        {
            self.nodes
                .get(&node_b)
                .unwrap()
                .lock()
                .unwrap()
                .insert(dist_to_a);
        }
        Ok(())
    }

    /// Removes an edge from the Graph.
    /// Since the add_edge method won't allow for self-connecting nodes, we don't check that here.
    /// Returns whether the edge was removed.
    pub fn remove_edge(&self, node_a: u32, dist: Node) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&dist.id) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        if (self.degree(node_a)? == 1) | (self.degree(dist.id)? == 1) {
            return Ok(false);
        }

        let a_rem = {
            self.nodes
                .get(&node_a)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&dist)
        };
        let b_rem = {
            self.nodes
                .get(&dist.id)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&&Node::new_with_dist(dist.dist.unwrap(), node_a))
        };
        Ok(a_rem & b_rem)
    }

    pub fn remove_edge_ignore_degree(&mut self, node_a: u32, dist: Node) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&dist.id) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        let a_rem = self
            .nodes
            .get(&node_a)
            .unwrap()
            .lock()
            .unwrap()
            .remove(&dist);
        let b_rem = self
            .nodes
            .get_mut(&dist.id)
            .unwrap()
            .lock()
            .unwrap()
            .remove(&Node::new_with_dist(dist.dist.unwrap(), node_a));
        Ok(a_rem & b_rem)
    }

    pub fn neighbors(&self, node_id: u32) -> Result<IntSet<Node>, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn replace_or_add_neighbors<I>(&self, node_id: u32, new_neighbors: I) -> Result<(), String>
    where
        I: Iterator<Item = Node>,
    {
        if self.degree(node_id)? == 0 {
            for dist in new_neighbors {
                self.add_edge(node_id, dist.id, dist)?;
            }
            return Ok(());
        }
        let news = IntSet::from_iter(new_neighbors);
        let olds = self.neighbors(node_id)?;

        let to_remove: Vec<Node> = olds.difference(&news).copied().collect();
        let to_add: Vec<Node> = news.difference(&olds).copied().collect();

        for dist in to_add {
            self.add_edge(node_id, dist.id, dist)?;
        }

        for dist in to_remove {
            self.remove_edge(node_id, dist)?;
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node_id: u32) -> Result<(), String> {
        // let neighbors = self.neighbors(node_id)?.clone();
        // for neigh in neighbors {
        //     self.remove_edge_ignore_degree(node_id, neigh)?;
        // }
        for dist in self.remove_neighbors(node_id) {
            self.nodes
                .get_mut(&dist.id)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&Node::new_with_dist(dist.dist.unwrap(), node_id));
        }
        self.nodes.remove(&node_id);
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: u32) {
        for dist in self.remove_neighbors(node_id) {
            self.nodes
                .get_mut(&dist.id)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&Node::new(node_id));
        }
    }

    fn remove_neighbors(&mut self, node_id: u32) -> Vec<Node> {
        match self.nodes.get_mut(&node_id) {
            Some(neighbors) => neighbors.lock().unwrap().drain().collect(),
            None => {
                panic!(
                    "Could not get the neighbors of {node_id}. The graph does not contain this node"
                );
            }
        }
    }

    pub fn degree(&self, node_id: u32) -> Result<usize, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().len()),
            None => Err(format!(
                "Error getting degree of {node_id}, it is not in the graph."
            )),
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node_id: &u32) -> bool {
        self.nodes.contains_key(node_id)
    }
}
