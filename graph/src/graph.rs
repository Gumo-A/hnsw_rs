use core::panic;
use nohash_hasher::{IntMap, IntSet};
use std::sync::{Arc, Mutex};

use crate::nodes::Node;

type Neighbors = IntSet<Node>;

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: IntMap<Node, Arc<Mutex<Neighbors>>>,
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

    pub fn iter_nodes(&self) -> impl Iterator<Item = Node> {
        self.nodes.keys().copied()
    }

    pub fn from_layer_data(data: IntMap<Node, IntMap<Node, Node>>) -> Graph {
        let mut nodes = IntMap::default();
        for (node_id, neighbors) in data.iter() {
            nodes.insert(
                *node_id,
                Arc::new(Mutex::new(IntSet::from_iter(neighbors.values().copied()))),
            );
        }
        Self { nodes }
    }

    pub fn add_node(&mut self, point_id: Node) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    pub fn add_edge(&self, node_a: Node, node_b: Node) -> Result<(), String> {
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

        self.nodes
            .get(&node_a)
            .unwrap()
            .lock()
            .unwrap()
            .insert(node_b);
        self.nodes
            .get(&node_b)
            .unwrap()
            .lock()
            .unwrap()
            .insert(node_a);
        Ok(())
    }

    /// Removes an edge from the Graph.
    /// Since the add_edge method won't allow for self-connecting nodes, we don't check that here.
    /// Returns whether the edge was removed.
    pub fn remove_edge(&self, node_a: Node, node_b: Node) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        if (self.degree(node_a)? == 1) | (self.degree(node_b)? == 1) {
            return Ok(false);
        }

        let ab_rem = self
            .nodes
            .get(&node_a)
            .unwrap()
            .lock()
            .unwrap()
            .remove(&node_b);
        let ba_rem = self
            .nodes
            .get(&node_b)
            .unwrap()
            .lock()
            .unwrap()
            .remove(&node_a);
        Ok(ab_rem & ba_rem)
    }

    pub fn neighbors(&self, node_id: Node) -> Result<Neighbors, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn neighbors_vec(&self, node_id: Node) -> Result<Vec<Node>, String> {
        let neighbors = match self.nodes.get(&node_id) {
            Some(neighbors) => neighbors,
            None => {
                return Err(format!(
                    "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
                ));
            }
        };

        let neighbors = neighbors.lock().unwrap().iter().cloned().collect();
        Ok(neighbors)
    }

    pub fn replace_neighbors<I>(&self, node: Node, new_neighbors: I) -> Result<(), String>
    where
        I: Iterator<Item = Node>,
    {
        if self.degree(node)? == 0 {
            for other in new_neighbors {
                self.add_edge(node, other)?;
            }
            return Ok(());
        }
        let news = IntSet::from_iter(new_neighbors);
        let olds = self.neighbors(node)?;

        let to_remove: Vec<Node> = olds.difference(&news).copied().collect();
        let to_add: Vec<Node> = news.difference(&olds).copied().collect();

        for new_neighbor in to_add {
            self.add_edge(node, new_neighbor)?;
        }

        for ex_neighbor in to_remove {
            self.remove_edge(node, ex_neighbor)?;
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node: Node) -> Result<(), String> {
        for ex_neighbor in self.neighbors(node)? {
            self.nodes
                .get_mut(&ex_neighbor)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node);
        }
        self.nodes.remove(&node);
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node: Node) {
        for ex_neighbor in self.remove_neighbors(node) {
            self.nodes
                .get_mut(&ex_neighbor)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node);
        }
    }

    fn remove_neighbors(&mut self, node_id: Node) -> IntSet<Node> {
        let removed = self.nodes.remove(&node_id);
        self.nodes
            .insert(node_id, Arc::new(Mutex::new(IntSet::default())));
        match removed {
            Some(neighbors) => Arc::into_inner(neighbors).unwrap().into_inner().unwrap(),
            None => {
                panic!(
                    "Could not get the neighbors of {node_id}. The graph does not contain this node"
                );
            }
        }
    }

    pub fn degree(&self, node_id: Node) -> Result<usize, String> {
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

    pub fn contains(&self, node_id: &Node) -> bool {
        self.nodes.contains_key(node_id)
    }
}
