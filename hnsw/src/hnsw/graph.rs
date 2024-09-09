use core::panic;
use nohash_hasher::{IntMap, IntSet};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use super::dist::Dist;

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: IntMap<usize, Arc<Mutex<IntSet<Dist>>>>,
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

    pub fn from_layer_data(data: IntMap<usize, IntMap<usize, Dist>>) -> Graph {
        let mut nodes = IntMap::default();
        for (node_id, neighbors) in data.iter() {
            nodes.insert(
                *node_id,
                Arc::new(Mutex::new(IntSet::from_iter(neighbors.values().copied()))),
            );
        }
        Self { nodes }
    }

    pub fn from_edge_list(list: &Vec<(u64, u64, f32)>) -> Self {
        let mut nodes: IntMap<usize, Arc<Mutex<IntSet<Dist>>>> = IntMap::default();
        for (node_a, node_b, weight) in list.iter() {
            let node_a = *node_a as usize;
            let node_b = *node_b as usize;
            nodes
                .entry(node_a)
                .and_modify(|e| {
                    e.lock().unwrap().insert(Dist::new(*weight, node_b));
                })
                .or_insert(Arc::new(Mutex::new(IntSet::from_iter([Dist::new(
                    *weight, node_b,
                )]))));

            nodes
                .entry(node_b)
                .and_modify(|e| {
                    e.lock().unwrap().insert(Dist::new(*weight, node_a));
                })
                .or_insert(Arc::new(Mutex::new(IntSet::from_iter([Dist::new(
                    *weight, node_a,
                )]))));
        }
        Self { nodes }
    }

    pub fn from_edge_list_bytes(list: &Vec<u8>) -> Self {
        let mut list_parsed = Vec::new();
        assert_eq!(list.len() % 20, 0);

        let mut cursor = 0;
        for _ in 0..(list.len() / 20) {
            let mut node_a_bytes: [u8; 8] = [0; 8];
            let mut node_b_bytes: [u8; 8] = [0; 8];
            let mut weight_bytes: [u8; 4] = [0; 4];
            for (idx, byte) in list[cursor..(cursor + 8)].iter().enumerate() {
                node_a_bytes[idx] = *byte;
            }
            for (idx, byte) in list[(cursor + 8)..(cursor + 16)].iter().enumerate() {
                node_b_bytes[idx] = *byte;
            }
            for (idx, byte) in list[(cursor + 16)..(cursor + 20)].iter().enumerate() {
                weight_bytes[idx] = *byte;
            }
            list_parsed.push((
                u64::from_be_bytes(node_a_bytes),
                u64::from_be_bytes(node_b_bytes),
                f32::from_be_bytes(weight_bytes),
            ));
            cursor += 20;
        }
        Self::from_edge_list(&list_parsed)
    }

    pub fn to_edge_list(&self) -> Vec<(u64, u64, f32)> {
        let mut list = HashSet::new();
        for (node, neighbors) in self.nodes.iter() {
            for semi_edge in neighbors.lock().unwrap().iter() {
                let node_min = node.min(&semi_edge.id);
                let node_max = node.max(&semi_edge.id);
                list.insert((*node_min, *node_max, *semi_edge));
            }
        }
        Vec::from_iter(
            list.iter()
                .map(|(a, b, dist)| (*a as u64, *b as u64, dist.dist)),
        )
    }

    pub fn to_edge_list_bytes(&self) -> Vec<Vec<u8>> {
        self.to_edge_list()
            .iter()
            .map(|(a, b, weight)| {
                let (a, b, weight) = (a.to_be_bytes(), b.to_be_bytes(), weight.to_be_bytes());
                let mut edge = Vec::with_capacity(20);
                edge.extend_from_slice(&a);
                edge.extend_from_slice(&b);
                edge.extend_from_slice(&weight);
                assert_eq!(edge.len(), 20);
                edge
            })
            .collect()
    }

    pub fn to_bytes(&self) -> (usize, Vec<u8>) {
        let edges = self.to_edge_list_bytes();
        let nb_edges = edges.len();
        let edges_stream = edges
            .iter()
            .map(|edge| edge.iter().copied())
            .flatten()
            .collect();
        (nb_edges, edges_stream)
    }

    pub fn add_node(&mut self, point_id: usize) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    pub fn add_edge(&self, node_a: usize, node_b: usize, dist: Dist) -> Result<(), String> {
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
    pub fn remove_edge(&self, node_a: usize, dist: Dist) -> Result<bool, String> {
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
                .remove(&Dist::new(dist.dist, node_a))
        };
        Ok(a_rem & b_rem)
    }

    pub fn remove_edge_ignore_degree(&mut self, node_a: usize, dist: Dist) -> Result<bool, String> {
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
            .remove(&Dist::new(dist.dist, node_a));
        Ok(a_rem & b_rem)
    }

    pub fn neighbors(&self, node_id: usize) -> Result<IntSet<Dist>, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn replace_or_add_neighbors<I>(
        &self,
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
        let to_add: Vec<Dist> = news.difference(&olds).copied().collect();

        for dist in to_add {
            self.add_edge(node_id, dist.id, dist)?;
        }

        for dist in to_remove {
            self.remove_edge(node_id, dist)?;
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node_id: usize) -> Result<(), String> {
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
                .remove(&Dist::new(dist.dist, node_id));
        }
        self.nodes.remove(&node_id);
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: usize) {
        for dist in self.remove_neighbors(node_id) {
            self.nodes
                .get_mut(&dist.id)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&Dist::new(0.0, node_id));
        }
    }

    fn remove_neighbors(&mut self, node_id: usize) -> Vec<Dist> {
        match self.nodes.get_mut(&node_id) {
            Some(neighbors) => neighbors.lock().unwrap().drain().collect(),
            None => {
                panic!("Could not get the neighbors of {node_id}. The graph does not contain this node");
            }
        }
    }

    pub fn degree(&self, node_id: usize) -> Result<usize, String> {
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

    pub fn contains(&self, node_id: &usize) -> bool {
        self.nodes.contains_key(node_id)
    }
}
