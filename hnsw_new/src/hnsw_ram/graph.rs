use core::panic;
use indicatif::ProgressBar;
use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: IntMap<u32, Arc<Mutex<IntSet<u32>>>>,
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

    pub fn from_layer_data(data: IntMap<u32, IntMap<u32, u32>>) -> Graph {
        let mut nodes = IntMap::default();
        for (node_id, neighbors) in data.iter() {
            nodes.insert(
                *node_id,
                Arc::new(Mutex::new(IntSet::from_iter(neighbors.values().copied()))),
            );
        }
        Self { nodes }
    }

    pub fn from_edge_list(list: &Vec<(u32, u32)>, bar: &ProgressBar) -> Self {
        let mut nodes: IntMap<u32, Arc<Mutex<IntSet<u32>>>> =
            IntMap::with_capacity_and_hasher(list.len(), BuildNoHashHasher::default());
        for (node_a, node_b) in list.iter() {
            let node_a = *node_a;
            let node_b = *node_b;
            nodes
                .entry(node_a)
                .and_modify(|e| {
                    e.lock().unwrap().insert(node_b);
                })
                .or_insert(Arc::new(Mutex::new(IntSet::from_iter([node_b]))));

            nodes
                .entry(node_b)
                .and_modify(|e| {
                    e.lock().unwrap().insert(node_a);
                })
                .or_insert(Arc::new(Mutex::new(IntSet::from_iter([node_a]))));
            bar.inc(12);
        }
        Self { nodes }
    }

    pub fn from_edge_list_bytes(list: &Vec<u8>, bar: &ProgressBar) -> Self {
        let mut list_parsed = Vec::new();
        assert_eq!(list.len() % 8, 0);

        let mut cursor = 0;
        for _ in 0..(list.len() / 8) {
            let mut node_a_bytes: [u8; 4] = [0; 4];
            let mut node_b_bytes: [u8; 4] = [0; 4];
            for (idx, byte) in list[cursor..(cursor + 4)].iter().enumerate() {
                node_a_bytes[idx] = *byte;
            }
            for (idx, byte) in list[(cursor + 4)..(cursor + 8)].iter().enumerate() {
                node_b_bytes[idx] = *byte;
            }
            list_parsed.push((
                u32::from_be_bytes(node_a_bytes),
                u32::from_be_bytes(node_b_bytes),
            ));
            cursor += 12;
        }
        Self::from_edge_list(&list_parsed, bar)
    }

    pub fn to_edge_list(&self) -> Vec<(u32, u32)> {
        let mut list = HashSet::new();
        for (node, neighbors) in self.nodes.iter() {
            for semi_edge in neighbors.lock().unwrap().iter() {
                let node_min = node.min(&semi_edge);
                let node_max = node.max(&semi_edge);
                list.insert((*node_min, *node_max));
            }
        }
        Vec::from_iter(list.iter().map(|(a, b)| (*a, *b)))
    }

    pub fn to_adjacency_list(&self) -> Vec<(u32, u32)> {
        let mut list = HashSet::new();
        for (node, neighbors) in self.nodes.iter() {
            for semi_edge in neighbors.lock().unwrap().iter() {
                let node_min = node.min(&semi_edge);
                let node_max = node.max(&semi_edge);
                list.insert((*node_min, *node_max));
            }
        }
        Vec::from_iter(list.iter().map(|(a, b)| (*a, *b)))
    }

    pub fn to_edge_list_bytes(&self) -> Vec<Vec<u8>> {
        self.to_edge_list()
            .iter()
            .map(|(a, b)| {
                let (a, b) = (a.to_be_bytes(), b.to_be_bytes());
                let mut edge = Vec::with_capacity(8);
                edge.extend_from_slice(&a);
                edge.extend_from_slice(&b);
                assert_eq!(edge.len(), 8);
                edge
            })
            .collect()
    }

    pub fn to_adjacency_list_bytes(&self) -> Vec<Vec<u8>> {
        self.to_adjacency_list()
            .iter()
            .map(|(a, b)| {
                let (a, b) = (a.to_be_bytes(), b.to_be_bytes());
                let mut edge = Vec::with_capacity(8);
                edge.extend_from_slice(&a);
                edge.extend_from_slice(&b);
                // removing weight: change edge size
                assert_eq!(edge.len(), 8);
                edge
            })
            .collect()
    }

    pub fn to_bytes_el(&self) -> (u64, Vec<u8>) {
        let edges = self.to_edge_list_bytes();
        let nb_edges = edges.len() as u64;
        let edges_stream = edges
            .iter()
            .map(|edge| edge.iter().copied())
            .flatten()
            .collect();
        (nb_edges, edges_stream)
    }

    pub fn to_bytes_al(&self) -> (u32, Vec<u8>) {
        let nodes = self.to_adjacency_list_bytes();
        let edges_stream = nodes
            .iter()
            .map(|node| node.iter().copied())
            .flatten()
            .collect();
        (self.nb_nodes() as u32, edges_stream)
    }

    pub fn add_node(&mut self, point_id: u32) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    pub fn add_edge(&self, node_a: u32, node_b: u32) -> Result<(), String> {
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
    pub fn remove_edge(&self, node_a: u32, node_b: u32) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        if (self.degree(node_a)? == 1) | (self.degree(node_b)? == 1) {
            return Ok(false);
        }

        let a_rem = {
            self.nodes
                .get(&node_a)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node_b)
        };
        let b_rem = {
            self.nodes
                .get(&node_b)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node_a)
        };
        // TODO: handle false case as error
        Ok(a_rem & b_rem)
    }

    pub fn remove_edge_ignore_degree(&mut self, node_a: u32, node_b: u32) -> Result<bool, String> {
        if !self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b) {
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
            .remove(&node_b);
        let b_rem = self
            .nodes
            .get_mut(&node_b)
            .unwrap()
            .lock()
            .unwrap()
            .remove(&node_a);
        Ok(a_rem & b_rem)
    }

    pub fn neighbors(&self, node_id: u32) -> Result<IntSet<u32>, String> {
        match self.nodes.get(&node_id) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        }
    }

    pub fn replace_or_add_neighbors<I>(&self, node_id: u32, new_neighbors: I) -> Result<(), String>
    where
        I: Iterator<Item = u32>,
    {
        if self.degree(node_id)? == 0 {
            for node in new_neighbors {
                self.add_edge(node_id, node)?;
            }
            return Ok(());
        }
        let news = IntSet::from_iter(new_neighbors);
        let olds = self.neighbors(node_id)?;

        let to_remove: Vec<u32> = olds.difference(&news).copied().collect();
        let to_add: Vec<u32> = news.difference(&olds).copied().collect();

        for node in to_add {
            self.add_edge(node_id, node)?;
        }

        for dist in to_remove {
            self.remove_edge(node_id, dist)?;
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node_id: u32) -> Result<(), String> {
        for node in self.remove_neighbors(node_id) {
            self.nodes
                .get_mut(&node)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node_id);
        }
        self.nodes.remove(&node_id);
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: u32) {
        for node in self.remove_neighbors(node_id) {
            self.nodes
                .get_mut(&node)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node_id);
        }
    }

    fn remove_neighbors(&mut self, node_id: u32) -> Vec<u32> {
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
