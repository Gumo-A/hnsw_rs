use core::panic;
use indicatif::ProgressBar;
use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use super::dist::Dist;

#[derive(Debug, Clone)]
pub struct Graph {
    id_manager: IntMap<u32, usize>,
    deleted_positions: IntSet<usize>,
    max_degree: u8,
    pub nodes: Vec<Option<Arc<Mutex<IntSet<Dist>>>>>,
}

impl Graph {
    pub fn new(max_degree: u8) -> Self {
        Graph {
            id_manager: IntMap::default(),
            deleted_positions: IntSet::default(),
            nodes: Vec::new(),
            max_degree,
        }
    }

    pub fn from_edge_list(list: &Vec<(u32, u32, f32)>, bar: &ProgressBar) -> Self {
        let mut id_manager: IntMap<u32, usize> = IntMap::default();
        let mut nodes = Vec::new();

        let mut max_degree = 0;
        for (node_a, node_b, weight) in list.iter() {
            let node_a = *node_a;
            let node_b = *node_b;
            if !id_manager.contains_key(&node_a) {
                id_manager.insert(node_a, nodes.len());
                nodes.push(Some(Arc::new(Mutex::new(IntSet::default()))));
            }
            if !id_manager.contains_key(&node_b) {
                id_manager.insert(node_b, nodes.len());
                nodes.push(Some(Arc::new(Mutex::new(IntSet::default()))));
            }

            let a_pos = id_manager.get(&node_a).unwrap();
            match nodes.get_mut(*a_pos).unwrap() {
                Some(neighbors) => {
                    let mut neighs = neighbors.lock().unwrap();
                    neighs.insert(Dist::new(*weight, node_b));
                    max_degree = max_degree.max(neighs.len());
                }
                None => {
                    panic!("Failed to build from edge list")
                }
            };

            let b_pos = id_manager.get(&node_b).unwrap();
            match nodes.get_mut(*b_pos).unwrap() {
                Some(neighbors) => {
                    let mut neighs = neighbors.lock().unwrap();
                    neighs.insert(Dist::new(*weight, node_a));
                    max_degree = max_degree.max(neighs.len());
                }
                None => {
                    panic!("Failed to build from edge list")
                }
            };

            bar.inc(12);
        }
        Self {
            id_manager,
            deleted_positions: IntSet::default(),
            nodes,
            max_degree: max_degree as u8,
        }
    }

    pub fn from_edge_list_bytes(list: &Vec<u8>, bar: &ProgressBar) -> Self {
        let mut list_parsed = Vec::new();
        assert_eq!(list.len() % 12, 0);

        let mut cursor = 0;
        for _ in 0..(list.len() / 12) {
            let mut node_a_bytes: [u8; 4] = [0; 4];
            let mut node_b_bytes: [u8; 4] = [0; 4];
            let mut weight_bytes: [u8; 4] = [0; 4];
            for (idx, byte) in list[cursor..(cursor + 4)].iter().enumerate() {
                node_a_bytes[idx] = *byte;
            }
            for (idx, byte) in list[(cursor + 4)..(cursor + 8)].iter().enumerate() {
                node_b_bytes[idx] = *byte;
            }
            for (idx, byte) in list[(cursor + 8)..(cursor + 12)].iter().enumerate() {
                weight_bytes[idx] = *byte;
            }
            list_parsed.push((
                u32::from_be_bytes(node_a_bytes),
                u32::from_be_bytes(node_b_bytes),
                f32::from_be_bytes(weight_bytes),
            ));
            cursor += 12;
        }
        Self::from_edge_list(&list_parsed, bar)
    }

    pub fn to_edge_list(&self) -> Vec<(u32, u32, f32)> {
        let mut list = HashSet::new();
        for (node_id, node_pos) in self.id_manager.iter() {
            let neighbors = match self.nodes.get(*node_pos) {
                Some(n) => match n {
                    Some(nn) => nn,
                    None => continue,
                },
                None => continue,
            };
            for semi_edge in neighbors.lock().unwrap().iter() {
                let node_min = node_id.min(&semi_edge.id);
                let node_max = node_id.max(&semi_edge.id);
                list.insert((*node_min, *node_max, *semi_edge));
            }
        }
        Vec::from_iter(list.iter().map(|(a, b, dist)| (*a, *b, dist.dist)))
    }

    /// The adjacency list is implemented as a tuple that contains:
    ///   1. A list of neighbors
    ///   2. An ID manager
    ///
    /// A neighbor here is a u32 (the neighbor ID) and an f32 (the neighbor's distance to the node).
    /// The list of neighbors is actually a list of lists of fixed size equal to the maximum number
    /// of neighbors in this Graph. This allows for the list of neighbors to be of size max_degree * nb_nodes,
    /// which facilitates looking for a node's neighbors. If a node does not have max_degree neighbors, its
    /// corresponding list will still be of length max_degree, only the last entries will contain a placeholder
    /// value (u32::MAX and f32::MAX, I think). On disk, this list of neighbors will thus take:
    ///   **(4 + 4) * max_degree * nb_nodes** bytes of space
    ///
    /// This list of neighbors is accompanied by the ID manager, which is supposed to be loaded on main memory
    /// as a HashMap. It consists of u32-u64 pairs, mapping the ID of the node to the position of its neighbors
    /// in the list of neighbors. Note that this mapping is with respect to the position of the node's list of
    /// neighbors in the outter list. So the position of a node's first neighbor in the outter list is the value
    /// returned by the ID manager * max_degree, and then, from that position, the next max_degree neighbors (u32,
    /// f32 pairs) are that node's neighbors.
    ///
    /// So, when this adjacency list is stored in the disk, and we want to find a node's neighbors, we will have
    /// to do the following:
    ///   - Load the ID manager into main memory
    ///   - Get the position of the node from the ID manager (refered here as 'node_pos')
    ///   - Compute the position of the node's first neighbor in the file:
    ///     - node_pos * max_degree * 8 (plus a offset, eventually)
    ///   - Use that to find the first neighbor
    ///   - Read from that position to the next max_degree * 8 bytes to get all that node's neighbors.
    pub fn to_adjacency_list(&self) -> (Vec<(u32, f32)>, Vec<(u32, usize)>) {}

    pub fn to_edge_list_bytes(&self) -> Vec<Vec<u8>> {
        self.to_edge_list()
            .iter()
            .map(|(a, b, weight)| {
                let (a, b, weight) = (a.to_be_bytes(), b.to_be_bytes(), weight.to_be_bytes());
                let mut edge = Vec::with_capacity(12);
                edge.extend_from_slice(&a);
                edge.extend_from_slice(&b);
                edge.extend_from_slice(&weight);
                assert_eq!(edge.len(), 12);
                edge
            })
            .collect()
    }

    pub fn to_adjacency_list_bytes(&self) -> Vec<Vec<u8>> {
        self.to_adjaceny_list()
            .iter()
            .map(|(a, b, weight)| {
                let (a, b, weight) = (a.to_be_bytes(), b.to_be_bytes(), weight.to_be_bytes());
                let mut edge = Vec::with_capacity(12);
                edge.extend_from_slice(&a);
                edge.extend_from_slice(&b);
                edge.extend_from_slice(&weight);
                assert_eq!(edge.len(), 12);
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
        if self.id_manager.contains_key(&point_id) {
            ()
        }
        let new_node_pos = self.nodes.len();
        self.nodes
            .push(Some(Arc::new(Mutex::new(IntSet::default()))));
        self.id_manager.insert(point_id, new_node_pos);
    }

    pub fn add_edge(&self, node_a: u32, node_b: u32, dist: Dist) -> Result<(), String> {
        // This if statatements garantee the unwraps() below won't fail.
        if node_a == node_b {
            return Ok(());
        }

        if !self.id_manager.contains_key(&node_a) | !self.id_manager.contains_key(&node_b) {
            println!(
                "node_a is in graph {}",
                self.id_manager.contains_key(&node_a)
            );
            println!(
                "node_b is in graph {}",
                self.id_manager.contains_key(&node_b)
            );
            return Err(format!(
                "Error adding edge, one of the nodes is not in the graph."
            ));
        }

        let dist_to_a = Dist::new(dist.dist, node_a);
        let dist_to_b = Dist::new(dist.dist, node_b);

        {
            let node_a_pos = self.id_manager.get(&node_a).unwrap();
            self.nodes
                .get(*node_a_pos)
                .unwrap()
                .as_ref()
                .expect("tried to add a neighbor to a deleted node")
                .lock()
                .unwrap()
                .insert(dist_to_b);
        }
        {
            let node_b_pos = self.id_manager.get(&node_b).unwrap();
            self.nodes
                .get(*node_b_pos)
                .unwrap()
                .as_ref()
                .expect("tried to add a neighbor to a deleted node")
                .lock()
                .unwrap()
                .insert(dist_to_a);
        }
        Ok(())
    }

    /// Removes an edge from the Graph.
    /// Since the add_edge method won't allow for self-connecting nodes, we don't check that here.
    /// Returns whether the edge was removed.
    pub fn remove_edge(&self, node_a: u32, dist: Dist) -> Result<bool, String> {
        if !self.id_manager.contains_key(&node_a) | !self.id_manager.contains_key(&dist.id) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        if (self.degree(node_a)? == 1) | (self.degree(dist.id)? == 1) {
            return Ok(false);
        }

        let node_a_pos = self.id_manager.get(&node_a).unwrap();
        let a_rem = {
            self.nodes
                .get(*node_a_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&dist)
        };
        let node_b_pos = self.id_manager.get(&dist.id).unwrap();
        let b_rem = {
            self.nodes
                .get(*node_b_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&Dist::new(dist.dist, node_a))
        };
        Ok(a_rem & b_rem)
    }

    pub fn remove_edge_ignore_degree(&mut self, node_a: u32, dist: Dist) -> Result<bool, String> {
        if !self.id_manager.contains_key(&node_a) | !self.id_manager.contains_key(&dist.id) {
            return Err(format!(
                "Error removing edge, one of the nodes don't exist in the graph."
            ));
        }

        let node_a_pos = self.id_manager.get(&node_a).unwrap();
        let a_rem = {
            self.nodes
                .get(*node_a_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&dist)
        };
        let node_b_pos = self.id_manager.get(&dist.id).unwrap();
        let b_rem = {
            self.nodes
                .get(*node_b_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&Dist::new(dist.dist, node_a))
        };
        Ok(a_rem & b_rem)
    }

    pub fn neighbors(&self, node_id: u32) -> Result<IntSet<Dist>, String> {
        let node_pos = match self.id_manager.get(&node_id) {
            Some(pos) => pos,
            None => return Err(format!(
                "Error getting neighbors of {node_id} (function 'neighbors'), it is not in the graph."
            )),
        };
        Ok(self
            .nodes
            .get(*node_pos)
            .unwrap()
            .as_ref()
            .expect("tried to get neighbors of a deleted node")
            .lock()
            .unwrap()
            .clone())
    }

    pub fn replace_or_add_neighbors<I>(&self, node_id: u32, new_neighbors: I) -> Result<(), String>
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

    pub fn remove_node(&mut self, node_id: u32) -> Result<(), String> {
        let node_pos = *self.id_manager.get(&node_id).unwrap();
        for dist in self.remove_neighbors(node_id) {
            let neighbor_pos = self.id_manager.get(&dist.id).unwrap();
            self.nodes
                .get_mut(*neighbor_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&Dist::new(dist.dist, node_id));
        }
        *self.nodes.get_mut(node_pos).unwrap() = None;
        self.deleted_positions.insert(node_pos);
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: u32) {
        for dist in self.remove_neighbors(node_id) {
            let neighbor_pos = self.id_manager.get(&dist.id).unwrap();
            self.nodes
                .get_mut(*neighbor_pos)
                .unwrap()
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .remove(&Dist::new(dist.dist, node_id));
        }
    }

    fn remove_neighbors(&mut self, node_id: u32) -> Vec<Dist> {
        let node_pos = self.id_manager.get(&node_id).unwrap();
        match self.nodes.get_mut(*node_pos) {
            Some(neighbors) => neighbors
                .as_ref()
                .expect("tried to remove the neighbor of a deleted node")
                .lock()
                .unwrap()
                .drain()
                .collect(),

            None => {
                panic!("Could not get the neighbors of {node_id}. The graph does not contain this node");
            }
        }
    }

    pub fn degree(&self, node_id: u32) -> Result<usize, String> {
        let node_pos = match self.id_manager.get(&node_id) {
            Some(pos) => pos,
            None => {
                return Err(format!(
                    "Error getting degree of {node_id}, it is not in the graph."
                ))
            }
        };
        match self.nodes.get(*node_pos) {
            Some(neighbors) => Ok(neighbors
                .as_ref()
                .expect("tried to get degree of a deleted node")
                .lock()
                .unwrap()
                .len()),
            None => Err(format!(
                "Error getting degree of {node_id}, it is not in the graph."
            )),
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node_id: &u32) -> bool {
        self.id_manager.contains_key(node_id)
    }
}
