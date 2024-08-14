use core::panic;
use nohash_hasher::BuildNoHashHasher;
use serde::{Deserialize, Serialize};
use std::collections::{hash_set::Drain, HashMap, HashSet};

#[derive(Debug, Serialize, Deserialize, Clone)]
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

    pub fn add_node(&mut self, point_id: usize) {
        if self.nodes.contains_key(&point_id) {
            return ();
        } else {
            self.nodes
                .insert(point_id, HashSet::with_hasher(BuildNoHashHasher::default()));
        };
    }

    pub fn add_edge(&mut self, node_a: usize, node_b: usize) -> Result<(), String> {
        if (node_a == node_b)
            | (!self.nodes.contains_key(&node_a) | !self.nodes.contains_key(&node_b))
        {
            return Ok(());
        }

        match self.nodes.get_mut(&node_a) {
            Some(a_n) => a_n.insert(node_b),
            None => {
                let msg = format!("Error adding edge: {node_a} is not in the graph.");
                return Err(msg);
            }
        };

        match self.nodes.get_mut(&node_b) {
            Some(b_n) => b_n.insert(node_a),
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

    pub fn neighbors(
        &self,
        node_id: usize,
    ) -> Result<&HashSet<usize, BuildNoHashHasher<usize>>, String> {
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
        new_neighbors: HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> Result<(), String> {
        let to_remove: Vec<usize> = match self.nodes.get_mut(&node_id) {
            Some(old_neighbors) => {
                let to_rem = old_neighbors
                    .difference(&new_neighbors)
                    .cloned()
                    .collect::<Vec<usize>>()
                    .clone();
                for i in to_rem.iter() {
                    old_neighbors.remove(&i);
                }
                to_rem
            }
            None => {
                return Err(format!(
                    "Error getting neighbors of {node_id}, (function 'replace_neighbors') it is not in the graph."
                ))
            }
        };
        let to_add = new_neighbors
            .difference(self.nodes.get(&node_id).unwrap())
            .cloned()
            .collect::<Vec<usize>>()
            .clone();
        for i in to_remove {
            self.remove_edge(node_id, i)?;
        }
        for i in to_add {
            self.add_edge(node_id, i)?;
        }
        Ok(())
    }

    pub fn remove_edges_with_node(&mut self, node_id: usize) {
        let drained: Vec<usize> = self._remove_neighbors(node_id).collect();
        for node in drained {
            self.nodes.get_mut(&node).unwrap().remove(&node_id);
        }
    }

    fn _remove_neighbors(&mut self, node_id: usize) -> Drain<'_, usize> {
        match self.nodes.get_mut(&node_id) {
            Some(neighbors) => neighbors.drain(),
            None => {
                panic!("Could not get the neighbors of {node_id}. The graph does not contain this node");
            }
        }
        // .expect(format!("Could not get the neighbors of {node_id}").as_str())
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
