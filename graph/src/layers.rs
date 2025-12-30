use core::panic;
use std::collections::BTreeSet;

use nohash_hasher::IntMap;

use crate::{
    errors::GraphError,
    graph::Graph,
    nodes::{Dist, NodeID},
};

#[derive(Debug, Clone)]
pub struct Layers {
    levels: Vec<Graph>,
    m: usize,
}

impl Layers {
    pub fn new(m: usize) -> Self {
        Self {
            levels: Vec::new(),
            m,
        }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn get_layer(&self, layer_nb: usize) -> &Graph {
        match self.levels.get(layer_nb) {
            Some(g) => g,
            None => panic!("Layer {layer_nb} not found in the structure."),
        }
    }

    pub fn get_layer_mut(&mut self, layer_nb: usize) -> &mut Graph {
        match self.levels.get_mut(layer_nb) {
            Some(g) => g,
            None => panic!("Layer {layer_nb} not found in the structure."),
        }
    }

    pub fn add_layer(&mut self, graph: Graph) {
        self.levels.push(graph);
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = &Graph> {
        self.levels.iter()
    }

    fn add_level(&mut self, level: usize) {
        while self.len() <= level {
            let m = if self.len() == 0 { self.m * 2 } else { self.m };
            self.levels.push(Graph::new(self.len(), m));
        }
    }

    /// Adds a Node to its layers, based on its maximum level
    /// Creates layers when needed.
    pub fn add_node_with_level(&mut self, point_id: NodeID, level: usize) {
        self.add_level(level);
        self.levels
            .iter_mut()
            .take(level + 1)
            .for_each(|layer| layer.add_node(point_id));
    }
}

#[cfg(test)]
mod test {
    // TODO
}
