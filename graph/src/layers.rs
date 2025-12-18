use core::panic;
use std::collections::BTreeSet;

use nohash_hasher::IntMap;

use crate::{
    errors::GraphError,
    graph::Graph,
    nodes::{Dist, Node},
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
            self.levels.push(Graph::new(self.len(), self.m));
        }
    }

    /// Adds a Node to its layers, based on its maximum level
    /// Creates layers when needed.
    pub fn add_node_with_level(&mut self, point_id: Node, layer_nb: usize) {
        self.add_level(layer_nb);
        self.levels
            .iter_mut()
            .take(layer_nb + 1)
            .for_each(|layer| layer.add_node(point_id));
    }

    pub fn apply_insertion_results(
        &self,
        layer_nb: usize,
        node_data: &IntMap<Node, BTreeSet<Dist>>,
    ) -> Result<(), String> {
        let layer = self.get_layer(layer_nb);
        for (node, neighbors) in node_data.iter() {
            match layer.replace_neighbors(*node, neighbors.iter().map(|dist| dist.id)) {
                Ok(()) => {}
                Err(e) => match e {
                    GraphError::NodeNotInGraph(n) => panic!(
                        "Trying to replace neighbors for {node}, node {n} wasn't found in the Graph"
                    ),
                    GraphError::SelfConnection(n) => panic!(
                        "Trying to replace neighbors for {node}, node {n} tried doing a self connection"
                    ),
                    GraphError::DegreeLimitReached(n) => panic!(
                        "Trying to replace neighbors for {node}, node {n} would exceed degree limit"
                    ),
                    GraphError::WouldIsolateNode(n) => {
                        panic!("Trying to replace neighbors for {node}, node {n} would be isolated")
                    }
                },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    // TODO
}
