use core::panic;
use std::collections::BinaryHeap;

use nohash_hasher::IntMap;

use crate::{
    graph::Graph,
    nodes::{Dist, Node},
};

pub struct Layers {
    levels: IntMap<u8, Graph>,
}

impl Layers {
    pub fn new() -> Self {
        Self {
            levels: IntMap::default(),
        }
    }

    pub fn len(&self) -> u8 {
        self.levels.len() as u8
    }

    pub fn get_layer(&self, layer_nb: &u8) -> &Graph {
        match self.levels.get(&layer_nb) {
            Some(g) => g,
            None => panic!("Layer {layer_nb} not found in the structure."),
        }
    }

    pub fn get_layer_mut(&mut self, layer_nb: &u8) -> &mut Graph {
        match self.levels.get_mut(&layer_nb) {
            Some(g) => g,
            None => panic!("Layer {layer_nb} not found in the structure."),
        }
    }

    pub fn add_layer(&mut self, layer_nb: u8, graph: Graph) -> bool {
        if layer_nb as usize == self.levels.len() {
            self.levels.insert(layer_nb, graph).unwrap();
            true
        } else {
            false
        }
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = (&u8, &Graph)> {
        self.levels.iter()
    }

    pub fn add_layer_with_node(&mut self, layer_nb: u8, point_id: Node) {
        let max_layer = self.len() - 1; // to make a useful error message below
        if layer_nb == max_layer + 1 {
            let mut layer = Graph::new();
            layer.add_node(point_id);
            self.levels.insert(layer_nb, layer).unwrap();
        } else {
            panic!("Tried to add layer {layer_nb}, while the current max. layer is {max_layer}");
        }
    }

    /// Adds a Node to the given layer, creating it if it didn't previously exist
    pub fn add_node_to_layer(&mut self, layer_nb: u8, point_id: Node) {
        let layer = self.levels.entry(layer_nb).or_insert(Graph::new());
        layer.add_node(point_id);
    }

    pub fn apply_insertion_results(
        &self,
        layer_nb: &u8,
        node_data: &IntMap<Node, BinaryHeap<Dist>>,
    ) -> Result<(), String> {
        let layer = self.get_layer(&layer_nb);
        for (node, neighbors) in node_data.iter() {
            layer.replace_neighbors(*node, neighbors.iter().map(|dist| dist.id))?;
        }
        Ok(())
    }
}
