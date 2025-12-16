use core::panic;
use std::collections::BTreeSet;

use nohash_hasher::IntMap;
use vectors::serializer::Serializer;

use crate::{
    errors::GraphError,
    graph::Graph,
    nodes::{Dist, Node},
};

#[derive(Debug, Clone)]
pub struct Layers {
    levels: Vec<Graph>,
}

impl Layers {
    pub fn new() -> Self {
        Self { levels: Vec::new() }
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
            self.levels.push(Graph::new(self.len() as u8));
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
                    GraphError::WouldIsolateNode(n) => {
                        panic!("Trying to replace neighbors for {node}, node {n} would be isolated")
                    }
                },
            }
        }
        Ok(())
    }
}

impl Serializer for Layers {
    fn size(&self) -> usize {
        let mut size = 0;
        for layer in self.iter_layers() {
            size += layer.size();
        }
        size
    }

    /// Val          Bytes
    /// nb_levels    1
    /// levels       variable
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size());
        bytes.push(self.len() as u8);
        for layer in self.iter_layers() {
            bytes.extend(layer.serialize());
        }
        bytes
    }

    /// Val          Bytes
    /// nb_levels    1
    /// levels       variable
    fn deserialize(data: Vec<u8>) -> Layers {
        let mut i = 0;
        let nb_levels = u8::from_be_bytes([data[i]]);
        i += 1;

        let mut levels = Vec::new();
        for _ in 0..nb_levels {
            let nb_bytes = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
            i += 4;
            let g = Graph::deserialize(data[i..i + nb_bytes].try_into().unwrap());
            i += nb_bytes;
            levels.push(g);
        }

        Layers { levels }
    }
}

#[cfg(test)]
mod test {

    use nohash_hasher::IntSet;

    use super::*;
    use crate::{
        graph::{make_rand_graph, simple_graph},
        layers,
    };

    // #[test]
    // fn serialization_round_trip() {
    //     let mut layers = Layers::new();
    //     for l in 0..3 {
    //         let mut g = make_rand_graph(128, 8);
    //         g.level = l as u8;
    //         layers.add_layer(g);
    //     }
    //     let restored = Layers::deserialize(layers.serialize());

    //     assert_eq!(layers.levels.len(), restored.levels.len());

    //     for l in 0..3 {
    //         let original = layers.get_layer(l);
    //         let ser = restored.get_layer(l);
    //         for node in original.iter_nodes() {
    //             assert!(ser.contains(node));
    //             let orig_neigh: IntSet<Node> =
    //                 original.neighbors(node).unwrap().into_iter().collect();
    //             let rest_neigh: IntSet<Node> = ser.neighbors(node).unwrap().into_iter().collect();
    //             assert_eq!(orig_neigh, rest_neigh);
    //         }
    //     }
    // }
}
