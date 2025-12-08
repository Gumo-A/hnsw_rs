use core::panic;
use std::collections::BTreeSet;

use nohash_hasher::IntMap;
use vectors::serializer::Serializer;

use crate::{
    graph::Graph,
    nodes::{Dist, Node},
};

#[derive(Debug, Clone)]
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
            self.levels.insert(layer_nb, graph);
            true
        } else {
            false
        }
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = (&u8, &Graph)> {
        self.levels.iter()
    }

    pub fn add_layer_with_node(&mut self, layer_nb: u8, point_id: Node) {
        let nb_layers = self.len();
        if layer_nb == nb_layers {
            let mut layer = Graph::new(layer_nb);
            layer.add_node(point_id);
            self.levels.insert(layer_nb, layer).unwrap();
        } else {
            panic!(
                "Tried to add layer {layer_nb}, while the current max. layer is {0}",
                nb_layers - 1
            );
        }
    }

    /// Adds a Node to the given layer, creating it if it didn't previously exist
    pub fn add_node_to_layer(&mut self, layer_nb: u8, point_id: Node) {
        let layer = self.levels.entry(layer_nb).or_insert(Graph::new(layer_nb));
        layer.add_node(point_id);
    }

    pub fn apply_insertion_results(
        &self,
        layer_nb: &u8,
        node_data: &IntMap<Node, BTreeSet<Dist>>,
    ) -> Result<(), String> {
        let layer = self.get_layer(&layer_nb);
        for (node, neighbors) in node_data.iter() {
            layer.replace_neighbors(*node, neighbors.iter().map(|dist| dist.id))?;
        }
        Ok(())
    }
}

impl Serializer for Layers {
    fn size(&self) -> usize {
        let mut size = 0;
        for (_, layer) in self.iter_layers() {
            size += layer.size();
        }
        size
    }

    /// Val          Bytes
    /// nb_levels    1
    /// levels       variable
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size());
        bytes.push(self.len());
        for (_, layer) in self.iter_layers() {
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

        let mut levels = IntMap::default();
        for _ in 0..nb_levels {
            let nb_bytes = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
            i += 4;
            let g = Graph::deserialize(data[i..i + nb_bytes].try_into().unwrap());
            i += nb_bytes;
            match levels.insert(g.level, g) {
                None => continue,
                Some(_) => {
                    panic!("Re-inserted an already loaded graph.");
                }
            };
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

    #[test]
    fn serialization_round_trip() {
        let mut layers = Layers::new();
        for l in 0..3 {
            let mut g = make_rand_graph(128, 8);
            g.level = l;
            layers.add_layer(l, g);
        }
        let restored = Layers::deserialize(layers.serialize());

        assert_eq!(layers.levels.len(), restored.levels.len());

        for l in 0..3 {
            let original = layers.get_layer(&l);
            let ser = restored.get_layer(&l);
            for node in original.iter_nodes() {
                assert!(ser.contains(node));
                let orig_neigh: IntSet<Node> =
                    original.neighbors(node).unwrap().into_iter().collect();
                let rest_neigh: IntSet<Node> = ser.neighbors(node).unwrap().into_iter().collect();
                assert_eq!(orig_neigh, rest_neigh);
            }
        }
    }
}
