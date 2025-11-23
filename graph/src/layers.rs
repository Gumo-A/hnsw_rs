use nohash_hasher::IntMap;

use crate::graph::Graph;

pub struct Layers {
    levels: IntMap<u8, Graph>,
}

impl Layers {
    pub fn new() -> Self {
        Self {
            levels: IntMap::default(),
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
}
