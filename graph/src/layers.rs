use core::panic;

use log::trace;
use serde::{Deserialize, Serialize};

use crate::{NodeID, graph::Graph};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        trace!("Adding a new Graph to the LayeredGraph");
        self.levels.push(graph);
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = &Graph> {
        self.levels.iter()
    }

    /// Adds a Node to its layers, based on its maximum level
    /// Creates layers when needed.
    pub fn add_node(&mut self, point_id: NodeID, level: usize) {
        trace!("Adding node {point_id} with level {level}");
        self.add_level(level);
        self.levels.iter_mut().take(level + 1).for_each(|layer| {
            trace!("Adding node {point_id} to level {}", layer.level);
            layer.add_node(point_id)
        });
    }

    fn add_level(&mut self, level: usize) {
        while self.len() <= level {
            let m = if self.len() == 0 { self.m * 2 } else { self.m };
            let g = Graph::new(self.len(), m);
            trace!(
                "LayeredGraph has {0} levels, the requested level is {1}",
                self.len(),
                level
            );
            self.add_layer(g);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{graph::gen_rand_graph, layers::Layers};

    fn gen_rand_layers() -> Layers {
        let mut layers = Layers::new(12);
        for _ in 0..2 {
            layers.add_layer(gen_rand_graph(4, 2));
        }
        layers
    }

    #[test]
    fn serialize() {
        let layers = gen_rand_layers();
        let l_s = rmp_serde::encode::to_vec(&layers).unwrap();
        let l_d: Layers = rmp_serde::from_slice(&l_s).unwrap();
        assert_eq!(layers.m, l_d.m);
        assert_eq!(layers.len(), l_d.len());
        assert_eq!(layers.get_layer(0).nb_nodes(), l_d.get_layer(0).nb_nodes());
        assert_eq!(layers.get_layer(1).nb_nodes(), l_d.get_layer(1).nb_nodes());
        assert_eq!(
            layers.get_layer(0).degree(0).unwrap(),
            l_d.get_layer(0).degree(0).unwrap()
        );
        assert_eq!(
            layers.get_layer(1).degree(0).unwrap(),
            l_d.get_layer(1).degree(0).unwrap()
        );
        assert_eq!(
            layers.get_layer(0).neighbors(0).unwrap(),
            l_d.get_layer(0).neighbors(0).unwrap()
        );
        assert_eq!(
            layers.get_layer(1).neighbors(0).unwrap(),
            l_d.get_layer(1).neighbors(0).unwrap()
        );
    }
}
