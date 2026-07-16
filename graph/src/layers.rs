use core::panic;

use log::trace;

use crate::{NodeID, graph::Graph};

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
        trace!("Adding a new Graph to the LayeredGraph");
        self.levels.push(graph);
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = &Graph> {
        self.levels.iter()
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
}

#[cfg(test)]
mod test {
    // TODO
}
