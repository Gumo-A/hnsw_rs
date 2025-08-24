use crate::hnsw_ram::graph::Graph;

pub struct Layers {
    graphs: Vec<Graph>,
}

impl Layers {
    pub fn new() -> Self {
        Layers { graphs: Vec::new() }
    }

    pub fn len(self) -> usize {
        self.graphs.len()
    }
}
