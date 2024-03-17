use std::collections::{HashMap, HashSet};

use crate::graph;
use crate::helpers::distance::norm_vector;
use ndarray::{Array, Dim};
use rand::Rng;

struct HNSW {
    m: i32,
    mmax: i32,
    mmax0: i32,
    ml: f32,
    ef_cons: i32,
    dist_cache: HashMap<(i32, i32), f32>,
    node_ids: HashSet<i32>,
    ep: i32,
    pub layers: Vec<graph::Graph>,
}

impl HNSW {
    pub fn new(
        m: i32,
        mmax: Option<i32>,
        mmax0: Option<i32>,
        ml: Option<f32>,
        ef_cons: Option<i32>,
    ) -> Self {
        Self {
            m,
            mmax: mmax.unwrap_or(m + m / 2),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).log(std::f32::consts::E)),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::new(),
            dist_cache: HashMap::new(),
            ep: -1,
            layers: vec![],
        }
    }

    fn define_new_layers(&mut self, current_layer_nb: i32, node_id: i32) -> i32 {
        let mut max_layer_nb: i32 = (self.layers.len() - 1).try_into().unwrap();
        while current_layer_nb > max_layer_nb {
            self.ep = node_id;
            self.layers.push(graph::Graph::new());
            max_layer_nb += 1;
        }
        max_layer_nb as i32
    }

    fn step_1(
        &self,
        node_id: i32,
        vector: Array<f32, Dim<[usize; 1]>>,
        ep: HashSet<i32>,
        max_layer_nb: i32,
        current_layer_number: i32,
    ) -> HashSet<i32> {
        let mut w: HashSet<i32> = HashSet::new();
        for layer_number in (current_layer_number - 1..max_layer_nb + 1).rev() {
            w = self.search_layer(layer_number, node_id, &vector, &ep, 1);
        }
        w
    }

    pub fn insert(&mut self, node_id: i32, mut vector: Array<f32, Dim<[usize; 1]>>) {
        if (self.layers.len() == 0) & (self.node_ids.is_empty()) {
            self.node_ids.insert(node_id);
            self.layers.push(graph::Graph::new());
            self.layers[0].add_node(node_id, vector);
            self.ep = node_id;
            return;
        } else if self.node_ids.contains(&node_id) {
            return;
        }

        let mut rng = rand::thread_rng();

        vector = norm_vector(vector);
        let current_layer_nb: i32 =
            (-rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as i32;
        let max_layer_nb = self.define_new_layers(current_layer_nb, node_id);

        let mut ep = HashSet::from([self.ep]);
        ep = self.step_1(node_id, vector, ep, max_layer_nb, current_layer_nb);
    }

    fn search_layer(
        &self,
        layer_nb: i32,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &HashSet<i32>,
        ef: i32,
    ) -> HashSet<i32> {
        HashSet::new()
    }
}
