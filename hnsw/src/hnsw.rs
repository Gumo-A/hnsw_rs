use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::helpers::distance::{norm_vector, v2v_dist};
use ndarray::{Array, Dim};
use rand::Rng;

pub struct HNSW {
    m: i32,
    mmax: i32,
    mmax0: i32,
    ml: f32,
    ef_cons: i32,
    pub dist_cache: HashMap<(i32, i32), f32>,
    pub node_ids: HashSet<i32>,
    ep: i32,
    pub layers: Vec<Graph>,
}

impl HNSW {
    pub fn new(m: i32) -> Self {
        Self {
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: m * 2,
            node_ids: HashSet::new(),
            dist_cache: HashMap::new(),
            ep: -1,
            layers: vec![],
        }
    }

    pub fn from_params(
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

    pub fn print_params(&self) {
        println!("m = {}", self.m);
        println!("mmax = {}", self.mmax);
        println!("mmax0 = {}", self.mmax0);
        println!("ml = {}", self.ml);
        println!("ef_cons = {}", self.ef_cons);
        println!("Nb. layers = {}", self.layers.len());
        println!("Nb. of nodes = {}", self.node_ids.len())
    }

    pub fn cache_distance(&mut self, node_a: i32, node_b: i32, distance: f32) {
        self.dist_cache
            .insert((node_a.min(node_b), node_a.max(node_b)), distance);
    }

    fn define_new_layers(&mut self, current_layer_nb: i32, node_id: i32) -> i32 {
        let mut max_layer_nb: i32 = (self.layers.len() - 1).try_into().unwrap();
        while current_layer_nb > max_layer_nb {
            self.ep = node_id;
            self.layers.push(Graph::new());
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
            self.layers.push(Graph::new());
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
        let layer_nb = layer_nb as usize;
        let ef = ef as usize;

        let mut visited = ep.clone();
        let mut candidates = ep.clone();
        let mut selected = ep.clone();

        while candidates.len() > 0 {
            let (candidate, cand2query_dist) =
                self.get_nearest(layer_nb, &candidates, &vector, node_id, false);
            candidates.remove(&candidate);

            let (furthest, f2q_dist) =
                self.get_nearest(layer_nb, &selected, &vector, node_id, true);

            if cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in self.layers[layer_nb].neighbors(candidate).iter() {
                if !visited.contains(neighbor) {
                    visited.insert(*neighbor);
                    let neighbor_vec = &self.layers[layer_nb].node(*neighbor).1;

                    let (furthest, f2q_dist) =
                        self.get_nearest(layer_nb, &selected, &vector, node_id, true);

                    let n2q_dist = v2v_dist(&vector, neighbor_vec);

                    if (n2q_dist < f2q_dist) | (selected.len() < ef) {
                        candidates.insert(*neighbor);
                        selected.insert(*neighbor);

                        if selected.len() > ef {
                            selected.remove(&furthest);
                        }
                    }
                }
            }
        }

        return selected;
    }

    fn get_nearest(
        &self,
        layer_nb: usize,
        candidates: &HashSet<i32>,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: i32,
        reverse: bool,
    ) -> (i32, f32) {
        let mut cand_dists: Vec<(i32, f32)> = Vec::new();
        for cand in candidates.iter() {
            let cand_vec = &self.layers[layer_nb].node(*cand).1;
            // TODO: cache dist and use cached dists
            cand_dists.push((*cand, v2v_dist(vector, cand_vec)));
        }
        cand_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if reverse {
            return *cand_dists.last().unwrap();
        } else {
            return cand_dists[0];
        }
    }
}
