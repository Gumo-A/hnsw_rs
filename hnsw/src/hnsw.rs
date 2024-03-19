use ndarray::{Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::helpers::distance::v2v_dist;

pub struct HNSW {
    max_layers: i32,
    m: i32,
    mmax: i32,
    mmax0: i32,
    ml: f32,
    ef_cons: i32,
    pub dist_cache: HashMap<(i32, i32), f32>,
    pub node_ids: HashSet<i32, BuildNoHashHasher<i32>>,
    ep: i32,
    pub layers: HashMap<i32, Graph, BuildNoHashHasher<i32>>,
    rng: rand::rngs::StdRng,
}

impl HNSW {
    pub fn new(max_layers: i32, m: i32) -> Self {
        Self {
            max_layers,
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: m * 2,
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            dist_cache: HashMap::new(),
            ep: -1,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            rng: StdRng::seed_from_u64(0),
        }
    }

    pub fn from_params(
        max_layers: i32,
        m: i32,
        mmax: Option<i32>,
        mmax0: Option<i32>,
        ml: Option<f32>,
        ef_cons: Option<i32>,
    ) -> Self {
        Self {
            max_layers,
            m,
            mmax: mmax.unwrap_or(m + m / 2),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).log(std::f32::consts::E)),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            dist_cache: HashMap::new(),
            ep: -1,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            rng: StdRng::seed_from_u64(0),
        }
    }

    pub fn print_params(&self) {
        println!("m = {}", self.m);
        println!("mmax = {}", self.mmax);
        println!("mmax0 = {}", self.mmax0);
        println!("ml = {}", self.ml);
        println!("ef_cons = {}", self.ef_cons);
        println!("Nb. layers = {}", self.layers.len());
        println!("Nb. of nodes = {}", self.node_ids.len());
        for (idx, layer) in self.layers.iter() {
            println!("NB. nodes in layer {idx}: {}", layer.order());
            // let mut nodes_alone = 0;
            // for (node, _) in layer.nodes.iter() {
            //     nodes_alone += if layer.degree(*node) == 0 { 1 } else { 0 };
            // }
            // println!("Nodes with degree 0 in layer {idx} = {nodes_alone}")
        }
        // let nodes: Vec<&i32> = self
        //     .layers
        //     .get(self.layers.keys().max().unwrap())
        //     .unwrap()
        //     .nodes
        //     .iter()
        //     .map(|x| x.0)
        //     .collect();
        // println!("Nodes of last layer {nodes:?}");
        println!("ep: {:?}", self.ep);
    }

    pub fn cache_distance(&mut self, node_a: i32, node_b: i32, distance: f32) {
        self.dist_cache
            .insert((node_a.min(node_b), node_a.max(node_b)), distance);
    }

    pub fn ann_by_vector(
        &mut self,
        vector: &Array<f32, Dim<[usize; 1]>>,
        n: i32,
        ef: i32,
    ) -> Vec<i32> {
        let mut ep = HashSet::from([self.ep]);
        let nb_layer = self.layers.len();

        // println!("{:?}", ep);
        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(layer_nb as i32, -1, vector, &ep, 1);
            // println!("{:?}", ep);
        }
        // for e in ep.iter() {
        // println!("{:?}", self.layers[0].neighbors(*e));
        // }

        let neighbors = self.search_layer(0, -1, vector, &ep, ef);
        let mut nearest_neighbors: Vec<(i32, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = self.get_dist(-1, -2, vector, neighbor_vec);
            nearest_neighbors.push((*neighbor, dist));
        }
        nearest_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // let anns: Vec<i32> = Vec::from_iter(nearest_neighbors.iter().map(|x| x.0));
        let mut anns: Vec<i32> = Vec::new();
        for (idx, neighbor) in nearest_neighbors.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            // +1 because dont want to include idx == 0
            // only to measure glove
            if idx == (n + 1) as usize {
                break;
            }
            anns.push(neighbor.0);
        }
        // anns[1..1 + n as usize].to_vec()
        anns
    }

    // fn define_new_layers(&mut self, current_layer_nb: i32, node_id: i32) -> i32 {
    //     if current_layer_nb >= self.max_layers - 1 {
    //         return self.max_layers - 1;
    //     }
    //     let mut max_layer_nb: i32 = (self.layers.len() - 1).try_into().unwrap();
    //     if current_layer_nb > max_layer_nb {
    //         self.ep = node_id;
    //         while current_layer_nb > max_layer_nb {
    //             max_layer_nb += 1;
    //             self.layers.insert(max_layer_nb, Graph::new());
    //         }
    //     }
    //     max_layer_nb
    // }

    fn step_1(
        &mut self,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<i32>,
        max_layer_nb: i32,
        current_layer_number: i32,
    ) -> HashSet<i32> {
        for layer_number in (current_layer_number + 1..max_layer_nb + 1).rev() {
            if self.layers.get(&layer_number).unwrap().order() == 0 {
                continue;
            }
            ep = self.search_layer(layer_number, node_id, vector, &ep, 1);
        }
        ep
    }

    fn step_2(
        &mut self,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<i32>,
        current_layer_number: i32,
    ) {
        for layer_nb in (0..current_layer_number + 1).rev() {
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector.clone());

            ep = self.search_layer(layer_nb, node_id, &vector, &ep, self.ef_cons);

            let neighbors_to_connect =
                self.select_heuristic(layer_nb, node_id, vector, &ep, self.m, false, true);

            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }
            self.prune_connexions(layer_nb, neighbors_to_connect);
        }
    }

    fn prune_connexions(&mut self, layer_nb: i32, connexions_made: HashSet<i32>) {
        for neighbor in connexions_made.iter() {
            if ((layer_nb == 0)
                & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax0))
                | ((layer_nb > 0)
                    & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax))
            {
                let limit = if layer_nb == 0 { self.mmax0 } else { self.mmax };

                let neighbor_vec = self
                    .layers
                    .get(&layer_nb)
                    .unwrap()
                    .node(*neighbor)
                    .1
                    .clone();
                let old_neighbors = self
                    .layers
                    .get(&layer_nb)
                    .unwrap()
                    .neighbors(*neighbor)
                    .clone();
                let new_neighbors = self.select_heuristic(
                    layer_nb,
                    *neighbor,
                    &neighbor_vec,
                    &old_neighbors,
                    limit,
                    false,
                    true,
                );

                for old_neighbor in old_neighbors.iter() {
                    self.layers
                        .get_mut(&layer_nb)
                        .unwrap()
                        .remove_edge(*neighbor, *old_neighbor);
                }
                for (idx, new_neighbor) in new_neighbors.iter().enumerate() {
                    self.layers
                        .get_mut(&layer_nb)
                        .unwrap()
                        .add_edge(*neighbor, *new_neighbor);
                    if idx + 1 == limit as usize {
                        break;
                    }
                }
            }
        }
    }

    fn select_heuristic(
        &mut self,
        layer_nb: i32,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        candidates: &HashSet<i32>,
        m: i32,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> HashSet<i32> {
        let mut candidates = candidates.clone();
        let mut selected = HashSet::new();

        if extend_cands {
            for cand in candidates.clone().iter() {
                for neighbor in self.layers.get(&layer_nb).unwrap().neighbors(*cand) {
                    if !candidates.contains(neighbor) {
                        candidates.insert(*neighbor);
                    }
                }
            }
        }
        let mut pruned_selected: HashSet<i32> = HashSet::new();

        while (candidates.len() > 0) & (selected.len() < m as usize) {
            let (e, dist_e) =
                self.get_nearest(layer_nb as usize, &candidates, vector, node_id, false);
            candidates.remove(&e);

            if selected.len() == 0 {
                selected.insert(e);
                continue;
            }

            let e_vector = self.layers.get(&layer_nb).unwrap().node(e).1.clone();
            let (_, dist_from_r) =
                self.get_nearest(layer_nb as usize, &selected, &e_vector, e, false);

            if dist_e < dist_from_r {
                selected.insert(e);
            } else {
                pruned_selected.insert(e);
            }

            if keep_pruned {
                while (pruned_selected.len() > 0) & (selected.len() < m as usize) {
                    let (e, _) = self.get_nearest(
                        layer_nb as usize,
                        &pruned_selected,
                        vector,
                        node_id,
                        false,
                    );
                    pruned_selected.remove(&e);
                    selected.insert(e);
                }
            }
        }

        return selected;
    }

    pub fn insert(&mut self, node_id: i32, vector: Array<f32, Dim<[usize; 1]>>) {
        if node_id < 0 {
            return;
        } // TODO: report the node is not being added because id is negative
        if (self.layers.len() == 0) & (self.node_ids.is_empty()) {
            self.node_ids.insert(node_id);

            // TODO: This is a cheap way to make the index work
            // but there is a problem linked to the addition of new layers.
            // If during insertion, I insert a new layer on top of the ones before,
            // (re-asaigning the entry point to the inserted node)
            // The results are all wrong.
            for lyr_nb in 0..self.max_layers {
                self.layers.insert(lyr_nb, Graph::new());
                self.layers
                    .get_mut(&(lyr_nb as i32))
                    .unwrap()
                    .add_node(node_id, vector.clone());
            }

            // self.layers.insert(0, Graph::new());
            // self.layers
            //     .get_mut(&0)
            //     .unwrap()
            //     .add_node(node_id, vector.clone());

            self.ep = node_id;
            return;
        } else if self.node_ids.contains(&node_id) {
            return;
        }

        // vector = norm_vector(vector);
        let mut current_layer_nb: i32 =
            (-self.rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as i32;
        // let max_layer_nb = self.define_new_layers(current_layer_nb, node_id);
        let max_layer_nb = self.layers.len() - 1;
        if current_layer_nb > max_layer_nb as i32 {
            current_layer_nb = max_layer_nb as i32;
        }

        let mut ep = HashSet::from([self.ep]);
        ep = self.step_1(node_id, &vector, ep, max_layer_nb as i32, current_layer_nb);
        self.step_2(node_id, &vector, ep, current_layer_nb);
        self.node_ids.insert(node_id);
    }

    pub fn remove_unused(&mut self) {
        for lyr_nb in 0..self.max_layers {
            if self.layers.contains_key(&lyr_nb) {
                if self.layers.get(&lyr_nb).unwrap().order() == 1 {
                    self.layers.remove(&lyr_nb);
                }
            }
        }
    }

    fn search_layer(
        &mut self,
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

            let (_, f2q_dist) = self.get_nearest(layer_nb, &selected, &vector, node_id, true);

            if cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in self
                .layers
                .get(&(layer_nb as i32))
                .unwrap()
                .neighbors(candidate)
                .clone()
                .iter()
            {
                if !visited.contains(neighbor) {
                    visited.insert(*neighbor);
                    let neighbor_vec = self
                        .layers
                        .get(&(layer_nb as i32))
                        .unwrap()
                        .node(*neighbor)
                        .1
                        .clone();

                    let (furthest, f2q_dist) =
                        self.get_nearest(layer_nb, &selected, &vector, node_id, true);

                    let n2q_dist = self.get_dist(node_id, *neighbor, &vector, &neighbor_vec);

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
        &mut self,
        layer_nb: usize,
        candidates: &HashSet<i32>,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: i32,
        reverse: bool,
    ) -> (i32, f32) {
        let mut cand_dists: Vec<(i32, f32)> = Vec::new();
        for cand in candidates.iter() {
            let cand_vec = &self
                .layers
                .get(&(layer_nb as i32))
                .unwrap()
                .node(*cand)
                .1
                .clone();
            let dist: f32 = self.get_dist(node_id, *cand, vector, cand_vec);
            cand_dists.push((*cand, dist));
        }
        cand_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if reverse {
            return *cand_dists.last().unwrap();
        } else {
            return *cand_dists.first().unwrap();
        }
    }

    fn get_dist(
        &mut self,
        node_a: i32,
        node_b: i32,
        a_vec: &Array<f32, Dim<[usize; 1]>>,
        b_vec: &Array<f32, Dim<[usize; 1]>>,
    ) -> f32 {
        let key = (node_a.min(node_b), node_a.max(node_b));
        if self.dist_cache.contains_key(&key) {
            return *self.dist_cache.get(&key).unwrap();
        } else {
            let dist = v2v_dist(a_vec, b_vec);
            if (key.0 >= 0) & (key.1 >= 0) {
                self.dist_cache.insert(key, dist);
            }
            dist
        }
    }
}
