// TODO: rayon crate for data parallelization
use ndarray::{Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use regex::Regex;
// use rand::rngs::StdRng;
// use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};

use std::thread;

use crate::graph::Graph;
use crate::helpers::distance::v2v_dist;

pub struct HNSW {
    max_layers: i32,
    m: i32,
    mmax: i32,
    mmax0: i32,
    ml: f32,
    ef_cons: i32,
    ep: i32,
    pub dist_cache: HashMap<(i32, i32), f32>,
    pub node_ids: HashSet<i32, BuildNoHashHasher<i32>>,
    pub layers: HashMap<i32, Graph, BuildNoHashHasher<i32>>,
    dim: u32,
    rng: rand::rngs::ThreadRng,
    // rng: rand::rngs::StdRng,
    nb_threads: u8,
}

impl HNSW {
    pub fn new(max_layers: i32, m: i32, ef_cons: Option<i32>, dim: u32) -> Self {
        Self {
            max_layers,
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            dist_cache: HashMap::new(),
            ep: -1,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            nb_threads: thread::available_parallelism().unwrap().get() as u8,
        }
    }

    pub fn from_params(
        max_layers: i32,
        m: i32,
        mmax: Option<i32>,
        mmax0: Option<i32>,
        ml: Option<f32>,
        ef_cons: Option<i32>,
        dim: u32,
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
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            nb_threads: thread::available_parallelism().unwrap().get() as u8,
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
            println!("NB. nodes in layer {idx}: {}", layer.nb_nodes());
        }
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
        let mut ep: Vec<i32> = Vec::new();
        ep.push(self.ep);
        let nb_layer = self.layers.len();

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(
                &self.layers.get(&(layer_nb as i32)).unwrap(),
                -1,
                vector,
                &ep,
                1,
            );
        }

        let neighbors = self.search_layer(&self.layers.get(&0).unwrap(), -1, vector, &ep, ef);
        let mut nearest_neighbors: Vec<(i32, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = self.get_dist(-1, -2, vector, neighbor_vec);
            nearest_neighbors.push((*neighbor, dist));
        }
        nearest_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
        anns
    }

    fn step_1(
        &mut self,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: Vec<i32>,
        max_layer_nb: i32,
        current_layer_number: i32,
    ) -> Vec<i32> {
        let mut ep: Vec<i32> = Vec::from_iter(ep.iter().map(|x| *x));
        for layer_number in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_number).unwrap();
            if layer.nb_nodes() == 0 {
                continue;
            }
            ep = self.search_layer(layer, node_id, vector, &ep, 1);
        }
        ep
    }

    fn step_2(
        &mut self,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: Vec<i32>,
        current_layer_number: i32,
    ) {
        let mut ep: Vec<i32> = Vec::from_iter(ep.iter().map(|x| *x));
        for layer_nb in (0..current_layer_number + 1).rev() {
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector.clone());
            let layer = &self.layers.get(&layer_nb).unwrap();

            ep = self.search_layer(layer, node_id, &vector, &ep, self.ef_cons);

            let neighbors_to_connect =
                self.select_heuristic(&layer, node_id, vector, &ep, self.m, false, true);

            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }
            self.prune_connexions(layer_nb, neighbors_to_connect);
        }
    }

    fn prune_connexions(&mut self, layer_nb: i32, connexions_made: Vec<i32>) {
        for neighbor in connexions_made.iter() {
            if ((layer_nb == 0)
                & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax0))
                | ((layer_nb > 0)
                    & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax))
            {
                let layer = &self.layers.get(&layer_nb).unwrap();
                let limit = if layer_nb == 0 { self.mmax0 } else { self.mmax };

                let neighbor_vec = layer.node(*neighbor).1.clone();
                let old_neighbors: Vec<i32> =
                    Vec::from_iter(layer.neighbors(*neighbor).iter().map(|x| *x));
                let new_neighbors = self.select_heuristic(
                    &layer,
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
        layer: &Graph,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        candidate_indices: &Vec<i32>,
        m: i32,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Vec<i32> {
        let max_node_id = *layer.nodes.keys().max().unwrap_or(&0) as usize;
        let mut selected: Vec<bool> = Vec::with_capacity(max_node_id);
        let mut visited: Vec<bool> = Vec::with_capacity(max_node_id);
        let mut candidates: Vec<bool> = Vec::with_capacity(max_node_id);
        for _ in 0..max_node_id {
            selected.push(false);
            visited.push(false);
            candidates.push(false);
        }

        if extend_cands {
            for cand in candidates.clone().iter() {
                for neighbor in self.layers.get(&layer_nb).unwrap().neighbors(*cand) {
                    if !candidates[neighbor] {
                        candidates.push(*neighbor);
                    }
                }
            }
        }
        let mut pruned_selected: HashSet<i32, BuildNoHashHasher<i32>> =
            HashSet::with_hasher(BuildNoHashHasher::default());

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

        return selected
            .iter()
            .enumerate()
            .filter(|x| *x.1)
            .map(|x| x.0 as i32)
            .collect();
    }

    pub fn insert(&mut self, node_id: i32, vector: &Array<f32, Dim<[usize; 1]>>) -> bool {
        if node_id < 0 {
            println!("Cannot insert node with index {node_id}, ids must be positive integers.");
            return false;
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

            self.ep = node_id;
            return true;
        } else if self.node_ids.contains(&node_id) {
            return false;
        }

        // vector = norm_vector(vector);
        let mut current_layer_nb: i32 =
            (-self.rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as i32;
        // let max_layer_nb = self.define_new_layers(current_layer_nb, node_id);
        let max_layer_nb = self.layers.len() - 1;
        if current_layer_nb > max_layer_nb as i32 {
            current_layer_nb = max_layer_nb as i32;
        }

        let mut ep = Vec::new();
        ep.push(self.ep);
        ep = self.step_1(node_id, &vector, ep, max_layer_nb as i32, current_layer_nb);
        self.step_2(node_id, &vector, ep, current_layer_nb);
        self.node_ids.insert(node_id);
        true
    }

    pub fn build_index(&mut self, node_ids: Vec<i32>, vectors: &Array<f32, Dim<[usize; 2]>>) {
        assert_eq!(node_ids.len(), vectors.dim().0);

        // TODO: parallelize data and pass it to the threads
    }

    pub fn remove_unused(&mut self) {
        for lyr_nb in 0..(self.layers.len() as i32) {
            if self.layers.contains_key(&lyr_nb) {
                if self.layers.get(&lyr_nb).unwrap().nb_nodes() == 1 {
                    self.layers.remove(&lyr_nb);
                }
            }
        }
    }

    fn search_layer(
        &mut self,
        layer: &Graph,
        node_id: i32,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &Vec<i32>,
        ef: i32,
    ) -> Vec<i32> {
        let max_node_id = *layer.nodes.keys().max().unwrap_or(&0) as usize;
        let ef = ef as usize;

        // let mut visited = ep.clone();
        // let mut candidates = ep.clone();
        // let mut selected = ep.clone();

        let mut selected: Vec<bool> = Vec::with_capacity(max_node_id);
        let mut visited: Vec<bool> = Vec::with_capacity(max_node_id);
        let mut candidates: Vec<bool> = Vec::with_capacity(max_node_id);
        for _ in 0..max_node_id {
            selected.push(false);
            visited.push(false);
            candidates.push(false);
        }
        for entry_point in ep.iter() {
            selected[*entry_point as usize] = true;
            visited[*entry_point as usize] = true;
            candidates[*entry_point as usize] = true;
        }

        while candidates.iter().filter(|x| **x).count() > 0 {
            let (candidate, cand2query_dist) =
                self.get_nearest(layer, &candidates, &vector, node_id, false);
            candidates[candidate as usize] = false;

            let (_, f2q_dist) = self.get_nearest(layer, &selected, &vector, node_id, true);

            if cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in layer
                .neighbors(candidate)
                .clone()
                .iter()
                .map(|x| *x as usize)
            {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let neighbor_vec = layer.node(neighbor as i32).1.clone();

                    let (furthest, f2q_dist) =
                        self.get_nearest(layer, &selected, &vector, node_id, true);

                    let n2q_dist = self.get_dist(node_id, neighbor as i32, &vector, &neighbor_vec);

                    if (n2q_dist < f2q_dist) | (selected.len() < ef) {
                        candidates[neighbor] = true;
                        selected[neighbor] = true;

                        if selected.len() > ef {
                            selected[neighbor] = false;
                        }
                    }
                }
            }
        }

        return selected
            .iter()
            .enumerate()
            .filter(|x| *x.1)
            .map(|x| x.0 as i32)
            .collect();
    }

    fn get_nearest(
        &mut self,
        layer: &Graph,
        candidates: &Vec<bool>,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: i32,
        reverse: bool,
    ) -> (i32, f32) {
        let mut cand_dists: Vec<(i32, f32)> = Vec::new();
        for (idx, cand) in candidates.iter().enumerate() {
            if *cand {
                let i = idx as i32;
                let cand_vec = &layer.node(i).1.clone();
                let dist: f32 = self.get_dist(node_id, i, vector, cand_vec);
                cand_dists.push((i, dist));
            }
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

    pub fn save(&self, index_dir: &str) -> std::io::Result<()> {
        let path = std::path::Path::new(index_dir);
        create_dir_all(path)?;

        let nodes_file = File::create(path.join("node_ids.json"))?;
        let mut writer = BufWriter::new(nodes_file);
        serde_json::to_writer(&mut writer, &self.node_ids)?;
        writer.flush()?;

        let params_file = File::create(path.join("params.json"))?;
        let mut writer = BufWriter::new(params_file);
        let params: HashMap<&str, f32> = HashMap::from([
            ("m", self.m as f32),
            ("mmax", self.mmax as f32),
            ("mmax0", self.mmax0 as f32),
            ("ef_cons", self.ef_cons as f32),
            ("ml", self.ml as f32),
            ("ep", self.ep as f32),
            ("dim", self.dim as f32),
        ]);
        serde_json::to_writer(&mut writer, &params)?;
        writer.flush()?;

        for layer_nb in 0..self.layers.len() {
            let layer_file = File::create(path.join(format!("layer_{layer_nb}.json")))?;
            let mut writer = BufWriter::new(layer_file);
            let mut layer_data: HashMap<i32, (&HashSet<i32, BuildNoHashHasher<i32>>, Vec<f32>)> =
                HashMap::new();

            for (node_id, node_data) in self.layers.get(&(layer_nb as i32)).unwrap().nodes.iter() {
                let neighbors: &HashSet<i32, BuildNoHashHasher<i32>> = &node_data.0;
                let vector: Vec<f32> = node_data.1.to_vec();
                layer_data.insert(*node_id, (neighbors, vector));
            }
            serde_json::to_writer(&mut writer, &layer_data)?;
            writer.flush()?;
        }

        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut params: HashMap<String, f32> = HashMap::new();
        let mut node_ids: HashSet<i32, BuildNoHashHasher<i32>> =
            HashSet::with_hasher(BuildNoHashHasher::default());
        let mut layers: HashMap<i32, Graph, BuildNoHashHasher<i32>> =
            HashMap::with_hasher(BuildNoHashHasher::default());

        let paths = std::fs::read_dir(path)?;
        for file_path in paths {
            let file_name = file_path?;
            let file = File::open(file_name.path())?;
            let reader = BufReader::new(file);
            if file_name.file_name().to_str().unwrap().contains("params") {
                let content: HashMap<String, f32> = serde_json::from_reader(reader)?;
                for (key, val) in content.iter() {
                    params.insert(String::from(key), *val);
                }
            }
        }

        let paths = std::fs::read_dir(path)?;
        for file_path in paths {
            let file_name = file_path?;
            let file = File::open(file_name.path())?;
            let reader = BufReader::new(file);

            if file_name.file_name().to_str().unwrap().contains("node_ids") {
                let content: HashSet<i32, BuildNoHashHasher<i32>> =
                    serde_json::from_reader(reader)?;
                for val in content.iter() {
                    node_ids.insert(*val);
                }
            } else if file_name.file_name().to_str().unwrap().contains("layer") {
                let re = Regex::new(r"\d+").unwrap();
                let layer_nb: u8 = re
                    .find(file_name.file_name().to_str().unwrap())
                    .unwrap()
                    .as_str()
                    .parse::<u8>()
                    .expect("Could not parse u32 from file.");
                let content: HashMap<i32, (HashSet<i32, BuildNoHashHasher<i32>>, Vec<f32>)> =
                    serde_json::from_reader(reader)?;
                let graph = Graph::from_layer_data(*params.get("dim").unwrap() as u32, content);
                layers.insert(layer_nb as i32, graph);
            }
        }

        Ok(Self {
            max_layers: 0,
            m: *params.get("m").unwrap() as i32,
            mmax: *params.get("mmax").unwrap() as i32,
            mmax0: *params.get("mmax0").unwrap() as i32,
            ml: *params.get("ml").unwrap() as f32,
            ef_cons: *params.get("ef_cons").unwrap() as i32,
            ep: *params.get("ep").unwrap() as i32,
            dist_cache: HashMap::new(),
            node_ids,
            layers,
            dim: *params.get("dim").unwrap() as u32,
            rng: rand::thread_rng(),
            nb_threads: thread::available_parallelism().unwrap().get() as u8,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::hnsw::HNSW;
    use ndarray::{Array1, Array2};
    use rand::Rng;

    #[test]
    fn hnsw_construction() {
        let _index: HNSW = HNSW::new(3, 12, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, Some(9), None, None, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, Some(18), None, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, None, Some(0.25), None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, None, None, Some(100), 100);
        let _index: HNSW = HNSW::from_params(3, 12, Some(9), Some(18), Some(0.25), Some(100), 100);
    }

    #[test]
    fn hnsw_insert() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(3, 12, None, dim);
        let n: usize = 100;

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            index.insert(i.try_into().unwrap(), &vector);
        }

        let already_in_index = 0;
        let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
        index.insert(already_in_index, &vector);
        assert_eq!(index.node_ids.len(), n);
    }

    #[test]
    fn hnsw_distance_caching() {
        let mut index: HNSW = HNSW::new(3, 12, None, 100);
        index.cache_distance(0, 1, 0.5);
        index.cache_distance(1, 0, 0.5);
        assert_eq!(index.dist_cache.len(), 1);
        assert_eq!(*index.dist_cache.get(&(0, 1)).unwrap(), 0.5);
    }

    #[test]
    fn build_multithreaded() {
        let mut index = HNSW::new(12, 12, None, 100);
        index.build_index(Vec::from_iter(0..10), &Array2::zeros((10, 100)));
    }
}
