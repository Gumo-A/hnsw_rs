use super::filter::{FilterVector, FilterVectorHolder};
use crate::graph::Graph;
use crate::helpers::bench::Bencher;
use crate::helpers::distance::v2v_dist;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use regex::Regex;
// use rand::rngs::StdRng;
// use rand::SeedableRng;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};

pub struct HNSW {
    max_layers: usize,
    m: usize,
    mmax: usize,
    mmax0: usize,
    ml: f32,
    ef_cons: usize,
    ep: usize,
    // pub dist_cache: RefCell<HashMap<(usize, usize), f32>>,
    pub node_ids: HashSet<usize, BuildNoHashHasher<usize>>,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
    dim: usize,
    rng: rand::rngs::ThreadRng,
    // rng: rand::rngs::StdRng,
    pub bencher: RefCell<Bencher>,
}

impl HNSW {
    pub fn new(max_layers: usize, m: usize, ef_cons: Option<usize>, dim: usize) -> Self {
        Self {
            max_layers,
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            // dist_cache: RefCell::new(HashMap::new()),
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            // _nb_threads: thread::available_parallelism().unwrap().get() as u8,
            bencher: RefCell::new(Bencher::new()),
        }
    }

    pub fn from_params(
        max_layers: usize,
        m: usize,
        mmax: Option<usize>,
        mmax0: Option<usize>,
        ml: Option<f32>,
        ef_cons: Option<usize>,
        dim: usize,
    ) -> Self {
        Self {
            max_layers,
            m,
            mmax: mmax.unwrap_or(m + m / 2),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).log(std::f32::consts::E)),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            // dist_cache: RefCell::new(HashMap::new()),
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            // _nb_threads: thread::available_parallelism().unwrap().get() as u8,
            bencher: RefCell::new(Bencher::new()),
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

    pub fn ann_by_vector(
        &self,
        vector: &Array<f32, Dim<[usize; 1]>>,
        n: usize,
        ef: usize,
    ) -> Vec<usize> {
        let mut ep: Vec<usize> = Vec::from([self.ep]);
        let mut filters = FilterVectorHolder::new(self.node_ids.len());
        let nb_layer = self.layers.len();
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(
                &self.layers.get(&(layer_nb)).unwrap(),
                usize::MAX,
                vector,
                &ep,
                1,
                &mut filters,
                &mut cache,
            );
        }

        let neighbors = self.search_layer(
            &self.layers.get(&0).unwrap(),
            usize::MAX,
            vector,
            &ep,
            ef,
            &mut filters,
            &mut cache,
        );
        let mut nearest_neighbors: Vec<(usize, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = self.get_dist(0, 0, vector, neighbor_vec, &mut cache);
            nearest_neighbors.push((*neighbor, dist));
        }
        nearest_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        // dbg!(&nearest_neighbors);

        let mut anns: Vec<usize> = Vec::new();
        for (idx, neighbor) in nearest_neighbors.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            // +1 because dont want to include idx == 0
            // only to measure glove
            if idx == (n + 1) {
                break;
            }
            anns.push(neighbor.0);
        }
        anns
    }

    fn step_1(
        &self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: Vec<usize>,
        max_layer_nb: usize,
        current_layer_number: usize,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> Vec<usize> {
        for layer_number in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_number).unwrap();
            if layer.nb_nodes() <= 1 {
                continue;
            }
            self.bencher.borrow_mut().start_timer("search_layer_1");
            ep = self.search_layer(layer, node_id, vector, &ep, 1, filters, cache);
            self.bencher.borrow_mut().end_timer("search_layer_1");
        }
        ep
    }

    fn step_2(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: Vec<usize>,
        current_layer_number: usize,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) {
        for layer_nb in (0..current_layer_number + 1).rev() {
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector.clone());
            let layer = &self.layers.get(&layer_nb).unwrap();

            self.bencher.borrow_mut().start_timer("search_layer_2");
            ep = self.search_layer(layer, node_id, &vector, &ep, self.ef_cons, filters, cache);
            self.bencher.borrow_mut().end_timer("search_layer_2");

            self.bencher
                .borrow_mut()
                .start_timer("heuristic_connect_prune");
            let neighbors_to_connect = self.select_heuristic(
                &layer, node_id, vector, &ep, self.m, false, true, filters, cache,
            );

            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }
            self.prune_connexions(layer_nb, neighbors_to_connect, filters, cache);
            self.bencher
                .borrow_mut()
                .end_timer("heuristic_connect_prune");
        }
    }

    fn prune_connexions(
        &mut self,
        layer_nb: usize,
        connexions_made: Vec<usize>,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) {
        for neighbor in connexions_made.iter() {
            if ((layer_nb == 0)
                & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax0))
                | ((layer_nb > 0)
                    & (self.layers.get(&layer_nb).unwrap().degree(*neighbor) > self.mmax))
            {
                let layer = &self.layers.get(&layer_nb).unwrap();
                let limit = if layer_nb == 0 { self.mmax0 } else { self.mmax };

                let neighbor_vec = layer.node(*neighbor).1.clone();
                let old_neighbors: Vec<usize> =
                    Vec::from_iter(layer.neighbors(*neighbor).iter().map(|x| *x));
                let new_neighbors = self.select_heuristic(
                    &layer,
                    *neighbor,
                    &neighbor_vec,
                    &old_neighbors,
                    limit,
                    false,
                    true,
                    filters,
                    cache,
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
                    if idx + 1 == limit {
                        break;
                    }
                }
            }
        }
    }

    fn select_heuristic(
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        candidate_indices: &Vec<usize>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> Vec<usize> {
        filters.set_entry_points(candidate_indices);

        if extend_cands {
            for (idx, _) in filters
                .candidates
                .bools
                .clone()
                .iter()
                .enumerate()
                .filter(|x| *x.1)
            {
                for neighbor in layer.neighbors(idx) {
                    if !filters.candidates.bools[*neighbor] {
                        filters.candidates.add(*neighbor);
                    }
                }
            }
        }

        while (filters.candidates.counter > 0) & (filters.selected.counter < m) {
            let (e, dist_e) =
                self.get_nearest(layer, &filters.candidates, vector, node_id, false, cache);
            filters.candidates.remove(e);

            if filters.selected.counter == 0 {
                filters.selected.add(e);
                continue;
            }

            let e_vector = &layer.node(e).1;
            let (_, dist_from_s) =
                self.get_nearest(layer, &filters.selected, &e_vector, e, false, cache);

            if dist_e < dist_from_s {
                filters.selected.add(e);
            } else {
                filters.visited.add(e);
            }

            if keep_pruned {
                while (filters.visited.counter > 0) & (filters.selected.counter < m) {
                    let (e, _) =
                        self.get_nearest(layer, &filters.visited, vector, node_id, false, cache);
                    filters.visited.remove(e);
                    filters.selected.add(e);
                }
            }
        }
        let mut found_nodes = 0;
        let final_selected = filters
            .selected
            .bools
            .iter()
            .enumerate()
            .take_while(|x| {
                found_nodes += if *x.1 { 1 } else { 0 };
                if found_nodes == m {
                    found_nodes += 1;
                    return true;
                }
                found_nodes < m
            })
            .filter(|x| *x.1)
            .map(|x| x.0)
            .collect();

        // let final_visited = filters
        //     .visited
        //     .vector
        //     .iter()
        //     .enumerate()
        //     .filter(|x| *x.1)
        //     .map(|x| x.0)
        //     .collect();

        // filters.clear(&final_visited);

        final_selected
    }

    pub fn insert(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> bool {
        self.bencher.borrow_mut().start_timer("insert");

        self.bencher.borrow_mut().start_timer("preliminaries");
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
                    .get_mut(&(lyr_nb))
                    .unwrap()
                    .add_node(node_id, vector.clone());
            }

            self.ep = node_id;
            return true;
        } else if self.node_ids.contains(&node_id) {
            return false;
        }

        // vector = norm_vector(vector);
        let mut current_layer_nb: usize =
            (-self.rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as usize;
        // let max_layer_nb = self.define_new_layers(current_layer_nb, node_id);
        let max_layer_nb = self.layers.len() - 1;
        if current_layer_nb > max_layer_nb {
            current_layer_nb = max_layer_nb;
        }

        let mut ep = Vec::from([self.ep]);
        self.bencher.borrow_mut().end_timer("preliminaries");
        ep = self.step_1(
            node_id,
            &vector,
            ep,
            max_layer_nb,
            current_layer_nb,
            filters,
            cache,
        );
        self.step_2(node_id, &vector, ep, current_layer_nb, filters, cache);
        self.node_ids.insert(node_id);

        self.bencher.borrow_mut().end_timer("insert");
        true
    }

    pub fn build_index(&mut self, node_ids: Vec<usize>, vectors: &Array<f32, Dim<[usize; 2]>>) {
        self.bencher.borrow_mut().start_timer("build");
        let lim = vectors.dim().0;
        assert_eq!(node_ids.len(), lim);

        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap());
        bar.set_message(format!("Inserting Embeddings"));

        let mut filter_vectors = FilterVectorHolder::new(vectors.dim().0);
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();
        for idx in node_ids {
            bar.inc(1);
            self.insert(
                idx,
                &vectors.slice(s![idx, ..]).to_owned(),
                &mut filter_vectors,
                &mut cache,
            );
        }
        self.bencher.borrow_mut().end_timer("build");
        self.remove_unused();
    }

    pub fn remove_unused(&mut self) {
        for lyr_nb in 0..(self.layers.len()) {
            if self.layers.contains_key(&lyr_nb) {
                if self.layers.get(&lyr_nb).unwrap().nb_nodes() == 1 {
                    self.layers.remove(&lyr_nb);
                }
            }
        }
    }

    fn search_layer(
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &Vec<usize>,
        ef: usize,
        filters: &mut FilterVectorHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> Vec<usize> {
        self.bencher.borrow_mut().start_timer("set_entry");
        filters.set_entry_points(ep);
        self.bencher.borrow_mut().end_timer("set_entry");

        while filters.candidates.counter > 0 {
            let (candidate, cand2query_dist) =
                self.get_nearest(layer, &filters.candidates, &vector, node_id, false, cache);
            filters.candidates.remove(candidate);

            let (_, f2q_dist) =
                self.get_nearest(layer, &filters.selected, &vector, node_id, true, cache);

            if cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in layer.neighbors(candidate).iter().map(|x| *x) {
                if !filters.visited.bools[neighbor] {
                    filters.visited.add(neighbor);
                    let neighbor_vec = &layer.node(neighbor).1;

                    let (furthest, f2q_dist) =
                        self.get_nearest(layer, &filters.selected, &vector, node_id, true, cache);

                    let n2q_dist = self.get_dist(node_id, neighbor, &vector, &neighbor_vec, cache);

                    if (n2q_dist < f2q_dist) | (filters.selected.counter < ef) {
                        filters.candidates.add(neighbor);
                        filters.selected.add(neighbor);

                        if filters.selected.counter > ef {
                            filters.selected.remove(furthest);
                        }
                    }
                }
            }
        }
        let mut found_nodes = 0;

        self.bencher
            .borrow_mut()
            .start_timer("final_iter_search_layer");
        let final_selected = filters
            .selected
            .bools
            .iter()
            .enumerate()
            .take_while(|x| {
                found_nodes += if *x.1 { 1 } else { 0 };
                if found_nodes == ef {
                    found_nodes += 1;
                    return true;
                }
                found_nodes < ef
            })
            .filter(|x| *x.1)
            .map(|x| x.0)
            .collect();
        self.bencher
            .borrow_mut()
            .end_timer("final_iter_search_layer");

        // let final_visited = filters
        //     .visited
        //     .vector
        //     .iter()
        //     .enumerate()
        //     .filter(|x| *x.1)
        //     .map(|x| x.0)
        //     .collect();

        // filters.clear(&final_visited);

        final_selected
    }

    fn get_nearest(
        &self,
        layer: &Graph,
        candidates: &FilterVector,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: usize,
        reverse: bool,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> (usize, f32) {
        self.bencher.borrow_mut().start_timer("get_nearest");
        let mut cand_dists: Vec<(usize, f32)> = Vec::new();
        let min_idx = candidates.min_idx.unwrap_or(0);
        let max_idx = candidates.max_idx.unwrap_or(candidates.bools.len() - 1);

        for idx in (min_idx..max_idx + 1).filter(|x| candidates.bools[*x]) {
            let cand_vec = &layer.node(idx).1.clone();
            let dist: f32 = self.get_dist(node_id, idx, vector, cand_vec, cache);
            cand_dists.push((idx, dist));
        }
        cand_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        self.bencher.borrow_mut().end_timer("get_nearest");
        if reverse {
            return *cand_dists.last().unwrap();
        } else {
            return *cand_dists.first().unwrap();
        }
    }

    fn get_dist(
        &self,
        node_a: usize,
        node_b: usize,
        a_vec: &Array<f32, Dim<[usize; 1]>>,
        b_vec: &Array<f32, Dim<[usize; 1]>>,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> f32 {
        let key = (node_a.min(node_b), node_a.max(node_b));
        if cache.contains_key(&key) & (node_a != node_b) {
            self.bencher.borrow_mut().count("cache_hits");
            return *cache.get(&key).unwrap();
        } else {
            self.bencher.borrow_mut().count("dist_computations");
            let dist = v2v_dist(a_vec, b_vec);
            cache.insert(key, dist);
            dist
        }

        // v2v_dist(a_vec, b_vec)
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
            let mut layer_data: HashMap<
                usize,
                (&HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>),
            > = HashMap::new();

            for (node_id, node_data) in self.layers.get(&(layer_nb)).unwrap().nodes.iter() {
                let neighbors: &HashSet<usize, BuildNoHashHasher<usize>> = &node_data.0;
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
        let mut node_ids: HashSet<usize, BuildNoHashHasher<usize>> =
            HashSet::with_hasher(BuildNoHashHasher::default());
        let mut layers: HashMap<usize, Graph, BuildNoHashHasher<usize>> =
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
                let content: HashSet<usize, BuildNoHashHasher<usize>> =
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
                let content: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)> =
                    serde_json::from_reader(reader)?;
                let graph = Graph::from_layer_data(*params.get("dim").unwrap() as usize, content);
                layers.insert(layer_nb as usize, graph);
            }
        }

        Ok(Self {
            max_layers: 0,
            m: *params.get("m").unwrap() as usize,
            mmax: *params.get("mmax").unwrap() as usize,
            mmax0: *params.get("mmax0").unwrap() as usize,
            ml: *params.get("ml").unwrap() as f32,
            ef_cons: *params.get("ef_cons").unwrap() as usize,
            ep: *params.get("ep").unwrap() as usize,
            // dist_cache: RefCell::new(HashMap::new()),
            node_ids,
            layers,
            dim: *params.get("dim").unwrap() as usize,
            rng: rand::thread_rng(),
            // _nb_threads: thread::available_parallelism().unwrap().get() as u8,
            bencher: RefCell::new(Bencher::new()),
        })
    }
}
