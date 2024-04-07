use crate::graph::Graph;
use crate::helpers::bench::Bencher;
use crate::helpers::distance::v2v_dist;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use rayon::prelude::IntoParallelRefIterator;
use regex::Regex;
// use rand::rngs::StdRng;
// use rand::SeedableRng;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

pub struct HNSW {
    m: usize,
    mmax: usize,
    mmax0: usize,
    ml: f32,
    ef_cons: usize,
    ep: usize,
    pub node_ids: HashSet<usize, BuildNoHashHasher<usize>>,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
    dim: usize,
    rng: rand::rngs::ThreadRng,
    // rng: rand::rngs::StdRng,
    pub bencher: RefCell<Bencher>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> Self {
        Self {
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
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
        m: usize,
        mmax: Option<usize>,
        mmax0: Option<usize>,
        ml: Option<f32>,
        ef_cons: Option<usize>,
        dim: usize,
    ) -> Self {
        Self {
            m,
            mmax: mmax.unwrap_or(m + m / 2),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).log(std::f32::consts::E)),
            ef_cons: ef_cons.unwrap_or(m * 2),
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
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
        let mut ep: HashSet<usize, BuildNoHashHasher<usize>> =
            HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);
        let nb_layer = self.layers.len();

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(&self.layers.get(&(layer_nb)).unwrap(), vector, &mut ep, 1);
        }

        let neighbors = self.search_layer(&self.layers.get(&0).unwrap(), vector, &mut ep, ef);
        let mut nearest_neighbors: Vec<(usize, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = v2v_dist(vector, neighbor_vec);
            nearest_neighbors.push((*neighbor, dist));
        }
        nearest_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
        vector: &Array<f32, Dim<[usize; 1]>>,
        max_layer_nb: usize,
        current_layer_number: usize,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        let mut ep = HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);

        for layer_nb in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_nb).unwrap();
            ep = self.search_layer(layer, vector, &mut ep, 1);
        }
        ep
    }

    fn step_2(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
    ) {
        let bound = (current_layer_number + 1).min(self.layers.len());
        for layer_nb in (0..bound).rev() {
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector);
            let layer = &self.layers.get(&layer_nb).unwrap();

            ep = self.search_layer(layer, &vector, &mut ep, self.ef_cons);

            let neighbors_to_connect =
                self.select_heuristic(&layer, vector, &mut ep, self.m, false, true);

            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }

            self.prune_connexions(layer_nb, neighbors_to_connect);
        }
    }

    fn prune_connexions(
        &mut self,
        layer_nb: usize,
        connexions_made: HashSet<usize, BuildNoHashHasher<usize>>,
    ) {
        let limit = if layer_nb == 0 { self.mmax0 } else { self.mmax };
        for neighbor in connexions_made.iter() {
            let layer = &self.layers.get(&layer_nb).unwrap();
            if ((layer_nb == 0) & (layer.degree(*neighbor) > self.mmax0))
                | ((layer_nb > 0) & (layer.degree(*neighbor) > self.mmax))
            {
                let neighbor_vec = &layer.node(*neighbor).1;
                let mut old_neighbors: HashSet<usize, BuildNoHashHasher<usize>> =
                    HashSet::with_hasher(BuildNoHashHasher::default());
                old_neighbors.clone_from(layer.neighbors(*neighbor));
                let new_neighbors = self.select_heuristic(
                    &layer,
                    &neighbor_vec,
                    &mut old_neighbors,
                    limit,
                    false,
                    false,
                );

                let layer_mut = self.layers.get_mut(&layer_nb).unwrap();
                for old_neighbor in old_neighbors.iter() {
                    layer_mut.remove_edge(*neighbor, *old_neighbor);
                }
                for (idx, new_neighbor) in new_neighbors.iter().enumerate() {
                    layer_mut.add_edge(*neighbor, *new_neighbor);
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
        vector: &Array<f32, Dim<[usize; 1]>>,
        cands_idx: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        if extend_cands {
            for idx in cands_idx.clone().iter() {
                for neighbor in layer.neighbors(*idx) {
                    cands_idx.insert(*neighbor);
                }
            }
        }
        let mut candidates = self.sort_by_distance(layer, vector, &cands_idx);
        let mut selected = BTreeMap::new();
        let mut visited = BTreeMap::new();
        let mut selected_set = HashSet::with_capacity_and_hasher(m, BuildNoHashHasher::default());

        while (candidates.len() > 0) & (selected.len() < m) {
            let (dist_e, e) = candidates.pop_first().unwrap();
            if selected.len() == 0 {
                selected.insert(dist_e, e);
                selected_set.insert(e);
                continue;
            }

            let e_vector = &layer.node(e).1;
            let (dist_from_s, _) = self
                .sort_by_distance(layer, &e_vector, &selected_set)
                .pop_first()
                .unwrap();

            if dist_e < dist_from_s {
                selected.insert(dist_e, e);
                selected_set.insert(e);
            } else {
                visited.insert(dist_e, e);
            }

            if keep_pruned {
                while (visited.len() > 0) & (selected.len() < m) {
                    let (dist_e, e) = visited.pop_first().unwrap();
                    selected_set.insert(e);
                    selected.insert(dist_e, e);
                }
            }
        }
        selected_set
        // filter_sets.selected.set.clone()
    }

    pub fn insert(&mut self, node_id: usize, vector: &Array<f32, Dim<[usize; 1]>>) -> bool {
        if self.node_ids.contains(&node_id) {
            return false;
        }

        let current_layer_nb: usize = self.get_new_node_layer();
        let max_layer_nb = self.layers.len() - 1;

        let ep = self.step_1(&vector, max_layer_nb, current_layer_nb);
        self.step_2(node_id, &vector, ep, current_layer_nb);
        self.node_ids.insert(node_id);
        if current_layer_nb > max_layer_nb {
            for layer_nb in max_layer_nb + 1..current_layer_nb + 1 {
                let mut layer = Graph::new();
                layer.add_node(node_id, vector);
                self.layers.insert(layer_nb, layer);
                self.node_ids.insert(node_id);
            }
            self.ep = node_id;
        }
        true
    }

    fn first_insert(&mut self, node_id: usize, vector: &Array<f32, Dim<[usize; 1]>>) {
        assert_eq!(self.node_ids.len(), 0);
        assert_eq!(self.layers.len(), 0);

        let current_layer_nb: usize = self.get_new_node_layer();
        for layer_nb in 0..current_layer_nb + 1 {
            let mut layer = Graph::new();
            layer.add_node(node_id, vector);
            self.layers.insert(layer_nb, layer);
            self.node_ids.insert(node_id);
        }
        self.ep = node_id;
    }

    pub fn build_index(
        &mut self,
        mut node_ids: Vec<usize>,
        vectors: &Array<f32, Dim<[usize; 2]>>,
        checkpoint: bool,
    ) -> std::io::Result<()> {
        let lim = vectors.dim().0;
        let dim = self.dim;
        let m = self.m;
        let checkpoint_path = format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}_m{m}");
        let mut copy_path = checkpoint_path.clone();
        copy_path.push_str("_copy");

        if checkpoint & Path::new(&checkpoint_path).exists() {
            self.load(&checkpoint_path)?;
        } else {
            println!("No checkpoint was loaded, building self from scratch.");
            self.first_insert(node_ids[0], &vectors.slice(s![0, ..]).to_owned());
            node_ids.remove(0);
        };

        let nb_nodes = self.node_ids.len();

        let bar = ProgressBar::new((lim - nb_nodes) as u64);
        bar.set_style(
                ProgressStyle::with_template(
                    "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed_precise} ] {per_sec} {wide_bar}",
                )
                .unwrap());
        bar.set_message(format!("Inserting vectors"));
        for idx in 1..lim {
            let inserted = self.insert(idx, &vectors.slice(s![idx, ..]).to_owned());
            if inserted {
                bar.inc(1);
            } else {
                bar.reset_eta();
            }
            if checkpoint {
                if ((idx % 10_000 == 0) & (inserted)) | (idx == lim - 1) {
                    println!("Checkpointing in {checkpoint_path}");
                    self.save(&checkpoint_path)?;
                    self.save(&copy_path)?;
                    bar.reset_eta();
                }
            }
        }
        self.save(format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}_m{m}").as_str())?;

        Ok(())
    }

    fn get_new_node_layer(&mut self) -> usize {
        (-self.rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as usize
    }

    fn sort_by_distance(
        &self,
        layer: &Graph,
        vector: &Array<f32, Dim<[usize; 1]>>,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> BTreeMap<usize, usize> {
        let result = others.iter().map(|idx| {
            (
                (v2v_dist(vector, &layer.node(*idx).1) * 10_000.0).trunc() as usize,
                *idx,
            )
        });
        BTreeMap::from_iter(result)
    }

    fn search_layer(
        &self,
        layer: &Graph,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        ef: usize,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        let mut candidates = self.sort_by_distance(layer, vector, &ep);
        let mut selected = self.sort_by_distance(layer, vector, &ep);

        while candidates.len() > 0 {
            let (cand2query_dist, candidate) = candidates.pop_first().unwrap();

            let (f2q_dist, _) = selected.last_key_value().unwrap();

            if &cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in layer.neighbors(candidate).iter().map(|x| *x) {
                if !ep.contains(&neighbor) {
                    ep.insert(neighbor);
                    let neighbor_vec = &layer.node(neighbor).1;

                    let (f2q_dist, _) = selected.last_key_value().unwrap().clone();

                    let n2q_dist = (v2v_dist(&vector, &neighbor_vec) * 10_000.0).trunc() as usize;

                    if (&n2q_dist < f2q_dist) | (selected.len() < ef) {
                        candidates.insert(n2q_dist, neighbor);
                        selected.insert(n2q_dist, neighbor);

                        if selected.len() > ef {
                            selected.pop_last().unwrap();
                        }
                    }
                }
            }
        }
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        for val in selected.values() {
            result.insert(*val);
        }
        result
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

    pub fn from_path(path: &str) -> std::io::Result<Self> {
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
            m: *params.get("m").unwrap() as usize,
            mmax: *params.get("mmax").unwrap() as usize,
            mmax0: *params.get("mmax0").unwrap() as usize,
            ml: *params.get("ml").unwrap() as f32,
            ef_cons: *params.get("ef_cons").unwrap() as usize,
            ep: *params.get("ep").unwrap() as usize,
            node_ids,
            layers,
            dim: *params.get("dim").unwrap() as usize,
            rng: rand::thread_rng(),
            bencher: RefCell::new(Bencher::new()),
        })
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        self.node_ids.clear();
        self.layers.clear();
        let paths = std::fs::read_dir(path)?;
        for file_path in paths {
            let file_name = file_path?;
            let file = File::open(file_name.path())?;
            let reader = BufReader::new(file);
            if file_name.file_name().to_str().unwrap().contains("params") {
                let content: HashMap<String, f32> = serde_json::from_reader(reader)?;
                self.m = *content.get("m").unwrap() as usize;
                self.mmax = *content.get("mmax").unwrap() as usize;
                self.mmax0 = *content.get("mmax0").unwrap() as usize;
                self.ep = *content.get("ep").unwrap() as usize;
                self.ef_cons = *content.get("ef_cons").unwrap() as usize;
                self.dim = *content.get("dim").unwrap() as usize;
                self.ml = *content.get("ml").unwrap() as f32;
            } else if file_name.file_name().to_str().unwrap().contains("node_ids") {
                let content: HashSet<usize, BuildNoHashHasher<usize>> =
                    serde_json::from_reader(reader)?;
                for val in content.iter() {
                    self.node_ids.insert(*val);
                }
            } else if file_name.file_name().to_str().unwrap().contains("layer") {
                let re = Regex::new(r"\d+").unwrap();
                let layer_nb: u8 = re
                    .find(file_name.file_name().to_str().unwrap())
                    .unwrap()
                    .as_str()
                    .parse::<u8>()
                    .expect("Could not parse u8 from file name.");
                let content: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)> =
                    serde_json::from_reader(reader)?;
                self.layers
                    .insert(layer_nb as usize, Graph::from_layer_data(self.dim, content));
            }
        }
        Ok(())
    }
}
