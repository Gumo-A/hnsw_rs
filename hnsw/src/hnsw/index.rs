use crate::graph::Graph;
use crate::helpers::bench::Bencher;
use crate::helpers::distance::v2v_dist;
use crate::hnsw::filter::{FilterSet, FilterSetHolder};

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use regex::Regex;
// use rand::rngs::StdRng;
// use rand::SeedableRng;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
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
        let mut filter_sets = FilterSetHolder::new(n);
        let nb_layer = self.layers.len();
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(
                &self.layers.get(&(layer_nb)).unwrap(),
                usize::MAX,
                vector,
                &ep,
                1,
                &mut filter_sets,
                &mut cache,
            );
        }

        let neighbors = self.search_layer(
            &self.layers.get(&0).unwrap(),
            usize::MAX,
            vector,
            &ep,
            ef,
            &mut filter_sets,
            &mut cache,
        );
        let mut nearest_neighbors: Vec<(usize, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = self.get_dist(None, None, vector, neighbor_vec, &mut cache);
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
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        max_layer_nb: usize,
        current_layer_number: usize,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        for layer_number in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_number).unwrap();
            if layer.nb_nodes() <= 1 {
                continue;
            }
            ep = self.search_layer(layer, node_id, vector, &ep, 1, filter_sets, cache);
        }
        ep
    }

    fn step_2(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) {
        // self.bencher.borrow_mut().start_timer("step_2");
        for layer_nb in (0..current_layer_number + 1).rev() {
            // self.bencher.borrow_mut().start_timer("add_node_get_layer");
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector);
            let layer = &self.layers.get(&layer_nb).unwrap();
            // self.bencher.borrow_mut().end_timer("add_node_get_layer");

            // self.bencher.borrow_mut().start_timer("search_layer");
            ep = self.search_layer(
                layer,
                node_id,
                &vector,
                &ep,
                self.ef_cons,
                filter_sets,
                cache,
            );
            // self.bencher.borrow_mut().end_timer("search_layer");

            // self.bencher.borrow_mut().start_timer("heuristic");
            let neighbors_to_connect = self.select_heuristic(
                &layer,
                node_id,
                vector,
                &ep,
                self.m,
                false,
                true,
                filter_sets,
                cache,
            );
            // self.bencher.borrow_mut().end_timer("heuristic");

            // self.bencher.borrow_mut().start_timer("connect");
            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }
            // self.bencher.borrow_mut().end_timer("connect");

            // self.bencher.borrow_mut().start_timer("prune");
            self.prune_connexions(layer_nb, neighbors_to_connect, filter_sets, cache);
            // self.bencher.borrow_mut().end_timer("prune");
        }
        // self.bencher.borrow_mut().end_timer("step_2");
    }

    fn prune_connexions(
        &mut self,
        layer_nb: usize,
        connexions_made: HashSet<usize, BuildNoHashHasher<usize>>,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
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
                    *neighbor,
                    &neighbor_vec,
                    &old_neighbors,
                    limit,
                    false,
                    false,
                    filter_sets,
                    cache,
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
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        candidate_indices: &HashSet<usize, BuildNoHashHasher<usize>>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        filter_sets.candidates.fill(candidate_indices);
        filter_sets.selected.clear();
        filter_sets.visited.clear();

        if extend_cands {
            for idx in filter_sets.candidates.set.clone().iter() {
                for neighbor in layer.neighbors(*idx) {
                    if !filter_sets.candidates.set.contains(neighbor) {
                        filter_sets.candidates.set.insert(*neighbor);
                    }
                }
            }
        }

        while (filter_sets.candidates.set.len() > 0) & (filter_sets.selected.set.len() < m) {
            let (e, dist_e) = self.get_nearest(
                layer,
                &filter_sets.candidates,
                vector,
                node_id,
                false,
                cache,
            );
            // filter_vecs.candidates.remove(e);
            filter_sets.candidates.set.remove(&e);

            if filter_sets.selected.set.len() == 0 {
                filter_sets.selected.set.insert(e);
                continue;
            }

            let e_vector = &layer.node(e).1;
            let (_, dist_from_s) =
                self.get_nearest(layer, &filter_sets.selected, &e_vector, e, false, cache);

            if dist_e < dist_from_s {
                filter_sets.selected.set.insert(e);
            } else {
                filter_sets.visited.set.insert(e);
            }

            if keep_pruned {
                while (filter_sets.visited.set.len() > 0) & (filter_sets.selected.set.len() < m) {
                    let (e, _) = self.get_nearest(
                        layer,
                        &filter_sets.visited,
                        vector,
                        node_id,
                        false,
                        cache,
                    );
                    filter_sets.visited.set.remove(&e);
                    filter_sets.selected.set.insert(e);
                }
            }
        }
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        result.clone_from(&filter_sets.selected.set);
        result
        // filter_sets.selected.set.clone()
    }

    fn init_index(&mut self, node_id: usize, vector: &Array<f32, Dim<[usize; 1]>>) {
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
                .add_node(node_id, vector);
        }

        self.ep = node_id;
    }

    pub fn insert(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> bool {
        // self.bencher.borrow_mut().start_timer("insert");
        if (self.layers.len() == 0) & (self.node_ids.is_empty()) {
            self.init_index(node_id, vector);
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

        let mut ep = HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);
        // self.bencher.borrow_mut().start_timer("step_1");
        ep = self.step_1(
            node_id,
            &vector,
            ep,
            max_layer_nb,
            current_layer_nb,
            filter_sets,
            cache,
        );
        // self.bencher.borrow_mut().end_timer("step_1");
        // self.bencher.borrow_mut().start_timer("step_2");
        self.step_2(node_id, &vector, ep, current_layer_nb, filter_sets, cache);
        // self.bencher.borrow_mut().end_timer("step_2");
        self.node_ids.insert(node_id);

        // self.bencher.borrow_mut().end_timer("insert");
        true
    }

    pub fn build_index(&mut self, node_ids: Vec<usize>, vectors: &Array<f32, Dim<[usize; 2]>>) {
        let lim = vectors.dim().0;
        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap());
        bar.set_message(format!("Inserting Embeddings"));

        let mut filter_sets = FilterSetHolder::new(vectors.dim().0);
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();
        for idx in node_ids {
            bar.inc(1);
            self.insert(
                idx,
                &vectors.slice(s![idx, ..]).to_owned(),
                &mut filter_sets,
                &mut cache,
            );
        }
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

    fn sort_by_distance(
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> BTreeMap<usize, usize> {
        let mut result: Vec<(usize, f32)> = others
            .iter()
            .map(|idx| {
                (
                    *idx,
                    self.get_dist(
                        Some(node_id),
                        Some(*idx),
                        vector,
                        &layer.node(*idx).1,
                        cache,
                    ),
                )
            })
            .collect();
        // result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        BTreeMap::from_iter(
            result
                .iter()
                .map(|x| ((x.1 * 10_000.0).trunc() as usize, x.0)),
        )
    }

    fn update_sorted_by_distance(
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        other: usize,
        mut sorted_others: BTreeMap<usize, usize>,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> BTreeMap<usize, usize> {
        let new_dist = self.get_dist(
            Some(node_id),
            Some(other),
            vector,
            &layer.node(other).1,
            cache,
        );
        sorted_others.insert((new_dist * 10_000.0).trunc() as usize, other);
        sorted_others
    }

    fn search_layer(
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &HashSet<usize, BuildNoHashHasher<usize>>,
        ef: usize,
        filter_sets: &mut FilterSetHolder,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        // TODO change to use BTreeMap instead of get_nearest all the time
        // self.bencher.borrow_mut().start_timer("search_layer");
        filter_sets.set_entry_points(ep);

        let mut candidates =
            self.sort_by_distance(layer, node_id, vector, &filter_sets.candidates.set, cache);
        let mut selected =
            self.sort_by_distance(layer, node_id, vector, &filter_sets.candidates.set, cache);

        while candidates.len() > 0 {
            // self.bencher.borrow_mut().start_timer("while block 1");

            let (cand2query_dist, candidate) = candidates.pop_first().unwrap();

            let (f2q_dist, _) = selected.last_key_value().unwrap();

            // self.bencher.borrow_mut().end_timer("while block 1");
            if &cand2query_dist > f2q_dist {
                // self.bencher.borrow_mut().end_timer("while block 1");
                break;
            }

            // self.bencher.borrow_mut().start_timer("while block 2");
            for neighbor in layer.neighbors(candidate).iter().map(|x| *x) {
                if !filter_sets.visited.set.contains(&neighbor) {
                    filter_sets.visited.set.insert(neighbor);
                    let neighbor_vec = &layer.node(neighbor).1;

                    let (f2q_dist, _) = selected.last_key_value().unwrap().clone();

                    let n2q_dist = (self.get_dist(
                        Some(node_id),
                        Some(neighbor),
                        &vector,
                        &neighbor_vec,
                        cache,
                    ) * 10_000.0)
                        .trunc() as usize;

                    if (&n2q_dist < f2q_dist) | (selected.len() < ef) {
                        candidates.insert(n2q_dist, neighbor);
                        selected.insert(n2q_dist, neighbor);

                        if selected.len() > ef {
                            selected.pop_last();
                        }
                    }
                }
            }
            // self.bencher.borrow_mut().end_timer("while block 2");
        }
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        for val in selected.values() {
            result.insert(*val);
        }
        // self.bencher.borrow_mut().end_timer("search_layer");
        result
    }

    fn get_nearest(
        &self,
        layer: &Graph,
        candidates: &FilterSet,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: usize,
        reverse: bool,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> (usize, f32) {
        // self.bencher.borrow_mut().start_timer("get_nearest");
        if reverse {
            let result = candidates
                .set
                .iter()
                .map(|idx| {
                    (
                        *idx,
                        self.get_dist(
                            Some(node_id),
                            Some(*idx),
                            vector,
                            &layer.node(*idx).1,
                            cache,
                        ),
                    )
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            // self.bencher.borrow_mut().end_timer("get_nearest");
            result
        } else {
            let result = candidates
                .set
                .iter()
                .map(|idx| {
                    (
                        *idx,
                        self.get_dist(
                            Some(node_id),
                            Some(*idx),
                            vector,
                            &layer.node(*idx).1,
                            cache,
                        ),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            // self.bencher.borrow_mut().end_timer("get_nearest");
            result
        }
    }

    fn get_dist(
        &self,
        node_a: Option<usize>,
        node_b: Option<usize>,
        a_vec: &Array<f32, Dim<[usize; 1]>>,
        b_vec: &Array<f32, Dim<[usize; 1]>>,
        cache: &mut HashMap<(usize, usize), f32>,
    ) -> f32 {
        match (node_a, node_b) {
            (Some(a), Some(b)) => {
                let key = (a.min(b), a.max(b));
                match cache.get(&key) {
                    Some(dist) => *dist,
                    None => {
                        let dist = v2v_dist(a_vec, b_vec);
                        cache.insert(key, dist);
                        dist
                    }
                }
            }
            (None, None) => {
                return v2v_dist(a_vec, b_vec);
            }
            (_, _) => {
                return f32::MAX;
            }
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
