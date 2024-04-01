use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use regex::Regex;
// use rand::rngs::StdRng;
// use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};

use std::cell::RefCell;
use std::thread;

use crate::graph::Graph;
use crate::helpers::distance::v2v_dist;

struct FilterVector {
    vector: Vec<bool>,
    counter: usize,
}
impl FilterVector {
    fn new(capacity: usize) -> Self {
        Self {
            vector: vec![false; capacity],
            counter: 0,
        }
    }
    fn fill(&mut self, entry_points: &Vec<usize>) {
        self.clear();
        for ep in entry_points {
            let ep = *ep as usize;
            self.add(ep);
        }
    }
    fn clear(&mut self) {
        self.vector.fill(false);
        self.counter = 0;
    }
    fn add(&mut self, node_id: usize) {
        self.counter += if self.vector[node_id] {
            0
        } else {
            self.vector[node_id] = true;
            1
        };
    }

    fn remove(&mut self, node_id: usize) {
        self.counter -= if self.vector[node_id] {
            self.vector[node_id] = false;
            1
        } else {
            0
        }
    }
}
pub struct FilterVectorHolder {
    candidates: FilterVector,
    visited: FilterVector,
    selected: FilterVector,
}

impl FilterVectorHolder {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: FilterVector::new(capacity),
            visited: FilterVector::new(capacity),
            selected: FilterVector::new(capacity),
        }
    }
    fn set_entry_points(&mut self, entry_points: &Vec<usize>) {
        self.candidates.fill(entry_points);
        self.visited.fill(entry_points);
        self.selected.fill(entry_points);
    }
}

pub struct HNSW {
    max_layers: usize,
    m: usize,
    mmax: usize,
    mmax0: usize,
    ml: f32,
    ef_cons: usize,
    ep: usize,
    pub dist_cache: RefCell<HashMap<(usize, usize), f32>>,
    pub node_ids: HashSet<usize, BuildNoHashHasher<usize>>,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
    dim: usize,
    rng: rand::rngs::ThreadRng,
    // rng: rand::rngs::StdRng,
    _nb_threads: u8,
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
            dist_cache: RefCell::new(HashMap::new()),
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            _nb_threads: thread::available_parallelism().unwrap().get() as u8,
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
            dist_cache: RefCell::new(HashMap::new()),
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            dim,
            rng: rand::thread_rng(),
            // rng: StdRng::seed_from_u64(0),
            _nb_threads: thread::available_parallelism().unwrap().get() as u8,
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

    pub fn cache_distance(&mut self, node_a: usize, node_b: usize, distance: f32) {
        self.dist_cache
            .borrow_mut()
            .insert((node_a.min(node_b), node_a.max(node_b)), distance);
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

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(
                &self.layers.get(&(layer_nb as usize)).unwrap(),
                usize::MAX,
                vector,
                &ep,
                1,
                &mut filters,
            );
        }

        let neighbors = self.search_layer(
            &self.layers.get(&0).unwrap(),
            usize::MAX,
            vector,
            &ep,
            ef,
            &mut filters,
        );
        let mut nearest_neighbors: Vec<(usize, f32)> = Vec::new();

        for neighbor in neighbors.iter() {
            let neighbor_vec = &self.layers.get(&0).unwrap().node(*neighbor).1.clone();
            let dist: f32 = self.get_dist(usize::MAX, usize::MAX, vector, neighbor_vec);
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
            if idx == (n + 1) as usize {
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
    ) -> Vec<usize> {
        for layer_number in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_number).unwrap();
            if layer.nb_nodes() <= 1 {
                continue;
            }
            ep = self.search_layer(layer, node_id, vector, &ep, 1, filters);
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
    ) {
        for layer_nb in (0..current_layer_number + 1).rev() {
            self.layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector.clone());
            let layer = &self.layers.get(&layer_nb).unwrap();

            ep = self.search_layer(layer, node_id, &vector, &ep, self.ef_cons, filters);

            let neighbors_to_connect =
                self.select_heuristic(&layer, node_id, vector, &ep, self.m, false, true, filters);

            for neighbor in neighbors_to_connect.iter() {
                self.layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }
            self.prune_connexions(layer_nb, neighbors_to_connect, filters);
        }
    }

    fn prune_connexions(
        &mut self,
        layer_nb: usize,
        connexions_made: Vec<usize>,
        filters: &mut FilterVectorHolder,
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
        &self,
        layer: &Graph,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        candidate_indices: &Vec<usize>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
        filters: &mut FilterVectorHolder,
    ) -> Vec<usize> {
        filters.set_entry_points(candidate_indices);

        if extend_cands {
            for (idx, _) in filters
                .candidates
                .vector
                .clone()
                .iter()
                .enumerate()
                .filter(|x| *x.1)
            {
                for neighbor in layer.neighbors(idx as usize) {
                    if !filters.candidates.vector[*neighbor as usize] {
                        filters.candidates.add(*neighbor as usize);
                    }
                }
            }
        }

        while (filters.candidates.counter > 0) & (filters.selected.counter < m as usize) {
            let (e, dist_e) =
                self.get_nearest(layer, &filters.candidates.vector, vector, node_id, false);
            filters.candidates.remove(e as usize);

            if filters.selected.counter == 0 {
                filters.selected.add(e as usize);
                continue;
            }

            let e_vector = &layer.node(e).1;
            let (_, dist_from_s) =
                self.get_nearest(layer, &filters.selected.vector, &e_vector, e, false);

            if dist_e < dist_from_s {
                filters.selected.add(e as usize);
            } else {
                filters.visited.add(e as usize);
            }

            if keep_pruned {
                while (filters.visited.counter > 0) & (filters.selected.counter < m as usize) {
                    let (e, _) =
                        self.get_nearest(layer, &filters.visited.vector, vector, node_id, false);
                    filters.visited.remove(e as usize);
                    filters.selected.add(e as usize);
                }
            }
        }
        let mut found_nodes = 0;

        return filters
            .selected
            .vector
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
            .map(|x| x.0 as usize)
            .collect();
    }

    pub fn insert(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        filters: &mut FilterVectorHolder,
    ) -> bool {
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
                    .get_mut(&(lyr_nb as usize))
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
        if current_layer_nb > max_layer_nb as usize {
            current_layer_nb = max_layer_nb as usize;
        }

        let mut ep = Vec::from([self.ep]);
        ep = self.step_1(
            node_id,
            &vector,
            ep,
            max_layer_nb as usize,
            current_layer_nb,
            filters,
        );
        self.step_2(node_id, &vector, ep, current_layer_nb, filters);
        self.node_ids.insert(node_id);
        true
    }

    pub fn build_index(&mut self, node_ids: Vec<usize>, vectors: &Array<f32, Dim<[usize; 2]>>) {
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
        for idx in node_ids {
            bar.inc(1);
            self.insert(
                idx as usize,
                &vectors.slice(s![idx, ..]).to_owned(),
                &mut filter_vectors,
            );
        }
        self.remove_unused();
    }

    pub fn remove_unused(&mut self) {
        for lyr_nb in 0..(self.layers.len() as usize) {
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
    ) -> Vec<usize> {
        filters.set_entry_points(ep);

        while filters.candidates.counter > 0 {
            let (candidate, cand2query_dist) =
                self.get_nearest(layer, &filters.candidates.vector, &vector, node_id, false);
            filters.candidates.remove(candidate as usize);

            let (_, f2q_dist) =
                self.get_nearest(layer, &filters.selected.vector, &vector, node_id, true);

            if cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in layer
                .neighbors(candidate)
                .clone()
                .iter()
                .map(|x| *x as usize)
            {
                if !filters.visited.vector[neighbor] {
                    filters.visited.add(neighbor);
                    let neighbor_vec = layer.node(neighbor as usize).1.clone();

                    let (furthest, f2q_dist) =
                        self.get_nearest(layer, &filters.selected.vector, &vector, node_id, true);

                    let n2q_dist =
                        self.get_dist(node_id, neighbor as usize, &vector, &neighbor_vec);

                    if (n2q_dist < f2q_dist) | (filters.selected.counter < ef as usize) {
                        filters.candidates.add(neighbor);
                        filters.selected.add(neighbor);

                        if filters.selected.counter > ef as usize {
                            filters.selected.remove(furthest as usize);
                        }
                    }
                }
            }
        }
        let mut found_nodes = 0;

        return filters
            .selected
            .vector
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
            .map(|x| x.0 as usize)
            .collect();
    }

    fn get_nearest(
        &self,
        layer: &Graph,
        candidates: &Vec<bool>,
        vector: &Array<f32, Dim<[usize; 1]>>,
        node_id: usize,
        reverse: bool,
    ) -> (usize, f32) {
        let mut cand_dists: Vec<(usize, f32)> = Vec::new();
        for (idx, _) in candidates.iter().enumerate().filter(|x| *x.1) {
            let i = idx as usize;
            let cand_vec = &layer.node(i).1.clone();
            let dist: f32 = self.get_dist(node_id, i, vector, cand_vec);
            cand_dists.push((i, dist));
        }
        cand_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
    ) -> f32 {
        let key = (node_a.min(node_b), node_a.max(node_b));
        if self.dist_cache.borrow().contains_key(&key) {
            return *self.dist_cache.borrow().get(&key).unwrap();
        } else {
            let dist = v2v_dist(a_vec, b_vec);
            self.dist_cache.borrow_mut().insert(key, dist);
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
            let mut layer_data: HashMap<
                usize,
                (&HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>),
            > = HashMap::new();

            for (node_id, node_data) in self.layers.get(&(layer_nb as usize)).unwrap().nodes.iter()
            {
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
            dist_cache: RefCell::new(HashMap::new()),
            node_ids,
            layers,
            dim: *params.get("dim").unwrap() as usize,
            rng: rand::thread_rng(),
            _nb_threads: thread::available_parallelism().unwrap().get() as u8,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::hnsw::{FilterVectorHolder, HNSW};
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
        let mut filters = FilterVectorHolder::new(n);

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            index.insert(i.try_into().unwrap(), &vector, &mut filters);
        }

        let already_in_index = 0;
        let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
        index.insert(already_in_index, &vector, &mut filters);
        assert_eq!(index.node_ids.len(), n);
    }

    #[test]
    fn ann() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(3, 12, None, dim);
        let n: usize = 100;
        let mut filters = FilterVectorHolder::new(n);

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            index.insert(i.try_into().unwrap(), &vector, &mut filters);
        }

        let n = 10;
        let vector = index.layers.get(&0).unwrap().node(n).1.to_owned();
        let anns = index.ann_by_vector(&vector, 10, 16);
        println!("ANNs of {:?}", n);
        for e in anns {
            println!("{:?}", e);
        }
    }

    #[test]
    fn hnsw_distance_caching() {
        let mut index: HNSW = HNSW::new(3, 12, None, 100);
        index.cache_distance(0, 1, 0.5);
        index.cache_distance(1, 0, 0.5);
        assert_eq!(index.dist_cache.borrow().len(), 1);
        assert_eq!(*index.dist_cache.borrow().get(&(0, 1)).unwrap(), 0.5);
    }

    #[test]
    fn build_multithreaded() {
        let mut index = HNSW::new(12, 12, None, 100);
        index.build_index(Vec::from_iter(0..10), &Array2::zeros((10, 100)));
    }
}
