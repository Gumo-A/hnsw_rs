use crate::graph::Graph;
use crate::helpers::data::split_ids;
// use crate::helpers::bench::Bencher;
use crate::helpers::distance::v2v_dist;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Dim};
use nohash_hasher::BuildNoHashHasher;
use rand::Rng;
use regex::Regex;
use std::sync::{Arc, RwLock};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug)]
pub struct HNSW {
    m: usize,
    mmax: usize,
    mmax0: usize,
    ml: f32,
    ef_cons: usize,
    ep: usize,
    dim: usize,
    pub node_ids: HashSet<usize, BuildNoHashHasher<usize>>,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
    // pub bencher: RefCell<Bencher>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> Self {
        Self {
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: ef_cons.unwrap_or(m * 2),
            ep: 0,
            dim,
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            // bencher: RefCell::new(Bencher::new()),
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
            ep: 0,
            dim,
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            // bencher: RefCell::new(Bencher::new()),
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

        let nearest_neighbors: BTreeMap<usize, usize> =
            BTreeMap::from_iter(neighbors.iter().map(|x| {
                let dist = (v2v_dist(vector, &self.layers.get(&0).unwrap().node(*x).1) * 10_000.0)
                    .trunc() as usize;
                (dist, *x)
            }));

        let anns: Vec<usize> = nearest_neighbors
            .values()
            .skip(1)
            .take(n)
            .map(|x| *x)
            .collect();
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

    fn step_2_par(
        index: &Arc<RwLock<Self>>,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
    ) {
        let bound = (current_layer_number + 1).min(index.read().unwrap().layers.len());
        for layer_nb in (0..bound).rev() {
            index
                .write()
                .unwrap()
                .layers
                .get_mut(&layer_nb)
                .unwrap()
                .add_node(node_id, vector);

            ep = Self::search_layer_par(
                index,
                layer_nb,
                &vector,
                &mut ep,
                index.read().unwrap().ef_cons,
            );

            let neighbors_to_connect = Self::select_heuristic_par(
                index,
                layer_nb,
                vector,
                &mut ep,
                index.read().unwrap().m,
                false,
                true,
            );

            for neighbor in neighbors_to_connect.iter() {
                index
                    .write()
                    .unwrap()
                    .layers
                    .get_mut(&layer_nb)
                    .unwrap()
                    .add_edge(node_id, *neighbor);
            }

            Self::prune_connexions_par(index, layer_nb, neighbors_to_connect);
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

    fn prune_connexions_par(
        index: &Arc<RwLock<Self>>,
        layer_nb: usize,
        connexions_made: HashSet<usize, BuildNoHashHasher<usize>>,
    ) {
        let read_ref = index.read().unwrap();
        let layer_read = read_ref.layers.get(&layer_nb).unwrap();

        let limit = if layer_nb == 0 {
            read_ref.mmax0
        } else {
            read_ref.mmax
        };
        for neighbor in connexions_made.iter() {
            // let layer = &read_ref.layers.get(&layer_nb).unwrap();
            if ((layer_nb == 0) & (layer_read.degree(*neighbor) > read_ref.mmax0))
                | ((layer_nb > 0) & (layer_read.degree(*neighbor) > read_ref.mmax))
            {
                let neighbor_vec = &layer_read.node(*neighbor).1;
                let mut old_neighbors: HashSet<usize, BuildNoHashHasher<usize>> =
                    HashSet::with_hasher(BuildNoHashHasher::default());
                old_neighbors.clone_from(layer_read.neighbors(*neighbor));
                let new_neighbors = read_ref.select_heuristic(
                    &layer_read,
                    &neighbor_vec,
                    &mut old_neighbors,
                    limit,
                    false,
                    false,
                );

                let mut write_ref = index.write().unwrap();
                let layer_write = write_ref.layers.get_mut(&layer_nb).unwrap();
                for old_neighbor in old_neighbors.iter() {
                    layer_write.remove_edge(*neighbor, *old_neighbor);
                }
                for (idx, new_neighbor) in new_neighbors.iter().enumerate() {
                    layer_write.add_edge(*neighbor, *new_neighbor);
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
        let mut visited = BTreeMap::new();
        let mut selected = BTreeMap::new();

        while (candidates.len() > 0) & (selected.len() < m) {
            let (dist_e, e) = candidates.pop_first().unwrap();
            if selected.len() == 0 {
                selected.insert(dist_e, e);
                continue;
            }

            let e_vector = &layer.node(e).1;
            let mut selected_set = HashSet::with_hasher(BuildNoHashHasher::default());
            selected_set.extend(selected.values());
            let (dist_from_s, _) = self
                .sort_by_distance(layer, &e_vector, &selected_set)
                .pop_first()
                .unwrap();

            if dist_e < dist_from_s {
                selected.insert(dist_e, e);
            } else {
                visited.insert(dist_e, e);
            }

            if keep_pruned {
                while (visited.len() > 0) & (selected.len() < m) {
                    let (dist_e, e) = visited.pop_first().unwrap();
                    selected.insert(dist_e, e);
                }
            }
        }
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        for val in selected.values() {
            result.insert(*val);
        }
        result
    }

    fn select_heuristic_par(
        index: &Arc<RwLock<Self>>,
        layer_nb: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        cands_idx: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        let read_ref = index.read().unwrap();
        let layer_read = read_ref.layers.get(&layer_nb).unwrap();

        if extend_cands {
            for idx in cands_idx.clone().iter() {
                for neighbor in layer_read.neighbors(*idx) {
                    cands_idx.insert(*neighbor);
                }
            }
        }
        let mut candidates = read_ref.sort_by_distance(layer_read, vector, &cands_idx);
        let mut visited = BTreeMap::new();
        let mut selected = BTreeMap::new();

        while (candidates.len() > 0) & (selected.len() < m) {
            let (dist_e, e) = candidates.pop_first().unwrap();
            if selected.len() == 0 {
                selected.insert(dist_e, e);
                continue;
            }

            let e_vector = &layer_read.node(e).1;
            let mut selected_set = HashSet::with_hasher(BuildNoHashHasher::default());
            selected_set.extend(selected.values());
            let (dist_from_s, _) = read_ref
                .sort_by_distance(layer_read, &e_vector, &selected_set)
                .pop_first()
                .unwrap();

            if dist_e < dist_from_s {
                selected.insert(dist_e, e);
            } else {
                visited.insert(dist_e, e);
            }

            if keep_pruned {
                while (visited.len() > 0) & (selected.len() < m) {
                    let (dist_e, e) = visited.pop_first().unwrap();
                    selected.insert(dist_e, e);
                }
            }
        }
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        for val in selected.values() {
            result.insert(*val);
        }
        result
    }

    pub fn insert(
        &mut self,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        level: Option<usize>,
    ) -> bool {
        if self.node_ids.contains(&node_id) {
            return false;
        }

        let current_layer_nb: usize = match level {
            Some(level) => level,
            None => self.get_new_node_layer(),
        };

        let max_layer_nb = self.layers.len() - 1;

        let ep = self.step_1(&vector, max_layer_nb, current_layer_nb);
        self.step_2(node_id, &vector, ep, current_layer_nb);
        self.node_ids.insert(node_id);
        if current_layer_nb > max_layer_nb {
            println!("Inserting new layers!");
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

    pub fn insert_par(
        index: &Arc<RwLock<Self>>,
        node_id: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        level: Option<usize>,
    ) -> bool {
        if index.read().unwrap().node_ids.contains(&node_id) {
            return false;
        }
        index.write().unwrap().node_ids.insert(node_id);

        let current_layer_nb: usize = match level {
            Some(level) => level,
            None => index.read().unwrap().get_new_node_layer(),
        };

        let max_layer_nb = index.read().unwrap().layers.len() - 1;

        let ep = index
            .read()
            .unwrap()
            .step_1(&vector, max_layer_nb, current_layer_nb);
        Self::step_2_par(index, node_id, &vector, ep, current_layer_nb);
        // index
        //     .write()
        //     .unwrap()
        //     .step_2(node_id, vector, ep, current_layer_nb);
        if current_layer_nb > max_layer_nb {
            println!("Inserting new layers!");
            for layer_nb in max_layer_nb + 1..current_layer_nb + 1 {
                let mut layer = Graph::new();
                layer.add_node(node_id, vector);
                index.write().unwrap().layers.insert(layer_nb, layer);
                index.write().unwrap().node_ids.insert(node_id);
            }
            index.write().unwrap().ep = node_id;
        }
        true
    }

    fn first_insert(&mut self, node_id: usize, vector: &Array<f32, Dim<[usize; 1]>>) {
        assert_eq!(self.node_ids.len(), 0);
        assert_eq!(self.layers.len(), 0);

        let mut layer = Graph::new();
        layer.add_node(node_id, vector);
        self.layers.insert(0, layer);
        self.node_ids.insert(node_id);
        self.ep = node_id;
    }

    pub fn build_index(
        &mut self,
        node_ids: Vec<usize>,
        vectors: &Array<f32, Dim<[usize; 2]>>,
        checkpoint: bool,
    ) -> std::io::Result<()> {
        let lim = vectors.dim().0;
        let dim = self.dim;
        let m = self.m;
        let efcons = self.ef_cons;

        let checkpoint_path =
            format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}_m{m}_efcons{efcons}");
        let mut copy_path = checkpoint_path.clone();
        copy_path.push_str("_copy");

        if checkpoint & Path::new(&checkpoint_path).exists() {
            self.load(&checkpoint_path)?;
            self.print_params();
        } else {
            println!("No checkpoint was loaded, building self from scratch.");
            self.first_insert(node_ids[0], &vectors.slice(s![0, ..]).to_owned());
        };

        let mut nodes_remaining = HashSet::with_hasher(BuildNoHashHasher::default());
        nodes_remaining.extend(node_ids.iter());
        let nodes_remaining: Vec<usize> = nodes_remaining
            .difference(&self.node_ids)
            .map(|x| *x)
            .collect();

        let vector_levels: Vec<(usize, usize)> = nodes_remaining
            .iter()
            .map(|x| (*x, self.get_new_node_layer()))
            .collect();

        let nb_nodes = self.node_ids.len();
        let remaining = lim - nb_nodes;
        let bar = get_progress_bar(remaining);
        for (node_id, level) in vector_levels.iter() {
            let inserted = self.insert(
                *node_id,
                &vectors.slice(s![*node_id, ..]).to_owned(),
                Some(*level),
            );
            if inserted {
                bar.inc(1);
            } else {
                bar.reset_eta();
            }
            if checkpoint {
                if ((*node_id != 0) & (node_id % 10_000 == 0) & (inserted)) | (*node_id == lim - 1)
                {
                    println!("Checkpointing in {checkpoint_path}");
                    self.save(&checkpoint_path)?;
                    self.save(&copy_path)?;
                    bar.reset_eta();
                }
            }
        }
        self.save(
            format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}_m{m}_efcons{efcons}")
                .as_str(),
        )?;

        Ok(())
    }

    pub fn build_index_par(
        m: usize,
        node_ids: Vec<usize>,
        vectors: Array<f32, Dim<[usize; 2]>>,
    ) -> Self {
        let (_lim, dim) = vectors.dim();
        let index = Arc::new(RwLock::new(Self::new(m, Some(500), dim)));
        // let vectors = Arc::new(RwLock::new(vectors));

        index
            .write()
            .unwrap()
            .first_insert(node_ids[0], &vectors.slice(s![0, ..]).to_owned());

        let mut handlers = vec![];

        // let nb_threads = std::thread::available_parallelism().unwrap().get();
        let nb_threads = 2;
        for thread_nb in 0..nb_threads {
            let node_ids_split = split_ids(node_ids.clone(), nb_threads as u8, thread_nb as u8);
            let vector_levels: Vec<(usize, usize)> = node_ids_split
                .iter()
                .map(|x| (*x, index.read().unwrap().get_new_node_layer()))
                .collect();
            let index_ref = index.clone();
            let vectors_ref = vectors.clone();
            if thread_nb == 0 {
                let bar = get_progress_bar(vector_levels.len());
                handlers.push(std::thread::spawn(move || {
                    for (node_id, level) in vector_levels.iter() {
                        let inserted = Self::insert_par(
                            &index_ref,
                            *node_id,
                            &vectors_ref.slice(s![*node_id, ..]).to_owned(),
                            Some(*level),
                        );
                        if inserted {
                            bar.inc(1);
                        }
                    }
                }));
            } else {
                handlers.push(std::thread::spawn(move || {
                    for (node_id, level) in vector_levels.iter() {
                        let _ = Self::insert_par(
                            &index_ref,
                            *node_id,
                            &vectors_ref.slice(s![*node_id, ..]).to_owned(),
                            Some(*level),
                        );
                    }
                }));
            }
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }
        Arc::try_unwrap(index).unwrap().into_inner().unwrap()
    }

    fn get_new_node_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        (-rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as usize
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
        let mut selected = candidates.clone();

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

    fn search_layer_par(
        index: &Arc<RwLock<Self>>,
        layer_nb: usize,
        vector: &Array<f32, Dim<[usize; 1]>>,
        ep: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        ef: usize,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        let read_ref = index.read().unwrap();
        let layer_read = read_ref.layers.get(&layer_nb).unwrap();
        let mut candidates = read_ref.sort_by_distance(layer_read, vector, &ep);
        let mut selected = candidates.clone();

        while candidates.len() > 0 {
            let (cand2query_dist, candidate) = candidates.pop_first().unwrap();

            let (f2q_dist, _) = selected.last_key_value().unwrap();

            if &cand2query_dist > f2q_dist {
                break;
            }

            for neighbor in layer_read.neighbors(candidate).iter().map(|x| *x) {
                if !ep.contains(&neighbor) {
                    ep.insert(neighbor);
                    let neighbor_vec = &layer_read.node(neighbor).1;

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
            // rng: rand::thread_rng(),
            // bencher: RefCell::new(Bencher::new()),
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

fn get_progress_bar(remaining: usize) -> ProgressBar {
    let bar = ProgressBar::new(remaining as u64);
    bar.set_style(
                ProgressStyle::with_template(
                    "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed_precise} ] {per_sec} {wide_bar}",
                )
                .unwrap());
    bar.set_message(format!("Inserting vectors"));
    bar
}
