use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::graph::Graph;
use crate::helpers::data::split_ids;
// use crate::helpers::bench::Bencher;
use crate::helpers::distance::v2v_dist;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, ArrayView, Dim};
use nohash_hasher::BuildNoHashHasher;
use parking_lot::RwLock;
use rand::Rng;
use regex::Regex;
use std::sync::Arc;

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
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
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
                let dist = (v2v_dist(vector, &self.layers.get(&0).unwrap().node(*x).1.view())
                    * 10_000.0)
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
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
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
        &self,
        node_id: usize,
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
    ) -> HashMap<usize, HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>> {
        let mut insertion_results = HashMap::new();
        let bound = (current_layer_number + 1).min(self.layers.len());

        for layer_nb in (0..bound).rev() {
            let layer = &self.layers.get(&layer_nb).unwrap();

            ep = self.search_layer(layer, &vector, &mut ep, self.ef_cons);

            let neighbors_to_connect =
                self.select_heuristic(&layer, vector, &mut ep, self.m, false, true);
            let prune_results = self.prune_connexions(layer_nb, &neighbors_to_connect);

            insertion_results.insert(layer_nb, HashMap::new());
            insertion_results
                .get_mut(&layer_nb)
                .unwrap()
                .insert(node_id, neighbors_to_connect);
            insertion_results
                .get_mut(&layer_nb)
                .unwrap()
                .extend(prune_results.iter().map(|x| (*x.0, x.1.to_owned())));
        }
        insertion_results
    }

    fn prune_connexions(
        &self,
        layer_nb: usize,
        connexions_made: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>> {
        let mut prune_results = HashMap::new();
        let limit = if layer_nb == 0 { self.mmax0 } else { self.mmax };

        for neighbor in connexions_made.iter() {
            let layer = &self.layers.get(&layer_nb).unwrap();
            if ((layer_nb == 0) & (layer.degree(*neighbor) > self.mmax0))
                | ((layer_nb > 0) & (layer.degree(*neighbor) > self.mmax))
            {
                let neighbor_vec = &layer.node(*neighbor).1.view();
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
                prune_results.insert(*neighbor, new_neighbors);
            }
        }
        prune_results
    }

    fn select_heuristic(
        &self,
        layer: &Graph,
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
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

            let e_vector = &layer.node(e).1.view();
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

    pub fn insert(
        &mut self,
        node_id: usize,
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
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
        let insertion_results = self.step_2(node_id, &vector, ep, current_layer_nb);

        self.node_ids.insert(node_id);
        for (layer_nb, node_data) in insertion_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                if *node == node_id {
                    layer.add_node(node_id, vector);
                }
                for old_neighbor in layer.neighbors(*node).clone() {
                    layer.remove_edge(*node, old_neighbor);
                }
                for neighbor in neighbors.iter() {
                    layer.add_edge(*node, *neighbor);
                }
            }
        }
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

    pub fn insert_par(
        index: &Arc<RwLock<Self>>,
        ids_levels: Vec<(usize, usize)>,
        vectors: &Array<f32, Dim<[usize; 2]>>,
        bar: ProgressBar,
    ) {
        let mut batch = Vec::new();
        let batch_size = 36;
        for (idx, (node_id, current_layer_nb)) in ids_levels.iter().enumerate() {
            bar.inc(1);
            let read_ref = index.read();
            if read_ref.node_ids.contains(&node_id) {
                continue;
            }
            let max_layer_nb = read_ref.layers.len() - 1;
            let vector = &vectors.slice(s![*node_id, ..]);
            let ep = read_ref.step_1(vector, max_layer_nb, *current_layer_nb);
            let insertion_results = read_ref.step_2(*node_id, vector, ep, *current_layer_nb);
            batch.push(insertion_results);
            drop(read_ref);

            let last_idx = idx == (ids_levels.len() - 1);
            let new_layer = current_layer_nb > &max_layer_nb;
            let full_batch = batch.len() >= batch_size;
            let have_to_write: bool = last_idx | new_layer | full_batch;

            let mut write_ref = if have_to_write {
                index.write()
            } else {
                continue;
            };
            for i in batch.iter() {
                for (layer_nb, node_data) in i.iter() {
                    write_ref.node_ids.extend(node_data.keys());
                    let layer = write_ref.layers.get_mut(&layer_nb).unwrap();
                    for (node, neighbors) in node_data.iter() {
                        layer.add_node(*node, &vectors.slice(s![*node, ..]));
                        for old_neighbor in layer.neighbors(*node).clone() {
                            layer.remove_edge(*node, old_neighbor);
                        }
                        for neighbor in neighbors.iter() {
                            layer.add_edge(*node, *neighbor);
                        }
                    }
                }
            }
            if current_layer_nb > &max_layer_nb {
                for layer_nb in max_layer_nb + 1..current_layer_nb + 1 {
                    let mut layer = Graph::new();
                    layer.add_node(*node_id, vector);
                    write_ref.layers.insert(layer_nb, layer);
                    write_ref.node_ids.insert(*node_id);
                }
                write_ref.ep = *node_id;
            }
            batch.clear();
        }
    }

    fn first_insert(&mut self, node_id: usize, vector: &ArrayView<f32, Dim<[usize; 1]>>) {
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
            self.first_insert(node_ids[0], &vectors.slice(s![0, ..]));
        };

        let mut nodes_remaining = HashSet::with_hasher(BuildNoHashHasher::default());
        nodes_remaining.extend(node_ids.iter());
        let nodes_remaining: Vec<usize> = nodes_remaining
            .difference(&self.node_ids)
            .map(|x| *x)
            .collect();

        let mut vector_levels: Vec<(usize, usize)> = nodes_remaining
            .iter()
            .map(|x| (*x, self.get_new_node_layer()))
            .collect();
        vector_levels.shuffle(&mut thread_rng());

        let nb_nodes = self.node_ids.len();
        let remaining = lim - nb_nodes;
        let bar = get_progress_bar(remaining, false);
        for (node_id, level) in vector_levels.iter() {
            let inserted = self.insert(*node_id, &vectors.slice(s![*node_id, ..]), Some(*level));
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

    // pub fn build_index_par_copy(
    //     m: usize,
    //     node_ids: Vec<usize>,
    //     vectors: Array<f32, Dim<[usize; 2]>>,
    // ) -> Self {
    //     let (_lim, dim) = vectors.dim();
    //     let index = Arc::new(RwLock::new(Self::new(m, Some(500), dim)));

    //     index
    //         .write()
    //         .first_insert(node_ids[0], &vectors.slice(s![0, ..]).to_owned());

    //     let mut handlers = vec![];

    //     // let nb_threads = 20;
    //     let nb_threads = std::thread::available_parallelism().unwrap().get();
    //     for thread_nb in 0..nb_threads {
    //         let node_ids_split = split_ids(node_ids.clone(), nb_threads as u8, thread_nb as u8);
    //         let vector_levels: Vec<(usize, usize)> = node_ids_split
    //             .iter()
    //             .map(|x| (*x, index.read().get_new_node_layer()))
    //             .collect();
    //         let index_ref = index.clone();
    //         let vectors_ref = vectors.clone();
    //         if thread_nb == (nb_threads - 1) {
    //             let bar = get_progress_bar(vector_levels.len());
    //             handlers.push(std::thread::spawn(move || {
    //                 for (node_id, level) in vector_levels.iter() {
    //                     let inserted = Self::insert_par(
    //                         &index_ref,
    //                         // *node_id,
    //                         // &vectors_ref.slice(s![*node_id, ..]).to_owned(),
    //                         &vector_levels,
    //                         &vectors_ref.to_owned(),
    //                         Some(*level),
    //                     );
    //                     if inserted {
    //                         bar.inc(1);
    //                     }
    //                 }
    //             }));
    //         } else {
    //             handlers.push(std::thread::spawn(move || {
    //                 for (node_id, level) in vector_levels.iter() {
    //                     let _ = Self::insert_par(
    //                         &index_ref,
    //                         *node_id,
    //                         // &vectors_ref.slice(s![*node_id, ..]).to_owned(),
    //                         &vectors_ref.to_owned(),
    //                         Some(*level),
    //                     );
    //                 }
    //             }));
    //         }
    //     }
    //     for handle in handlers {
    //         let _ = handle.join().unwrap();
    //     }
    //     Arc::try_unwrap(index).unwrap().into_inner()
    // }

    pub fn build_index_par(
        m: usize,
        node_ids: Vec<usize>,
        vectors: &Array<f32, Dim<[usize; 2]>>,
    ) -> Self {
        let (_lim, dim) = vectors.dim();
        let index = Arc::new(RwLock::new(Self::new(m, None, dim)));
        // let index = Arc::new(RwLock::new(Self::new(m, Some(500), dim)));
        let vectors = Arc::new(vectors.to_owned());

        index
            .write()
            .first_insert(node_ids[0], &vectors.slice(s![0, ..]));

        let mut handlers = vec![];

        let nb_threads = std::thread::available_parallelism().unwrap().get();
        for thread_nb in 0..nb_threads {
            let node_ids_split = split_ids(node_ids.clone(), nb_threads as u8, thread_nb as u8);
            let mut vector_levels: Vec<(usize, usize)> = node_ids_split
                .iter()
                .map(|x| (*x, index.read().get_new_node_layer()))
                .collect();
            vector_levels.sort_by_key(|x| x.1);
            let index_ref = index.clone();
            let vectors_ref = vectors.clone();
            let bar = get_progress_bar(vector_levels.len(), thread_nb != (nb_threads - 1));
            handlers.push(std::thread::spawn(move || {
                Self::insert_par(&index_ref, vector_levels, &vectors_ref, bar);
            }));
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }
        Arc::try_unwrap(index).unwrap().into_inner()
    }

    fn get_new_node_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        (-rng.gen::<f32>().log(std::f32::consts::E) * self.ml).floor() as usize
    }

    fn sort_by_distance(
        &self,
        layer: &Graph,
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> BTreeMap<usize, usize> {
        let result = others.iter().map(|idx| {
            (
                (v2v_dist(vector, &layer.node(*idx).1.view()) * 10_000.0).trunc() as usize,
                *idx,
            )
        });
        BTreeMap::from_iter(result)
    }

    fn search_layer(
        &self,
        layer: &Graph,
        vector: &ArrayView<f32, Dim<[usize; 1]>>,
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
                    let neighbor_vec = &layer.node(neighbor).1.view();

                    let (f2q_dist, _) = selected.last_key_value().unwrap().clone();
                    let n2q_dist = (v2v_dist(&vector, neighbor_vec) * 10_000.0).trunc() as usize;

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

fn get_progress_bar(remaining: usize, hidden: bool) -> ProgressBar {
    let bar = if hidden {
        return ProgressBar::hidden();
    } else {
        ProgressBar::new(remaining as u64)
    };
    bar.set_style(
                ProgressStyle::with_template(
                    "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed_precise} ] {per_sec} {wide_bar}",
                )
                .unwrap());
    bar.set_message(format!("Inserting vectors"));
    bar
}
