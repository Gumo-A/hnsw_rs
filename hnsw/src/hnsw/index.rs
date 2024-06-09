// use crate::helpers::bench::Bencher;
use crate::helpers::data::split;
// use crate::helpers::distance::{l2_compressed, v2v_dist};
use super::{
    distid::Dist,
    points::{Points, Vector},
};
use crate::hnsw::params::Params;
use crate::hnsw::points::Point;
use crate::hnsw::{graph::Graph, lvq::LVQVec};

use indicatif::{ProgressBar, ProgressStyle};
use nohash_hasher::BuildNoHashHasher;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize)]
pub struct HNSW {
    ep: usize,
    pub params: Params,
    pub points: Points,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> HNSW {
        let params = Params::from_m_efcons(m, ef_cons.unwrap_or(2 * m), dim);
        HNSW {
            points: Points::Empty,
            params,
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }

    pub fn from_params(params: Params) -> HNSW {
        HNSW {
            points: Points::Empty,
            params,
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }

    pub fn print_params(&self) {
        println!("m = {}", self.params.m);
        println!("mmax = {}", self.params.mmax);
        println!("mmax0 = {}", self.params.mmax0);
        println!("ml = {}", self.params.ml);
        println!("ef_cons = {}", self.params.ef_cons);
        println!("Nb. layers = {}", self.layers.len());
        println!("Nb. of nodes = {}", self.points.len());
        for (idx, layer) in self.layers.iter() {
            println!("NB. nodes in layer {idx}: {}", layer.nb_nodes());
        }
        println!("ep: {:?}", self.ep);
    }

    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
        // bencher: &mut Bencher,
    ) -> Vec<usize> {
        let mut ep: HashSet<usize, BuildNoHashHasher<usize>> =
            HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);
        let nb_layer = self.layers.len();

        let vector = Vector::Full(vector.clone());

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(
                &self.layers.get(&(layer_nb)).unwrap(),
                &vector,
                &mut ep,
                1,
                // bencher,
            );
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        let neighbors = self.search_layer(
            layer_0, &vector, &mut ep, ef,
            // bencher
        );

        let nearest_neighbors: BTreeMap<Dist, usize> =
            BTreeMap::from_iter(neighbors.iter().map(|x| {
                let dist = self.points.get_point(*x).dist2vec(&vector);
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
        point: &Point,
        max_layer_nb: usize,
        current_layer_number: usize,
        // bencher: &mut Bencher,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        let mut ep = HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);

        for layer_nb in (current_layer_number + 1..max_layer_nb + 1).rev() {
            let layer = &self.layers.get(&layer_nb).unwrap();
            ep = self.search_layer(
                layer,
                &point.vector,
                &mut ep,
                1,
                // bencher
            );
        }
        ep
    }

    fn step_2(
        &self,
        point: &Point,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
        // bencher: &mut Bencher,
    ) -> HashMap<usize, HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>> {
        // bencher.start_timer("step_2");
        let mut insertion_results = HashMap::new();
        let bound = (current_layer_number + 1).min(self.layers.len());

        for layer_nb in (0..bound).rev() {
            let layer = &self.layers.get(&layer_nb).unwrap();

            // bencher.start_timer("search_layer");
            ep = self.search_layer(
                layer,
                &point.vector,
                &mut ep,
                self.params.ef_cons,
                // bencher
            );

            // bencher.end_timer("search_layer");

            // bencher.start_timer("heuristic");
            let neighbors_to_connect =
                self.select_heuristic(&layer, &point.vector, &mut ep, self.params.m, false, true);
            // bencher.end_timer("heuristic");

            // bencher.start_timer("prune");
            let prune_results = self.prune_connexions(layer_nb, &neighbors_to_connect);
            // bencher.end_timer("prune");

            // bencher.start_timer("load");
            insertion_results.insert(layer_nb, HashMap::new());
            insertion_results
                .get_mut(&layer_nb)
                .unwrap()
                .insert(point.id, neighbors_to_connect);
            insertion_results
                .get_mut(&layer_nb)
                .unwrap()
                .extend(prune_results.iter().map(|x| (*x.0, x.1.to_owned())));
            // bencher.end_timer("load");
        }
        // bencher.end_timer("step_2");
        insertion_results
    }

    fn prune_connexions(
        &self,
        layer_nb: usize,
        connexions_made: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>> {
        let mut prune_results = HashMap::new();
        let limit = if layer_nb == 0 {
            self.params.mmax0
        } else {
            self.params.mmax
        };

        for neighbor in connexions_made.iter() {
            let layer = &self.layers.get(&layer_nb).unwrap();
            if ((layer_nb == 0) & (layer.degree(*neighbor) > self.params.mmax0))
                | ((layer_nb > 0) & (layer.degree(*neighbor) > self.params.mmax))
            {
                let neighbor_vec = &self.points.get_point(*neighbor).vector;
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
        vector: &Vector,
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

        let (dist_e, e) = candidates.pop_first().unwrap();
        selected.insert(dist_e, e);
        while (candidates.len() > 0) & (selected.len() < m) {
            let (dist_e, e) = candidates.pop_first().unwrap();
            let e_vector = &self.points.get_point(e).vector;

            // let mut selected_set = HashSet::with_hasher(BuildNoHashHasher::default());
            // selected_set.extend(selected.values());
            let (dist_from_s, _) =
                self.get_nearest(layer, &e_vector, selected.values().map(|x| *x));

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
        mut point: Point,
        level: usize,
        // bencher: &mut Bencher
    ) -> bool {
        // bencher.start_timer("insert");

        // let point = self.points.get_point(point_idx);
        if self.points.contains(&point.id) {
            return false;
        }

        if self.layers.len() == 0 {
            self.first_insert(point);
            return true;
        }

        let max_layer_nb = self.layers.len() - 1;

        // bencher.start_timer("step_1");
        let ep = self.step_1(
            &point,
            max_layer_nb,
            level,
            // bencher
        );
        // bencher.end_timer("step_1");

        // bencher.start_timer("step_2");
        let insertion_results = self.step_2(
            &point, ep, level,
            // bencher
        );
        // bencher.end_timer("step_2");

        // bencher.start_timer("load_data");
        for (layer_nb, node_data) in insertion_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                if *node == point.id {
                    layer.add_node(&point);
                }
                for old_neighbor in layer.neighbors(*node).clone() {
                    layer.remove_edge(*node, old_neighbor);
                }
                for neighbor in neighbors.iter() {
                    layer.add_edge(*node, *neighbor);
                }
            }
        }
        if level > max_layer_nb {
            for layer_nb in max_layer_nb + 1..level + 1 {
                let mut layer = Graph::new();
                layer.add_node(&point);
                self.layers.insert(layer_nb, layer);
                // self.points.insert(point.id);
            }
            self.ep = point.id;
        }
        // bencher.end_timer("load_data");
        // bencher.end_timer("insert");
        point.quantize();
        self.points.insert(point);
        true
    }

    pub fn insert_par(index: &Arc<RwLock<Self>>, mut points: Vec<Point>, bar: ProgressBar) {
        let batch_size = 16;
        let points_len = points.len();
        let mut batch = Vec::new();
        for idx in 0..points_len {
            let point = points.pop().unwrap();
            let point_id = point.id;
            let read_ref = index.read();
            if read_ref.points.contains(&point.id) {
                continue;
            }
            let point_max_layer = get_new_node_layer(read_ref.params.ml);
            let max_layer_nb = read_ref.layers.len() - 1;

            let ep = read_ref.step_1(&point, max_layer_nb, point_max_layer);
            let insertion_results = read_ref.step_2(&point, ep, point_max_layer);
            batch.push((point, insertion_results));

            let last_idx = idx == (points_len - 1);
            let new_layer = point_max_layer > max_layer_nb;
            let full_batch = batch.len() >= batch_size;
            let have_to_write: bool = last_idx | new_layer | full_batch;

            let mut write_ref = if have_to_write {
                drop(read_ref);
                index.write()
            } else {
                continue;
            };
            if new_layer {
                for layer_nb in max_layer_nb + 1..point_max_layer + 1 {
                    let mut layer = Graph::new();
                    layer.add_node_by_id(point_id);
                    write_ref.layers.insert(layer_nb, layer);
                    write_ref.ep = point_id;
                }
            }
            let batch_len = batch.len();
            write_ref.write_batch(&mut batch);
            if !bar.is_hidden() {
                bar.inc(batch_len as u64);
            }
            batch.clear();
        }
    }

    fn write_batch(
        &mut self,
        batch: &mut Vec<(
            Point,
            HashMap<usize, HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>>,
        )>,
    ) {
        let batch_len = batch.len();
        for _ in 0..batch_len {
            let batch_content = batch.pop().unwrap();
            let point = batch_content.0;
            let batch_data = batch_content.1;
            for (layer_nb, node_data) in batch_data.iter() {
                let layer = self.layers.get_mut(&layer_nb).unwrap();
                for (node, neighbors) in node_data.iter() {
                    layer.add_node_by_id(*node);
                    for old_neighbor in layer.neighbors(*node).clone() {
                        layer.remove_edge(*node, old_neighbor);
                    }
                    for neighbor in neighbors.iter() {
                        layer.add_edge(*node, *neighbor);
                    }
                }
            }
            self.points.insert(point);
        }
    }

    // fn store_vectors(&mut self, vectors: Vec<Vec<f32>>) {
    //     let points: Vec<Point> = (0..vectors.len())
    //         .map(|idx| Point::new(idx, vectors[idx].clone(), true))
    //         .collect();
    //     self.points.extend_or_fill(points)
    // }

    fn make_points(&mut self, vectors: Vec<Vec<f32>>) -> Vec<Point> {
        (0..vectors.len())
            .map(|idx| Point::new(idx, vectors[idx].clone(), true))
            .collect()
    }

    fn first_insert(&mut self, point: Point) {
        // let point: &Point = self.points.get_point(idx);
        let mut layer = Graph::new();
        layer.add_node(&point);
        self.layers.insert(0, layer);
        self.ep = point.id;
        self.points.insert(point);
    }

    pub fn build_index(
        &mut self,
        vectors: Vec<Vec<f32>>,
        // bencher: &mut Bencher
    ) {
        let lim = vectors.len();
        let mut points = self.make_points(vectors);

        assert_eq!(self.points.len(), 0);
        assert_eq!(self.layers.len(), 0);

        self.first_insert(points.pop().unwrap());

        let levels: Vec<usize> = (1..lim)
            .map(|_| get_new_node_layer(self.params.ml))
            .collect();
        let bar = get_progress_bar(lim, false);
        for level in levels {
            let inserted = self.insert(
                points.pop().unwrap(),
                level,
                // bencher
            );
            if inserted {
                bar.inc(1);
            } else {
                bar.reset_eta();
            }
        }
    }

    pub fn build_index_par(m: usize, vectors: Vec<Vec<f32>>) -> Self {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        let (dim, lim) = (vectors[0].len(), vectors.len());
        let index = Arc::new(RwLock::new(HNSW::new(m, None, dim)));
        let mut points = index.write().make_points(vectors);
        index.write().first_insert(points.pop().unwrap());

        let mut points_split = split(points, nb_threads);

        let mut handlers = vec![];
        for thread_nb in 0..nb_threads {
            let index_ref = index.clone();
            let points_thread = points_split.pop().unwrap();
            let bar = get_progress_bar(points_thread.len(), thread_nb != 0);
            handlers.push(std::thread::spawn(move || {
                Self::insert_par(&index_ref, points_thread, bar);
            }));
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }
        // Verify all points were inserted
        // let mut index_ref = index.write();
        // for idx in 0..lim {
        //     if !index_ref.node_ids.contains(&idx) {
        //         let point = Point::new(idx, vectors.slice(s![idx, ..]), None, None);
        //         index_ref.insert(&point, Some(0));
        //         if index_ref.ep == idx {
        //             let nb_layers = index_ref.layers.len();
        //             index_ref.ep = *index_ref
        //                 .layers
        //                 .get(&(nb_layers - 1))
        //                 .unwrap()
        //                 .nodes
        //                 .keys()
        //                 .next()
        //                 .unwrap();
        //         }
        //     }
        // }
        // drop(index_ref);
        Arc::try_unwrap(index).unwrap().into_inner()
    }

    fn sort_by_distance(
        &self,
        _layer: &Graph,
        vector: &Vector,
        // others: &T,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> BTreeMap<Dist, usize> {
        let result = others
            .iter()
            .map(|idx| (self.points.get_point(*idx).dist2vec(vector), *idx));
        BTreeMap::from_iter(result)
    }

    fn get_nearest<I>(&self, _layer: &Graph, vector: &Vector, others: I) -> (Dist, usize)
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| (self.points.get_point(idx).dist2vec(vector), idx))
            .min_by_key(|x| x.0)
            .unwrap()
    }

    fn search_layer(
        &self,
        layer: &Graph,
        vector: &Vector,
        ep: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        ef: usize,
        // bencher: &mut Bencher,
    ) -> HashSet<usize, BuildNoHashHasher<usize>> {
        // bencher.start_timer("search_layer");

        // bencher.start_timer("initial_sort");
        let mut candidates = self.sort_by_distance(layer, vector, &ep);
        let mut selected = candidates.clone();
        // bencher.end_timer("initial_sort");

        while let Some((cand2q_dist, candidate)) = candidates.pop_first() {
            // bencher.start_timer("while_1");

            let (furthest2q_dist, _) = selected.last_key_value().unwrap();

            if &cand2q_dist > furthest2q_dist {
                break;
            }

            // bencher.end_timer("while_1");
            for (n2q_dist, neighbor_point) in layer
                .neighbors(candidate)
                .iter()
                .filter(|idx| ep.insert(**idx))
                .map(|idx| {
                    let point = self.points.get_point(*idx);
                    let dist = point.dist2vec(vector);
                    (dist, point)
                })
            {
                let (f2q_dist, _) = selected.last_key_value().unwrap();

                // bencher.start_timer("while_2_2");
                if (&n2q_dist < f2q_dist) | (selected.len() < ef) {
                    candidates.insert(n2q_dist, neighbor_point.id);
                    selected.insert(n2q_dist, neighbor_point.id);

                    if selected.len() > ef {
                        selected.pop_last();
                    }
                }
                // bencher.end_timer("while_2_2");
            }
        }
        // bencher.start_timer("end_results");
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        result.extend(selected.values());
        // bencher.end_timer("end_results");
        // bencher.end_timer("search_layer");
        result
    }

    pub fn save(&self, index_path: &str) -> std::io::Result<()> {
        let file = File::create(index_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn from_path(index_path: &str) -> std::io::Result<Self> {
        let file = File::open(index_path)?;
        let reader = BufReader::new(file);
        let content: serde_json::Value = serde_json::from_reader(reader)?;

        let ep = match content
            .get("ep")
            .expect("Error: entry point could not be loaded.")
        {
            serde_json::Value::Number(ep) => ep.as_i64().unwrap() as usize,
            _ => panic!("Error: unexpected type of entry point."),
        };

        let params = match content
            .get("params")
            .expect("Error: key 'params' is not in the index file.")
        {
            serde_json::Value::Object(params_map) => extract_params(&params_map),
            _ => panic!("Something went wrong reading parameters of the index file."),
        };

        let layers = match content
            .get("layers")
            .expect("Error: key 'layers' could not be loaded.")
        {
            serde_json::Value::Object(layers_map) => {
                let mut intermediate_layers = HashMap::with_hasher(BuildNoHashHasher::default());
                for (key, value) in layers_map {
                    let layer_nb: usize = key
                        .parse()
                        .expect("Error: could not load key {key} into layer number");
                    let layer_content = match value
                        .get("nodes")
                        .expect("Error: could not load 'key' nodes for layer {key}")
                    {
                        serde_json::Value::Object(layer_content) => {
                            let mut this_layer = HashMap::new();
                            for (node_id, neighbors) in layer_content.iter() {
                                let neighbors = match neighbors {
                                    serde_json::Value::Array(neighbors_arr) => {
                                        let mut final_neighbors = Vec::new();
                                        let mut neighbors_set = HashSet::with_hasher(BuildNoHashHasher::default());
                                        for neighbor in neighbors_arr {
                                            match neighbor {
                                                serde_json::Value::Number(num) => final_neighbors.push(num.as_u64().unwrap() as usize),
            _ => panic!("Something went wrong reading neighbors of node {node_id} in layer {layer_nb}"),
                                            }
                                        }
                                        neighbors_set.extend(final_neighbors);
                                        neighbors_set
                                    },
            _ => panic!("Something went wrong reading neighbors of node {node_id} in layer {layer_nb}"),
                                };
                                this_layer.insert(node_id.parse::<usize>().unwrap(), neighbors);
                            }
                            Graph::from_layer_data(this_layer)
                        }
                        _ => panic!(
                            "Something went wrong reading layer {layer_nb} of the index file."
                        ),
                    };
                    intermediate_layers.insert(layer_nb, layer_content);
                }
                intermediate_layers
            }
            _ => panic!("Something went wrong reading layers of the index file."),
        };

        let points = match content
            .get("points")
            .expect("Error: key 'points' is not in the index file.")
        {
            serde_json::Value::Object(points_map) => {
                let err_msg =
                    "Error reading index file: could not find key 'Collection' in 'points', maybe the index is empty.";
                match points_map.get("Collection").expect(err_msg) {
                    serde_json::Value::Object(points_final) => extract_points(points_final),
                    _ => panic!("Something went wrong reading parameters of the index file."),
                }
            }
            _ => panic!("Something went wrong reading parameters of the index file."),
        };

        Ok(HNSW {
            ep,
            params,
            layers,
            points: Points::Collection(points),
        })
    }

    // TODO: see todo in graph.rs
    // pub fn load(&mut self, path: &str) -> std::io::Result<()> {
    //     self.node_ids.clear();
    //     self.layers.clear();
    //     let paths = std::fs::read_dir(path)?;
    //     for file_path in paths {
    //         let file_name = file_path?;
    //         let file = File::open(file_name.path())?;
    //         let reader = BufReader::new(file);
    //         if file_name.file_name().to_str().unwrap().contains("params") {
    //             let content: HashMap<String, f32> = serde_json::from_reader(reader)?;
    //             self.params.m = *content.get("m").unwrap() as usize;
    //             self.params.mmax = *content.get("mmax").unwrap() as usize;
    //             self.params.mmax0 = *content.get("mmax0").unwrap() as usize;
    //             self.ep = *content.get("ep").unwrap() as usize;
    //             self.params.ef_cons = *content.get("ef_cons").unwrap() as usize;
    //             self.params.dim = *content.get("dim").unwrap() as usize;
    //             self.params.ml = *content.get("ml").unwrap() as f32;
    //         } else if file_name.file_name().to_str().unwrap().contains("node_ids") {
    //             let content: HashSet<usize, BuildNoHashHasher<usize>> =
    //                 serde_json::from_reader(reader)?;
    //             for val in content.iter() {
    //                 self.node_ids.insert(*val);
    //             }
    //         } else if file_name.file_name().to_str().unwrap().contains("layer") {
    //             let re = Regex::new(r"\d+").unwrap();
    //             let layer_nb: u8 = re
    //                 .find(file_name.file_name().to_str().unwrap())
    //                 .unwrap()
    //                 .as_str()
    //                 .parse::<u8>()
    //                 .expect("Could not parse u8 from file name.");
    //             let content: HashMap<usize, (HashSet<usize, BuildNoHashHasher<usize>>, Vec<f32>)> =
    //                 serde_json::from_reader(reader)?;
    //             self.layers
    //                 .insert(layer_nb as usize, Graph::from_layer_data(content));
    //         }
    //     }
    //     Ok(())
    // }
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

fn get_new_node_layer(ml: f32) -> usize {
    let mut rng = rand::thread_rng();
    (-rng.gen::<f32>().log(std::f32::consts::E) * ml).floor() as usize
}
fn extract_params(params: &serde_json::Map<String, serde_json::Value>) -> Params {
    let hnsw_params = Params::from(
        params
            .get("m")
            .unwrap()
            .as_number()
            .unwrap()
            .as_i64()
            .unwrap() as usize,
        Some(
            params
                .get("ef_cons")
                .unwrap()
                .as_number()
                .unwrap()
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("mmax")
                .unwrap()
                .as_number()
                .unwrap()
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("mmax0")
                .unwrap()
                .as_number()
                .unwrap()
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("ml")
                .unwrap()
                .as_number()
                .unwrap()
                .as_f64()
                .unwrap() as f32,
        ),
        params
            .get("dim")
            .unwrap()
            .as_number()
            .unwrap()
            .as_i64()
            .unwrap() as usize,
    );
    hnsw_params
}

fn extract_points(
    points_data: &serde_json::Map<String, serde_json::Value>,
) -> HashMap<usize, Point, BuildNoHashHasher<usize>> {
    let mut points = HashMap::with_hasher(BuildNoHashHasher::default());

    for (id, value) in points_data.iter() {
        let id: usize = id.parse().unwrap();
        let vector_content = value.get("vector").unwrap();
        let vector = match vector_content {
            serde_json::Value::Object(vector_content_map) => {
                if vector_content_map.contains_key("Compressed") {
                    let delta = vector_content_map
                        .get("Compressed")
                        .unwrap()
                        .get("delta")
                        .unwrap()
                        .as_number()
                        .unwrap()
                        .as_f64()
                        .unwrap() as f32;

                    let lower = vector_content_map
                        .get("Compressed")
                        .unwrap()
                        .get("lower")
                        .unwrap()
                        .as_number()
                        .unwrap()
                        .as_f64()
                        .unwrap() as f32;

                    let quantized_vector: Vec<u8> = vector_content_map
                        .get("Compressed")
                        .unwrap()
                        .get("quantized_vec")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u8)
                        .collect();
                    Vector::Compressed(LVQVec::from_quantized(quantized_vector, delta, lower))
                } else {
                    let full_vector = vector_content_map
                        .get("Full")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect();
                    Vector::Full(full_vector)
                }
            }
            _ => panic!(),
        };
        let point = Point::from_vector(id, vector);
        points.insert(id, point);
    }
    points
}
