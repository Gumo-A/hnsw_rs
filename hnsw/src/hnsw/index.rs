// use crate::helpers::bench::Bencher;
// use crate::helpers::data::split;
// use crate::helpers::distance::{l2_compressed, v2v_dist};
use crate::hnsw::graph::Graph;
use crate::hnsw::params::Params;
use crate::hnsw::points::Point;

use indicatif::{ProgressBar, ProgressStyle};
use nohash_hasher::BuildNoHashHasher;
// use parking_lot::RwLock;
use rand::Rng;
use regex::Regex;
// use std::sync::Arc;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::points::{Points, Vector};

#[derive(Debug)]
pub struct HNSW {
    points: Points,
    params: Params,
    ep: usize,
    pub node_ids: HashSet<usize, BuildNoHashHasher<usize>>,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
    // pub bencher: Bencher,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> HNSW {
        let params = Params::from_m_efcons(m, ef_cons.unwrap_or(2 * m), dim);
        HNSW {
            points: Points::Empty,
            params,
            ep: 0,
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            // bencher: Bencher::new(),
        }
    }

    pub fn from_params(params: Params) -> HNSW {
        HNSW {
            points: Points::Empty,
            params,
            ep: 0,
            node_ids: HashSet::with_hasher(BuildNoHashHasher::default()),
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
            // bencher: Bencher::new(),
        }
    }

    pub fn print_params(&self) {
        println!("m = {}", self.params.m);
        println!("mmax = {}", self.params.mmax);
        println!("mmax0 = {}", self.params.mmax0);
        println!("ml = {}", self.params.ml);
        println!("ef_cons = {}", self.params.ef_cons);
        println!("Nb. layers = {}", self.layers.len());
        println!("Nb. of nodes = {}", self.node_ids.len());
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

        let nearest_neighbors: BTreeMap<usize, usize> =
            BTreeMap::from_iter(neighbors.iter().map(|x| {
                let dist = (&layer_0.node(*x).dist2vec(&vector) * 10_000.0) as usize;
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
                let neighbor_vec = &layer.node(*neighbor).vector;
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
            let e_vector = &layer.node(e).vector;

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
        point: &mut Point,
        level: Option<usize>,
        // bencher: &mut Bencher,
    ) -> bool {
        // bencher.start_timer("insert");
        if self.node_ids.contains(&point.id) {
            return false;
        }

        let current_layer_nb: usize = match level {
            Some(level) => level,
            None => self.get_new_node_layer(),
        };

        let max_layer_nb = self.layers.len() - 1;

        // bencher.start_timer("step_1");
        let ep = self.step_1(
            &point,
            max_layer_nb,
            current_layer_nb,
            // bencher
        );
        // bencher.end_timer("step_1");

        // bencher.start_timer("step_2");
        let insertion_results = self.step_2(
            &point,
            ep,
            current_layer_nb,
            // bencher
        );
        // bencher.end_timer("step_2");

        point.quantize();
        // bencher.start_timer("load_data");
        self.node_ids.insert(point.id);
        for (layer_nb, node_data) in insertion_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                if *node == point.id {
                    layer.add_node(point);
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
                layer.add_node(point);
                self.layers.insert(layer_nb, layer);
                self.node_ids.insert(point.id);
            }
            self.ep = point.id;
        }
        // bencher.end_timer("load_data");
        // bencher.end_timer("insert");
        true
    }

    // pub fn insert_par(index: &Arc<RwLock<Self>>, points: Vec<(Point, usize)>, bar: ProgressBar) {
    //     let mut batch = Vec::new();
    //     let batch_size = 16;
    //     for (idx, (point, point_max_layer)) in points.iter().enumerate() {
    //         let read_ref = index.read();
    //         if read_ref.node_ids.contains(&point.id) {
    //             continue;
    //         }
    //         let max_layer_nb = read_ref.layers.len() - 1;
    //         let ep = read_ref.step_1(&point.vector.view(), max_layer_nb, *point_max_layer);
    //         let insertion_results = read_ref.step_2(&point, ep, *point_max_layer);
    //         batch.push((idx, insertion_results));

    //         let last_idx = idx == (points.len() - 1);
    //         let new_layer = point_max_layer > &max_layer_nb;
    //         let full_batch = batch.len() >= batch_size;
    //         let have_to_write: bool = last_idx | new_layer | full_batch;

    //         let mut write_ref = if have_to_write {
    //             drop(read_ref);
    //             index.write()
    //         } else {
    //             continue;
    //         };
    //         write_ref.write_batch(&batch, &points);
    //         if new_layer {
    //             for layer_nb in max_layer_nb + 1..point_max_layer + 1 {
    //                 let mut layer = Graph::new();
    //                 layer.add_node(point);
    //                 write_ref.layers.insert(layer_nb, layer);
    //                 write_ref.node_ids.insert(point.id);
    //                 write_ref.ep = point.id;
    //             }
    //         }
    //         if !bar.is_hidden() {
    //             bar.inc(batch.len() as u64);
    //         }
    //         batch.clear();
    //     }
    // }

    // fn write_batch(
    //     &mut self,
    //     batch: &Vec<(
    //         usize,
    //         HashMap<usize, HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>>>,
    //     )>,
    //     points: &Vec<(Point, usize)>,
    // ) {
    //     for (point_nb, batch_data) in batch.iter() {
    //         for (layer_nb, node_data) in batch_data.iter() {
    //             self.node_ids.extend(node_data.keys());
    //             let layer = self.layers.get_mut(&layer_nb).unwrap();
    //             for (node, neighbors) in node_data.iter() {
    //                 let (point, _) = &points[*point_nb];
    //                 layer.add_node(point);
    //                 for old_neighbor in layer.neighbors(*node).clone() {
    //                     // if neighbors.contains(&old_neighbor) {
    //                     //     // neighbors.remove(&old_neighbor);
    //                     //     continue;
    //                     // }
    //                     layer.remove_edge(*node, old_neighbor);
    //                 }
    //                 for neighbor in neighbors.iter() {
    //                     layer.add_edge(*node, *neighbor);
    //                 }
    //             }
    //         }
    //     }
    // }

    fn first_insert(&mut self, point: &Point) {
        assert_eq!(self.node_ids.len(), 0);
        assert_eq!(self.layers.len(), 0);

        let mut layer = Graph::new();
        layer.add_node(point);
        self.layers.insert(0, layer);
        self.node_ids.insert(point.id);
        self.ep = point.id;
    }

    pub fn build_index(
        &mut self,
        vectors: Vec<Vec<f32>>,
        checkpoint: bool,
        // bencher: &mut Bencher,
    ) -> std::io::Result<()> {
        let lim = vectors.len();
        let dim = self.params.dim;
        let m = self.params.m;
        let efcons = self.params.ef_cons;

        let mut points: Vec<(Point, usize)> = (0..vectors.len())
            .map(|idx| {
                (
                    Point::new(idx, vectors[idx].clone(), None, false),
                    self.get_new_node_layer(),
                )
            })
            .collect();

        drop(vectors);

        let checkpoint_path =
            format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}_m{m}_efcons{efcons}");
        let mut copy_path = checkpoint_path.clone();
        copy_path.push_str("_copy");

        if checkpoint & Path::new(&checkpoint_path).exists() {
            self.load(&checkpoint_path)?;
            self.print_params();
        } else {
            println!("No checkpoint was loaded, building self from scratch.");
            self.first_insert(&points[0].0);
        };

        let nb_nodes = self.node_ids.len();
        let remaining = lim - nb_nodes;
        let bar = get_progress_bar(remaining, false);
        for (point, level) in points.iter_mut() {
            let inserted = self.insert(
                point,
                Some(*level),
                // bencher,
            );
            if inserted {
                bar.inc(1);
            } else {
                bar.reset_eta();
            }
            if checkpoint {
                if ((point.id != 0) & (point.id % 10_000 == 0) & (inserted)) | (point.id == lim - 1)
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

    // pub fn build_index_par(
    //     m: usize,
    //     vectors: &Array<f32, Dim<[usize; 2]>>,
    //     filters: &Option<Vec<Payload>>,
    // ) -> Self {
    //     let nb_threads = std::thread::available_parallelism().unwrap().get();
    //     let (lim, dim) = vectors.dim();
    //     let index = Arc::new(RwLock::new(HNSW::new(m, None, dim)));
    //     let filters = filters.as_ref().unwrap();
    //     let points: Vec<(Point, usize)> = (0..lim)
    //         .map(|idx| {
    //             (
    //                 Point::new(
    //                     idx,
    //                     vectors.slice(s![idx, ..]),
    //                     None,
    //                     Some(filters[idx].clone()),
    //                 ),
    //                 index.read().get_new_node_layer(),
    //             )
    //         })
    //         .collect();
    //     index.write().first_insert(&points[0].0);
    //     let mut points_split = split(points, nb_threads);

    //     let mut handlers = vec![];
    //     for thread_nb in 0..nb_threads {
    //         let index_ref = index.clone();
    //         let points_ref = points_split.pop().unwrap();
    //         let bar = get_progress_bar(points_ref.len(), thread_nb != 0);
    //         handlers.push(std::thread::spawn(move || {
    //             Self::insert_par(&index_ref, points_ref, bar);
    //         }));
    //     }
    //     for handle in handlers {
    //         let _ = handle.join().unwrap();
    //     }
    //     let mut index_ref = index.write();
    //     for idx in 0..lim {
    //         if !index_ref.node_ids.contains(&idx) {
    //             let point = Point::new(idx, vectors.slice(s![idx, ..]), None, None);
    //             index_ref.insert(&point, Some(0));
    //             if index_ref.ep == idx {
    //                 let nb_layers = index_ref.layers.len();
    //                 index_ref.ep = *index_ref
    //                     .layers
    //                     .get(&(nb_layers - 1))
    //                     .unwrap()
    //                     .nodes
    //                     .keys()
    //                     .next()
    //                     .unwrap();
    //             }
    //         }
    //     }
    //     drop(index_ref);
    //     Arc::try_unwrap(index).unwrap().into_inner()
    // }

    fn get_new_node_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        (-rng.gen::<f32>().log(std::f32::consts::E) * self.params.ml).floor() as usize
    }

    fn sort_by_distance(
        &self,
        layer: &Graph,
        vector: &Vector,
        // others: &T,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> BTreeMap<usize, usize> {
        let result = others.iter().map(|idx| {
            (
                (&layer.node(*idx).dist2vec(vector) * 10_000.0) as usize,
                *idx,
            )
        });
        BTreeMap::from_iter(result)
    }

    fn get_nearest<I>(&self, layer: &Graph, vector: &Vector, others: I) -> (usize, usize)
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| ((&layer.node(idx).dist2vec(vector) * 10_000.0) as usize, idx))
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

            // let cand_neighbors: HashSet<usize, BuildNoHashHasher<usize>> = layer
            //     .neighbors(candidate)
            //     .iter()
            //     .filter(|x| ep.insert(**x))
            //     .map(|x| *x)
            //     .collect();
            // let cand_neighbors_sorted = self.sort_by_distance(layer, vector, &cand_neighbors);

            // for (n2q_dist, neighbor) in cand_neighbors_sorted.iter() {
            //     let (f2q_dist, _) = selected.last_key_value().unwrap().clone();

            //     if (n2q_dist < f2q_dist) | (selected.len() < ef) {
            //         candidates.insert(*n2q_dist, *neighbor);
            //         selected.insert(*n2q_dist, *neighbor);

            //         if selected.len() > ef {
            //             selected.pop_last().unwrap();
            //         }
            //         continue;
            //     }
            //     break;
            // }

            // bencher.end_timer("while_1");
            for neighbor in layer.neighbors(candidate).iter().map(|x| *x) {
                if ep.insert(neighbor) {
                    let neighbor_point = &layer.node(neighbor);

                    let (f2q_dist, _) = selected.last_key_value().unwrap().clone();

                    // bencher.start_timer("while_2_1");
                    let n2q_dist = (neighbor_point.dist2vec(vector) * 10_000.0) as usize;
                    // bencher.end_timer("while_2_1");

                    // bencher.start_timer("while_2_2");
                    if (&n2q_dist < f2q_dist) | (selected.len() < ef) {
                        candidates.insert(n2q_dist, neighbor);
                        selected.insert(n2q_dist, neighbor);

                        if selected.len() > ef {
                            selected.pop_last().unwrap();
                        }
                    }
                    // bencher.end_timer("while_2_2");
                }
            }
        }
        // bencher.start_timer("end_results");
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        result.extend(selected.values());
        // bencher.end_timer("end_results");
        // bencher.end_timer("search_layer");
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
            ("m", self.params.m as f32),
            ("mmax", self.params.mmax as f32),
            ("mmax0", self.params.mmax0 as f32),
            ("ef_cons", self.params.ef_cons as f32),
            ("ml", self.params.ml as f32),
            ("ep", self.ep as f32),
            ("dim", self.params.dim as f32),
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
                let neighbors: &HashSet<usize, BuildNoHashHasher<usize>> = &node_data.neighbors;
                let vector: Vec<f32> = match &node_data.vector {
                    Vector::Full(full) => full.clone(),
                    Vector::Compressed(compressed) => compressed.reconstruct(),
                };
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

        let hnsw_params = Params::from(
            *params.get("m").unwrap() as usize,
            Some(*params.get("ef_cons").unwrap() as usize),
            Some(*params.get("mmax").unwrap() as usize),
            Some(*params.get("mmax0").unwrap() as usize),
            Some(*params.get("ml").unwrap() as f32),
            *params.get("dim").unwrap() as usize,
        );
        Ok(HNSW {
            params: hnsw_params,
            ep: *params.get("ep").unwrap() as usize,
            node_ids,
            layers,
            // bencher: Bencher::new(),
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
                self.params.m = *content.get("m").unwrap() as usize;
                self.params.mmax = *content.get("mmax").unwrap() as usize;
                self.params.mmax0 = *content.get("mmax0").unwrap() as usize;
                self.ep = *content.get("ep").unwrap() as usize;
                self.params.ef_cons = *content.get("ef_cons").unwrap() as usize;
                self.params.dim = *content.get("dim").unwrap() as usize;
                self.params.ml = *content.get("ml").unwrap() as f32;
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
                self.layers.insert(
                    layer_nb as usize,
                    Graph::from_layer_data(self.params.dim, content),
                );
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
