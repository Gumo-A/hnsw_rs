use super::{
    dist::Dist,
    points::{Point, PointsV2, Vector},
};
use crate::hnsw::params::Params;
use crate::hnsw::{graph::Graph, lvq::LVQVec};

use indicatif::{ProgressBar, ProgressStyle};

use nohash_hasher::BuildNoHashHasher;

use parking_lot::{RwLock, RwLockReadGuard};
use std::sync::Arc;

use rand::thread_rng;
use rand::Rng;
use rand::{rngs::ThreadRng, seq::SliceRandom};

use serde::{Deserialize, Serialize};

use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HNSW {
    ep: usize,
    pub params: Params,
    pub points: PointsV2,
    pub layers: HashMap<usize, Graph, BuildNoHashHasher<usize>>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> HNSW {
        let params = Params::from_m_efcons(m, ef_cons.unwrap_or(2 * m), dim);
        HNSW {
            points: PointsV2::Empty,
            params,
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }

    pub fn from_params(params: Params) -> HNSW {
        HNSW {
            points: PointsV2::Empty,
            params,
            ep: 0,
            layers: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }

    fn assert_param_compliance(&self) {
        let mut is_ok = true;
        for (layer_nb, layer) in self.layers.iter() {
            let max_degree = if *layer_nb > 0 {
                self.params.mmax
            } else {
                self.params.mmax0
            };
            for (node, neighbors) in layer.nodes.iter() {
                // I allow degrees to exceed the limit by one,
                // because I am too lazy to change the current methods.
                if neighbors.len() > (max_degree + 1) {
                    is_ok = false;
                    println!(
                        "In layer {layer_nb}, node {node} has degree {0}, but limit is {1}",
                        neighbors.len(),
                        max_degree
                    );
                }

                if (neighbors.len() == 0) & (layer.nb_nodes() > 1) {
                    is_ok = false;
                    println!("In layer {layer_nb}, node {node} has degree 0",);
                }
            }
        }
        if is_ok {
            println!("Index complies with params.")
        }
    }

    pub fn print_index(&self) {
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

    // TODO: add multithreaded functionality for ANN search.
    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<usize>, String> {
        let mut ep: HashSet<usize, BuildNoHashHasher<usize>> =
            HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);
        let nb_layer = self.layers.len();

        // let vector = self.center_vector(vector)?;

        let point = Point::new_quantized(0, 0, &vector);

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(&self.layers.get(&(layer_nb)).unwrap(), &point, &mut ep, 1)?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        let neighbors = self.search_layer(layer_0, &point, &mut ep, ef)?;

        let nearest_neighbors: BTreeMap<Dist, usize> =
            BTreeMap::from_iter(neighbors.iter().map(|x| {
                let dist = self.points.get_point(*x).unwrap().dist2other(&point);
                (dist, *x)
            }));

        let anns: Vec<usize> = nearest_neighbors.values().take(n).cloned().collect();
        Ok(anns)
    }

    // TODO: multithreaded
    pub fn anns_by_vectors(&self, vector: &Vec<Vec<f32>>, n: usize, ef: usize) -> () {
        // Result<Vec<Vec<usize>>, String> {
        // todo!("multithreaded");
        // let mut ep: HashSet<usize, BuildNoHashHasher<usize>> =
        //     HashSet::with_hasher(BuildNoHashHasher::default());
        // ep.insert(self.ep);
        // let nb_layer = self.layers.len();

        // let point = Point::new_quantized(0, None, vector);

        // for layer_nb in (0..nb_layer).rev() {
        //     ep = self.search_layer(&self.layers.get(&(layer_nb)).unwrap(), &point, &mut ep, 1)?;
        // }

        // let layer_0 = &self.layers.get(&0).unwrap();
        // let neighbors = self.search_layer(layer_0, &point, &mut ep, ef)?;

        // let nearest_neighbors: BTreeMap<Dist, usize> =
        //     BTreeMap::from_iter(neighbors.iter().map(|x| {
        //         let dist = self.points.get_point(*x).unwrap().dist2other(&point);
        //         (dist, *x)
        //     }));

        // let anns: Vec<usize> = nearest_neighbors
        //     .values()
        //     .skip(1)
        //     .take(n)
        //     .cloned()
        //     .collect();
        // Ok(anns)
    }

    fn step_1(
        &self,
        point: &Point,
        max_layer_nb: usize,
        current_layer_number: usize,
        stop_at_layer: Option<usize>,
    ) -> Result<HashSet<usize, BuildNoHashHasher<usize>>, String> {
        let mut ep = HashSet::with_hasher(BuildNoHashHasher::default());
        ep.insert(self.ep);

        ep = match stop_at_layer {
            None => {
                for layer_nb in (current_layer_number + 1..max_layer_nb + 1).rev() {
                    let layer = match self.layers.get(&layer_nb) {
                        Some(l) => l,
                        None => return Err(format!("Could not get layer {layer_nb} in step 1.")),
                    };
                    ep = self.search_layer(layer, point, &mut ep, 1)?;
                }
                ep
            }
            Some(target_layer) => {
                for layer_nb in (current_layer_number + 1..max_layer_nb + 1).rev() {
                    let layer = match self.layers.get(&layer_nb) {
                        Some(l) => l,
                        None => return Err(format!("Could not get layer {layer_nb} in step 1.")),
                    };
                    ep = self.search_layer(layer, point, &mut ep, 1)?;
                    if layer_nb == target_layer {
                        break;
                    }
                }
                ep
            }
        };
        Ok(ep)
    }

    fn step_2(
        &self,
        point: &Point,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
        current_layer_number: usize,
    ) -> Result<
        HashMap<
            usize,
            HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>, BuildNoHashHasher<usize>>,
            BuildNoHashHasher<usize>,
        >,
        String,
    > {
        let mut insertion_results = HashMap::with_hasher(BuildNoHashHasher::default());
        let bound = (current_layer_number + 1).min(self.layers.len());

        for layer_nb in (0..bound).rev() {
            let layer = self.layers.get(&layer_nb).unwrap();

            ep = self.search_layer(layer, point, &mut ep, self.params.ef_cons)?;

            let neighbors_to_connect =
                self.select_heuristic(layer, point, &mut ep, self.params.m, false, true)?;

            // let prune_results = self.prune_connexions(limit, layer, &neighbors_to_connect)?;

            let mut layer_result = HashMap::with_hasher(BuildNoHashHasher::default());
            layer_result.insert(point.id, neighbors_to_connect);
            // layer_result.extend(
            //     prune_results.iter().map(|x| (*x.0, x.1.to_owned())).chain(
            //         [(point.id, neighbors_to_connect)]
            //             .iter()
            //             .map(|x| (x.0, x.1.to_owned())),
            //     ),
            // );
            insertion_results.insert(layer_nb, layer_result);
        }
        Ok(insertion_results)
    }

    fn step_2_layer0(
        index: &RwLockReadGuard<'_, HNSW>,
        layer0: &mut Graph,
        point: &Point,
        mut ep: HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> Result<(), String> {
        ep = index.search_layer(layer0, point, &mut ep, index.params.ef_cons)?;

        let neighbors_to_connect =
            index.select_heuristic(layer0, point, &mut ep, index.params.m, false, true)?;

        let prune_results =
            index.prune_connexions(index.params.mmax0, layer0, &neighbors_to_connect)?;

        let mut layer_result: HashMap<
            usize,
            HashSet<usize, BuildNoHashHasher<usize>>,
            BuildNoHashHasher<usize>,
        > = HashMap::with_hasher(BuildNoHashHasher::default());
        layer_result.extend(
            prune_results.iter().map(|x| (*x.0, x.1.to_owned())).chain(
                [(point.id, neighbors_to_connect)]
                    .iter()
                    .map(|x| (x.0, x.1.to_owned())),
            ),
        );

        for (node, neighbors) in layer_result.drain() {
            if node == point.id {
                layer0.add_node(point.id);
            }
            layer0.replace_neighbors(node, neighbors)?;
        }

        Ok(())
    }

    fn prune_connexions(
        &self,
        limit: usize,
        layer: &Graph,
        nodes_to_prune: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> Result<
        HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>, BuildNoHashHasher<usize>>,
        String,
    > {
        let mut prune_results = HashMap::with_hasher(BuildNoHashHasher::default());
        for node in nodes_to_prune.iter() {
            // if layer.degree(*node)? > limit {
            let point = &self.points.get_point(*node).unwrap();
            let mut old_neighbors = HashSet::with_hasher(BuildNoHashHasher::default());
            old_neighbors.extend(
                layer
                    .neighbors(*node)?
                    .iter()
                    .filter(|old_n| layer.degree(**old_n).unwrap() > 1)
                    .cloned(),
            );
            let new_neighbors =
                self.select_heuristic(&layer, &point, &mut old_neighbors, limit, false, false)?;
            prune_results.insert(*node, new_neighbors);
            // }
        }
        Ok(prune_results)
    }

    fn select_heuristic(
        &self,
        layer: &Graph,
        point: &Point,
        cands_idx: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<HashSet<usize, BuildNoHashHasher<usize>>, String> {
        if extend_cands {
            for idx in cands_idx.clone().iter() {
                for neighbor in layer.neighbors(*idx)? {
                    cands_idx.insert(*neighbor);
                }
            }
        }
        let mut candidates = self.sort_by_distance(point, &cands_idx)?;
        let mut visited = BTreeMap::new();
        let mut selected = BTreeMap::new();

        let (dist_e, e) = candidates.pop_first().unwrap();
        selected.insert(dist_e, e);
        while (candidates.len() > 0) & (selected.len() < m) {
            let (dist_e, e) = candidates.pop_first().unwrap();
            let e_vector = &self.points.get_point(e).unwrap().vector;

            let (dist_from_s, _) = self.get_nearest(&e_vector, selected.values().cloned());

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
        Ok(result)
    }

    pub fn insert(&mut self, point_id: usize, reinsert: bool) -> Result<bool, String> {
        let point = match self.points.get_point(point_id) {
            Some(p) => p,
            None => return Err(format!("{point_id} not in points given to the index.")),
        };

        if self.layers.len() == 0 {
            self.first_insert(point_id);
            return Ok(true);
        }

        if self.layers.get(&0).unwrap().contains(&point_id) & !reinsert {
            return Ok(true);
        }

        let level = point.level;
        let max_layer_nb = self.layers.len() - 1;
        let ep = self.step_1(point, max_layer_nb, level, None)?;
        let insertion_results = self.step_2(point, ep, level)?;
        let nodes_to_prune = insertion_results.clone();

        self.write_results(insertion_results, point_id, level, max_layer_nb)?;

        let mut pruned_results = HashMap::with_hasher(BuildNoHashHasher::default());
        for (layer_nb, layer) in self.layers.iter() {
            let limit = if *layer_nb == 0 {
                self.params.mmax0
            } else {
                self.params.mmax
            };
            if nodes_to_prune.contains_key(layer_nb) {
                let neighbors = nodes_to_prune
                    .get(layer_nb)
                    .unwrap()
                    .get(&point_id)
                    .unwrap();
                if neighbors.len() > limit {
                    let new_node = self.prune_connexions(limit, layer, neighbors)?;
                    pruned_results.insert(*layer_nb, new_node);
                }
            }
        }

        self.write_results(pruned_results, point_id, level, max_layer_nb)?;

        Ok(true)
    }

    pub fn insert_with_ep(
        &mut self,
        point_id: usize,
        ep: HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> Result<bool, String> {
        if !self.points.contains(&point_id) {
            return Ok(false);
        }

        if self.layers.len() == 0 {
            self.first_insert(0);
            return Ok(true);
        }

        let point = match self.points.get_point(point_id) {
            Some(p) => p,
            None => {
                println!("Tried to insert node wirh id {point_id}, but it wasn't found in the index storage.");
                return Ok(false);
            }
        };

        let insertion_results = self.step_2(point, ep, 0)?;

        for (layer_nb, node_data) in insertion_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                if *node == point.id {
                    layer.add_node(point_id);
                }
                layer.remove_edges_with_node(*node);
                for neighbor in neighbors.iter() {
                    layer.add_edge(*node, *neighbor)?;
                }
            }
        }

        Ok(true)
    }

    pub fn insert_par(
        index: &Arc<RwLock<Self>>,
        ids_levels: Vec<(usize, usize)>,
        bar: ProgressBar,
    ) -> Result<(), String> {
        let batch_size = 16;
        let points_len = ids_levels.len();
        let mut batch = Vec::with_capacity(batch_size);

        for (idx, (point_id, level)) in ids_levels.iter().enumerate() {
            let read_ref = index.read();

            let point = read_ref.points.get_point(*point_id).unwrap();
            let max_layer_nb = read_ref.layers.len() - 1;
            let ep = read_ref.step_1(&point, max_layer_nb, *level, None)?;
            let insertion_results = read_ref.step_2(&point, ep, *level)?;

            drop(read_ref);

            batch.push(insertion_results);
            let last_idx = idx == (points_len - 1);
            let new_layer = *level > max_layer_nb;
            let full_batch = batch.len() >= batch_size;

            let have_to_write: bool = last_idx | new_layer | full_batch;

            let mut write_ref = if have_to_write {
                index.write()
            } else {
                continue;
            };
            if new_layer {
                for layer_nb in max_layer_nb + 1..level + 1 {
                    let mut layer = Graph::new();
                    layer.add_node(*point_id);
                    write_ref.layers.insert(layer_nb, layer);
                    write_ref.ep = *point_id;
                }
            }
            if !bar.is_hidden() {
                bar.inc(batch.len() as u64);
            }
            write_ref.write_batch(&mut batch)?;
            batch.clear();
        }
        Ok(())
    }

    pub fn insert_par_v2(
        index: &Arc<RwLock<Self>>,
        ids: Vec<usize>,
        bar: ProgressBar,
    ) -> Result<(), String> {
        let points_len = ids.len();

        let read_ref = index.read();
        let mut layer0 = read_ref.layers.get(&0).unwrap().clone();
        drop(read_ref);

        for (idx, point_id) in ids.iter().enumerate() {
            if index.is_locked_exclusive() {
                index.read().update_thread_layer(&mut layer0);
            }

            let read_ref = index.read();

            let point = read_ref.points.get_point(*point_id).unwrap();
            let level = point.level;
            let max_layer_nb = read_ref.layers.len() - 1;
            let ep = read_ref.step_1(&point, max_layer_nb, level, None)?;
            HNSW::step_2_layer0(&read_ref, &mut layer0, &point, ep)?;
            drop(read_ref);

            bar.inc(1);

            let last_idx = idx == (points_len - 1);

            let mut write_ref = if last_idx {
                index.write()
            } else {
                continue;
            };

            write_ref.update_layer0(&layer0);
        }
        Ok(())
    }

    fn prune_layer(&self, layer: &mut Graph) {
        let mut to_prune = HashSet::with_hasher(BuildNoHashHasher::default());
        for (node, neighbors) in layer.nodes.iter() {
            if neighbors.len() > self.params.mmax0 {
                to_prune.insert(*node);
            }
        }
        let prune_results = self
            .prune_connexions(self.params.mmax0, &layer, &to_prune)
            .unwrap();
        let pr_keys: HashSet<&usize> = HashSet::from_iter(prune_results.keys());
        for (node, neighbors) in layer.nodes.iter_mut() {
            if pr_keys.contains(node) {
                *neighbors = prune_results.get(node).unwrap().clone();
            }
        }
    }

    fn update_thread_layer(&self, thread_layer: &mut Graph) {
        for (node, new_neighbors) in self.layers.get(&0).unwrap().nodes.iter() {
            let entry = thread_layer.nodes.entry(*node);
            entry
                .and_modify(|neighbors| {
                    neighbors.extend(new_neighbors);
                })
                .or_insert(new_neighbors.clone());
        }

        self.prune_layer(thread_layer);
    }

    fn update_layer0(&mut self, thread_layer: &Graph) {
        let true_layer0 = self.layers.get_mut(&0).unwrap();
        let layer0_nodes = thread_layer
            .nodes
            .keys()
            .collect::<HashSet<&usize, BuildNoHashHasher<usize>>>();
        let true_layer_nodes = true_layer0
            .nodes
            .keys()
            .collect::<HashSet<&usize, BuildNoHashHasher<usize>>>();
        let connexions_made = layer0_nodes
            .difference(&true_layer_nodes)
            .map(|x| **x)
            .collect();
        for (node, new_neighbors) in thread_layer.nodes.iter() {
            let entry = true_layer0.nodes.entry(*node);
            entry
                .and_modify(|neighbors| {
                    neighbors.extend(new_neighbors);
                })
                .or_insert(new_neighbors.clone());
        }

        let true_layer0 = self.layers.get(&0).unwrap();
        let pruned_connextions = self
            .prune_connexions(self.params.mmax0, true_layer0, &connexions_made)
            .unwrap();
        let true_layer0 = self.layers.get_mut(&0).unwrap();

        for (node, new_neighbors) in pruned_connextions.iter() {
            let entry = true_layer0.nodes.entry(*node);
            entry
                .and_modify(|neighbors| {
                    *neighbors = new_neighbors.clone();
                })
                .or_insert(new_neighbors.clone());
        }
        // println!(
        //     "thread {tn} nb of nodes end of update {}",
        //     true_layer0.nb_nodes()
        // );
        // println!("thread {tn} inserted {}", true_layer0.nb_nodes() - start);
    }

    fn write_batch(
        &mut self,
        batch: &mut Vec<
            HashMap<
                usize,
                HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>, BuildNoHashHasher<usize>>,
                BuildNoHashHasher<usize>,
            >,
        >,
    ) -> Result<(), String> {
        let batch_len = batch.len();
        for _ in 0..batch_len {
            let batch_data = batch.pop().unwrap();
            for (layer_nb, node_data) in batch_data.iter() {
                let layer = self.layers.get_mut(&layer_nb).unwrap();
                for (node, neighbors) in node_data.iter() {
                    layer.add_node(*node);
                    for old_neighbor in layer.neighbors(*node)?.clone() {
                        layer.remove_edge(*node, old_neighbor)?;
                    }
                    for neighbor in neighbors.iter() {
                        layer.add_edge(*node, *neighbor)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_results(
        &mut self,
        mut insertion_results: HashMap<
            usize,
            HashMap<usize, HashSet<usize, BuildNoHashHasher<usize>>, BuildNoHashHasher<usize>>,
            BuildNoHashHasher<usize>,
        >,
        point_id: usize,
        level: usize,
        max_layer_nb: usize,
    ) -> Result<(), String> {
        for (layer_nb, mut node_data) in insertion_results.drain() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.drain() {
                if node == point_id {
                    layer.add_node(point_id);
                }
                layer.replace_neighbors(node, neighbors)?;
            }
        }

        if level > max_layer_nb {
            for layer_nb in max_layer_nb + 1..level + 1 {
                let mut layer = Graph::new();
                layer.add_node(point_id);
                self.layers.insert(layer_nb, layer);
            }
            self.ep = point_id;
        }
        Ok(())
    }

    /// Assigns IDs to all vectors (usize).
    /// Creates Point structs, giving a level to each Point.
    /// Stores the Point structs in a Points struct, in index.points
    fn store_points(&mut self, vectors: Vec<Vec<f32>>) {
        // center_vectors(&mut vectors);

        // let mut rng = rand::thread_rng();
        // let collection = Vec::from_iter(vectors.iter().enumerate().map(|(id, v)| {
        //     Point::new_quantized(id, get_new_node_layer(self.params.ml, &mut rng), v)
        // }));
        // let points = PointsV2::Collection(collection);

        let points = PointsV2::from_vecs(vectors, self.params.ml);

        self.points.extend_or_fill(points);
    }

    fn first_insert(&mut self, point_id: usize) {
        let mut layer = Graph::new();
        layer.add_node(point_id);
        self.layers.insert(0, layer);
        self.ep = point_id;
    }

    fn reinsert_with_degree_zero(&mut self) {
        println!("Reinserting nodes with degree 0");
        for _ in 0..3 {
            for (_, layer) in self.layers.clone().iter() {
                for (node, neighbors) in layer.nodes.iter() {
                    if neighbors.len() == 0 {
                        self.insert(*node, true).unwrap();
                    }
                }
            }
        }
    }

    pub fn build_index(
        m: usize,
        ef_cons: Option<usize>,
        vectors: Vec<Vec<f32>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let dim = match vectors.first() {
            Some(vector) => vector.len(),
            None => return Err("Could not read vector dimension.".to_string()),
        };
        let mut index = HNSW::new(m, ef_cons, dim);
        index.store_points(vectors);

        let bar = get_progress_bar("Inserting Vectors".to_string(), index.points.len(), verbose);

        let ids_levels: Vec<(usize, usize)> = index.points.ids_levels().collect();
        for (id, _) in ids_levels {
            index.insert(id, false)?;
            bar.inc(1);
        }
        index.reinsert_with_degree_zero();
        index.assert_param_compliance();
        Ok(index)
    }

    pub fn build_index_par(
        m: usize,
        ef_cons: Option<usize>,
        vectors: Vec<Vec<f32>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        let dim = match vectors.first() {
            Some(vector) => vector.len(),
            None => return Err("Could not read vector dimension.".to_string()),
        };
        let mut index = HNSW::new(m, ef_cons, dim);
        index.store_points(vectors);
        index.first_insert(0);

        let mut ids_levels: Vec<(usize, usize)> = index.points.ids_levels().collect();
        ids_levels.shuffle(&mut thread_rng());

        let mut points_split = split_ids_levels(ids_levels, nb_threads);
        let index_arc = Arc::new(RwLock::new(index));

        let multibar = Arc::new(indicatif::MultiProgress::new());

        let mut handlers = Vec::new();
        for thread_idx in 0..nb_threads {
            let index_copy = index_arc.clone();
            let ids_levels: Vec<(usize, usize)> = points_split.pop().unwrap();
            let bar = multibar.insert(
                thread_idx,
                get_progress_bar(format!("Thread {}:", thread_idx), ids_levels.len(), verbose),
            );
            handlers.push(std::thread::spawn(move || {
                Self::insert_par(&index_copy, ids_levels, bar).unwrap();
            }));
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }

        index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    pub fn build_index_par_v2(
        m: usize,
        ef_cons: Option<usize>,
        vectors: Vec<Vec<f32>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        // let nb_threads = 16;
        let dim = match vectors.first() {
            Some(vector) => vector.len(),
            None => return Err("Could not read vector dimension.".to_string()),
        };
        let mut index = HNSW::new(m, ef_cons, dim);
        index.store_points(vectors);
        index.first_insert(0);
        index = HNSW::insert_non_zero(index, verbose)?;

        let (index, eps_ids_map) = HNSW::find_layer_eps(index, 1, verbose)?;

        let mut points_split = index.partition_points(eps_ids_map, nb_threads, 1);

        let index_arc = Arc::new(RwLock::new(index));
        let multibar = Arc::new(indicatif::MultiProgress::new());

        let mut handlers = Vec::new();
        for thread_idx in 0..nb_threads {
            let index_copy = index_arc.clone();
            let ids: Vec<usize> = points_split.pop().unwrap().iter().cloned().collect();
            let bar = multibar.insert(
                thread_idx,
                get_progress_bar(format!("Thread {}:", thread_idx), ids.len(), verbose),
            );
            handlers.push(std::thread::spawn(move || {
                Self::insert_par_v2(&index_copy, ids, bar).unwrap();
            }));
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }
        index_arc.write().reinsert_with_degree_zero();
        index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    fn partition_points(
        &self,
        mut eps_ids_map: HashMap<usize, HashSet<usize>>,
        nb_splits: usize,
        layer_nb: usize,
    ) -> Vec<HashSet<usize>> {
        let nb_eps = eps_ids_map.keys().count();
        let eps_per_split = nb_eps / nb_splits;
        let mut splits: Vec<HashSet<usize>> =
            Vec::from_iter((0..nb_splits).map(|_| HashSet::new()));
        let mut inserted = HashSet::new();
        let mut next_ep = None;

        while inserted.len() < nb_eps {
            for split_nb in 0..nb_splits {
                if inserted.len() == nb_eps {
                    break;
                }
                let reference_ep = match next_ep {
                    None => *eps_ids_map
                        .keys()
                        .filter(|x| !inserted.contains(*x))
                        .take(1)
                        .next()
                        .unwrap(),
                    Some(n_ep) => n_ep,
                };
                let ep_point = self.points.get_point(reference_ep).unwrap();

                let split_mut = splits.get_mut(split_nb).unwrap();
                split_mut.extend(eps_ids_map.get(&reference_ep).unwrap().iter());
                inserted.insert(reference_ep);
                eps_ids_map.remove(&reference_ep);
                let mut inserted_eps = 1;

                let ep_neighbors = self
                    .layers
                    .get(&layer_nb)
                    .unwrap()
                    .neighbors(reference_ep)
                    .unwrap();

                let mut sorted_neighbors = self.sort_by_distance(ep_point, ep_neighbors).unwrap();
                while inserted_eps < eps_per_split {
                    let (_, nearest_neighbor) = match sorted_neighbors.pop_first() {
                        Some(key_value) => key_value,
                        None => break,
                    };
                    if inserted.contains(&nearest_neighbor) {
                        continue;
                    }
                    let point_ids_to_insert = match eps_ids_map.get(&nearest_neighbor) {
                        Some(p_ids) => p_ids,
                        None => continue,
                    };
                    split_mut.extend(point_ids_to_insert.iter());
                    inserted.insert(nearest_neighbor);
                    eps_ids_map.remove(&nearest_neighbor);
                    inserted_eps += 1;
                    if inserted.len() == nb_eps {
                        break;
                    }
                }
                next_ep = match sorted_neighbors.pop_first() {
                    Some(key_value) => {
                        let key = key_value.1;
                        if eps_ids_map.contains_key(&key) {
                            Some(key)
                        } else {
                            None
                        }
                    }
                    None => None,
                };
            }
            next_ep = None;
        }
        splits
    }

    /// Inserts the points that will be present in layer 1 or above.
    fn insert_non_zero(index: Self, verbose: bool) -> Result<Self, String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        let ids_levels: Vec<(usize, usize)> =
            index.points.ids_levels().filter(|x| x.1 > 0).collect();
        let mut points_split = split_ids_levels(ids_levels, nb_threads);
        let index_arc = Arc::new(RwLock::new(index));

        let mut handlers = Vec::new();
        for thread_idx in 0..nb_threads {
            let index_copy = index_arc.clone();
            let ids_levels: Vec<(usize, usize)> = points_split.pop().unwrap();
            let bar = get_progress_bar(
                "Inserting non-zeros:".to_string(),
                ids_levels.len(),
                (thread_idx == 0) & verbose,
            );
            handlers.push(std::thread::spawn(move || {
                Self::insert_par(&index_copy, ids_levels, bar).unwrap();
            }));
        }
        for handle in handlers {
            let _ = handle.join().unwrap();
        }

        index_arc.write().reinsert_with_degree_zero();
        index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    /// Finds the entry points in layer for all points that
    /// have not been inserted.
    ///
    /// Returns a HashMap pointing every entry point in the layer to
    /// the points it inserts.
    fn find_layer_eps(
        index: Self,
        target_layer_nb: usize,
        verbose: bool,
    ) -> Result<(Self, HashMap<usize, HashSet<usize>>), String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();

        let to_insert = index.points.ids_levels().filter(|x| x.1 == 0).collect();
        let mut to_insert_split = split_ids_levels(to_insert, nb_threads);

        let mut handlers = Vec::new();
        let index_arc = Arc::new(RwLock::new(index));
        for thread_nb in 0..nb_threads {
            let thread_split = to_insert_split.pop().unwrap();
            let index_ref = index_arc.clone();

            handlers.push(std::thread::spawn(
                move || -> HashMap<usize, HashSet<usize>> {
                    let mut thread_results = HashMap::new();
                    let read_ref = index_ref.read();
                    let bar = get_progress_bar(
                        "Finding entry points".to_string(),
                        thread_split.len(),
                        (thread_nb == nb_threads - 1) & verbose,
                    );
                    for (id, level) in thread_split {
                        let point = read_ref.points.get_point(id).unwrap();
                        let max_layer_nb = read_ref.layers.len() - 1;
                        let ep = read_ref
                            .step_1(point, max_layer_nb, level, Some(target_layer_nb))
                            .unwrap();
                        thread_results
                            .entry(*ep.iter().next().unwrap())
                            .and_modify(|e: &mut HashSet<usize>| {
                                e.insert(id);
                            })
                            .or_insert(HashSet::from([id]));
                        bar.inc(1);
                    }
                    thread_results
                },
            ));
        }

        let mut eps_ids = HashMap::new();
        for handle in handlers {
            let result = handle.join().unwrap();
            for (ep, point_ids) in result.iter() {
                eps_ids
                    .entry(*ep)
                    .and_modify(|e: &mut HashSet<usize>| e.extend(point_ids.iter().cloned()))
                    .or_insert(HashSet::from_iter(point_ids.iter().cloned()));
            }
        }

        let index = Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner();

        Ok((index, eps_ids))
    }

    fn sort_by_distance(
        &self,
        point: &Point,
        others: &HashSet<usize, BuildNoHashHasher<usize>>,
    ) -> Result<BTreeMap<Dist, usize>, String> {
        let result = others.iter().map(|idx| {
            let dist = self.points.get_point(*idx).unwrap().dist2other(point);
            (dist, *idx)
        });
        Ok(BTreeMap::from_iter(result))
    }

    fn get_nearest<I>(&self, vector: &Vector, others: I) -> (Dist, usize)
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| (self.points.get_point(idx).unwrap().dist2vec(vector), idx))
            .min_by_key(|x| x.0)
            .unwrap()
    }

    // TODO:
    // 1. implement a searcher struct, to avoid the creation of btmaps at every call to search_layer.
    // 2. separate the behaviour of search_layer into ef == 1 and ef > 1, the amount of work changes
    // drasticatly, so the method should change. The amount of work seems to be almost the same regardless of
    // the choice of M in the case of ef == 1, maybe I can take advantage of that.
    // 2.1 maybe I can pre-allocate memory in the searcher struct of idea (1) based on M. I can study the
    // relationship of amount of work and memory required as a function of M and pre-allocate.
    pub fn search_layer(
        &self,
        layer: &Graph,
        point: &Point,
        ep: &mut HashSet<usize, BuildNoHashHasher<usize>>,
        ef: usize,
    ) -> Result<HashSet<usize, BuildNoHashHasher<usize>>, String> {
        let mut candidates = self.sort_by_distance(point, &ep)?;
        let mut selected = candidates.clone();

        while let Some((cand2q_dist, candidate)) = candidates.pop_first() {
            let (furthest2q_dist, _) = selected.last_key_value().unwrap();

            if &cand2q_dist > furthest2q_dist {
                break;
            }
            for (n2q_dist, neighbor_point) in layer
                .neighbors(candidate)?
                .iter()
                .filter(|idx| ep.insert(**idx))
                .map(|idx| {
                    let (dist, point) = match self.points.get_point(*idx) {
                        Some(p) => (p.dist2other(point), p),
                        None => {
                            println!(
                                "Tried to get node with id {idx} from index, but it doesn't exist"
                            );
                            panic!("Tried to get a node that doesn't exist.")
                        }
                    };
                    (dist, point)
                })
            {
                let (f2q_dist, _) = selected.last_key_value().unwrap();

                if (&n2q_dist < f2q_dist) | (selected.len() < ef) {
                    candidates.insert(n2q_dist, neighbor_point.id);
                    selected.insert(n2q_dist, neighbor_point.id);

                    if selected.len() > ef {
                        selected.pop_last();
                    }
                }
            }
        }
        // println!("{ef},{counter}");
        let mut result = HashSet::with_hasher(BuildNoHashHasher::default());
        result.extend(selected.values());
        Ok(result)
    }

    /// Mean-centers each vector using each dimension's mean over the entire matrix.
    fn center_vector(&self, vector: &Vec<f32>) -> Result<Vec<f32>, String> {
        let means = self.points.get_means()?;

        Ok(vector
            .iter()
            .enumerate()
            .map(|(idx, x)| x - means[idx])
            .collect())
    }

    /// Saves the index to the specified path.
    /// Creates the path to the file if it didn't exist before.
    pub fn save(&self, index_path: &str) -> std::io::Result<()> {
        let index_path = std::path::Path::new(index_path);
        if !index_path.parent().unwrap().exists() {
            std::fs::create_dir_all(index_path.parent().unwrap())?;
        }
        let file = File::create(index_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn from_path(index_path: &str) -> std::io::Result<Self> {
        let file = File::open(index_path)?;
        let reader = BufReader::new(file);
        let content: serde_json::Value = serde_json::from_reader(reader)?;

        let ep = content
            .get("ep")
            .expect("Error: key 'ep' is not in the index file.")
            .as_number()
            .expect("Error: entry point could not be parsed as a number.")
            .as_i64()
            .unwrap() as usize;

        let params = match content
            .get("params")
            .expect("Error: key 'params' is not in the index file.")
        {
            serde_json::Value::Object(params_map) => extract_params(&params_map),
            err => {
                println!("{err}");
                panic!("Something went wrong reading parameters of the index file.");
            }
        };

        let layers_unparsed = content
            .get("layers")
            .expect("Error: key 'layers' is not in the index file.")
            .as_object()
            .expect("Error: expected key 'layers' to be an Object, but couldn't parse it as such.");
        let mut layers = HashMap::with_hasher(BuildNoHashHasher::default());
        for (layer_nb, layer_content) in layers_unparsed {
            let layer_nb: usize = layer_nb
                .parse()
                .expect("Error: could not load key {key} into layer number");
            let layer_content = layer_content
                .get("nodes")
                .expect("Error: could not load 'key' nodes for layer {key}").as_object().expect("Error: expected key 'nodes' for layer {layer_nb} to be an Object, but couldl not be parsed as such.");
            let mut this_layer = HashMap::new();
            for (node_id, neighbors) in layer_content.iter() {
                let neighbors = neighbors
                    .as_array()
                    .expect("Error: could not load the neighbors of node {node_id} in layer {layer_nb} as an Array.")
                    .iter()
                    .map(|neighbor_id| neighbor_id.as_number().unwrap().as_u64().unwrap() as usize)
                    .collect();
                this_layer.insert(node_id.parse::<usize>().unwrap(), neighbors);
            }
            layers.insert(layer_nb, Graph::from_layer_data(this_layer));
        }

        let (points, means) = match content
            .get("points")
            .expect("Error: key 'points' is not in the index file.")
        {
            serde_json::Value::Object(points_vec) => {
                let err_msg =
                    "Error reading index file: could not find key 'Collection' in 'points', maybe the index is empty.";
                match points_vec.get("Collection").expect(err_msg) {
                    serde_json::Value::Array(points_final) => {
                        extract_points_and_means(points_final)
                    }
                    _ => panic!("Something went wrong reading parameters of the index file."),
                }
            }
            serde_json::Value::String(s) => {
                if s == "Empty" {
                    (Vec::new(), Vec::new())
                } else {
                    panic!("Something went wrong reading parameters of the index file.");
                }
            }
            err => {
                println!("{err:?}");
                panic!("Something went wrong reading parameters of the index file.");
            }
        };

        Ok(HNSW {
            ep,
            params,
            layers,
            points: PointsV2::Collection((points, means)),
        })
    }
}

fn get_progress_bar(message: String, remaining: usize, verbose: bool) -> ProgressBar {
    let bar = if verbose {
        ProgressBar::new(remaining as u64)
    } else {
        return ProgressBar::hidden();
    };
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {bar:60} {percent}% of {len} Elapsed: {elapsed} {per_sec}\n",
        )
        .unwrap()
        .progress_chars(">>-"),
    );
    bar.set_message(message);
    bar
}

pub fn get_new_node_layer(ml: f32, rng: &mut ThreadRng) -> usize {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.gen::<f32>();
        } else {
            break;
        }
    }
    let num = (-rand_nb.log(std::f32::consts::E) * ml).floor() as usize;
    num
}

fn extract_params(params: &serde_json::Map<String, serde_json::Value>) -> Params {
    let hnsw_params = Params::from(
        params
            .get("m")
            .expect("Error: could not find key 'm' in 'params'.")
            .as_number()
            .expect("Error: 'm' could not be parsed as a number.")
            .as_i64()
            .unwrap() as usize,
        Some(
            params
                .get("ef_cons")
                .expect("Error: could not find key 'ef_cons' in 'params'.")
                .as_number()
                .expect("Error: 'ef_cons' could not be parsed as a number.")
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("mmax")
                .expect("Error: could not find key 'mmax' in 'params'.")
                .as_number()
                .expect("Error: 'mmax' could not be parsed as a number.")
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("mmax0")
                .expect("Error: could not find key 'mmax0' in 'params'.")
                .as_number()
                .expect("Error: 'mmax0' could not be parsed as a number.")
                .as_i64()
                .unwrap() as usize,
        ),
        Some(
            params
                .get("ml")
                .expect("Error: could not find key 'ml' in 'params'.")
                .as_number()
                .expect("Error: 'ml' could not be parsed as a number.")
                .as_f64()
                .unwrap() as f32,
        ),
        params
            .get("dim")
            .expect("Error: could not find key 'dim' in 'params'.")
            .as_number()
            .expect("Error: 'dim' could not be parsed as a number.")
            .as_i64()
            .unwrap() as usize,
    );
    hnsw_params
}

fn extract_points_and_means(points_data: &Vec<serde_json::Value>) -> (Vec<Point>, Vec<f32>) {
    let mut points = Vec::new();
    let points_vec = points_data[0].as_array().unwrap();
    let points_means = points_data[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v {
            serde_json::Value::Number(n) => n.as_f64().unwrap() as f32,
            _ => panic!("Not a number."),
        })
        .collect();

    // 0 is the vec of Point, because PointsV2 holds the points and their means
    for (idx, value) in points_vec.iter().enumerate() {
        let id = value.get("id").unwrap().as_u64().unwrap() as usize;
        assert_eq!(id, idx);
        let level = value.get("level").unwrap().as_u64().unwrap() as usize;
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
        let point = Point::from_vector(id, level, vector);
        points.push(point);
    }
    (points, points_means)
}

fn _compute_stats(points: &PointsV2) -> (f32, f32) {
    let mut dists: HashMap<(usize, usize), f32> = HashMap::new();
    for (id, point) in points.iterate() {
        for (idx, pointx) in points.iterate() {
            if id == idx {
                continue;
            }
            dists
                .entry((id.min(idx), id.max(idx)))
                .or_insert(point.dist2vec(&pointx.vector).dist);
        }
    }

    _write_stats(&dists).unwrap();

    let mut mean = 0.0;
    let mut std = 0.0;
    for (_, dist) in dists.iter() {
        mean += dist;
    }
    mean /= dists.len() as f32;

    for (_, dist) in dists.iter() {
        std += (dist - mean).powi(2);
    }
    std /= dists.len() as f32;

    (mean, std.sqrt())
}

fn _write_stats(dists: &HashMap<(usize, usize), f32>) -> std::io::Result<()> {
    std::fs::remove_file("./dist_stats.json")?;
    let file = File::create("./dist_stats.json")?;
    let mut writer = BufWriter::new(file);
    let dists = Vec::from_iter(dists.values());
    serde_json::to_writer_pretty(&mut writer, &dists)?;
    writer.flush()?;
    Ok(())
}

fn split_ids(ids: Vec<usize>, nb_splits: usize) -> Vec<Vec<usize>> {
    let mut split_vector = Vec::new();

    let per_split = ids.len() / nb_splits;

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push(ids[buffer..].to_vec());
        } else {
            split_vector.push(ids[buffer..(buffer + per_split)].to_vec());
            buffer += per_split;
        }
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == ids.len(), "sum: {sum_lens}");

    split_vector
}

fn split_ids_levels(ids_levels: Vec<(usize, usize)>, nb_splits: usize) -> Vec<Vec<(usize, usize)>> {
    let mut split_vector = Vec::new();

    let per_split = ids_levels.len() / nb_splits;

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push(ids_levels[buffer..].to_vec());
        } else {
            split_vector.push(ids_levels[buffer..(buffer + per_split)].to_vec());
            buffer += per_split;
        }
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == ids_levels.len(), "sum: {sum_lens}");

    split_vector
}
