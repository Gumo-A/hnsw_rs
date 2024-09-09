use super::{
    dist::Dist,
    graph::Graph,
    points::{Point, Points},
};
use crate::{helpers::data::split_ids, hnsw::params::Params};

use indicatif::{ProgressBar, ProgressStyle};
use nohash_hasher::{IntMap, IntSet};

use std::{cmp::Reverse, collections::BTreeSet, sync::Arc};

use rand::rngs::ThreadRng;
use rand::Rng;

use serde::{Deserialize, Serialize};

use core::panic;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Searcher {
    selected: BinaryHeap<Dist>,
    candidates: BinaryHeap<Reverse<Dist>>,
    visited: IntSet<usize>,
    visited_heuristic: BinaryHeap<Reverse<Dist>>,
    insertion_results: IntMap<usize, IntMap<usize, BinaryHeap<Dist>>>,
    prune_results: IntMap<usize, IntMap<usize, BinaryHeap<Dist>>>,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            selected: BinaryHeap::new(),
            candidates: BinaryHeap::new(),
            visited: IntSet::default(),
            visited_heuristic: BinaryHeap::new(),
            insertion_results: IntMap::default(),
            prune_results: IntMap::default(),
        }
    }

    fn clear_all(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
        self.insertion_results.clear();
        self.prune_results.clear();
    }

    fn clear_searchers(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
    }
}

#[derive(Debug, Clone)]
// #[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HNSW {
    ep: usize,
    pub params: Params,
    pub points: Points,
    pub layers: IntMap<usize, Graph>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> Self {
        // limit m to 128 so the serialization file's header is cleaner
        if m > 128 {
            panic!("The 'm' parameter is limited to 128 in this implementation. Increase ef_cons if you want a higher quality index.")
        }
        let params = if ef_cons.is_some() {
            Params::from_m_efcons(m, ef_cons.unwrap(), dim)
        } else {
            Params::from_m(m, dim)
        };
        HNSW {
            points: Points::Empty,
            params,
            ep: 0,
            layers: IntMap::default(),
        }
    }

    pub fn assert_param_compliance(&self) {
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
                if neighbors.lock().unwrap().len() > ((max_degree as f32) * 1.1).ceil() as usize {
                    is_ok = false;
                    println!(
                        "layer {layer_nb}, {node} degree = {0}, limit = {1}",
                        neighbors.lock().unwrap().len(),
                        max_degree
                    );
                }

                if (neighbors.lock().unwrap().is_empty()) & (layer.nb_nodes() > 1) {
                    is_ok = false;
                    println!("layer {layer_nb}, {node} degree = 0",);
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

    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<usize>, String> {
        let mut searcher = Searcher::new();

        let point = Point::new_quantized(0, 0, &self.center_vector(vector)?);

        searcher
            .selected
            .push(point.dist2other(self.points.get_point(self.ep).unwrap()));
        let nb_layer = self.layers.len();

        for layer_nb in (1..nb_layer).rev() {
            self.search_layer(
                &mut searcher,
                self.layers.get(&(layer_nb)).unwrap(),
                &point,
                1,
            )?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        self.search_layer(&mut searcher, layer_0, &point, ef)?;

        let anns: Vec<Dist> = searcher.selected.into_sorted_vec();
        Ok(anns.iter().take(n).map(|x| x.id).collect())
    }

    // TODO: multithreaded
    // pub fn anns_by_vectors(
    //     index: &Self,
    //     vectors: &Vec<Vec<f32>>,
    //     n: usize,
    //     ef: usize,
    //     verbose: bool,
    // ) -> Result<Vec<Vec<usize>>, String> {
    //     let nb_threads = std::thread::available_parallelism().unwrap().get();
    //     let per_thread = vectors.len() / nb_threads;
    //     let index_arc = Arc::new(index);
    //     let vectors_arc = Arc::new(vectors);

    //     let mut handlers = Vec::new();
    //     std::thread::scope(|s| {
    //         let mut buffer = 0;
    //         for thread_idx in 0..nb_threads {
    //             let vectors_ref = Arc::clone(&vectors_arc);
    //             let thread_vector_ids = if thread_idx == (nb_threads - 1) {
    //                 buffer..vectors_ref.len()
    //             } else {
    //                 buffer..(buffer + per_thread)
    //             };
    //             let thread_len = thread_vector_ids.clone().count();
    //             buffer += per_thread;
    //             let index_ref = Arc::clone(&index_arc);
    //             handlers.push(
    //                 s.spawn(move || -> Result<(usize, Vec<Vec<usize>>), String> {
    //                     let mut thread_results = Vec::with_capacity(thread_len);
    //                     let bar = get_progress_bar(
    //                         format!("T{thread_idx}: Finding ANNs with ef{ef}"),
    //                         thread_len,
    //                         verbose,
    //                     );
    //                     for i in thread_vector_ids {
    //                         thread_results.push(index_ref.ann_by_vector(
    //                             vectors_ref.get(i).unwrap(),
    //                             n,
    //                             ef,
    //                         )?);
    //                         bar.inc(1);
    //                     }
    //                     Ok((thread_idx, thread_results))
    //                 }),
    //             );
    //         }
    //     });

    //     let mut intermediate_results: Vec<(usize, Vec<Vec<usize>>)> =
    //         Vec::with_capacity(nb_threads);
    //     for handle in handlers {
    //         let thread_results = handle.join().unwrap()?;
    //         intermediate_results.push(thread_results);
    //     }
    //     intermediate_results.sort_by_key(|(thread_idx, _)| *thread_idx);
    //     let results = Vec::from_iter(
    //         intermediate_results
    //             .iter()
    //             .map(|(_, thread_results)| thread_results.iter().cloned())
    //             .flatten(),
    //     );
    //     Ok(results)
    // }

    fn step_1(
        &self,
        searcher: &mut Searcher,
        point: &Point,
        max_layer_nb: usize,
        level: usize,
        stop_at_layer: Option<usize>,
    ) -> Result<(), String> {
        // let mut ep = IntSet::default();
        // ep.insert(self.ep);

        // let mut step_1_results: BinaryHeap<Dist> = BinaryHeap::from([point.dist2other(self.points.get_point(self.ep).unwrap())]);
        let target_layer = stop_at_layer.unwrap_or(0);
        for layer_nb in (level + 1..=max_layer_nb).rev() {
            let layer = match self.layers.get(&layer_nb) {
                Some(l) => l,
                None => return Err(format!("Could not get layer {layer_nb} in step 1.")),
            };
            self.search_layer(searcher, layer, point, 1)?;
            if layer_nb == target_layer {
                break;
            }
        }
        Ok(())
    }

    fn step_2(&self, searcher: &mut Searcher, point: &Point, level: usize) -> Result<(), String> {
        let bound = (level).min(self.layers.len() - 1);

        for layer_nb in (0..=bound).rev() {
            let layer = self.layers.get(&layer_nb).unwrap();

            self.search_layer(searcher, layer, point, self.params.ef_cons)?;

            self.select_heuristic(searcher, layer, point, self.params.m, false, true)?;

            let layer_result = searcher
                .insertion_results
                .entry(layer_nb)
                .or_insert(IntMap::default());
            let point_neighbors = searcher.selected.clone();
            layer_result.insert(point.id, point_neighbors);
        }
        Ok(())
    }

    // fn step_2_layer0(
    //     index: &Arc<Self>,
    //     searcher: &mut Searcher,
    //     layer0: &mut Graph,
    //     point: &Point,
    // ) -> Result<(), String> {
    //     index.search_layer(searcher, layer0, point, index.params.ef_cons)?;
    //     index.select_heuristic(searcher, layer0, point, index.params.m, false, true)?;
    //     let layer_result = searcher
    //         .insertion_results
    //         .entry(0)
    //         .or_insert(IntMap::default());
    //     layer_result.insert(point.id, searcher.selected.clone());

    //     Ok(())
    // }

    fn prune_connexions(&self, searcher: &mut Searcher) -> Result<(), String> {
        searcher.prune_results.clear();

        for (layer_nb, node_neighbors) in searcher.insertion_results.clone().iter() {
            let layer = self.layers.get(layer_nb).unwrap();
            let limit = if *layer_nb == 0 {
                self.params.mmax0
            } else {
                self.params.mmax
            };

            for (_, neighbors) in node_neighbors.iter() {
                let to_prune = neighbors
                    .iter()
                    .filter(|x| layer.degree(x.id).unwrap() > limit)
                    .map(|x| *x);

                for dist in to_prune {
                    // searcher.clear_searchers();
                    let nearest =
                        HNSW::select_simple(layer.neighbors(dist.id)?.iter().copied(), limit)?;
                    let entry = searcher
                        .prune_results
                        .entry(*layer_nb)
                        .or_insert(IntMap::default());
                    entry.insert(dist.id, nearest);
                }
            }
        }
        Ok(())
    }

    fn select_heuristic(
        &self,
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        // let s1 = Instant::now();
        searcher.visited_heuristic.clear();
        searcher.candidates.clear();
        searcher
            .candidates
            .extend(searcher.selected.iter().map(|dist| Reverse(*dist)));
        searcher.selected.clear();
        // println!("s1 {}", s1.elapsed().as_nanos());

        if extend_cands {
            for dist in searcher
                .candidates
                .iter()
                .copied()
                .collect::<Vec<Reverse<Dist>>>()
            {
                for neighbor_dist in layer.neighbors(dist.0.id)? {
                    searcher.candidates.push(Reverse(
                        point.dist2other(self.points.get_point(neighbor_dist.id).unwrap()),
                    ));
                }
            }
        }

        // let s2 = Instant::now();
        let dist_e = searcher.candidates.pop().unwrap();
        searcher.selected.push(dist_e.0);
        // println!("s2 {}", s2.elapsed().as_nanos());

        while (!searcher.candidates.is_empty()) & (searcher.selected.len() < m) {
            // let s3 = Instant::now();
            let dist_e = searcher.candidates.pop().unwrap();
            let e_point = self.points.get_point(dist_e.0.id).unwrap();
            // println!("s3 {}", s3.elapsed().as_nanos());

            // let s4 = Instant::now();
            let dist_from_s = self.get_nearest(e_point, searcher.selected.iter().map(|x| x.id));
            // println!("s4 {}", s4.elapsed().as_nanos());

            // let s5 = Instant::now();
            if dist_e.0 < dist_from_s {
                searcher.selected.push(dist_e.0);
            } else if keep_pruned {
                searcher.visited_heuristic.push(dist_e);
            }
            // println!("s5 {}", s5.elapsed().as_nanos());
        }

        // let s6 = Instant::now();
        if keep_pruned {
            while (!searcher.visited_heuristic.is_empty()) & (searcher.selected.len() < m) {
                let dist_e = searcher.visited_heuristic.pop().unwrap();
                searcher.selected.push(dist_e.0);
            }
        }
        // println!("s6 {}", s6.elapsed().as_nanos());

        Ok(())
    }

    fn select_simple<I>(
        // searcher: &mut Searcher,
        candidate_dists: I,
        m: usize,
    ) -> Result<BinaryHeap<Dist>, String>
    where
        I: Iterator<Item = Dist>,
    {
        // searcher.candidates.clear();
        // searcher.selected.clear();
        // searcher.candidates.extend(candidate_dists.map(|dist| Reverse(dist)));

        // while (!searcher.candidates.is_empty()) & (searcher.selected.len() < m) {
        //     let dist_e = searcher.candidates.pop().unwrap();
        //     searcher.selected.push(dist_e.0);
        // }
        let cands = BTreeSet::from_iter(candidate_dists);
        Ok(BinaryHeap::from_iter(cands.iter().copied().take(m)))
    }

    pub fn insert(&mut self, point_id: usize, searcher: &mut Searcher) -> Result<bool, String> {
        // let s0 = Instant::now();
        searcher.clear_all();

        let point = match self.points.get_point(point_id) {
            Some(p) => p,
            None => return Err(format!("{point_id} not in points given to the index.")),
        };

        if self.layers.is_empty() {
            self.first_insert(point_id);
            return Ok(true);
        }

        searcher
            .selected
            .push(point.dist2other(self.points.get_point(self.ep).unwrap()));

        let level = point.level;
        let max_layer_nb = self.layers.len() - 1;
        // println!("s0 {}", s0.elapsed().as_nanos());

        // let s1 = Instant::now();
        self.step_1(searcher, point, max_layer_nb, level, None)?;
        // println!("s1 {}", s1.elapsed().as_nanos());

        // let s2 = Instant::now();
        self.step_2(searcher, point, level)?;
        // println!("s2 {}", s2.elapsed().as_nanos());

        // let s3 = Instant::now();
        self.write_results(searcher, point_id, level, max_layer_nb)?;
        // println!("s3 {}", s3.elapsed().as_nanos());

        self.prune_connexions(searcher)?;

        self.write_results_prune(searcher)?;

        Ok(true)
    }

    // pub fn insert_with_ep(&mut self, point_id: usize, ep: IntSet<usize>) -> Result<bool, String> {
    //     if !self.points.contains(&point_id) {
    //         return Ok(false);
    //     }

    //     if self.layers.is_empty() {
    //         self.first_insert(0);
    //         return Ok(true);
    //     }

    //     let point = match self.points.get_point(point_id) {
    //         Some(p) => p,
    //         None => {
    //             println!("Tried to insert node wirh id {point_id}, but it wasn't found in the index storage.");
    //             return Ok(false);
    //         }
    //     };

    //     let insertion_results = self.step_2(point, ep, 0)?;

    //     for (layer_nb, node_data) in insertion_results.iter() {
    //         let layer = self.layers.get_mut(layer_nb).unwrap();
    //         for (node, neighbors) in node_data.iter() {
    //             if *node == point.id {
    //                 layer.add_node(point_id);
    //             }
    //             layer.remove_edges_with_node(*node);
    //             for (neighbor, dist) in neighbors.iter() {
    //                 layer.add_edge(*node, *neighbor, *dist)?;
    //             }
    //         }
    //     }

    //     Ok(true)
    // }

    // pub fn insert_par(index: Arc<Self>, ids: Vec<usize>, bar: ProgressBar) -> Result<(), String> {
    //     let mut searcher = Searcher::new();

    //     for point_id in ids.iter() {
    //         searcher.clear_searchers();

    //         let max_layer_nb = index.layers.len() - 1;

    //         let point = index.points.get_point(*point_id).unwrap();
    //         let level = point.level;
    //         searcher
    //             .selected
    //             .push(point.dist2other(index.points.get_point(index.ep).unwrap()));

    //         match index.step_1(&mut searcher, point, max_layer_nb, level, None) {
    //             Ok(()) => (),
    //             Err(msg) => return Err(format!("Error in step 1: {msg}")),
    //         };

    //         match index.step_2(&mut searcher, point, level) {
    //             Ok(()) => (),
    //             Err(msg) => return Err(format!("Error in step 2: {msg}")),
    //         };

    //         index.write_results(&searcher, *point_id, level, max_layer_nb)?;
    //         index.prune_connexions(&mut searcher)?;
    //         index.write_results_prune(&searcher)?;
    //         searcher.clear_all();

    //         if !bar.is_hidden() {
    //             bar.inc(1);
    //         }
    //     }
    //     Ok(())
    // }

    pub fn insert_par(index: Arc<Self>, ids: Vec<usize>, bar: ProgressBar) -> Result<(), String> {
        let mut searcher = Searcher::new();
        let max_layer_nb = index.layers.len() - 1;

        for point_id in ids.iter() {
            searcher.clear_searchers();

            let point = index.points.get_point(*point_id).unwrap();
            let level = point.level;
            searcher
                .selected
                .push(point.dist2other(index.points.get_point(index.ep).unwrap()));

            match index.step_1(&mut searcher, point, max_layer_nb, level, None) {
                Ok(()) => (),
                Err(msg) => return Err(format!("Error in step 1: {msg}")),
            };

            match index.step_2(&mut searcher, point, level) {
                Ok(()) => (),
                Err(msg) => return Err(format!("Error in step 2: {msg}")),
            };

            index.write_results_v2(&searcher)?;
            index.prune_connexions(&mut searcher)?;
            index.write_results_prune_v2(&searcher)?;
            searcher.clear_all();

            if !bar.is_hidden() {
                bar.inc(1);
            }
        }
        Ok(())
    }

    fn write_results(
        &mut self,
        searcher: &Searcher,
        point_id: usize,
        level: usize,
        max_layer_nb: usize,
    ) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.insertion_results.iter() {
            let layer = self.layers.get(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
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

    fn write_results_v2(
        &self,
        searcher: &Searcher,
        // point_id: usize,
        // level: usize,
        // max_layer_nb: usize,
    ) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.insertion_results.iter() {
            let layer = self.layers.get(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
            }
        }
        Ok(())
    }

    fn write_results_prune(&mut self, searcher: &Searcher) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.prune_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.add_node(*node);
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
            }
        }

        Ok(())
    }

    fn write_results_prune_v2(&self, searcher: &Searcher) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.prune_results.iter() {
            let layer = self.layers.get(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
            }
        }

        Ok(())
    }

    /// Assigns IDs to all vectors (usize).
    /// Creates Point structs, giving a level to each Point.
    /// Creates all the layers and stores the points in their levels.
    /// Stores the Point structs in index.points.
    fn store_points(&mut self, vectors: Vec<Vec<f32>>) {
        let points = Points::from_vecs(vectors, self.params.ml);
        // TODO: if this is a bulk update, make sure to use the max between
        // the current max level and the new points' max level.
        let max_layer_nb = points
            .iterate()
            .map(|(_, point)| point.level)
            .max()
            .unwrap();
        for layer_nb in 0..=max_layer_nb {
            self.layers.entry(layer_nb).or_insert(Graph::new());
        }
        for (point_id, point) in points.iterate() {
            for l in 0..=point.level {
                self.layers.get_mut(&l).unwrap().add_node(point_id);
            }
        }
        self.points.extend_or_fill(points);
        self.ep = *self
            .layers
            .get(&max_layer_nb)
            .unwrap()
            .nodes
            .keys()
            .next()
            .unwrap();
    }

    fn first_insert(&mut self, point_id: usize) {
        let mut layer = Graph::new();
        layer.add_node(point_id);
        self.layers.insert(0, layer);
        self.ep = point_id;
    }

    fn delete_one_node_layer(&mut self) {
        let max_layer_nb = self.layers.len() - 1;
        if self.layers.get(&max_layer_nb).unwrap().nb_nodes() == 1 {
            println!("Removing top layer, it only contains one point");
            self.layers.remove(&max_layer_nb);
        }
    }

    pub fn build_index(
        m: usize,
        ef_cons: Option<usize>,
        vectors: Vec<Vec<f32>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let mut searcher = Searcher::new();
        let dim = match vectors.first() {
            Some(vector) => vector.len(),
            None => return Err("Could not read vector dimension.".to_string()),
        };
        let mut index = HNSW::new(m, ef_cons, dim);
        index.store_points(vectors);

        let bar = get_progress_bar("Inserting Vectors".to_string(), index.points.len(), verbose);

        let ids: Vec<usize> = index.points.ids().collect();
        for id in ids.iter() {
            index.insert(*id, &mut searcher)?;
            if !bar.is_hidden() {
                bar.inc(1);
            }
        }
        index.delete_one_node_layer();
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
        let index_arc = Arc::new(index);

        for layer_nb in (0..index_arc.layers.len()).rev() {
            let ids: Vec<usize> = index_arc
                .points
                .iterate()
                .filter(|(_, point)| point.level == layer_nb)
                .map(|(id, _)| id)
                .collect();

            let mut points_split = split_ids(ids, nb_threads);

            let mut handlers = Vec::new();
            for thread_idx in 0..nb_threads {
                let index_copy = Arc::clone(&index_arc);
                let ids_split: Vec<usize> = points_split.pop().unwrap();
                let bar = get_progress_bar(
                    format!("Layer {layer_nb}:"),
                    ids_split.len(),
                    (verbose) & (thread_idx == 0),
                );
                handlers.push(std::thread::spawn(move || {
                    Self::insert_par(index_copy, ids_split, bar).unwrap();
                }));
            }
            for handle in handlers {
                handle.join().unwrap();
            }
        }

        // index_arc.assert_param_compliance();

        let mut index =
            Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");
        index.delete_one_node_layer();
        Ok(index)
    }

    // // // TODO: Implementation is faster, but index quality is not good enough
    // pub fn build_index_par_v2(
    //     m: usize,
    //     ef_cons: Option<usize>,
    //     vectors: Vec<Vec<f32>>,
    //     verbose: bool,
    // ) -> Result<Self, String> {
    //     let nb_threads = std::thread::available_parallelism().unwrap().get();

    //     let dim = match vectors.first() {
    //         Some(vector) => vector.len(),
    //         None => return Err("Could not read vector dimension.".to_string()),
    //     };

    //     let mut index = HNSW::new(m, ef_cons, dim);
    //     index.store_points(vectors);

    //     index.first_insert(0);
    //     index = HNSW::insert_non_zero(index, verbose)?;

    //     let (index, eps_ids_map) = HNSW::find_layer_eps(index, 0, verbose)?;
    //     let ids_eps = IntMap::from_iter(
    //         eps_ids_map
    //             .iter()
    //             .map(|(ep, ids)| ids.iter().map(|id| (*id, *ep)))
    //             .flatten(),
    //     );
    //     let all_eps = IntSet::from_iter(eps_ids_map.keys().copied());

    //     let (mut points_split, split_eps) = index.partition_points(eps_ids_map, nb_threads, 1);

    //     let ids_eps_arc = Arc::new(ids_eps);
    //     let index_arc = Arc::new(index);
    //     let multibar = Arc::new(indicatif::MultiProgress::new());

    //     let mut handlers = Vec::new();
    //     for thread_idx in 0..nb_threads {
    //         let index_clone = index_arc.clone();
    //         let ids_eps_clone = ids_eps_arc.clone();
    //         let ids: Vec<usize> = points_split
    //             .remove(&thread_idx)
    //             .unwrap()
    //             .iter()
    //             .copied()
    //             .collect();
    //         let bar = multibar.insert(
    //             thread_idx,
    //             get_progress_bar(format!("Thread {}:", thread_idx), ids.len(), verbose),
    //         );
    //         handlers.push(std::thread::spawn(
    //             move || -> (usize, Result<Graph, String>) {
    //                 (
    //                     thread_idx,
    //                     Self::insert_par_v2(index_clone, ids, ids_eps_clone, bar),
    //                 )
    //             },
    //         ));
    //     }
    //     let mut thread_results = IntMap::default();
    //     for handle in handlers {
    //         let (thread_idx, result) = handle.join().unwrap();
    //         thread_results.insert(thread_idx, result?);
    //     }
    //     let mut index =
    //         Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");

    //     let mut merged = IntSet::default();
    //     for (idx, (thread_idx, layer0)) in thread_results.iter_mut().enumerate() {
    //         if idx == 0 {
    //             *index.layers.get_mut(&0).unwrap() = layer0.clone();
    //             merged.insert(*thread_idx);
    //             continue;
    //         }
    //         index.join_thread_result(
    //             layer0,
    //             *thread_idx,
    //             &all_eps,
    //             split_eps.get(thread_idx).unwrap(),
    //             verbose,
    //         )?;
    //         merged.insert(*thread_idx);
    //     }

    //     Ok(index)
    // }

    fn get_nearest<I>(&self, point: &Point, others: I) -> Dist
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| point.dist2other(self.points.get_point(idx).unwrap()))
            .min()
            .unwrap()
    }

    pub fn search_layer(
        &self,
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point,
        ef: usize,
    ) -> Result<(), String> {
        searcher
            .candidates
            .extend(searcher.selected.iter().map(|x| Reverse(*x)));
        searcher
            .visited
            .extend(searcher.selected.iter().map(|dist| dist.id));

        while !searcher.candidates.is_empty() {
            let cand_dist = searcher.candidates.pop().unwrap();
            let furthest2q_dist = searcher.selected.peek().unwrap();
            if cand_dist.0 > *furthest2q_dist {
                break;
            }
            let cand_neighbors = match layer.neighbors(cand_dist.0.id) {
                Ok(neighs) => neighs,
                Err(msg) => return Err(format!("Error in search_layer: {msg}")),
            };

            // pre-compute distances to candidate neighbors to take advantage of
            // caches and to prevent the re-construction of the query to a full vector
            let q2cand_neighbors_dists = point.dist2others(
                cand_neighbors
                    .iter()
                    .filter(|dist| searcher.visited.insert(dist.id))
                    .map(|dist| match self.points.get_point(dist.id) {
                        Some(p) => p,
                        None => panic!("nope!"),
                    }),
            );
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = searcher.selected.peek().unwrap();
                if (n2q_dist < *f2q_dist) | (searcher.selected.len() < ef) {
                    searcher.selected.push(n2q_dist);
                    searcher.candidates.push(Reverse(n2q_dist));

                    if searcher.selected.len() > ef {
                        searcher.selected.pop();
                    }
                }
            }
        }
        searcher.candidates.clear();
        searcher.visited.clear();
        Ok(())
    }

    // pub fn search_layer_s(
    //     &self,
    //     layer: &Graph,
    //     searcher: &mut Searcher,
    //     ef: usize,
    // ) -> Result<(), String> {
    //     let searcher_point = searcher.point.unwrap();
    //     searcher.sort_candidates_selected(self.points.get_points(&searcher.ep));

    //     while let Some((cand2q_dist, candidate)) = searcher.search_candidates.pop_first() {
    //         let (furthest2q_dist, _) = searcher.search_selected.last_key_value().unwrap();

    //         if &cand2q_dist > furthest2q_dist {
    //             break;
    //         }
    //         for (n2q_dist, neighbor_point) in layer
    //             .neighbors(candidate)?
    //             .iter()
    //             .filter(|idx| searcher.search_seen.insert(**idx))
    //             .map(|idx| {
    //                 let (dist, point) = match self.points.get_point(*idx) {
    //                     Some(p) => (p.dist2other(searcher_point), p),
    //                     None => {
    //                         println!(
    //                             "Tried to get node with id {idx} from index, but it doesn't exist"
    //                         );
    //                         panic!("Tried to get a node that doesn't exist.")
    //                     }
    //                 };
    //                 (dist, point)
    //             })
    //         {
    //             let (f2q_dist, _) = searcher.search_selected.last_key_value().unwrap();

    //             // TODO: do this inside the searcher struct
    //             if (&n2q_dist < f2q_dist) | (searcher.search_selected.len() < ef) {
    //                 searcher
    //                     .search_candidates
    //                     .insert(n2q_dist, neighbor_point.id);
    //                 searcher.search_selected.insert(n2q_dist, neighbor_point.id);

    //                 if searcher.search_selected.len() > ef {
    //                     searcher.search_selected.pop_last();
    //                 }
    //             }
    //         }
    //     }
    //     searcher.set_next_ep();
    //     Ok(())
    // }

    /// Mean-centers each vector using each dimension's mean over the entire matrix.
    fn center_vector(&self, vector: &Vec<f32>) -> Result<Vec<f32>, String> {
        let means = self.points.get_means()?;

        Ok(vector
            .iter()
            .enumerate()
            .map(|(idx, x)| x - means[idx])
            .collect())
    }

    /// Saves the index to the specified path in a custom binary format.
    /// Creates the path to the file if it didn't exist before.
    /// SQLite uses big-endian, so I'll stick to that standard.
    pub fn save(&self, index_path: &str) -> std::io::Result<()> {
        let index_path = std::path::Path::new(index_path);
        if !index_path.parent().unwrap().exists() {
            std::fs::create_dir_all(index_path.parent().unwrap())?;
        }
        let file = File::create(index_path)?;
        let mut writer = BufWriter::new(file);

        // We start the file with 34 bytes of header
        let header = self.make_header_bytes();
        writer.write(&header)?;

        // Next, the contents of the layers,
        // each layer is:
        //   - One byte for the layer number
        //   - A u64 for the number of edges
        //   - A stream of bytes containing the edges,
        //   each edge is 20 bytes long, so this stream's
        //   length is divisible by 20.
        for (layer_nb, layer) in self.layers.iter() {
            let (nb_edges, layer_bytes) = layer.to_bytes();
            writer.write(&(*layer_nb as u8).to_be_bytes())?;
            writer.write(&(nb_edges as u64).to_be_bytes())?;
            writer.write(&layer_bytes)?;
        }

        // TODO: Write points content

        writer.flush()?;
        Ok(())
    }

    fn make_header_bytes(&self) -> Vec<u8> {
        let mut header = Vec::with_capacity(34);

        // 0..1
        header.push(self.params.m as u8);
        // 1..2
        header.push(self.layers.len() as u8);

        // 2..10
        for byte in (self.points.dim() as u64).to_be_bytes() {
            header.push(byte);
        }

        // 10..18
        for byte in (self.points.len() as u64).to_be_bytes() {
            header.push(byte)
        }

        // 18..26
        for byte in (self.params.ef_cons as u64).to_be_bytes() {
            header.push(byte)
        }

        // 26..34
        for byte in (self.ep as u64).to_be_bytes() {
            header.push(byte)
        }

        header
    }

    fn read_header_bytes(bytes: Vec<u8>) -> (Params, usize, usize, usize) {
        let m = bytes[0];
        let nb_layers = bytes[1] as usize;

        let mut dim_bytes = [0u8; 8];
        for (idx, byte) in bytes[2..10].iter().enumerate() {
            dim_bytes[idx] = *byte;
        }
        let dim = usize::from_be_bytes(dim_bytes);

        let mut nb_points_bytes = [0u8; 8];
        for (idx, byte) in bytes[10..18].iter().enumerate() {
            nb_points_bytes[idx] = *byte;
        }
        let nb_points = usize::from_be_bytes(dim_bytes);

        let mut ef_cons_bytes = [0u8; 8];
        for (idx, byte) in bytes[18..26].iter().enumerate() {
            ef_cons_bytes[idx] = *byte;
        }
        let ef_cons = usize::from_be_bytes(ef_cons_bytes);

        let mut ep_bytes = [0u8; 8];
        for (idx, byte) in bytes[26..34].iter().enumerate() {
            ep_bytes[idx] = *byte;
        }
        let ep = usize::from_be_bytes(ep_bytes);

        let params = Params::from_m_efcons(m as usize, ef_cons, dim);

        (params, nb_layers, nb_points, ep)
    }

    pub fn from_path(index_path: &str) -> std::io::Result<Self> {
        let index_path = std::path::Path::new(index_path);
        let mut file = File::open(index_path)?;

        // These two lines read the header of the file (first 34 bytes)
        let header: Vec<u8> = file
            .try_clone()
            .unwrap()
            .bytes()
            .take(34)
            .map(|x| x.unwrap())
            .collect();
        let (params, nb_layers, nb_points, ep) = Self::read_header_bytes(header);

        // Move to the start of the layers
        // Read the layer contents
        file.seek(SeekFrom::Start(34))?;
        let mut layers = IntMap::default();
        for _ in 0..nb_layers {
            let file_copy = file.try_clone()?;
            let mut bytes = file_copy.bytes();
            let layer_nb = bytes.next().unwrap()? as usize;
            let mut nb_edges_bytes = [0u8; 8];
            for idx in 0..8 {
                nb_edges_bytes[idx] = bytes.next().unwrap()?;
            }
            let nb_edges = u64::from_be_bytes(nb_edges_bytes);
            let edges_bytes = bytes
                .take((nb_edges * 20) as usize)
                .map(|x| x.unwrap())
                .collect();
            let layer = Graph::from_edge_list_bytes(&edges_bytes);
            layers.insert(layer_nb, layer);
        }

        // TODO: Read points content

        Ok(Self {
            ep,
            params,
            points: Points::Empty,
            layers,
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
            "{msg} {bar:60} {percent}% of {len} Elapsed: {elapsed} | ETA: {eta} | {per_sec}\n",
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

    (-rand_nb.log(std::f32::consts::E) * ml).floor() as usize
}

fn _compute_stats(points: &Points) -> (f32, f32) {
    let mut dists: HashMap<(usize, usize), f32> = HashMap::new();
    for (id, point) in points.iterate() {
        for (idx, pointx) in points.iterate() {
            if id == idx {
                continue;
            }
            dists
                .entry((id.min(idx), id.max(idx)))
                .or_insert(point.dist2vec(&pointx.vector, idx).dist);
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

// fn split_ids(ids: Vec<usize>, nb_splits: usize) -> Vec<Vec<usize>> {
//     let mut split_vector = Vec::new();

//     let per_split = ids.len() / nb_splits;

//     let mut buffer = 0;
//     for idx in 0..nb_splits {
//         if idx == nb_splits - 1 {
//             split_vector.push(ids[buffer..].to_vec());
//         } else {
//             split_vector.push(ids[buffer..(buffer + per_split)].to_vec());
//             buffer += per_split;
//         }
//     }

//     let mut sum_lens = 0;
//     for i in split_vector.iter() {
//         sum_lens += i.len();
//     }

//     assert!(sum_lens == ids.len(), "sum: {sum_lens}");

//     split_vector
// }
