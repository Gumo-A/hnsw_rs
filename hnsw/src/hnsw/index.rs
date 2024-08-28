use super::{
    dist::Dist,
    points::{Point, PointsV2, Vector},
};
use crate::hnsw::params::Params;
// use crate::hnsw::searcher::Searcher;
use crate::hnsw::{graph::Graph, lvq::LVQVec};

use indicatif::{ProgressBar, ProgressStyle};

use nohash_hasher::{IntMap, IntSet};

use parking_lot::{RwLock, RwLockReadGuard};
use std::{sync::Arc, cmp::Reverse, time::Instant};

use rand::Rng;
use rand::{rngs::ThreadRng, seq::SliceRandom};
use rand::thread_rng;

use serde::{Deserialize, Serialize};

use core::panic;
use std::collections::{HashMap, BinaryHeap, BTreeMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Searcher {
    selected: BinaryHeap<Dist>,
    candidates: BinaryHeap<Reverse<Dist>>,
    visited: IntSet<usize>,
    visited_heuristic: BinaryHeap<Reverse<Dist>>,
    insertion_results: IntMap<usize, IntMap<usize, BinaryHeap<Dist>>>,
    prune_results: IntMap<usize, IntMap<usize, BinaryHeap<Dist>>>
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            selected: BinaryHeap::new(),
            candidates: BinaryHeap::new(),
            visited: IntSet::default(),
            visited_heuristic: BinaryHeap::new(),
            insertion_results: IntMap::default(),
            prune_results: IntMap::default()
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HNSW {
    ep: usize,
    pub params: Params,
    pub points: PointsV2,
    pub layers: IntMap<usize, Graph>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> HNSW {
        let params = if ef_cons.is_some() {
            Params::from_m_efcons(m, ef_cons.unwrap(), dim)
        } else {
            Params::from_m(m, dim)
        };
        HNSW {
            points: PointsV2::Empty,
            params,
            ep: 0,
            layers: IntMap::default(),
        }
    }

    pub fn from_params(params: Params) -> HNSW {
        HNSW {
            points: PointsV2::Empty,
            params,
            ep: 0,
            layers: IntMap::default(),
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
                if neighbors.len() > ((max_degree as f32) * 1.1).ceil() as usize {
                    is_ok = false;
                    println!(
                        "layer {layer_nb}, {node} degree = {0}, limit = {1}",
                        neighbors.len(),
                        max_degree
                    );
                }

                if (neighbors.is_empty()) & (layer.nb_nodes() > 1) {
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

    // TODO: add multithreaded functionality for ANN search.
    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<usize>, String> {

        let mut searcher = Searcher::new();

        let point = Point::new_quantized(0, 0, &self.center_vector(vector)?);

        searcher.selected.push(point.dist2other(self.points.get_point(self.ep).unwrap()));
        let nb_layer = self.layers.len();

        for layer_nb in (1..nb_layer).rev() {
            self.search_layer(&mut searcher, self.layers.get(&(layer_nb)).unwrap(), &point, 1)?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        self.search_layer(&mut searcher, layer_0, &point, ef)?;

        let anns: Vec<Dist> = searcher.selected.into_sorted_vec();
        Ok(anns.iter().take(n).map(|x| x.id).collect())
    }

    // TODO: multithreaded
    pub fn anns_by_vectors(&self, _vector: &Vec<Vec<f32>>, _n: usize, _ef: usize) {
        // Result<Vec<Vec<usize>>, String> {
        // todo!("multithreaded");
        // let mut ep: IntSet<usize, > =
        //     IntSet::with_hasher(:));
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
        for layer_nb in (level + 1..max_layer_nb + 1).rev() {
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

    fn step_2(
        &self,
        searcher: &mut Searcher,
        point: &Point,
        level: usize,
    ) -> Result<(), String> {
        // let s1 = Instant::now();
        let bound = (level + 1).min(self.layers.len());
        // println!("s1 {}", s1.elapsed().as_nanos());

        for layer_nb in (0..bound).rev() {
            // let mmax = if layer_nb == 0 {
            //     self.params.mmax0
            // } else {
            //     self.params.mmax
            // };
        //     let s2 = Instant::now();
            let layer = self.layers.get(&layer_nb).unwrap();
        //     println!("s2 {}", s2.elapsed().as_nanos());

        //     let s3 = Instant::now();
            self.search_layer(searcher, layer, point, self.params.ef_cons)?;
        //     println!("s3 {}", s3.elapsed().as_nanos());


        //     let s4 = Instant::now();
            self.select_heuristic(searcher, layer, point, self.params.m, false, true)?;
        //     println!("s4 {}", s4.elapsed().as_nanos());

        //     let s5 = Instant::now();
            let layer_result = searcher.insertion_results.entry(layer_nb).or_insert(IntMap::default());
        //     println!("s5 {}", s5.elapsed().as_nanos());

            // Pruning
        //     let s6 = Instant::now();
            let point_neighbors = searcher.selected.clone();
            // let exceeding_neighbors: Vec<Dist> = searcher.selected.iter().filter(|x| layer.degree(x.id).unwrap() >= mmax).copied().collect();
            // for exc_neigh in exceeding_neighbors {
            //     let worst_neighbor = layer.neighbors(exc_neigh.id)?.iter().max().unwrap();
            //     if *worst_neighbor >= exc_neigh {
            //         // remove worst neighbor from that node
            //         layer_result.insert(exc_neigh.id, BinaryHeap::from_iter(layer.neighbors(exc_neigh.id)?.iter().filter(|dist| dist.id != worst_neighbor.id).copied()));
            //     } else {
            //         // dont connect point.id with that node
            //         point_neighbors = BinaryHeap::from_iter(searcher.selected.iter().filter(|x| x.id != exc_neigh.id).map(|x| *x));
            //     }
            // }
        //     println!("s6 {}", s6.elapsed().as_nanos());

        //     let s7 = Instant::now();
            layer_result.insert(point.id, point_neighbors);
        //     println!("s7 {}", s7.elapsed().as_nanos());
        }
        Ok(())
    }

    fn step_2_layer0(
        index: &RwLockReadGuard<'_, HNSW>,
        searcher: &mut Searcher,
        layer0: &mut Graph,
        point: &Point,
    ) -> Result<(), String> {
        index.search_layer(searcher, layer0, point, index.params.ef_cons)?;

        index.select_heuristic(searcher, layer0, point, index.params.m, false, true)?;
        let layer_result = searcher.insertion_results.entry(0).or_insert(IntMap::default());
        let point_neighbors = searcher.selected.clone();
        layer_result.insert(point.id, point_neighbors);

        // index.write_results(searcher, point.id, 0, 0);

        // layer0.add_node(point.id);
        // layer0.replace_neighbors(point.id, searcher.selected.iter().copied())?;

        // index.prune_connexions(searcher)?;

        // index.write_results_prune(searcher);

        Ok(())
    }

    fn prune_connexions(
        &self,
        searcher: &mut Searcher,
    ) -> Result<(), String>  {
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
                    searcher.clear_searchers();
                    HNSW::select_simple(searcher, layer.neighbors(dist.id)?.iter().copied(), limit)?;
                    let entry = searcher.prune_results.entry(*layer_nb).or_insert(IntMap::default());
                    entry.insert(dist.id, searcher.selected.clone());
                }
            }

        }
        Ok(())
    }
    
    fn prune_connexions_layer(
        searcher: &mut Searcher,
        layer: &Graph,
        mmax: usize,
    ) -> Result<(), String>  {
        searcher.prune_results.clear();

        for (layer_nb, node_neighbors) in searcher.insertion_results.clone().iter() {
            for (_, neighbors) in node_neighbors.iter() {

                let to_prune = neighbors
                        .iter()
                        .filter(|x| layer.degree(x.id).unwrap() > mmax)
                        .map(|x| *x);

                for dist in to_prune {
                    searcher.clear_searchers();
                    HNSW::select_simple(searcher, layer.neighbors(dist.id)?.iter().copied(), mmax)?;
                    let entry = searcher.prune_results.entry(*layer_nb).or_insert(IntMap::default());
                    entry.insert(dist.id, searcher.selected.clone());
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
        searcher.candidates.extend(searcher.selected.iter().map(|dist| Reverse(*dist)));
        searcher.selected.clear();
        // println!("s1 {}", s1.elapsed().as_nanos());

        if extend_cands {
            for dist in searcher.candidates.iter().copied().collect::<Vec<Reverse<Dist>>>() {
                for neighbor_dist in layer.neighbors(dist.0.id)? {
                    searcher.candidates.push(Reverse(point.dist2other(self.points.get_point(neighbor_dist.id).unwrap())));
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
        searcher: &mut Searcher,
        candidate_dists: I,
        m: usize,
    ) -> Result<(), String> where I: Iterator<Item = Dist>{

        searcher.candidates.clear();
        searcher.selected.clear();
        searcher.candidates.extend(candidate_dists.map(|dist| Reverse(dist)));

        while (!searcher.candidates.is_empty()) & (searcher.selected.len() < m) {
            let dist_e = searcher.candidates.pop().unwrap();
            searcher.selected.push(dist_e.0);
        }
        Ok(())
    }
       
    pub fn insert(&mut self, point_id: usize, searcher: &mut Searcher, reinsert: bool) -> Result<bool, String> {

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

        if self.layers.get(&0).unwrap().contains(&point_id) & !reinsert {
            return Ok(true);
        }
        searcher.selected.push(point.dist2other(self.points.get_point(self.ep).unwrap()));

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

    pub fn insert_par(
        index: &Arc<RwLock<Self>>,
        ids_levels: Vec<(usize, usize)>,
        bar: ProgressBar,
    ) -> Result<(), String> {
        let mut searcher = Searcher::new();

        for (_, (point_id, level)) in ids_levels.iter().enumerate() {

            searcher.clear_searchers();

            let read_ref = index.read();
            let max_layer_nb = read_ref.layers.len() - 1;

            let point = read_ref.points.get_point(*point_id).unwrap();
            searcher.selected.push(point.dist2other(read_ref.points.get_point(read_ref.ep).unwrap()));

            match read_ref.step_1(&mut searcher, point, max_layer_nb, *level, None) {
                Ok(()) => (),
                Err(msg) => return Err(format!("Error in step 1: {msg}"))
            };

            match read_ref.step_2(&mut searcher, point, *level) {
                Ok(()) => (),
                Err(msg) => return Err(format!("Error in step 2: {msg}"))
            };

            drop(read_ref);

            // batch += 1;
            // let last_idx = idx == (points_len - 1);
            // let new_layer = *level > max_layer_nb;
            // let full_batch = batch >= batch_size;

            // let have_to_write: bool = last_idx | new_layer | full_batch;

            // let mut write_ref = if have_to_write {
            //     index.write()
            // } else {
            //     continue;
            // };

            index.write().write_results(&searcher, *point_id, *level, max_layer_nb)?;
            index.read().prune_connexions(&mut searcher)?;
            index.write().write_results_prune(&searcher)?;

            // write_ref.write_batch(&mut searcher)?;
            // if new_layer {
            //     for layer_nb in max_layer_nb + 1..level + 1 {
            //         let mut layer = Graph::new();
            //         layer.add_node(*point_id);
            //         write_ref.layers.insert(layer_nb, layer);
            //         write_ref.ep = *point_id;
            //     }
            // }
            searcher.clear_all();

            if !bar.is_hidden() {
                bar.inc(1);
            }
            // batch = 0;
        }
        Ok(())
    }

    pub fn insert_par_v2(
        index: &Arc<RwLock<Self>>,
        ids: Vec<usize>,
        bar: ProgressBar,
    ) -> Result<(), String> {
        let points_len = ids.len();

        let mut thread_layer0 = index.read().layers.get(&0).unwrap().clone();
        let mut searcher = Searcher::new();

        for (idx, point_id) in ids.iter().enumerate() {
           
            // let s0 = Instant::now();
            let read_ref = if index.is_locked_exclusive() {
                let reference = index.read();
                reference.update_thread_layer(&mut thread_layer0)?;
                reference
            } else {
                index.read()
            };
            // println!("s0 {}", s0.elapsed().as_nanos());

            // let s1 = Instant::now();
            searcher.clear_all();
            let point = read_ref.points.get_point(*point_id).unwrap();
            searcher.selected.push(point.dist2other(read_ref.points.get_point(read_ref.ep).unwrap()));
            let level = point.level;
            let max_layer_nb = read_ref.layers.len() - 1;
            // println!("s1 {}", s1.elapsed().as_nanos());

            // let s2 = Instant::now();
            read_ref.step_1(&mut searcher, point, max_layer_nb, level, None)?;
            // println!("s2 {}", s2.elapsed().as_nanos());

            // let s3 = Instant::now();
            HNSW::step_2_layer0(&read_ref, &mut searcher, &mut thread_layer0, point)?;
            // println!("s3 {}", s3.elapsed().as_nanos());

            // let s4 = Instant::now();
            HNSW::write_results_layer(&searcher, &mut thread_layer0)?;
            // println!("s4 {}", s4.elapsed().as_nanos());

            HNSW::prune_connexions_layer(&mut searcher, &thread_layer0, read_ref.params.mmax0)?;

            HNSW::write_prune_layer(&searcher, &mut thread_layer0)?;

            if !bar.is_hidden() {
                bar.inc(1);
            }
            drop(read_ref);

            // add other conditions if you wish to sync more frequently
            let last_idx = idx == (points_len - 1);

            // let s5 = Instant::now();
            if last_idx {
                index.write().update_layer0(&thread_layer0)?;
            } else {
                continue;
            };
            // println!("s5 {}", s5.elapsed().as_nanos());
        }
        Ok(())
    }

    // fn prune_layer(&self, layer: &mut Graph) {
    //     let mut to_prune = IntSet::default();
    //     for (node, neighbors) in layer.nodes.iter() {
    //         if neighbors.len() > self.params.mmax0 {
    //             to_prune.insert(*node);
    //         }
    //     }
    //     let prune_results = self
    //         .prune_connexions(self.params.mmax0, layer, &to_prune)
    //         .unwrap();
    //     for (node, neighbors) in prune_results.iter() {
    //         layer.add_node(*node);
    //         layer.replace_neighbors(*node, neighbors).unwrap();
    //         assert!(neighbors.len() <= self.params.mmax0);
    //     }
    // }

    fn update_thread_layer(&self, thread_layer: &mut Graph) -> Result<(), String> {
        let true_layer0 = self.layers.get(&0).unwrap();

        let thread_layer_nodes = thread_layer
            .nodes
            .keys()
            .cloned()
            .collect::<IntSet<usize>>();
        let true_layer_nodes = true_layer0.nodes.keys().cloned().collect::<IntSet<usize>>();

        let new_nodes: Vec<usize> = true_layer_nodes
            .difference(&thread_layer_nodes).copied()
            .collect();

        for node in new_nodes.iter() {
            thread_layer.add_node(*node);
        }

        for node in new_nodes.iter() {
            let new_neighbors = true_layer0.neighbors(*node).unwrap().iter().copied();
            thread_layer
                .replace_or_add_neighbors(*node, new_neighbors)?;
        }
        Ok(())
    }

    fn update_layer0(&mut self, thread_layer: &Graph) -> Result<(), String> {
        let true_layer0 = self.layers.get_mut(&0).unwrap();

        let thread_layer_nodes = thread_layer
            .nodes
            .keys()
            .copied()
            .collect::<IntSet<usize>>();
        let true_layer_nodes = true_layer0.nodes.keys().copied().collect::<IntSet<usize>>();

        let new_nodes: Vec<usize> = thread_layer_nodes
            .difference(&true_layer_nodes).copied()
            .collect();

        for node in new_nodes.iter() {
            true_layer0.add_node(*node);
        }

        for node in new_nodes.iter() {
            let new_neighbors = thread_layer.neighbors(*node)?.iter().copied();
            true_layer0.replace_or_add_neighbors(*node, new_neighbors)?;
        }
        Ok(())

        // let mut to_prune = IntSet::default();
        // for (node, neighbors) in true_layer0.nodes.iter() {
        //     if neighbors.len() > self.params.mmax0 {
        //         to_prune.insert(*node);
        //     }
        // }

        // let true_layer0 = self.layers.get(&0).unwrap();
        // let prune_results = self
        //     .prune_connexions(self.params.mmax0, true_layer0, &to_prune)
        //     .unwrap();
        // let true_layer0 = self.layers.get_mut(&0).unwrap();

        // for (node, new_neighbors) in prune_results.iter() {
        //     assert!(new_neighbors.len() <= self.params.mmax0);
        //     true_layer0.replace_neighbors(*node, new_neighbors).unwrap();
        // }
        // println!(
        //     "thread {tn} nb of nodes end of update {}",
        //     true_layer0.nb_nodes()
        // );
        // println!("thread {tn} inserted {}", true_layer0.nb_nodes() - start);
    }

    // fn write_batch(
    //     &mut self,
    //     searcher: &mut Searcher,
    // ) -> Result<(), String> {

    //     for (layer_nb, node_data) in searcher.insertion_results.iter() {
    //         let layer = self.layers.get_mut(&layer_nb).unwrap();
    //         for (node, neighbors) in node_data.iter() {
    //             layer.add_node(*node);
    //             layer.replace_neighbors(*node, neighbors.iter().copied())?;
    //         }
    //     }
    //     Ok(())
    // }

    fn write_results(
        &mut self,
        searcher: &Searcher,
        point_id: usize,
        level: usize,
        max_layer_nb: usize,
    ) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.insertion_results.iter() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.add_node(*node);
                // if neighbors.iter().filter(|dist| dist.id == *node).count() != 0 {
                //     println!("{neighbors:?}");
                //     println!("{node}");
                //     std::process::exit(1);
                // }
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
    
    fn write_results_layer(
        searcher: &Searcher,
        layer: &mut Graph
    ) -> Result<(), String> {
        for (_, node_data) in searcher.insertion_results.iter() {
            for (node, neighbors) in node_data.iter() {
                layer.add_node(*node);
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
            }
        }
        Ok(())
    }
    
    fn write_results_prune(
        &mut self,
        searcher: &Searcher,
    ) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.prune_results.iter() {

        //     let s1 = Instant::now();
            let layer = self.layers.get_mut(&layer_nb).unwrap();
        //     println!("s1 {}", s1.elapsed().as_nanos());

            // println!("1 {}", node_data.len());
            for (node, neighbors) in node_data.iter() {
                // println!("2 {}", neighbors.len());
                // println!("3 {}", layer.degree(*node)?);

        //         let s2 = Instant::now();
                layer.add_node(*node);
        //         println!("s2 {}", s2.elapsed().as_nanos());

        //         let s4 = Instant::now();
                // if neighbors.iter().filter(|dist| dist.id == *node).count() != 0 {
                //     println!("{neighbors:?}");
                //     println!("{node}");
                //     std::process::exit(1);
                // }
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
        //         println!("s4 {}", s4.elapsed().as_nanos());
                // println!("4 {}", layer.degree(*node)?);
            }
        }

        Ok(())
    }
    
    fn write_prune_layer(
        searcher: &Searcher,
        layer: &mut Graph
    ) -> Result<(), String> {
        for (_, node_data) in searcher.prune_results.iter() {
            for (node, neighbors) in node_data.iter() {
                layer.add_node(*node);
                layer.replace_or_add_neighbors(*node, neighbors.iter().copied())?;
            }
        }

        Ok(())
    }

    /// Assigns IDs to all vectors (usize).
    /// Creates Point structs, giving a level to each Point.
    /// Stores the Point structs in a Points struct, in index.points
    fn store_points(&mut self, vectors: Vec<Vec<f32>>) {
        let points = PointsV2::from_vecs(vectors, self.params.ml);

        self.points.extend_or_fill(points);
    }

    fn first_insert(&mut self, point_id: usize) {
        let mut layer = Graph::new();
        layer.add_node(point_id);
        self.layers.insert(0, layer);
        self.ep = point_id;
    }

    // fn reinsert_with_degree_zero(&mut self) {
    //     // println!("Reinserting nodes with degree 0");
    //     let mut searcher = Searcher::new();
    //     for _ in 0..3 {
    //         for (_, layer) in self.layers.clone().iter() {
    //             for (node, neighbors) in layer.nodes.iter() {
    //                 if neighbors.is_empty() {
    //                     self.insert(*node, &mut searcher, true).unwrap();
    //                 }
    //             }
    //         }
    //     }
    // }

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

        // let mut searcher = Searcher::new();
        let ids: Vec<usize> = index.points.ids().collect();
        for (idx, id) in ids.iter().enumerate() {
            // index.insert_with_searcher(id, &mut searcher, false)?;
            index.insert(*id, &mut searcher, false)?;
            if !bar.is_hidden() {
                bar.inc(1);
            }
            if idx % 10_000 == 0 {
                // println!("{}", bar.per_sec());
            }
        }
        // index.reinsert_with_degree_zero();
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
        // let nb_threads = 2;
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
            handle.join().unwrap();
        }

        index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    // // TODO: Implementation is faster, but index quality is not good enough
    pub fn build_index_par_v2(
        m: usize,
        ef_cons: Option<usize>,
        vectors: Vec<Vec<f32>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        // let nb_threads = 1;
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
            handlers.push(std::thread::spawn(move || -> Result<(), String> {
                Self::insert_par_v2(&index_copy, ids, bar)?;
                Ok(())
            }));
        }
        for handle in handlers {
            handle.join().unwrap()?;
        }
        // TODO prune connexions of all nodes in layer 0 before ending
        index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    // TODO: this was quickly done and without much thought, try to find a smarter way
    fn partition_points(
        &self,
        mut eps_ids_map: IntMap<usize, IntSet<usize>>,
        nb_splits: usize,
        layer_nb: usize,
    ) -> Vec<IntSet<usize>> {
        let nb_eps = eps_ids_map.keys().count();
        let eps_per_split = nb_eps / nb_splits;
        let mut splits: Vec<IntSet<usize>> =
            Vec::from_iter((0..nb_splits).map(|_| IntSet::default()));
        let mut inserted = IntSet::default();
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
                let _ep_point = self.points.get_point(reference_ep).unwrap();

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

                // let mut sorted_neighbors = self.sort_by_distance(ep_point, ep_neighbors).unwrap();
                let mut sorted_neighbors =
                    BTreeMap::from_iter(ep_neighbors.iter().map(|dist| (*dist, dist.id)));
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
    fn insert_non_zero(mut index: Self, verbose: bool) -> Result<Self, String> {
        let ids_levels: Vec<(usize, usize)> = index.points.ids_levels().filter(|x| x.1 > 0).collect();
        let bar = get_progress_bar(
            "Inserting non-zeros:".to_string(),
            ids_levels.len(),
            verbose,
        );
        let mut searcher = Searcher::new();
        for (id, _) in ids_levels {
            index.insert(id, &mut searcher, false)?;
            if verbose {
                bar.inc(1);
            }
        }

        // index_arc.write().reinsert_with_degree_zero();
        // index_arc.read().assert_param_compliance();

        Ok(index)
    }

    /// Finds the entry points in layer for all points that
    /// have not been inserted.
    ///
    /// Returns a IntMap pointing every entry point in the layer to
    /// the points it inserts.
    fn find_layer_eps(
        index: Self,
        target_layer_nb: usize,
        verbose: bool,
    ) -> Result<(Self, IntMap<usize, IntSet<usize>>), String> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();

        let to_insert = index.points.ids_levels().filter(|x| x.1 == 0).collect();
        let mut to_insert_split = split_ids_levels(to_insert, nb_threads);

        let mut handlers = Vec::new();
        let index_arc = Arc::new(RwLock::new(index));
        for thread_nb in 0..nb_threads {
            let thread_split = to_insert_split.pop().unwrap();
            let index_ref = index_arc.clone();

            handlers.push(std::thread::spawn(
                move || -> IntMap<usize, IntSet<usize>> {
                    let mut thread_results = IntMap::default();
                    let read_ref = index_ref.read();
                    let bar = get_progress_bar(
                        "Finding entry points".to_string(),
                        thread_split.len(),
                        (thread_nb == nb_threads - 1) & verbose,
                    );
                    let mut searcher = Searcher::new();
                    for (id, level) in thread_split {
                        searcher.clear_searchers();
                        let point = read_ref.points.get_point(id).unwrap();
                        searcher.selected.push(point.dist2other(read_ref.points.get_point(read_ref.ep).unwrap()));
                        let max_layer_nb = read_ref.layers.len() - 1;
                        read_ref
                            .step_1(&mut searcher, point, max_layer_nb, level, Some(target_layer_nb))
                            .unwrap();
                        thread_results
                            .entry(searcher.selected.pop().unwrap().id)
                            .and_modify(|e: &mut IntSet<usize>| {
                                e.insert(id);
                            })
                            .or_insert(IntSet::from_iter([id].iter().cloned()));
                        if verbose {
                            bar.inc(1);
                        }
                    }
                    thread_results
                },
            ));
        }

        let mut eps_ids = IntMap::default();
        for handle in handlers {
            let result = handle.join().unwrap();
            for (ep, point_ids) in result.iter() {
                eps_ids
                    .entry(*ep)
                    .and_modify(|e: &mut IntSet<usize>| e.extend(point_ids.iter().cloned()))
                    .or_insert(IntSet::from_iter(point_ids.iter().cloned()));
            }
        }

        let index = Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner();

        Ok((index, eps_ids))
    }
    
    fn get_nearest<I>(&self, point: &Point, others: I) -> Dist
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| {
                point.dist2other(self.points.get_point(idx).unwrap())
            })
            .min()
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
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point,
        ef: usize,
    ) -> Result<(), String> {

        // let s1 = Instant::now();
        searcher.candidates.extend(searcher.selected.iter().map(|x| Reverse(*x)));
        searcher.visited.extend(searcher.selected.iter().map(|dist| dist.id));
        // println!("s1 {}", s1.elapsed().as_nanos());

        // let s2 = Instant::now();
        while !searcher.candidates.is_empty() {

            // let s2 = Instant::now();
            let cand_dist = searcher.candidates.pop().unwrap();
            let furthest2q_dist = searcher.selected.peek().unwrap();
            if cand_dist.0 > *furthest2q_dist {
                break;
            }
            // println!("s2 {}", s2.elapsed().as_nanos());

            // let s3 = Instant::now();
            let cand_neighbors = match layer.neighbors(cand_dist.0.id) {
                Ok(neighs) => neighs,
                Err(msg) => return Err(format!("Error in search_layer: {msg}"))
            };

            // pre-compute distances to candidate neighbors to take advantage of
            // caches and to prevent the re-construction of the query to a full vector
            let q2cand_neighbors_dists = point.dist2others(cand_neighbors
                .iter()
                .filter(|dist| searcher.visited.insert(dist.id))
                .map(|dist| {
                    match self.points.get_point(dist.id) {
                        Some(p) => p,
                        None => panic!("nope!")
                    }
            }));
            // println!("s3 {}", s3.elapsed().as_nanos());

            // let s4 = Instant::now();
            for n2q_dist in q2cand_neighbors_dists
            {
                // let s2 = Instant::now();
                // let (n2q_dist, _) = res?;
                let f2q_dist = searcher.selected.peek().unwrap();
                // println!("s2 {}", s2.elapsed().as_nanos());

                // let s3 = Instant::now();
                if (n2q_dist < *f2q_dist) | (searcher.selected.len() < ef) {
                    searcher.selected.push(n2q_dist);
                    searcher.candidates.push(Reverse(n2q_dist));

                    if searcher.selected.len() > ef {
                        searcher.selected.pop();
                    }
                }
                // println!("s3 {}", s3.elapsed().as_nanos());
            }
            // println!("s4 {}", s4.elapsed().as_nanos());
        }
        // println!("s2 {}", s2.elapsed().as_nanos());

        // let s5 = Instant::now();
        searcher.candidates.clear();
        searcher.visited.clear();
        // println!("s5 {}", s5.elapsed().as_nanos());

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
            serde_json::Value::Object(params_map) => extract_params(params_map),
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
        let mut layers = IntMap::default();
        for (layer_nb, layer_content) in layers_unparsed {
            let layer_nb: usize = layer_nb
                .parse()
                .expect("Error: could not load key {key} into layer number");
            let layer_content = layer_content
                .get("nodes")
                .expect("Error: could not load 'key' nodes for layer {key}").as_object().expect("Error: expected key 'nodes' for layer {layer_nb} to be an Object, but couldl not be parsed as such.");
            let mut this_layer = IntMap::default();
            for (node_id, neighbors) in layer_content.iter() {
                let neighbors = IntMap::from_iter(neighbors
                    .as_array()
                    .expect("Error: could not load the neighbors of node {node_id} in layer {layer_nb} as an Array.")
                    .iter()
                    .map(|neighbor| {
                        let id_dist = neighbor.as_array().unwrap(); 
                        let id = id_dist.first().unwrap().as_u64().unwrap() as usize; 
                        let dist = id_dist.get(1).unwrap().as_f64().unwrap() as f32;
                        (id, Dist { dist, id })
                    }));
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
