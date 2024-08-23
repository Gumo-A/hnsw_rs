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
use std::{sync::Arc, cmp::Reverse};

use rand::Rng;
use rand::{rngs::ThreadRng, seq::SliceRandom};
use rand::thread_rng;

use serde::{Deserialize, Serialize};

use core::panic;
use std::collections::{BTreeMap, HashMap, BinaryHeap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HNSW {
    ep: usize,
    pub params: Params,
    pub points: PointsV2,
    pub layers: IntMap<usize, Graph>,
}

impl HNSW {
    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> HNSW {
        let params = Params::from_m_efcons(m, ef_cons.unwrap_or(2 * m), dim);
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
                if neighbors.len() > (max_degree + 1) {
                    is_ok = false;
                    println!(
                        "In layer {layer_nb}, node {node} has degree {0}, but limit is {1}",
                        neighbors.len(),
                        max_degree
                    );
                }

                if (neighbors.is_empty()) & (layer.nb_nodes() > 1) {
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

        // let vector = self.center_vector(vector)?;

        let point = Point::new_quantized(0, 0, vector);

        let mut ep: BinaryHeap<Dist> = BinaryHeap::from([point.dist2other(self.points.get_point(self.ep).unwrap())]);
        let nb_layer = self.layers.len();

        for layer_nb in (0..nb_layer).rev() {
            ep = self.search_layer(self.layers.get(&(layer_nb)).unwrap(), &point, &mut ep, 1)?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        let nearest_neighbors = self.search_layer(layer_0, &point, &mut ep, ef)?;

        // let nearest_neighbors: BTreeMap<Dist, usize> =
        //     BTreeMap::from_iter(neighbors.iter().map(|x| {
        //         let dist = self.points.get_point(*x).unwrap().dist2other(&point);
        //         (dist, *x)
        //     }));

        let mut anns: Vec<Dist> = nearest_neighbors.iter().take(n).copied().collect();
        anns.sort();
        Ok(anns.iter().map(|x| x.id).collect())
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
        point: &Point,
        max_layer_nb: usize,
        level: usize,
        stop_at_layer: Option<usize>,
    ) -> Result<BinaryHeap<Dist>, String> {
        // let mut ep = IntSet::default();
        // ep.insert(self.ep);

        let mut ep: BinaryHeap<Dist> = BinaryHeap::from([point.dist2other(self.points.get_point(self.ep).unwrap())]);
        let nb_layer = self.layers.len();

        ep = match stop_at_layer {
            None => {
                for layer_nb in (level + 1..max_layer_nb + 1).rev() {
                    let layer = match self.layers.get(&layer_nb) {
                        Some(l) => l,
                        None => return Err(format!("Could not get layer {layer_nb} in step 1.")),
                    };
                    ep = self.search_layer(layer, point, &mut ep, 1)?;
                }
                ep
            }
            Some(target_layer) => {
                for layer_nb in (level + 1..max_layer_nb + 1).rev() {
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

    // fn step_1_s(
    //     &self,
    //     searcher: &mut Searcher,
    //     max_layer_nb: usize,
    //     level: usize,
    // ) -> Result<(), String> {
    //     for layer_nb in (level + 1..max_layer_nb + 1).rev() {
    //         let layer = match self.layers.get(&layer_nb) {
    //             Some(l) => l,
    //             None => return Err(format!("Could not get layer {layer_nb} in step 1.")),
    //         };
    //         self.search_layer_s(layer, searcher, 1)?;
    //     }
    //     Ok(())
    // }

    fn step_2(
        &self,
        point: &Point,
        mut ep: BinaryHeap<Dist>,
        current_layer_number: usize,
    ) -> Result<IntMap<usize, IntMap<usize, IntMap<usize, Dist>>>, String> {
        // let s1 = Instant::now();
        let mut insertion_results = IntMap::default();
        let bound = (current_layer_number + 1).min(self.layers.len());
        // println!("s1 {}", s1.elapsed().as_nanos());

        for layer_nb in (0..bound).rev() {
            // let s2 = Instant::now();
            let layer = self.layers.get(&layer_nb).unwrap();
            // println!("s2 {}", s2.elapsed().as_nanos());

            // let s3 = Instant::now();
            ep = self.search_layer(layer, point, &mut ep, self.params.ef_cons)?;
            // println!("s3 {}", s3.elapsed().as_nanos());

            // let s4 = Instant::now();
            let neighbors_to_connect =
                self.select_heuristic(layer, point, &mut ep, self.params.m, false, true)?;
            // println!("s4 {}", s4.elapsed().as_nanos());

            // let s5 = Instant::now();
            let mut layer_result = IntMap::default();
            layer_result.insert(point.id, neighbors_to_connect);
            insertion_results.insert(layer_nb, layer_result);
            // println!("s5 {}", s5.elapsed().as_nanos());
        }
        Ok(insertion_results)
    }

    // fn step_2_s(
    //     &self,
    //     searcher: &mut Searcher,
    //     level: usize,
    // ) -> Result<IntMap<usize, IntMap<usize, IntSet<usize>>>, String> {
    //     let mut insertion_results = IntMap::default();
    //     let bound = (level + 1).min(self.layers.len());

    //     for layer_nb in (0..bound).rev() {
    //         let layer = self.layers.get(&layer_nb).unwrap();

    //         self.search_layer_s(layer, searcher, self.params.ef_cons)?;

    //         self.select_heuristic_s(layer, searcher, self.params.m, false, true)?;

    //         let mut layer_result = IntMap::default();
    //         layer_result.insert(
    //             searcher.point.unwrap().id,
    //             searcher.heuristic_selected.values().cloned().collect(),
    //         );
    //         insertion_results.insert(layer_nb, layer_result);
    //     }
    //     Ok(insertion_results)
    // }

    fn step_2_layer0(
        index: &RwLockReadGuard<'_, HNSW>,
        layer0: &mut Graph,
        point: &Point,
        mut ep: BinaryHeap<Dist>,
    ) -> Result<(), String> {
        ep = index.search_layer(layer0, point, &mut ep, index.params.ef_cons)?;

        let neighbors_to_connect =
            index.select_heuristic(layer0, point, &mut ep, index.params.m, false, true)?;

        layer0.add_node(point.id);
        layer0.replace_neighbors(point.id, &neighbors_to_connect)?;

        let prune_results = index.prune_connexions(
            index.params.mmax0,
            layer0,
            &neighbors_to_connect.keys().copied().collect(),
        )?;

        for (node_id, neighbors) in prune_results {
            assert!(neighbors.len() <= index.params.mmax0);
            layer0.replace_neighbors(node_id, &neighbors)?;
        }

        Ok(())
    }

    fn prune_connexions(
        &self,
        limit: usize,
        layer: &Graph,
        nodes_to_prune: &IntSet<usize>,
    ) -> Result<IntMap<usize, IntMap<usize, Dist>>, String> {
        let mut prune_results = IntMap::default();
        for node in nodes_to_prune.iter() {
            if layer.degree(*node)? > limit {
                let point = &self.points.get_point(*node).unwrap();
                let mut old_neighbors =
                    BinaryHeap::from_iter(layer.neighbors(*node)?.iter().map(|(_, dist)| *dist));
                // old_neighbors.extend(layer.neighbors(*node)?.iter().cloned());
                let new_neighbors =
                    self.select_heuristic(layer, point, &mut old_neighbors, limit, false, false)?;
                assert!(new_neighbors.len() <= limit);
                prune_results.insert(*node, new_neighbors);
            }
        }
        Ok(prune_results)
    }

    fn select_heuristic(
        &self,
        layer: &Graph,
        point: &Point,
        ep: &mut BinaryHeap<Dist>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<IntMap<usize, Dist>, String> {
        let mut candidates = BinaryHeap::from_iter(ep.iter().map(|x| Reverse(*x)));
        if extend_cands {
            for dist in candidates.iter().copied().collect::<Vec<Reverse<Dist>>>() {
                for neighbor in layer.neighbors(dist.0.id)?.keys() {
                    candidates.push(Reverse(point.dist2other(self.points.get_point(*neighbor).unwrap())));
                }
            }
        }
        // let mut candidates = self.sort_by_distance(point, cands)?;
        let mut visited = BinaryHeap::new();
        let mut selected = BinaryHeap::new();

        let dist_e = candidates.pop().unwrap();
        selected.push(dist_e.0);
        while (!candidates.is_empty()) & (selected.len() < m) {
            let dist_e = candidates.pop().unwrap();
            let e_point = &self.points.get_point(dist_e.0.id).unwrap();

            let dist_from_s = self.get_nearest(e_point, selected.iter().map(|x| x.id));

            if dist_e.0 < dist_from_s {
                selected.push(dist_e.0);
            } else {
                visited.push(dist_e);
            }

            if keep_pruned {
                while (!visited.is_empty()) & (selected.len() < m) {
                    let dist_e = visited.pop().unwrap();
                    selected.push(dist_e.0);
                }
            }
        }
        let result = IntMap::from_iter(selected.iter().take(m).map(|dist| (dist.id, *dist)));
        // for val in selected.values().take(m) {
        //     result.insert(*val);
        // }
        Ok(result)
    }

    // fn select_heuristic_s(
    //     &self,
    //     layer: &Graph,
    //     searcher: &mut Searcher,
    //     m: usize,
    //     extend_cands: bool,
    //     keep_pruned: bool,
    // ) -> Result<(), String> {
    //     let searcher_point = searcher.point.unwrap();
    //     searcher.init_heuristic();

    //     if extend_cands {
    //         for (_, idx) in searcher.search_selected.iter() {
    //             for neighbor in layer.neighbors(*idx)? {
    //                 let neighbor_point = self.points.get_point(*neighbor).unwrap();
    //                 searcher
    //                     .heuristic_candidates
    //                     .insert(searcher_point.dist2other(neighbor_point), *neighbor);
    //             }
    //         }
    //     }

    //     let (dist_e, e) = searcher.heuristic_candidates.pop_first().unwrap();
    //     searcher.heuristic_selected.insert(dist_e, e);
    //     while (searcher.search_candidates.len() > 0) & (searcher.heuristic_selected.len() < m) {
    //         let (dist_e, e) = searcher.heuristic_candidates.pop_first().unwrap();
    //         let e_point = &self.points.get_point(e).unwrap();

    //         let (dist_from_s, _) =
    //             self.get_nearest(&e_point, searcher.heuristic_selected.values().cloned());

    //         if dist_e < dist_from_s {
    //             searcher.heuristic_selected.insert(dist_e, e);
    //         } else {
    //             searcher.heuristic_visited.insert(dist_e, e);
    //         }

    //         if keep_pruned {
    //             while (searcher.heuristic_visited.len() > 0)
    //                 & (searcher.heuristic_selected.len() < m)
    //             {
    //                 let (dist_e, e) = searcher.heuristic_visited.pop_first().unwrap();
    //                 searcher.search_selected.insert(dist_e, e);
    //             }
    //         }
    //     }
    //     Ok(())
    // }

    pub fn insert(&mut self, point_id: usize, reinsert: bool) -> Result<bool, String> {
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

        let level = point.level;
        let max_layer_nb = self.layers.len() - 1;
        let ep = self.step_1(point, max_layer_nb, level, None)?;
        let insertion_results = self.step_2(point, ep, level)?;

        let nodes_to_prune = insertion_results.clone();

        self.write_results(insertion_results, point_id, level, max_layer_nb)?;

        let mut pruned_results = IntMap::default();
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
                let mut to_prune = IntSet::default();
                to_prune.extend(
                    neighbors
                        .iter()
                        .filter(|(node_id, _)| layer.degree(**node_id).unwrap() > limit)
                        .map(|(node_id, _)| *node_id),
                );

                let new_nodes = self.prune_connexions(limit, layer, &to_prune)?;
                pruned_results.insert(*layer_nb, new_nodes);
            }
        }

        self.write_results(pruned_results, point_id, level, max_layer_nb)?;

        Ok(true)
    }

    // pub fn insert_with_searcher<'i, 's>(
    //     &'i mut self,
    //     point_id: usize,
    //     searcher: &mut Searcher<'s>,
    //     reinsert: bool,
    // ) -> Result<bool, String>
    // where
    //     'i: 's,
    // {
    //     todo!();
    //     if self.layers.len() == 0 {
    //         self.first_insert(point_id);
    //         return Ok(true);
    //     }

    //     let point = match self.points.get_point(point_id) {
    //         Some(p) => p,
    //         None => return Err(format!("{point_id} not in points given to the index.")),
    //     };

    //     if self.layers.get(&0).unwrap().contains(&point_id) & !reinsert {
    //         return Ok(true);
    //     }

    //     // searcher will store everything so we dont have to create values on the fly
    //     searcher.init(point, self.ep);

    //     let level = point.level;
    //     let max_layer_nb = self.layers.len() - 1;
    //     self.step_1_s(searcher, max_layer_nb, level)?;
    //     let insertion_results = self.step_2_s(searcher, level)?;
    //     let nodes_to_prune = insertion_results.clone();

    //     self.write_results(insertion_results, point_id, level, max_layer_nb)?;

    //     let mut pruned_results = IntMap::default();
    //     for (layer_nb, layer) in self.layers.iter() {
    //         let limit = if *layer_nb == 0 {
    //             self.params.mmax0
    //         } else {
    //             self.params.mmax
    //         };
    //         if nodes_to_prune.contains_key(layer_nb) {
    //             let neighbors = nodes_to_prune
    //                 .get(layer_nb)
    //                 .unwrap()
    //                 .get(&point_id)
    //                 .unwrap();
    //             let mut to_prune = IntSet::default();
    //             to_prune.extend(
    //                 neighbors
    //                     .iter()
    //                     .filter(|node_id| layer.degree(**node_id).unwrap() > limit)
    //                     .map(|node_id| *node_id),
    //             );

    //             let new_nodes = self.prune_connexions(limit, layer, &to_prune)?;
    //             pruned_results.insert(*layer_nb, new_nodes);
    //         }
    //     }

    //     self.write_results(pruned_results, point_id, level, max_layer_nb)?;

    //     Ok(true)
    // }

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
        let batch_size = 16;
        let points_len = ids_levels.len();
        let mut batch = Vec::with_capacity(batch_size);

        for (idx, (point_id, level)) in ids_levels.iter().enumerate() {
            let read_ref = index.read();

            let point = read_ref.points.get_point(*point_id).unwrap();
            let max_layer_nb = read_ref.layers.len() - 1;
            let ep = read_ref.step_1(point, max_layer_nb, *level, None)?;
            let insertion_results = read_ref.step_2(point, ep, *level)?;

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
            let ep = read_ref.step_1(point, max_layer_nb, level, None)?;
            HNSW::step_2_layer0(&read_ref, &mut layer0, point, ep)?;
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
        let mut to_prune = IntSet::default();
        for (node, neighbors) in layer.nodes.iter() {
            if neighbors.len() > self.params.mmax0 {
                to_prune.insert(*node);
            }
        }
        let prune_results = self
            .prune_connexions(self.params.mmax0, layer, &to_prune)
            .unwrap();
        for (node, neighbors) in prune_results.iter() {
            layer.add_node(*node);
            layer.replace_neighbors(*node, neighbors).unwrap();
            assert!(neighbors.len() <= self.params.mmax0);
        }
    }

    fn update_thread_layer(&self, thread_layer: &mut Graph) {
        let true_layer0 = self.layers.get(&0).unwrap();
        let layer0_nodes = thread_layer
            .nodes
            .keys()
            .cloned()
            .collect::<IntSet<usize>>();
        let true_layer_nodes = true_layer0.nodes.keys().cloned().collect::<IntSet<usize>>();
        let new_nodes: Vec<usize> = true_layer_nodes
            .difference(&layer0_nodes).copied()
            .collect();
        for node in new_nodes.iter() {
            let new_neighbors = true_layer0.neighbors(*node).unwrap();
            thread_layer.add_node(*node);
            thread_layer
                .replace_neighbors(*node, new_neighbors)
                .unwrap();
        }

        self.prune_layer(thread_layer);
    }

    fn update_layer0(&mut self, thread_layer: &Graph) {
        let true_layer0 = self.layers.get_mut(&0).unwrap();
        let layer0_nodes = thread_layer
            .nodes
            .keys()
            .cloned()
            .collect::<IntSet<usize>>();
        let true_layer_nodes = true_layer0.nodes.keys().cloned().collect::<IntSet<usize>>();
        let new_nodes: Vec<usize> = layer0_nodes
            .difference(&true_layer_nodes).copied()
            .collect();
        for node in new_nodes.iter() {
            let new_neighbors = thread_layer.neighbors(*node).unwrap();
            true_layer0.add_node(*node);
            true_layer0.replace_neighbors(*node, new_neighbors).unwrap();
        }

        let mut to_prune = IntSet::default();
        for (node, neighbors) in true_layer0.nodes.iter() {
            if neighbors.len() > self.params.mmax0 {
                to_prune.insert(*node);
            }
        }

        let true_layer0 = self.layers.get(&0).unwrap();
        let prune_results = self
            .prune_connexions(self.params.mmax0, true_layer0, &to_prune)
            .unwrap();
        let true_layer0 = self.layers.get_mut(&0).unwrap();

        for (node, new_neighbors) in prune_results.iter() {
            assert!(new_neighbors.len() <= self.params.mmax0);
            true_layer0.replace_neighbors(*node, new_neighbors).unwrap();
        }
        // println!(
        //     "thread {tn} nb of nodes end of update {}",
        //     true_layer0.nb_nodes()
        // );
        // println!("thread {tn} inserted {}", true_layer0.nb_nodes() - start);
    }

    fn write_batch(
        &mut self,
        batch: &mut Vec<IntMap<usize, IntMap<usize, IntMap<usize, Dist>>>>,
    ) -> Result<(), String> {
        let batch_len = batch.len();
        for _ in 0..batch_len {
            let batch_data = batch.pop().unwrap();
            for (layer_nb, node_data) in batch_data.iter() {
                let layer = self.layers.get_mut(layer_nb).unwrap();
                for (node, neighbors) in node_data.iter() {
                    layer.add_node(*node);
                    for (old_neighbor, _dist) in layer.neighbors(*node)?.clone() {
                        layer.remove_edge(*node, old_neighbor)?;
                    }
                    for (neighbor, dist) in neighbors.iter() {
                        layer.add_edge(*node, *neighbor, *dist)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_results(
        &mut self,
        mut insertion_results: IntMap<usize, IntMap<usize, IntMap<usize, Dist>>>,
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
                layer.replace_neighbors(node, &neighbors)?;
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
        // println!("Reinserting nodes with degree 0");
        for _ in 0..3 {
            for (_, layer) in self.layers.clone().iter() {
                for (node, neighbors) in layer.nodes.iter() {
                    if neighbors.is_empty() {
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

        // let mut searcher = Searcher::new();
        let ids: Vec<usize> = index.points.ids().collect();
        for (_idx, id) in ids.iter().enumerate() {
            // index.insert_with_searcher(id, &mut searcher, false)?;
            index.insert(*id, false)?;
            bar.inc(1);
        }
        // index.reinsert_with_degree_zero();
        // index.assert_param_compliance();
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
            handle.join().unwrap();
        }

        // index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
    }

    // TODO: Implementation is faster, but index quality is not good enough
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
            handle.join().unwrap();
        }
        // TODO prune connexions of all nodes in all layers before ending
        // index_arc.read().assert_param_compliance();

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
                    BTreeMap::from_iter(ep_neighbors.iter().map(|(id, dist)| (*dist, *id)));
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
            handle.join().unwrap();
        }

        index_arc.write().reinsert_with_degree_zero();
        // index_arc.read().assert_param_compliance();

        Ok(Arc::into_inner(index_arc)
            .expect("Could not get index out of Arc reference")
            .into_inner())
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
                    for (id, level) in thread_split {
                        let point = read_ref.points.get_point(id).unwrap();
                        let max_layer_nb = read_ref.layers.len() - 1;
                        let ep = read_ref
                            .step_1(point, max_layer_nb, level, Some(target_layer_nb))
                            .unwrap();
                        thread_results
                            .entry(read_ref.get_nearest(point, ep.iter().map(|x| x.id)).id)
                            .and_modify(|e: &mut IntSet<usize>| {
                                e.insert(id);
                            })
                            .or_insert(IntSet::from_iter([id].iter().cloned()));
                        bar.inc(1);
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

    fn sort_by_distance(
        &self,
        point: &Point,
        others: &IntSet<usize>,
    ) -> Result<BTreeMap<Dist, usize>, String> {
        let result = others.iter().map(|idx| {
            // println!("1");
            let dist = self.points.get_point(*idx).unwrap().dist2other(point);
            (dist, *idx)
        });
        Ok(BTreeMap::from_iter(result))
    }

    fn get_nearest<I>(&self, point: &Point, others: I) -> Dist
    where
        I: Iterator<Item = usize>,
    {
        others
            .map(|idx| {
                // println!("1");
                self.points.get_point(idx).unwrap().dist2other(point)
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
        layer: &Graph,
        point: &Point,
        ep: &BinaryHeap<Dist>,
        ef: usize,
    ) -> Result<BinaryHeap<Dist>, String> {
        // let s1 = Instant::now();
        // let mut candidates = self.sort_by_distance(point, ep)?;
        // let mut selected = candidates.clone();
        let mut selected = ep.clone();
        let mut candidates: BinaryHeap<Reverse<Dist>> = BinaryHeap::from_iter(ep.iter().map(|x| Reverse(*x)));
        let mut seen = IntSet::from_iter(ep.iter().map(|dist| dist.id));
        // println!("s1 {}", s1.elapsed().as_nanos());

        while let Some(cand_dist) = candidates.pop() {
            // let s2 = Instant::now();
            let furthest2q_dist = selected.peek().unwrap();

            if cand_dist.0 > *furthest2q_dist {
                break;
            }
            // if ((cand2q_dist.dist * 10000.0) as usize) % 9 == 0 {
            //     println!("cand2q_dist {cand2q_dist}");
            //     println!("{:#?}", layer.neighbors(candidate)?);
            //     println!();
            // }           
            // println!("s2 {}", s2.elapsed().as_nanos());
            // let s3 = Instant::now();
            for (n2q_dist, _) in layer
                .neighbors(cand_dist.0.id)?
                .iter()
                .filter(|(idx, _)| seen.insert(**idx))
                // .filter(|(_, cand2neigh_dist)| { 
                //     let skip = !(cand2neigh_dist.dist > (2.0 * cand2q_dist.dist)) | !((cand2neigh_dist.dist - cand2q_dist.dist).abs() > furthest2q_dist.dist); 
                //     if !skip {
                //         println!("skipped thanks to filter");
                //     }  
                //     skip 
                // }) // Triangle inequality filter
                .map(|(idx, _)| {
                    // let s1 = Instant::now();
                    // println!("1");
                    let (dist, point) = match self.points.get_point(*idx) {
                        Some(p) => (point.dist2other(p), p),
                        None => {
                            println!(
                                "Tried to get node with id {idx} from index, but it doesn't exist"
                            );
                            panic!("Tried to get a node that doesn't exist.")
                        }
                    };
                    // println!("s1 {}", s1.elapsed().as_nanos());
                    (dist, point)
                })
            {
                // println!("0");
                // let s2 = Instant::now();
                let f2q_dist = selected.peek().unwrap();
                // println!("s2 {}", s2.elapsed().as_nanos());

                // let s3 = Instant::now();
                if (n2q_dist < *f2q_dist) | (selected.len() < ef) {
                    selected.push(n2q_dist);
                    candidates.push(Reverse(n2q_dist));

                    if selected.len() > ef {
                        selected.pop();
                    }
                }
                // println!("s3 {}", s3.elapsed().as_nanos());
            }
            // println!("s3 {}", s3.elapsed().as_nanos());
        }
        // let s4 = Instant::now();
        // let mut result = IntSet::default();
        // result.extend(selected.values());
        // println!("s4 {}", s4.elapsed().as_nanos());
        // Ok(result)
        Ok(selected)
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
