// use crate::helpers::bench::Bencher;
use super::{
    dist::Dist,
    kmeans::{get_frontier_points, kmeans},
    points::{Points, Vector},
};
use crate::hnsw::params::Params;
use crate::hnsw::points::Point;
use crate::hnsw::{graph::Graph, lvq::LVQVec};
use crate::{
    helpers::data::{split, split_eps},
    hnsw::kmeans::partition_space,
};

use indicatif::{ProgressBar, ProgressStyle};
use nohash_hasher::BuildNoHashHasher;
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

const PREQUANTIZE: bool = true;

#[derive(Debug, Serialize, Deserialize, Clone)]
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

    // TODO: add multithreaded functionality for ANN search.
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
        let mut candidates = self.sort_by_distance(vector, &cands_idx);
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

    fn reinsert(&mut self, id: usize, level: usize) -> bool {
        if !self.points.contains(&id) {
            return false;
        }

        let point = self.points.get_point(id);
        let max_layer_nb = self.layers.len() - 1;

        let ep = self.step_1(point, max_layer_nb, level);

        let insertion_results = self.step_2(point, ep, level);

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
        true
    }

    pub fn insert_par(
        index: &Arc<RwLock<Self>>,
        mut points: Vec<Point>,
        levels: Vec<usize>,
        bar: ProgressBar,
    ) {
        let batch_size = 16;
        let points_len = points.len();
        let mut batch = Vec::new();

        for (idx, level) in (0..points_len).zip(levels) {
            let point = points.pop().unwrap();
            let point_id = point.id;
            let read_ref = index.read();
            if read_ref.points.contains(&point.id) {
                continue;
            }
            let max_layer_nb = read_ref.layers.len() - 1;

            let ep = read_ref.step_1(&point, max_layer_nb, level);
            let insertion_results = read_ref.step_2(&point, ep, level);
            batch.push((point, insertion_results));

            let last_idx = idx == (points_len - 1);
            let new_layer = level > max_layer_nb;
            let full_batch = batch.len() >= batch_size;
            let have_to_write: bool = last_idx | new_layer | full_batch;

            let mut write_ref = if have_to_write {
                drop(read_ref);
                index.write()
            } else {
                continue;
            };
            if new_layer {
                for layer_nb in max_layer_nb + 1..level + 1 {
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

    pub fn insert_par_2(
        index: Arc<RwLock<Self>>,
        mut points: Vec<(usize, Point)>,
        bar: ProgressBar,
    ) {
        let batch_size = 16;
        let points_len = points.len();
        let mut batch = Vec::new();

        for idx in 0..points_len {
            let (ep, point) = points.pop().unwrap();
            let mut entry = HashSet::with_hasher(BuildNoHashHasher::default());
            entry.insert(ep);
            if index.read().points.contains(&point.id) {
                continue;
            }

            let insertion_results = index.write().step_2(&point, entry, 0);
            batch.push((point, insertion_results));

            let last_idx = idx == (points_len - 1);
            let full_batch = batch.len() >= batch_size;
            let have_to_write: bool = last_idx | full_batch;

            if !have_to_write {
                continue;
            }
            let batch_len = batch.len() as u64;
            index.write().write_batch(&mut batch);
            if !bar.is_hidden() {
                bar.inc(batch_len);
            }
            batch.clear();
        }
    }

    pub fn insert_par_3(&mut self, mut points: Vec<(usize, Point)>, bar: ProgressBar) {
        let batch_size = 16;
        let points_len = points.len();
        let mut batch = Vec::new();

        for idx in 0..points_len {
            let (ep, point) = points.pop().unwrap();
            let mut entry = HashSet::with_hasher(BuildNoHashHasher::default());
            entry.insert(ep);
            if self.points.contains(&point.id) {
                continue;
            }

            let insertion_results = self.step_2(&point, entry, 0);
            bar.inc(1);
            batch.push((point, insertion_results));

            let last_idx = idx == (points_len - 1);
            let full_batch = batch.len() >= batch_size;
            let have_to_write: bool = last_idx | full_batch;

            if !have_to_write {
                continue;
            }
            self.write_batch(&mut batch);
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

    fn make_points(&mut self, mut vectors: Vec<Vec<f32>>) -> (Points, Vec<usize>) {
        let mut points_idx: Vec<usize> = (0..vectors.len()).collect();
        points_idx.shuffle(&mut thread_rng());

        center_vectors(&mut vectors);
        // let points = points_idx
        //     .iter()
        //     .map(|idx| Point::new(*idx, vectors[*idx].clone(), PREQUANTIZE))
        //     .collect();

        let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
        collection.extend(
            vectors
                .iter()
                .enumerate()
                .map(|(id, x)| (id, Point::new(id, x.clone(), PREQUANTIZE))),
        );

        // From 1 and not 0 because first insert is always in layer 0.
        let mut levels: Vec<usize> = (1..vectors.len())
            .map(|_| get_new_node_layer(self.params.ml))
            .collect();
        levels.sort();
        levels.reverse();

        (Points::Collection(collection), levels)
    }

    fn make_vec_points(&self, mut vectors: Vec<Vec<f32>>) -> (Vec<Point>, Vec<usize>) {
        let mut points_idx: Vec<usize> = (0..vectors.len()).collect();
        points_idx.shuffle(&mut thread_rng());

        center_vectors(&mut vectors);
        let points = points_idx
            .iter()
            .map(|idx| Point::new(*idx, vectors[*idx].clone(), PREQUANTIZE))
            .collect();

        // From 1 and not 0 because first insert is always in layer 0.
        let mut levels: Vec<usize> = (1..vectors.len())
            .map(|_| get_new_node_layer(self.params.ml))
            .collect();
        levels.sort();
        levels.reverse();

        (points, levels)
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
        let (mut points, levels) = self.make_vec_points(vectors);

        assert_eq!(self.points.len(), 0);
        assert_eq!(self.layers.len(), 0);

        self.first_insert(points.pop().unwrap());

        let bar = get_progress_bar("Inserting Vectors", lim, false);
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

    // TODO: Try to be smarter about parallelized insertion
    //       There is probably some things I can take advantage of thanks to the
    //       fact that I have all the vectors in advance.
    pub fn build_index_par_001(m: usize, vectors: Vec<Vec<f32>>) -> Self {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        let dim = vectors[0].len();

        let mut index = HNSW::new(m, None, dim);
        let (mut points, levels) = index.make_vec_points(vectors);

        let non_zero: Vec<usize> = levels.iter().take_while(|x| **x > 0).map(|x| *x).collect();
        let bar = get_progress_bar("Inserting non-zero layer points", non_zero.len(), false);

        index.first_insert(points.pop().unwrap());
        for level in non_zero {
            index.insert(points.pop().unwrap(), level);
            bar.inc(1);
        }

        let eps_points = index.determine_eps(points.clone());

        let mut points_map = HashMap::new();
        for point in points.iter() {
            points_map.insert(point.id, point.clone());
        }

        let (mean, std) = compute_stats(&index.points);

        println!("Distances stats:");
        println!("Mean {mean}, Std {std}");

        let mut entry_points: HashSet<&usize> = eps_points.keys().collect();

        let mut rng = rand::thread_rng();
        let min_dist = 0.0;
        let bar = get_progress_bar("Inserting Vectors", points_map.len(), false);

        while points_map.len() > 0 {
            let mut eps_insert: Vec<Vec<usize>> = Vec::new();
            let mut trials = 0;
            let mut retry: bool;

            while eps_insert.len() < nb_threads {
                let mut thread_list = Vec::new();
                if entry_points.len() == 0 {
                    break;
                }
                let entry_points_vec: Vec<&&usize> = entry_points.iter().collect();
                retry = false;
                let ep_idx = rng.gen_range(0..entry_points_vec.len());
                let ep_id = entry_points_vec
                    .get(ep_idx)
                    .expect(format!("{ep_idx} not in 'entry_points'").as_str());
                let possible_ep = index.points.get_point(***ep_id);

                // for already_taken_list in eps_insert.iter() {
                //     for entry_point in already_taken_list.iter() {
                //         let ep_point = index.points.get_point(*entry_point);
                //         if ep_point.dist2vec(&possible_ep.vector).dist < min_dist {
                //             retry = true;
                //             break;
                //         }
                //     }
                // }

                // if (trials > 32) & (eps_insert.len() > 0) {
                //     break;
                // }
                // if retry {
                //     trials += 1;
                //     continue;
                // }
                // trials = 0;

                thread_list.push(possible_ep.id);
                entry_points.remove(&possible_ep.id);
                for neighbor in index
                    .layers
                    .get(&1)
                    .unwrap()
                    .neighbors(possible_ep.id)
                    .iter()
                {
                    if eps_points.contains_key(neighbor) {
                        thread_list.push(*neighbor);
                        entry_points.remove(neighbor);
                    }
                }

                eps_insert.push(thread_list);
            }

            let mut update = 0;
            let mut handlers = Vec::new();
            for thread_list in eps_insert.iter() {
                let mut index_copy = index.clone();
                let mut points_thread = Vec::new();
                for ep_to_insert in thread_list.iter() {
                    for point_id in eps_points
                        .get(ep_to_insert)
                        .expect(format!("{ep_to_insert} not a key in 'eps_points'").as_str())
                        .iter()
                    {
                        if points_map.contains_key(point_id) {
                            points_thread.push((
                                *ep_to_insert,
                                points_map
                                    .get(point_id)
                                    .expect(
                                        format!("{point_id} not a key in 'points_map'").as_str(),
                                    )
                                    .clone(),
                            ));
                            points_map.remove(point_id);
                            update += 1;
                        }
                    }
                }
                let bar = get_progress_bar("Inserting Vectors", points_map.len(), true);
                handlers.push(std::thread::spawn(move || {
                    index_copy.insert_par_3(points_thread, bar);
                    index_copy
                }));
            }
            let mut thread_results = Vec::new();
            for handle in handlers {
                let thread_output = handle.join().unwrap();
                thread_results.push(thread_output);
            }
            index.merge_threads(thread_results);
            eps_insert.clear();
            bar.inc(update);
        }

        index
    }

    pub fn build_index_par_002(m: usize, vectors: Vec<Vec<f32>>) -> Self {
        // let nb_threads = std::thread::available_parallelism().unwrap().get();
        let nb_threads = 64;
        let dim = vectors[0].len();

        let mut index = HNSW::new(m, None, dim);
        let (mut points, levels) = index.make_vec_points(vectors);

        let non_zero = levels.iter().take_while(|x| **x > 0).map(|x| *x);
        let bar = get_progress_bar(
            "Inserting non-zero layer points",
            non_zero.clone().count(),
            false,
        );

        let first_insert = points.pop().unwrap();
        index.first_insert(first_insert.clone());
        for level in non_zero {
            index.insert(points.pop().unwrap(), level);
            bar.inc(1);
        }

        let eps_points = index.determine_eps(points.clone());

        let mut points_map = HashMap::new();
        for point in points.iter() {
            points_map.insert(point.id, point.clone());
        }

        let (eps_splits, frontier_eps) = index.split_eps(nb_threads, &eps_points);

        let biggest_thread = eps_splits
            .iter()
            .enumerate()
            .max_by_key(|(_, split)| split.len())
            .unwrap();

        let frac_frontier = (frontier_eps.len() as f32) / (eps_points.len() as f32);

        println!("Frac frontier eps: {frac_frontier}");
        println!("Biggest split {} eps", biggest_thread.1.len());
        println!(
            "There are {} non-empty splits",
            eps_splits.iter().filter(|x| x.len() > 0).count()
        );

        let mut handlers = Vec::new();
        for (thread_idx, thread_list) in eps_splits.iter().enumerate() {
            let mut index_copy = index.clone();
            let mut points_thread = Vec::new();
            for ep_to_insert in thread_list.iter() {
                for point_id in eps_points
                    .get(ep_to_insert)
                    .expect(format!("{ep_to_insert} not a key in 'eps_points'").as_str())
                    .iter()
                {
                    if points_map.contains_key(point_id) {
                        points_thread.push((
                            *ep_to_insert,
                            points_map
                                .get(point_id)
                                .expect(format!("{point_id} not a key in 'points_map'").as_str())
                                .clone(),
                        ));
                        points_map.remove(point_id);
                    }
                }
            }
            let bar = get_progress_bar(
                "Inserting Vectors",
                points_thread.len(),
                thread_idx != biggest_thread.0,
            );
            handlers.push(std::thread::spawn(move || {
                index_copy.insert_par_3(points_thread, bar);
                index_copy
            }));
        }
        let mut thread_results = Vec::new();
        for handle in handlers {
            let thread_output = handle.join().unwrap();
            thread_results.push(thread_output);
        }
        index.merge_threads(thread_results);

        let mut points_thread = Vec::new();
        for ep_to_insert in frontier_eps.iter() {
            for point_id in eps_points
                .get(ep_to_insert)
                .expect(format!("{ep_to_insert} not a key in 'eps_points'").as_str())
                .iter()
            {
                if points_map.contains_key(point_id) {
                    points_thread.push((
                        *ep_to_insert,
                        points_map
                            .get(point_id)
                            .expect(format!("{point_id} not a key in 'points_map'").as_str())
                            .clone(),
                    ));
                    points_map.remove(point_id);
                }
            }
        }
        let bar = get_progress_bar("Inserting Frontier Vectors", points_thread.len(), false);
        index.insert_par_3(points_thread, bar);
        index
    }

    /// Splits eps according to randomized halt-and-sync logic (obsidian)
    fn split_eps(
        &self,
        nb_threads: usize,
        eps: &HashMap<usize, HashSet<usize>>,
    ) -> (Vec<Vec<usize>>, Vec<usize>) {
        let min_dist = 1.0;
        let mut eps_ids: Vec<&usize> = eps.keys().collect();
        eps_ids.shuffle(&mut thread_rng());
        let eps_ids_set: HashSet<&&usize> = HashSet::from_iter(eps_ids.iter());
        let mut seen: HashSet<usize> = HashSet::new();

        let mut eps_splits: Vec<Vec<usize>> = Vec::from_iter((0..nb_threads).map(|_| Vec::new()));
        let mut frontier_eps: Vec<usize> = Vec::new();
        let mut empty_splits = nb_threads;

        let bar = get_progress_bar("Splitting eps", eps_ids.len(), false);

        for id in eps_ids.iter() {
            bar.inc(1);
            if seen.contains(&id) {
                continue;
            }
            seen.insert(**id);
            let mut ep_neighbors = self.layers.get(&1).unwrap().neighbors(**id).clone();
            let mut to_remove = Vec::new();
            for i in ep_neighbors.iter() {
                if !eps_ids_set.contains(&i) {
                    to_remove.push(*i);
                }
            }
            for i in to_remove {
                ep_neighbors.remove(&i);
            }
            seen.extend(ep_neighbors.iter());

            if empty_splits == nb_threads {
                let split = eps_splits.get_mut(0).unwrap();
                split.push(**id);
                split.extend(ep_neighbors.iter());
                empty_splits -= 1;
                continue;
            }

            let mut close_to = Vec::from_iter((0..nb_threads).map(|_| false));

            let ep_point = self.points.get_point(**id);
            for (idx, split) in eps_splits.iter().enumerate() {
                for inserted_ep in split.iter() {
                    let inserted_point = self.points.get_point(*inserted_ep);
                    let dist = ep_point.dist2vec(&inserted_point.vector).dist;
                    if dist < min_dist {
                        *close_to.get_mut(idx).unwrap() = true;
                        break;
                    }
                }
            }

            let nb_close = close_to.iter().filter(|x| **x).count();

            // means ep is a frontier point; its close to more than one cluster
            if nb_close > 1 {
                frontier_eps.push(ep_point.id);
                frontier_eps.extend(ep_neighbors.iter());
            // means we will insert in one of the threads
            } else if nb_close == 1 {
                let insert_in = close_to
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| **x)
                    .next()
                    .unwrap()
                    .0;
                assert!(insert_in < eps_splits.len());
                eps_splits.get_mut(insert_in).unwrap().push(**id);
                eps_splits
                    .get_mut(insert_in)
                    .unwrap()
                    .extend(ep_neighbors.iter());
            } else if nb_close == 0 {
                let mut insert_in: usize = 999;
                for (idx, split) in eps_splits.iter().enumerate() {
                    if split.is_empty() {
                        insert_in = idx;
                        break;
                    }
                }
                // means the point is far from all clusters, there is no empty thread
                // put it in the nearest cluster
                if insert_in == 999 {
                    let mut min_dist = (999, 999.0);
                    for (idx, split) in eps_splits.iter().enumerate() {
                        for inserted_ep in split.iter() {
                            let inserted_point = self.points.get_point(*inserted_ep);
                            let dist = ep_point.dist2vec(&inserted_point.vector).dist;
                            if dist < min_dist.1 {
                                min_dist = (idx, dist);
                            }
                        }
                    }
                    insert_in = min_dist.0;
                }

                assert!(insert_in < eps_splits.len());
                eps_splits.get_mut(insert_in).unwrap().push(**id);
                eps_splits
                    .get_mut(insert_in)
                    .unwrap()
                    .extend(ep_neighbors.iter());
            }
        }

        (eps_splits, frontier_eps)
    }

    fn partition_layer_1(
        &self,
        partitions: usize,
        exclude: HashSet<usize>,
        point_ids: HashSet<&usize>,
    ) -> HashSet<usize> {
        // fn partition_layer_1(&self, partitions: usize) -> BTreeMap<Dist, (usize, usize)> {
        // let points: Vec<Point> = self.points.iterate().map(|(_, x)| x.clone()).collect();
        // let (centers, clusters) = partition_space(partitions, 10, &points);

        let len = (((self.points.len() as f32).powi(2)) / 2.0) - self.points.len() as f32;
        let mut biggest = ((0, 0), 0.0);
        let mut mean = 0.0;
        let mut distances_ord = BTreeMap::new();
        let mut distances = HashMap::new();
        let mut visited_pairs: HashSet<(usize, usize)> = HashSet::new();
        let bar = get_progress_bar("distances", len as usize, false);

        for idx in point_ids.iter().filter(|x| !exclude.contains(x)) {
            let point_i = self.points.get_point(**idx);
            for jdx in point_ids.iter().filter(|x| !exclude.contains(x)) {
                let point_j = self.points.get_point(**jdx);
                if idx == jdx {
                    continue;
                }
                let pair = (point_i.id.min(point_j.id), point_i.id.max(point_j.id));
                if visited_pairs.contains(&pair) {
                    continue;
                }
                bar.inc(1);
                let dist = point_i.dist2vec(&point_j.vector);
                if dist.dist > biggest.1 {
                    biggest = (pair, dist.dist);
                }
                visited_pairs.insert(pair);
                distances_ord.insert(dist, pair);
                distances.insert(pair, dist);
                mean += dist.dist;
            }
        }

        // mean /= len;
        // let std = (distances_ord
        //     .iter()
        //     .map(|(x, _)| (x.dist - mean).powi(2))
        //     .sum::<f32>()
        //     / len)
        //     .powf(0.5);
        // let limit = 1.32;
        // let frac = (distances_ord
        //     .iter()
        //     .filter(|(x, _)| x.dist >= limit)
        //     .count() as f32)
        //     / len;

        // println!("mean dist: {}", mean);
        // println!("std of dists: {}", std);
        // println!("largest dist: {biggest:?}");
        // println!("fraction of pair of points with dists > {limit}: {frac}");

        let threshold = 1.4;
        let mut points_appart: HashSet<usize> = HashSet::new();
        while points_appart.len() < partitions {
            let (dist, (i, j)) = distances_ord.pop_last().unwrap();
            if dist.dist < threshold {
                println!("broke form while loop because next dist was too low");
                break;
            }
            for new_point in [i, j] {
                let mut can_insert: bool = true;
                for point_appart in points_appart.clone().iter() {
                    if new_point == *point_appart {
                        continue;
                    }
                    let pair = (new_point.min(*point_appart), new_point.max(*point_appart));
                    if distances.get(&pair).unwrap().dist < threshold {
                        can_insert = false;
                        break;
                    }
                }
                if can_insert {
                    points_appart.insert(new_point);
                }
            }
        }

        // let mut v = HashSet::new();
        // for i in points_appart.iter() {
        //     for j in points_appart.iter() {
        //         if i == j {
        //             continue;
        //         }
        //         let pair = (*i.min(j), *i.max(j));
        //         if v.contains(&pair) {
        //             continue;
        //         }
        //         v.insert(pair);
        //         let dist = distances.get(&pair).unwrap();
        //         println!("Selected pair is {pair:?} with distance {dist}");
        //     }
        // }

        // distances_ord
        points_appart
    }

    /// Returns a HashMap linking each entry point to the points it inserts in layer 0.
    fn determine_eps(&self, points: Vec<Point>) -> HashMap<usize, HashSet<usize>> {
        let nb_threads = std::thread::available_parallelism().unwrap().get();
        let self_ref = Arc::new(RwLock::new(self.clone()));
        let mut points_split = split(points, nb_threads);

        let max_layer_nb = self.layers.len() - 1;
        let mut handlers = Vec::new();
        for thread_nb in 0..nb_threads {
            let index_copy = self_ref.clone();
            let points_thread = points_split.pop().unwrap();
            let bar = get_progress_bar(
                "Finding entry points",
                points_thread.len(),
                thread_nb != (nb_threads - 1),
            );

            handlers.push(std::thread::spawn(move || {
                let mut eps_thread = HashMap::new();
                for (_, point) in points_thread.iter().enumerate() {
                    let ep = *index_copy
                        .read()
                        .step_1(point, max_layer_nb, 0)
                        .iter()
                        .next()
                        .unwrap();

                    eps_thread
                        .entry(ep)
                        .or_insert(HashSet::from([point.id]))
                        .insert(point.id);
                    bar.inc(1);
                }
                eps_thread
            }));
        }
        let mut added = 0;
        let mut seen = 0;
        let mut eps = HashMap::new();
        for handle in handlers {
            let eps_thread = handle.join().unwrap();
            for (ep, nodes) in eps_thread {
                for node in nodes {
                    let i = eps.entry(ep).or_insert(HashSet::from([node])).insert(node);
                    if i {
                        added += 1;
                    }
                    seen += 1;
                }
            }
        }
        println!("added nodes {}", added);
        println!("seen nodes {}", seen);

        let mut in_map = 0;
        for (_, ps) in eps.iter() {
            in_map += ps.len();
        }
        println!("in map {}", in_map);

        //ok
        eps
    }
    fn merge_threads(&mut self, thread_results: Vec<Self>) {
        let layer = self.layers.get_mut(&0).unwrap();
        for result in thread_results {
            let thread_layer = result.layers.get(&0).unwrap();
            for (node_id, neighbors) in thread_layer.nodes.iter() {
                if layer.nodes.contains_key(node_id) {
                    layer
                        .nodes
                        .get_mut(node_id)
                        .unwrap()
                        .extend(neighbors.clone());
                } else {
                    layer.nodes.insert(*node_id, neighbors.clone());
                }
            }
            for (_, point) in result.points.iterate() {
                self.points.insert(point.clone());
            }
        }
    }

    fn sort_by_distance(
        &self,
        vector: &Vector,
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
        let mut candidates = self.sort_by_distance(vector, &ep);
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
            _ => panic!("Something went wrong reading parameters of the index file."),
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
}

fn get_progress_bar(message: &'static str, remaining: usize, hidden: bool) -> ProgressBar {
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
    bar.set_message(message);
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

/// Mean-centers each vector using each dimension's mean over the entire matrix.
fn center_vectors(vectors: &mut Vec<Vec<f32>>) {
    let mut means = Vec::from_iter((0..vectors[0].len()).map(|_| 0.0));
    for vector in vectors.iter() {
        for (idx, val) in vector.iter().enumerate() {
            means[idx] += val
        }
    }
    for idx in 0..means.len() {
        means[idx] /= vectors.len() as f32;
    }

    vectors.iter_mut().for_each(|v| {
        v.iter_mut()
            .enumerate()
            .for_each(|(idx, x)| *x -= means[idx])
    });
}

fn compute_stats(points: &Points) -> (f32, f32) {
    let mut dists: HashMap<(usize, usize), f32> = HashMap::new();
    for (id, point) in points.iterate() {
        for (idx, pointx) in points.iterate() {
            if id == idx {
                continue;
            }
            dists
                .entry((*id.min(idx), *id.max(idx)))
                .or_insert(point.dist2vec(&pointx.vector).dist);
        }
    }

    write_stats(&dists).unwrap();

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

fn write_stats(dists: &HashMap<(usize, usize), f32>) -> std::io::Result<()> {
    std::fs::remove_file("./dist_stats.json")?;
    let file = File::create("./dist_stats.json")?;
    let mut writer = BufWriter::new(file);
    let dists = Vec::from_iter(dists.values());
    serde_json::to_writer_pretty(&mut writer, &dists)?;
    writer.flush()?;
    Ok(())
}
