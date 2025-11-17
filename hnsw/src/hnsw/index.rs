use super::{dist::Node, graph::Graph};
use crate::{
    helpers::data::split_ids,
    hnsw::{
        params::Params,
        points::{
            point::Point,
            point_collection::{Points, Storage},
        },
    },
};

use indicatif::{ProgressBar, ProgressStyle};
use nohash_hasher::{IntMap, IntSet};

use std::{cmp::Reverse, collections::BTreeSet, sync::Arc};

use core::panic;
use std::collections::BinaryHeap;

fn select_simple<I>(candidate_dists: I, m: usize) -> Result<BinaryHeap<Node>, String>
where
    I: Iterator<Item = Node>,
{
    let cands = BTreeSet::from_iter(candidate_dists);
    Ok(BinaryHeap::from_iter(cands.iter().copied().take(m)))
}

#[derive(Debug, Clone)]
pub struct Searcher {
    selected: BinaryHeap<Node>,
    candidates: BinaryHeap<Reverse<Node>>,
    visited: IntSet<u32>,
    visited_heuristic: BinaryHeap<Reverse<Node>>,
    insertion_results: IntMap<u8, IntMap<u32, BinaryHeap<Node>>>,
    prune_results: IntMap<u8, IntMap<u32, BinaryHeap<Node>>>,
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
pub struct HNSW {
    ep: u32,
    pub params: Params,
    pub points: Points,
    pub layers: IntMap<u8, Graph>,
}

impl HNSW {
    pub fn new(m: u8, ef_cons: Option<u32>, dim: u32) -> Self {
        let params = if ef_cons.is_some() {
            Params::from_m_efcons(m, ef_cons.unwrap(), dim)
        } else {
            Params::from_m(m, dim)
        };
        HNSW {
            params,
            ep: 0,
            points: Points::new(),
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

    pub fn ann_by_vector(&self, point: &Point, n: usize, ef: u32) -> Result<Vec<u32>, String> {
        let mut searcher = Searcher::new();
        searcher.selected.push(Node::new_with_dist(
            point.distance(self.points.get_point(self.ep)),
            self.ep,
        ));
        let nb_layer = self.layers.len();

        for layer_nb in (1..nb_layer).rev().map(|x| x as u8) {
            self.search_layer(
                &mut searcher,
                self.layers.get(&(layer_nb)).unwrap(),
                &point,
                1,
            )?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        self.search_layer(&mut searcher, layer_0, &point, ef)?;

        let anns: Vec<Node> = searcher.selected.into_sorted_vec();
        Ok(anns.iter().take(n).map(|x| x.id).collect())
    }

    fn step_1(
        &self,
        searcher: &mut Searcher,
        point: &Point,
        max_layer_nb: u8,
        level: u8,
        stop_at_layer: Option<u8>,
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

    fn step_2(&self, searcher: &mut Searcher, point: &Point, level: u8) -> Result<(), String> {
        let bound = (level).min((self.layers.len() - 1) as u8);

        for layer_nb in (0..=bound).rev().map(|x| x as u8) {
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
                self.params.mmax0 as usize
            } else {
                self.params.mmax as usize
            };

            for (_, neighbors) in node_neighbors.iter() {
                let to_prune = neighbors
                    .iter()
                    .filter(|x| layer.degree(x.id).unwrap() > limit)
                    .map(|x| *x);

                for dist in to_prune {
                    // searcher.clear_searchers();
                    let nearest = select_simple(layer.neighbors(dist.id)?.iter().copied(), limit)?;
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
        m: u8,
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
                .collect::<Vec<Reverse<Node>>>()
            {
                for neighbor_dist in layer.neighbors(dist.0.id)? {
                    searcher.candidates.push(Reverse(Node::new_with_dist(
                        point.distance(self.points.get_point(neighbor_dist.id)),
                        neighbor_dist.id,
                    )));
                }
            }
        }

        // let s2 = Instant::now();
        let dist_e = searcher.candidates.pop().unwrap();
        searcher.selected.push(dist_e.0);
        // println!("s2 {}", s2.elapsed().as_nanos());

        while (!searcher.candidates.is_empty()) & (searcher.selected.len() < m as usize) {
            // let s3 = Instant::now();
            let dist_e = searcher.candidates.pop().unwrap();
            let e_point = self.points.get_point(dist_e.0.id);
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
            while (!searcher.visited_heuristic.is_empty()) & (searcher.selected.len() < m as usize)
            {
                let dist_e = searcher.visited_heuristic.pop().unwrap();
                searcher.selected.push(dist_e.0);
            }
        }
        // println!("s6 {}", s6.elapsed().as_nanos());

        Ok(())
    }

    pub fn insert(&mut self, point_id: u32, searcher: &mut Searcher) -> Result<bool, String> {
        // let s0 = Instant::now();
        searcher.clear_all();

        let point = self.points.get_point(point_id);

        if self.layers.is_empty() {
            self.first_insert(point_id);
            return Ok(true);
        }

        searcher.selected.push(Node::new_with_dist(
            point.distance(self.points.get_point(self.ep)),
            self.ep,
        ));

        let level = point.level;
        let max_layer_nb = (self.layers.len() - 1) as u8;
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

    pub fn insert_par(index: Arc<Self>, ids: Vec<u32>, bar: ProgressBar) -> Result<(), String> {
        let mut searcher = Searcher::new();
        let max_layer_nb = (index.layers.len() - 1) as u8;

        for point_id in ids.iter() {
            searcher.clear_searchers();

            let point = index.points.get_point(*point_id);
            let level = point.level;
            searcher.selected.push(Node::new_with_dist(
                point.distance(index.points.get_point(index.ep)),
                index.ep,
            ));

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
        point_id: u32,
        level: u8,
        max_layer_nb: u8,
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

    /// Stores the points in the internal `points` field
    /// and adds them to the index's layers.
    ///
    /// This is only the storing part, no indexing can
    /// be done on these points after this operation,
    fn store_points(&mut self, points: Points) {
        let max_layer_nb = points.iter_points().map(|point| point.level).max().unwrap();
        for layer_nb in 0..=max_layer_nb {
            self.layers.entry(layer_nb).or_insert(Graph::new());
        }
        for point in points.iter_points() {
            for l in 0..=point.level {
                self.layers.get_mut(&l).unwrap().add_node(point.id);
            }
        }
        self.ep = *self
            .layers
            .get(&max_layer_nb)
            .unwrap()
            .nodes
            .keys()
            .next()
            .unwrap();

        self.points.extend(points);
    }

    fn first_insert(&mut self, point_id: u32) {
        let mut layer = Graph::new();
        layer.add_node(point_id);
        self.layers.insert(0, layer);
        self.ep = point_id;
    }

    fn delete_one_node_layer(&mut self) {
        let max_layer_nb = (self.layers.len() - 1) as u8;
        if self.layers.get(&max_layer_nb).unwrap().nb_nodes() == 1 {
            self.layers.remove(&max_layer_nb);
        }
    }

    pub fn build_index(
        m: u8,
        ef_cons: Option<u32>,
        points: Points,
        verbose: bool,
    ) -> Result<Self, String> {
        let mut searcher = Searcher::new();
        let dim = match points.dim() {
            None => panic!("No points in the collection"),
            Some(d) => d,
        };

        let mut index = HNSW::new(m, ef_cons, dim as u32);
        index.store_points(points);

        let bar = get_progress_bar("Inserting Vectors".to_string(), index.points.len(), verbose);

        let ids: Vec<u32> = index.points.ids().collect();
        for id in ids.iter() {
            index.insert(*id, &mut searcher)?;
            if !bar.is_hidden() {
                bar.inc(1);
            }
        }
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

    fn get_nearest<I>(&self, point: &Point, others: I) -> Node
    where
        I: Iterator<Item = u32>,
    {
        others
            .map(|idx| Node::new_with_dist(point.distance(self.points.get_point(idx)), idx))
            .min()
            .unwrap()
    }

    pub fn search_layer(
        &self,
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point,
        ef: u32,
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
            let q2cand_neighbors_dists = point.dist2many(
                cand_neighbors
                    .iter()
                    .filter(|dist| searcher.visited.insert(dist.id))
                    .map(|dist| self.points.get_point(dist.id)),
            );
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = searcher.selected.peek().unwrap();
                if (n2q_dist < *f2q_dist) | (searcher.selected.len() < ef as usize) {
                    searcher.selected.push(n2q_dist);
                    searcher.candidates.push(Reverse(n2q_dist));

                    if searcher.selected.len() > ef as usize {
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

pub fn build_index_par(
    m: u8,
    ef_cons: Option<u32>,
    points: Points,
    verbose: bool,
) -> Result<HNSW, String> {
    let nb_threads = std::thread::available_parallelism().unwrap().get() as u8;
    let dim = match points.dim() {
        None => panic!("No points in the collection"),
        Some(d) => d,
    };

    let mut index = HNSW::new(m, ef_cons, dim as u32);
    index.store_points(points);
    let index_arc = Arc::new(index);

    for layer_nb in (0..index_arc.layers.len()).rev().map(|x| x as u8) {
        let ids: Vec<u32> = index_arc
            .points
            .iter_points()
            .filter(|point| point.level == layer_nb)
            .map(|point| point.id)
            .collect();

        let mut points_split = split_ids(ids, nb_threads);

        let mut handlers = Vec::new();
        for thread_idx in 0..nb_threads {
            let index_copy = Arc::clone(&index_arc);
            let ids_split: Vec<u32> = points_split.pop().unwrap();
            let bar = get_progress_bar(
                format!("Layer {layer_nb}:"),
                ids_split.len(),
                (verbose) & (thread_idx == 0),
            );
            handlers.push(std::thread::spawn(move || {
                HNSW::insert_par(index_copy, ids_split, bar).unwrap();
            }));
        }
        for handle in handlers {
            handle.join().unwrap();
        }
    }

    // index_arc.assert_param_compliance();

    let mut index = Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");
    index.delete_one_node_layer();
    Ok(index)
}
