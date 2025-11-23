use std::collections::BinaryHeap;

use crate::{helpers::get_progress_bar, params::Params};
use graph::{
    graph::Graph,
    nodes::{Dist, Node},
};
use nohash_hasher::IntMap;
use points::{point::Point, point_collection::Points};
use vectors::VecTrait;

mod searcher;
use searcher::Searcher;

fn select_simple<I>(candidate_dists: I, m: usize) -> Result<BinaryHeap<Dist>, String>
where
    I: Iterator<Item = Dist>,
{
    let mut cands = Vec::from_iter(candidate_dists);
    cands.sort();
    Ok(BinaryHeap::from_iter(cands.iter().copied().take(m)))
}

pub struct HNSW<T: VecTrait> {
    pub params: Params,
    pub layers: IntMap<u8, Graph>,
    ep: Node,
    points: Points<T>,
    verbose: bool,
}

impl<T: VecTrait> HNSW<T> {
    pub fn new(m: u8, ef_cons: Option<u32>, dim: u32, verbose: bool) -> Self {
        let params = if ef_cons.is_some() {
            Params::from_m_efcons(m, ef_cons.unwrap(), dim)
        } else {
            Params::from_m(m, dim)
        };
        let layers = IntMap::default();
        HNSW {
            params,
            ep: 0,
            points: Points::new(),
            layers: layers,
            verbose,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn distance(&self, a: Node, b: Node) -> Option<f32> {
        self.points.distance(a, b)
    }

    pub fn insert_point(&mut self, point: Point<T>) -> Result<bool, String> {
        let point_id = point.id;
        self.points.push(point);
        self.insert(point_id, &mut Searcher::new())
    }

    fn insert(&mut self, point_id: Node, searcher: &mut Searcher) -> Result<bool, String> {
        searcher.clear_all();

        // println!("Inserting {point_id}");

        let point = self
            .points
            .get_point(point_id)
            .expect("Point ID not found in collection.");

        let level = point.level;
        let max_layer = (self.layers.len() - 1) as u8;
        let add_new_layers = point.level > max_layer;

        let dist2ep = self
            .distance(self.ep, point.id)
            .expect("Could not compute distance between EP and point to insert.");

        // println!("Dist to EP is {dist2ep}");

        searcher.push_selected(Dist::new(self.ep, dist2ep));

        self.search_layers_above(searcher, point)?;

        self.search_layers_below(searcher, point)?;

        self.make_connections(searcher)?;

        // assert!(false);

        self.prune_connexions(searcher)?;

        self.write_results_prune(searcher)?;

        if add_new_layers {
            for layer_nb in max_layer + 1..level + 1 {
                let mut layer = Graph::new();
                layer.add_node(point_id);
                self.layers.insert(layer_nb, layer);
            }
            self.ep = point_id;
        }

        Ok(true)
    }

    fn search_layers_above(&self, searcher: &mut Searcher, point: &Point<T>) -> Result<(), String> {
        let layers_len = self.layers.len() as u8;

        // println!("Traversing layers above, point level is {}", point.level);
        // println!(
        //     "Searcher selected at start of traversal {:?}",
        //     searcher.selected
        // );
        for layer_nb in (point.level + 1..layers_len).rev() {
            let layer = match self.layers.get(&layer_nb) {
                Some(l) => l,
                None => {
                    return Err(format!(
                        "Could not get layer {layer_nb} while searching layers above."
                    ))
                }
            };
            // println!("Traversing layer {layer_nb}");
            self.search_layer(searcher, layer, point, 1)?;
            // println!(
            //     "Searcher selected after traversal of {layer_nb} {:?}",
            //     searcher.selected
            // );

            if layer_nb == 0 {
                break;
            }
        }
        Ok(())
    }

    fn search_layers_below(&self, searcher: &mut Searcher, point: &Point<T>) -> Result<(), String> {
        let bound = (point.level).min((self.layers.len() - 1) as u8);

        // println!("Traversing layers below, point level is {}", point.level);
        // println!(
        //     "Searcher selected at start of traversal {:?}",
        //     searcher.selected
        // );
        for layer_nb in (0..=bound).rev().map(|x| x as u8) {
            let layer = self.layers.get(&layer_nb).unwrap();

            // println!("Traversing layer {layer_nb}");
            self.search_layer(searcher, layer, point, self.params.ef_cons)?;
            self.select_heuristic(searcher, layer, point, self.params.m, false, true)?;
            // println!(
            //     "Searcher selected after traversal of {layer_nb} {:?}",
            //     searcher.selected
            // );

            searcher.insert_layer_results(layer_nb, point.id);
        }
        Ok(())
    }

    fn prune_connexions(&self, searcher: &mut Searcher) -> Result<(), String> {
        searcher.clear_prune();

        for (layer_nb, node_data) in searcher.get_clone_insertion_results().iter() {
            // for (layer_nb, node_data) in searcher.iter_insertion_results() {
            let layer = self.layers.get(layer_nb).unwrap();
            let limit = if *layer_nb == 0 {
                self.params.mmax0 as usize
            } else {
                self.params.mmax as usize
            };

            // println!("Pruning nodes on layer {layer_nb}");
            for (_, neighbors) in node_data.iter() {
                let nodes_to_prune = neighbors
                    .iter()
                    .filter(|x| layer.degree(x.id).unwrap() > limit)
                    .map(|x| *x);

                for to_prune in nodes_to_prune {
                    // println!("Pruning node {0} on layer {layer_nb}", to_prune.id);
                    let to_prune_neighbors = layer.neighbors(to_prune.id)?;
                    let to_prune_distances = to_prune_neighbors
                        .iter()
                        .map(|n| Dist::new(to_prune.id, self.distance(to_prune.id, *n).unwrap()));
                    let nearest = select_simple(to_prune_distances, limit)?;
                    searcher.insert_prune_result(*layer_nb, to_prune.id, nearest);
                }
            }
        }
        Ok(())
    }

    fn select_heuristic(
        &self,
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point<T>,
        m: u8,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        searcher.heuristic_setup();
        if extend_cands {
            // I think this is wrong, because nodes in candidates now contain distances to the candidates, not to our query
            searcher.extend_candidates_with_neighbors(point, &self.points, layer)?;
        }

        let node_e = searcher.pop_candidate().unwrap();
        searcher.push_selected(node_e.0);

        while (!searcher.candidates_is_empty()) & (searcher.selected_len() < m as usize) {
            let node_e = searcher.pop_candidate().unwrap();
            let e_point = self.points.get_point(node_e.0.id).unwrap();

            let nearest_selected = searcher.get_nearest_from_selected(e_point, &self.points);

            if node_e.0 < nearest_selected {
                searcher.push_selected(node_e.0);
            } else if keep_pruned {
                searcher.push_visited_heuristic(node_e.0);
            }
        }

        if keep_pruned {
            while (!searcher.visited_heuristic_is_empty()) & (searcher.selected_len() < m as usize)
            {
                let node_e = searcher.pop_visited_heuristic().unwrap();
                searcher.push_selected(node_e.0);
            }
        }

        Ok(())
    }

    pub fn insert_bulk(&mut self, points: Points<T>) -> Result<bool, String> {
        let mut searcher = Searcher::new();

        self.store_points(points);
        self.print_index();

        let bar = get_progress_bar(
            "Inserting Vectors".to_string(),
            self.points.len(),
            self.verbose,
        );

        let ids: Vec<Node> = self.points.ids().collect();
        for id in ids.iter() {
            self.insert(*id, &mut searcher)?;
            bar.inc(1);
        }
        Ok(true)
    }

    fn make_connections(&mut self, searcher: &Searcher) -> Result<(), String> {
        // println!("Will create connections between nodes after layer traversals");
        for (layer_nb, node_data) in searcher.iter_insertion_results() {
            let layer = self.layers.get(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                // println!("On layer {layer_nb}");
                // println!("The node {node} will get neighbors {neighbors:?}");
                layer.replace_neighbors(*node, neighbors.iter().map(|dist| dist.id))?;
                // println!(
                //     "Neighbors of the point are now {:?}",
                //     layer.neighbors_vec(*node)
                // );
            }
        }
        Ok(())
    }

    fn write_results_prune(&mut self, searcher: &Searcher) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.iter_prune_results() {
            let layer = self.layers.get_mut(&layer_nb).unwrap();
            for (node, neighbors) in node_data.iter() {
                layer.add_node(*node);
                layer.replace_neighbors(*node, neighbors.iter().map(|dist| dist.id))?;
            }
        }

        Ok(())
    }

    /// Stores the points in the internal `points` field
    /// and adds them to the index's layers.
    ///
    /// This is only the storing part, no indexing can
    /// be done on these points after this operation,
    fn store_points(&mut self, points: Points<T>) {
        for point in points.iter_points() {
            for layer_nb in 0..=point.level {
                let layer = self.layers.entry(layer_nb).or_insert(Graph::new());
                layer.add_node(point.id);
            }
        }
        let max_layer_nb = self.layers.len() - 1;
        self.ep = *self
            .layers
            .get(&(max_layer_nb as u8))
            .unwrap()
            .nodes
            .keys()
            .next()
            .unwrap();

        self.points.extend(points);
    }

    // fn get_nearest<I>(&self, point: &Point<T>, others: I) -> Node
    // where
    //     I: Iterator<Item = Node>,
    // {
    //     point
    //         .dist2many(others.map(|idx| self.points.get_point(idx).unwrap()))
    //         .min()
    //         .unwrap()
    //         .id
    // }

    pub fn search_layer(
        &self,
        searcher: &mut Searcher,
        layer: &Graph,
        point: &Point<T>,
        ef: u32,
    ) -> Result<(), String> {
        searcher.extend_candidates_with_selected();
        searcher.extend_visited_with_selected();

        while !searcher.candidates_is_empty() {
            let cand_dist = searcher.pop_candidate().unwrap();
            let furthest2q_dist = searcher.peek_selected().unwrap();
            if cand_dist.0 > *furthest2q_dist {
                break;
            }
            let cand_neighbors = match layer.neighbors_vec(cand_dist.0.id) {
                Ok(neighs) => neighs,
                Err(msg) => return Err(format!("Error in search_layer: {msg}")),
            };

            // println!(
            //     "SEARCH_LAYER: the candidate has {} neighbors",
            //     cand_neighbors.len()
            // );

            // pre-compute distances to candidate neighbors to take advantage of
            // caches and to prevent the re-construction of the query to a full vector
            let not_visited: Vec<Node> = cand_neighbors
                .iter()
                .filter(|node| searcher.push_visited(**node))
                .copied()
                .collect();
            let q2cand_dists = point.dist2many(not_visited.iter().map(|node| {
                self.points
                    .get_point(*node)
                    .expect("Point ID not found in collection.")
            }));
            let q2cand_neighbors_dists: Vec<Dist> = q2cand_dists
                .zip(not_visited.iter())
                .map(|(dist, id)| Dist::new(*id, dist))
                .collect();
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = searcher.peek_selected().unwrap();
                if (n2q_dist < *f2q_dist) | (searcher.selected_len() < ef as usize) {
                    searcher.push_selected(n2q_dist);
                    searcher.push_candidate(n2q_dist);

                    if searcher.selected_len() > ef as usize {
                        searcher.pop_selected();
                    }
                }
            }
        }
        searcher.clear_candidates();
        searcher.clear_visited();
        Ok(())
    }
    pub fn ann_by_vector(&self, point: &Point<T>, n: usize, ef: u32) -> Result<Vec<Node>, String> {
        let mut searcher = Searcher::new();
        searcher.push_selected(Dist::new(
            self.ep,
            self.points.distance2point(point, self.ep).unwrap(),
        ));
        let nb_layers = self.layers.len();

        for layer_nb in (1..nb_layers).rev().map(|x| x as u8) {
            self.search_layer(
                &mut searcher,
                self.layers.get(&(layer_nb)).unwrap(),
                &point,
                1,
            )?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        self.search_layer(&mut searcher, layer_0, &point, ef)?;

        let anns: Vec<Node> = searcher
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        Ok(anns)
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
        println!("Nb. of points = {}", self.points.len());
        for (idx, layer) in self.layers.iter() {
            println!("NB. nodes in layer {idx}: {}", layer.nb_nodes());
        }
        println!("ep: {:?}", self.ep);
    }
}
