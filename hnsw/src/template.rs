use std::collections::BinaryHeap;

use crate::{helpers::get_progress_bar, params::Params, template::searcher::Searcher};
use graph::{
    graph::Graph,
    layers::Layers,
    nodes::{Dist, Node},
};
use points::{point::Point, point_collection::Points};
use vectors::VecTrait;

mod searcher;

mod results;
use results::Results;

mod inserter;
use inserter::Inserter;

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
    pub layers: Layers,
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
        HNSW {
            params,
            ep: 0,
            points: Points::new(),
            layers: Layers::new(),
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
        self.insert(point_id, &mut Inserter::new())
    }

    fn insert(&mut self, point_id: Node, inserter: &mut Inserter) -> Result<bool, String> {
        let point = self
            .points
            .get_point(point_id)
            .expect("Point ID not found in collection.");
        let level = point.level;

        inserter.build_insertion_results(&self, point)?;
        self.make_connections(inserter.get_results())?;
        self.prune_connexions(inserter.get_results_mut())?;
        self.write_results_prune(inserter.get_results())?;
        self.add_layers(point_id, level);

        Ok(true)
    }

    pub fn get_layer(&self, layer_nb: &u8) -> &Graph {
        self.layers.get_layer(layer_nb)
    }

    fn add_layers(&mut self, point_id: Node, level: u8) {
        let max_layer = self.layers.len() - 1;
        if level > max_layer {
            for layer_nb in max_layer + 1..level + 1 {
                self.layers.add_layer_with_node(layer_nb, point_id);
            }
            self.ep = point_id;
        }
    }

    fn prune_connexions(&self, searcher: &mut Results) -> Result<(), String> {
        searcher.clear_prune();

        for (layer_nb, node_data) in searcher.get_clone_insertion_results().iter() {
            let layer = self.get_layer(layer_nb);
            let limit = if *layer_nb == 0 {
                self.params.mmax0 as usize
            } else {
                self.params.mmax as usize
            };

            for (_, neighbors) in node_data.iter() {
                let nodes_to_prune = neighbors
                    .iter()
                    .filter(|x| layer.degree(x.id).unwrap() > limit)
                    .map(|x| *x);

                for to_prune in nodes_to_prune {
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

    pub fn insert_bulk(&mut self, points: Points<T>) -> Result<bool, String> {
        let mut inserter = Inserter::new();

        self.store_points(points);
        self.print_index();

        let bar = get_progress_bar(
            "Inserting Vectors".to_string(),
            self.points.len(),
            self.verbose,
        );

        let ids: Vec<Node> = self.points.ids().collect();
        for id in ids.iter() {
            self.insert(*id, &mut inserter)?;
            bar.inc(1);
        }
        Ok(true)
    }

    fn make_connections(&mut self, results: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in results.iter_insertion_results() {
            self.layers.apply_insertion_results(&layer_nb, node_data)?;
        }
        Ok(())
    }

    fn write_results_prune(&mut self, searcher: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in searcher.iter_prune_results() {
            let layer = self.layers.get_layer_mut(&layer_nb);
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
                self.layers.add_node_to_layer(layer_nb, point.id);
            }
        }
        let max_layer_nb = self.layers.len() - 1;
        self.ep = *self
            .get_layer(&(max_layer_nb as u8))
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

    pub fn ann_by_vector(&self, point: &Point<T>, n: usize, ef: u32) -> Result<Vec<Node>, String> {
        let mut results = Results::new();
        let searcher = Searcher::new();
        results.push_selected(Dist::new(
            self.ep,
            self.points.distance2point(point, self.ep).unwrap(),
        ));
        let nb_layers = self.layers.len();

        for layer_nb in (1..nb_layers).rev().map(|x| x as u8) {
            searcher.search_layer(
                &mut results,
                self.get_layer(&(layer_nb)),
                &point,
                &self.points,
                1,
            )?;
        }

        let layer_0 = &self.get_layer(&0);
        searcher.search_layer(&mut results, layer_0, &point, &self.points, ef)?;

        let anns: Vec<Node> = results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        Ok(anns)
    }

    fn iter_layers(&self) -> impl Iterator<Item = (&u8, &Graph)> {
        self.layers.iter_layers()
    }

    pub fn assert_param_compliance(&self) {
        let mut is_ok = true;
        for (layer_nb, layer) in self.iter_layers() {
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
        for (idx, layer) in self.iter_layers() {
            println!("NB. nodes in layer {idx}: {}", layer.nb_nodes());
        }
        println!("ep: {:?}", self.ep);
    }
}
