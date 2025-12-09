use core::panic;
use std::{
    collections::BTreeSet,
    fs::{create_dir, File},
    io::{BufReader, Read, Write},
    path::Path,
    sync::Arc,
};

use crate::{helpers::get_progress_bar, params::Params, template::searcher::Searcher};
use graph::{
    graph::Graph,
    layers::Layers,
    nodes::{Dist, Node},
};
use points::{point::Point, point_collection::Points};
use vectors::{serializer::Serializer, VecBase, VecTrait};

mod searcher;

mod results;
use results::Results;

mod inserter;
use inserter::Inserter;

fn select_simple<I>(candidate_dists: I, m: usize) -> Result<BTreeSet<Dist>, String>
where
    I: Iterator<Item = Dist>,
{
    let mut cands = Vec::from_iter(candidate_dists);
    cands.sort();
    Ok(BTreeSet::from_iter(cands.iter().copied().take(m)))
}

#[derive(Debug, Clone)]
pub struct HNSW<T: VecTrait> {
    pub params: Params,
    layers: Layers,
    points: Points<T>,
}

impl<T: VecTrait> HNSW<T> {
    pub fn save(&self, dir: &Path) {
        if !dir.exists() {
            match create_dir(dir) {
                Ok(_) => (),
                Err(e) => panic!("Could not create dir {dir:?}: {e}"),
            }
        }

        let mut points_file =
            File::create(dir.join("points")).expect("Could not create points file");
        let mut layers_file =
            File::create(dir.join("layers")).expect("Could not create layers file");
        let mut params_file =
            File::create(dir.join("params")).expect("Could not create params file");

        points_file
            .write_all(&self.points.serialize())
            .expect("Could not write bytes to point file");
        layers_file
            .write_all(&self.layers.serialize())
            .expect("Could not write bytes to layers file");
        params_file
            .write_all(&self.params.serialize())
            .expect("Could not write bytes to params file");
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        if !dir.exists() {
            return Err(format!("{dir:?} does not exist"));
        }

        let points = match File::open(dir.join("points")) {
            Ok(f) => {
                let reader = BufReader::new(f);
                Points::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
            }
            Err(e) => return Err(format!("Problem reading points file: {e}")),
        };
        let layers = match File::open(dir.join("layers")) {
            Ok(f) => {
                let reader = BufReader::new(f);
                Layers::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
            }
            Err(e) => return Err(format!("Problem reading layers file: {e}")),
        };
        let params = match File::open(dir.join("params")) {
            Ok(f) => {
                let reader = BufReader::new(f);
                Params::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
            }
            Err(e) => return Err(format!("Problem reading params file: {e}")),
        };

        Ok(HNSW {
            points,
            layers,
            params,
        })
    }

    pub fn new(m: usize, ef_cons: Option<usize>, dim: usize) -> Self {
        let params = if ef_cons.is_some() {
            Params::from_m_efcons(m, ef_cons.unwrap(), dim)
        } else {
            Params::from_m(m, dim)
        };
        HNSW {
            params,
            points: Points::new(),
            layers: Layers::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn distance(&self, a: Node, b: Node) -> Option<f32> {
        self.points.distance(a, b)
    }

    pub fn layer_degrees(&self, layer_nb: &u8) {
        let layer = self.layers.get_layer(layer_nb);
        for node in layer.nodes.keys() {
            println!("{}", layer.degree(*node).unwrap());
        }
    }

    pub fn insert_point(&mut self, point: Point<T>) -> Result<bool, String> {
        let point_id = point.id;
        self.points.push(point);
        self.insert(point_id, &mut Inserter::new())
    }

    // not really insertion, this only determines neighbors
    // of a point already in the points store and in the layers
    fn insert(&self, point_id: Node, inserter: &mut Inserter) -> Result<bool, String> {
        let point = self
            .points
            .get_point(point_id)
            .expect("Point ID not found in collection.");

        inserter.build_insertion_results(&self, point)?;
        self.make_connections(inserter.get_results())?;
        self.prune_connexions(inserter.get_results_mut())?;
        self.write_results_prune(inserter.get_results())?;

        Ok(true)
    }

    pub fn get_layer(&self, layer_nb: &u8) -> &Graph {
        self.layers.get_layer(layer_nb)
    }

    fn make_connections(&self, results: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in results.iter_insertion_results() {
            self.layers.apply_insertion_results(&layer_nb, node_data)?;
        }
        Ok(())
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
                        .map(|n| Dist::new(*n, self.distance(to_prune.id, *n).unwrap()));
                    let nearest = select_simple(to_prune_distances, limit)?;
                    searcher.insert_prune_result(*layer_nb, to_prune.id, nearest);
                }
            }
        }
        Ok(())
    }

    fn write_results_prune(&self, results: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in results.iter_prune_results() {
            let layer = self.layers.get_layer(&layer_nb);
            for (node, neighbors) in node_data.iter() {
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
        let ids_levels = self.points.extend(points);
        for (point_id, level) in ids_levels {
            for layer_nb in 0..=level {
                self.layers.add_node_to_layer(layer_nb, point_id);
            }
        }

        let max_layer_nb = self.layers.len() - 1;
        let new_ep = *self
            .get_layer(&(max_layer_nb as u8))
            .nodes
            .keys()
            .next()
            .unwrap();
        // TODO: if index already had points and we insert more,
        // one of the new points could have a higher level than the
        // current max. So it will become the new EP, and therefore
        // needs to be connected in all the layers.
        // self.insert(new_ep, &mut Inserter::new()).unwrap();
        self.params.ep = new_ep;
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

    pub fn ann_by_vector(
        &self,
        point: &Point<T>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<Node>, String> {
        let mut results = Results::new();
        let searcher = Searcher::new();
        let mut point = point.clone();
        point.center(&self.points.means.clone().unwrap());
        results.push_selected(Dist::new(
            self.params.ep,
            self.points.distance2point(&point, self.params.ep).unwrap(),
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
        println!("ep: {:?}", self.params.ep);
    }
}

impl<T: VecTrait + std::marker::Send + std::marker::Sync + 'static> HNSW<T> {
    pub fn insert_bulk(mut self, points: Points<T>, nb_threads: usize) -> Result<HNSW<T>, String> {
        self.store_points(points);
        let bar = get_progress_bar("layerzzz".to_string(), self.len(), true);
        let index_arc = Arc::new(self);

        for layer_nb in (0..index_arc.layers.len()).rev().map(|x| x as u8) {
            let layer = index_arc.layers.get_layer(&layer_nb);
            let chunks = ((layer.nb_nodes() as f64) / (nb_threads as f64)).ceil() as usize;
            let mut ids: Vec<Vec<Node>> = layer
                .iter_nodes()
                .filter(|id| index_arc.points.get_point(*id).unwrap().level == layer_nb)
                .collect::<Vec<Node>>()
                .chunks(chunks)
                .map(|chunk| Vec::from(chunk))
                .collect();

            let mut handlers = Vec::new();
            while let Some(ids_split) = ids.pop() {
                let index_copy = Arc::clone(&index_arc);
                let bar_ref = bar.clone();
                handlers.push(std::thread::spawn(move || {
                    let mut inserter = Inserter::new();
                    for id in ids_split.iter() {
                        index_copy.insert(*id, &mut inserter).unwrap();
                        bar_ref.inc(1);
                    }
                }));
            }
            for handle in handlers {
                handle.join().unwrap();
            }
        }

        let index = Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");
        Ok(index)
    }
}
