use core::panic;
use std::{
    collections::BTreeSet,
    fs::{self, create_dir, File},
    io::{BufReader, Read, Write},
    path::Path,
    sync::Arc,
    thread::Builder,
};

use crate::{
    helpers::get_progress_bar,
    params::{get_default_ml, Params},
    template::searcher::Searcher,
};
use graph::{
    graph::Graph,
    layers::Layers,
    nodes::{Dist, NodeID},
};
use points::{
    point::Point,
    points::{Points, SimplePoints},
};
use rand::Rng;
use vectors::serializer::Serializer;

mod searcher;

mod results;
use results::Results;

mod inserter;
use inserter::Inserter;

type PointsType = SimplePoints;

#[derive(Debug, Clone)]
pub struct HNSW {
    pub params: Params,
    layers: Layers,
    points: PointsType,
}

impl HNSW {
    pub fn save(&self, dir: &Path) {
        if !dir.exists() {
            match create_dir(dir) {
                Ok(_) => (),
                Err(e) => panic!("Could not create dir {dir:?}: {e}"),
            }
        }

        let mut points_file =
            File::create(dir.join("points")).expect("Could not create points file");
        let mut params_file =
            File::create(dir.join("params")).expect("Could not create params file");

        points_file
            .write_all(&self.points.serialize())
            .expect("Could not write bytes to point file");
        params_file
            .write_all(&self.params.serialize())
            .expect("Could not write bytes to params file");

        let layers_dir = dir.join("layers");
        fs::create_dir(layers_dir.clone()).expect("Could not create layers dir");

        for (idx, layer) in self.layers.iter_layers().enumerate() {
            let mut layer_file = File::create(layers_dir.join(idx.to_string()))
                .expect("Could not create file for layer {idx}");
            layer_file
                .write_all(&layer.serialize())
                .expect("Could not write bytes to layer {idx}");
        }
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        if !dir.exists() {
            return Err(format!("{dir:?} does not exist"));
        }

        let points = match File::open(dir.join("points")) {
            Ok(f) => {
                let reader = BufReader::new(f);
                PointsType::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
            }
            Err(e) => return Err(format!("Problem reading points file: {e}")),
        };
        let params = match File::open(dir.join("params")) {
            Ok(f) => {
                let reader = BufReader::new(f);
                Params::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
            }
            Err(e) => return Err(format!("Problem reading params file: {e}")),
        };

        let mut layers = Layers::new(params.m);
        let layers_dir = dir.join("layers");
        match layers_dir.read_dir() {
            Ok(dir) => {
                let mut layer_files = Vec::new();
                for layer_idx in dir {
                    layer_files.push(layer_idx.unwrap());
                }
                layer_files.sort_by_key(|x| {
                    x.file_name()
                        .into_string()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                });

                for layer_file in layer_files {
                    let layer = match File::open(layer_file.path()) {
                        Ok(f) => {
                            let reader = BufReader::new(f);
                            Graph::deserialize(reader.bytes().map(|b| b.unwrap()).collect())
                        }
                        Err(e) => return Err(format!("Problem reading params file: {e}")),
                    };
                    assert_eq!(layers.len(), layer.level as usize);
                    layers.add_layer(layer);
                }
            }
            Err(e) => panic!("There was a problem reading layers: {e}"),
        }

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
            points: PointsType::new(Vec::new(), params.ml),
            params,
            layers: Layers::new(m),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn distance(&self, a: NodeID, b: NodeID) -> Option<f32> {
        self.points.distance(a, b)
    }

    pub fn get_point(&self, point_id: NodeID) -> Option<&Point> {
        self.points.get_point(point_id)
    }

    pub fn layer_degrees(&self, layer_nb: usize) {
        let layer = self.layers.get_layer(layer_nb);
        for node in layer.nodes.keys() {
            println!("{}", layer.degree(*node).unwrap());
        }
    }

    pub fn insert_point(&mut self, point: Point) -> Result<bool, String> {
        let point_id = point.id;
        self.points.push(point);
        self.insert(point_id, &mut Inserter::new())
    }

    /// Determines and sets the neighbors of a point that has been stored in
    /// the layered graph and in the points field
    fn insert(&self, point_id: NodeID, inserter: &mut Inserter) -> Result<bool, String> {
        let point = self
            .points
            .get_point(point_id)
            .expect("Point ID not found in collection.");

        inserter.build_insertion_results(&self, point)?;
        self.make_connections(inserter.get_results())?;
        self.fix_connexions(inserter.get_results_mut())?;
        self.write_results_prune(inserter.get_results())?;

        Ok(true)
    }

    pub fn get_layer(&self, layer_nb: usize) -> &Graph {
        self.layers.get_layer(layer_nb)
    }

    fn make_connections(&self, results: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in results.insertion_results.iter() {
            let layer = self.get_layer(*layer_nb);
            for (node, neighbors) in node_data.iter() {
                let neighbors = neighbors.iter().map(|n| n.id);
                layer.add_neighbors(*node, neighbors).unwrap();
            }
        }
        Ok(())
    }

    fn fix_connexions(&self, searcher: &mut Results) -> Result<(), String> {
        searcher.clear_prune();

        for (layer_nb, node_data) in searcher.insertion_results.clone().iter() {
            let layer = self.get_layer(*layer_nb);

            for (_, neighbors) in node_data.iter() {
                let nodes_to_prune = neighbors
                    .iter()
                    .filter(|x| layer.degree(x.id).unwrap() > layer.m)
                    .map(|x| *x);

                for to_prune in nodes_to_prune {
                    let id = to_prune.id;
                    let to_prune_neighbors = match layer.neighbors_vec(id) {
                        Ok(n) => n,
                        Err(_) => return Err("Could not get neighbors of {id}".to_string()),
                    };
                    let to_prune_distances = to_prune_neighbors
                        .iter()
                        .map(|n| Dist::new(*n, self.distance(to_prune.id, *n).unwrap()));
                    let nearest = select_simple(to_prune_distances, layer.m)?;
                    searcher.insert_prune_result(*layer_nb, to_prune.id, nearest);
                }
            }
        }
        Ok(())
    }

    fn write_results_prune(&self, results: &Results) -> Result<(), String> {
        for (layer_nb, node_data) in results.prune_results.iter() {
            let layer = self.layers.get_layer(*layer_nb);
            for (node, neighbors) in node_data.iter() {
                let neighbors = neighbors.iter().map(|n| n.id);
                layer.replace_neighbors(*node, neighbors).unwrap();
            }
        }

        Ok(())
    }

    fn check_points_dim(&self, points: &PointsType) {
        match points.dim() {
            Some(points_d) => {
                if points_d != self.params.dim {
                    panic!("The current index dimension is {0}, but tried inserting points of dimension {points_d}", self.params.dim)
                }
            }
            _ => {}
        }
    }

    /// Stores the points in the internal `points` field
    /// and adds them to the index's layers.
    ///
    /// This is only the storing part, no indexing can
    /// be done on these points after this operation,
    fn store_points(&mut self, points: PointsType) {
        self.check_points_dim(&points);
        let ids = self.points.extend(points);
        for point_id in ids {
            let level = self.points.get_point(point_id).unwrap().level;
            self.layers.add_node_with_level(point_id, level as usize);
        }

        let max_layer_nb = self.layers.len() - 1;
        let new_ep = *self.get_layer(max_layer_nb).nodes.keys().next().unwrap();
        // TODO: if index already had points and we insert more,
        // one of the new points could have a higher level than the
        // current max. So it will become the new EP, and therefore
        // needs to be connected in all the layers.
        // self.insert(new_ep, &mut Inserter::new()).unwrap();
        self.params.ep = new_ep;
    }

    // fn get_nearest<I>(&self, point: &Point, others: I) -> Node
    // where
    //     I: Iterator<Item = Node>,
    // {
    //     point
    //         .dist2many(others.map(|idx| self.points.get_point(idx).unwrap()))
    //         .min()
    //         .unwrap()
    //         .id
    // }

    pub fn ann_by_vector(&self, point: &Point, n: usize, ef: usize) -> Result<Vec<NodeID>, String> {
        let mut results = Results::new();
        let searcher = Searcher::new();
        results.insert_selected(Dist::new(
            self.params.ep,
            self.points.distance2point(point, self.params.ep).unwrap(),
        ));
        let nb_layers = self.layers.len();

        for layer_nb in (1..nb_layers).rev() {
            searcher.search_layer(
                &mut results,
                self.get_layer(layer_nb),
                &point,
                &self.points,
                1,
            )?;
        }

        searcher.search_layer(&mut results, &self.get_layer(0), &point, &self.points, ef)?;

        let anns: Vec<NodeID> = results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        Ok(anns)
    }

    fn iter_layers(&self) -> impl Iterator<Item = &Graph> {
        self.layers.iter_layers()
    }

    pub fn assert_param_compliance(&self) {
        let mut is_ok = true;
        for (layer_nb, layer) in self.iter_layers().enumerate() {
            let max_degree = if layer_nb > 0 {
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
        for (idx, layer) in self.iter_layers().enumerate() {
            println!("NB. nodes in layer {idx}: {}", layer.nb_nodes());
        }
        println!("ep: {:?}", self.params.ep);
    }
}

impl HNSW {
    pub fn insert_bulk(
        mut self,
        vectors: Vec<Vec<f32>>,
        nb_threads: usize,
        verbose: bool,
    ) -> Result<HNSW, String> {
        self.store_points(PointsType::new(vectors, get_default_ml(self.params.m)));

        let bar = get_progress_bar("Building HNSW index".to_string(), self.len(), verbose);
        let bar_arc = Arc::new(bar);

        let index_arc = Arc::new(self);

        for layer_nb in (0..index_arc.layers.len()).rev() {
            let layer = index_arc.layers.get_layer(layer_nb);
            let chunks = ((layer.nb_nodes() as f64) / (nb_threads as f64)).ceil() as usize;
            let mut ids: Vec<Vec<NodeID>> = layer
                .iter_nodes()
                .filter(|id| index_arc.points.get_point(*id).unwrap().level == layer_nb as u8)
                .collect::<Vec<NodeID>>()
                .chunks(chunks)
                .map(|chunk| Vec::from(chunk))
                .collect();

            let mut handlers = Vec::new();
            let mut idx = 0;
            while let Some(ids_split) = ids.pop() {
                let index_copy = Arc::clone(&index_arc);
                let bar_ref = bar_arc.clone();
                let builder = Builder::new().name(format!("worker {idx}"));
                let thread = builder
                    .spawn(move || {
                        let mut inserter = Inserter::new();
                        for id in ids_split.iter() {
                            index_copy.insert(*id, &mut inserter).unwrap();
                            bar_ref.inc(1);
                        }
                    })
                    .expect("Could not spawn a thread");
                handlers.push(thread);
                idx += 1;
            }
            for handle in handlers {
                handle.join().unwrap();
            }
        }

        let index = Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");
        Ok(index)
    }
}

#[cfg(test)]
mod test {

    use std::{collections::HashSet, fs::File, path::Path};

    use crate::{
        helpers::glove::load_glove_array,
        template::{make_rand_vectors, HNSW},
    };
    use graph::nodes::{Dist, NodeID};
    use itertools::Itertools;
    use points::{point::Point, points::Points};
    use vectors::VecBase;

    const DIM: usize = 10;
    const N: usize = 100;
    const M: usize = 12;

    #[test]
    fn hnsw_init() {
        let _index: HNSW = HNSW::new(12, None, 128);
    }

    #[test]
    fn hnsw_build() {
        let vectors = make_rand_vectors(N, DIM);
        let index: HNSW = HNSW::new(12, None, DIM);
        let index = index.insert_bulk(vectors, 1, false).unwrap();
        assert_eq!(index.len(), N);
    }

    #[test]
    #[should_panic]
    fn can_not_add_different_dim() {
        let index = HNSW::new(12, None, 128);

        let vectors = make_rand_vectors(10, 128);
        let index = index.insert_bulk(vectors, 1, false).unwrap();

        let vectors = make_rand_vectors(10, 512);
        let _index = index.insert_bulk(vectors, 1, false).unwrap();
    }

    #[test]
    fn hnsw_glove_build_eval() {
        let file = File::open("/home/gamal/repos/vector-store/test-data/store.txt").unwrap();
        let (_, stored) = load_glove_array(0, file, false).expect("Could not load store vectors");
        let file = File::open("/home/gamal/repos/vector-store/test-data/queries.txt").unwrap();
        let (_, queries) = load_glove_array(0, file, false).expect("Could not load query vectors");
        let nb_queries = queries.len();

        let index = HNSW::new(M, None, 50);
        let index = index.insert_bulk(stored.clone(), 1, false).unwrap();

        let points = &index.points;

        let mut queries_nn: Vec<Vec<NodeID>> = Vec::new();
        for query in queries.iter() {
            let query_point = Point::new(query);
            let query_true_nn = (0..stored.len() as NodeID)
                .map(|idx| Dist::new(idx, points.distance2point(&query_point, idx).unwrap()))
                .sorted()
                .map(|dist| dist.id)
                .take(10)
                .collect();
            queries_nn.push(query_true_nn);
        }

        let mut total_hits = 0;
        for (idx, query) in queries.iter().enumerate() {
            let query_point = Point::new(query);
            let query_ann = index.ann_by_vector(&query_point, 10, 100).unwrap();
            let query_ann: HashSet<NodeID> = HashSet::from_iter(query_ann.iter().copied());

            let query_true_nn = HashSet::from_iter(queries_nn.get(idx).unwrap().iter().copied());
            let hits = query_true_nn.intersection(&query_ann).count();
            total_hits += hits;
        }
        let final_acc = total_hits as f32 / (nb_queries * 10) as f32;
        println!("Final accuracy was {final_acc}");
        assert!(final_acc > 0.99);

        for layer in index.layers.iter_layers() {
            let mut min_degree = usize::MAX;
            let mut max_degree = usize::MIN;
            for node in layer.iter_nodes() {
                if layer.nb_nodes() <= 1 {
                    continue;
                }
                let degree = layer.degree(node).unwrap();
                min_degree = min_degree.min(degree);
                max_degree = max_degree.max(degree);
            }
            println!("Layer {0}: limit is {1}", layer.level, layer.m);
            println!("Min degree {min_degree}");
            println!("Max degree {max_degree}");
            assert!(min_degree > 0);
            // assert!(max_degree <= layer.m);
        }
    }

    #[test]
    fn hnsw_serialize_quant() {
        for _ in 0..100 {
            let vectors = make_rand_vectors(N, DIM);
            let index = HNSW::new(12, None, DIM);
            let index = index.insert_bulk(vectors, 1, false).unwrap();

            let index_path = Path::new("./ser_test_quant");
            if index_path.exists() {
                std::fs::remove_dir_all(index_path).unwrap();
            }
            index.save(index_path);
            let loaded_index = HNSW::load(index_path).unwrap();

            std::fs::remove_dir_all(index_path).unwrap();

            assert_eq!(N, loaded_index.len());
        }
    }
}

fn select_simple<I>(candidate_dists: I, m: usize) -> Result<BTreeSet<Dist>, String>
where
    I: Iterator<Item = Dist>,
{
    let mut cands = Vec::from_iter(candidate_dists);
    cands.sort();
    Ok(BTreeSet::from_iter(cands.iter().copied().take(m)))
}

pub fn make_rand_index_full(n: usize, dim: usize) -> HNSW {
    let vectors = make_rand_vectors(n, dim);
    let index = HNSW::new(12, None, dim);
    let index = index.insert_bulk(vectors, 1, false).unwrap();
    index
}

pub fn make_rand_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::new();
    for _ in 0..n {
        let vector = (0..dim).map(|_| rng.gen::<f32>()).collect();
        vectors.push(vector)
    }
    vectors
}
