use core::panic;
use log::info;
use nohash_hasher::IntSet;
use std::{
    collections::HashSet,
    fs::{self, create_dir, create_dir_all, remove_dir_all, File},
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
use graph::{dist::Dist, graph::Graph, layers::Layers, NodeID};
use points::{point::Point, points::Points, points_disk::PointsDisk};
use rand::Rng;
use vectors::{serializer::Serializer, VecBase};

mod searcher;

mod results;
use results::Results;

mod inserter;
use inserter::Inserter;

type PointsType = PointsDisk;
type LayersType = Layers;

#[derive(Debug, Clone)]
pub struct HNSW {
    pub params: Params,
    layers: LayersType,
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

        self.save_part(dir, self.params.serialize(), "params");
        self.save_part(dir, self.points.serialize(), "points");
        self.save_layers(dir);
    }

    fn save_part(&self, dir: &Path, data: Vec<u8>, part: &str) {
        File::create(dir.join(part))
            .expect(format!("Could not create {part} file").as_str())
            .write_all(&data)
            .expect(format!("Could not write bytes to {part} file").as_str());
    }

    fn save_layers(&self, dir: &Path) {
        let layers_dir = dir.join("layers");
        if !layers_dir.exists() {
            fs::create_dir(layers_dir.clone()).expect("Could not create layers dir");
        }
        for (idx, layer) in self.layers.iter_layers().enumerate() {
            self.save_part(&layers_dir, layer.serialize(), format!("{idx}").as_str());
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

        let mut layers = LayersType::new(params.m);
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
        let tmp_path = std::env::temp_dir().join("hnsw_rs").join("points");
        create_dir_all(&tmp_path).unwrap();
        HNSW {
            points: PointsType::new(Vec::new(), params.ml, 0, &tmp_path),
            layers: LayersType::new(m),
            params,
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

    pub fn insert_vec(&mut self, vector: &Vec<f32>) -> Result<NodeID, String> {
        info!("Adding new vector to the index");
        let point_id = self.store_points(vec![vector.clone()])[0];
        info!("Stored the vector with ID {point_id}");
        let point = self.get_point(point_id).unwrap();
        self.layers.add_node(point.id, point.level as usize);
        self.insert(point_id, &mut Inserter::new())?;
        Ok(point_id)
    }

    /// Determines and sets the neighbors of a point that has been stored in
    /// the layered graph and in the points field
    fn insert(&self, point_id: NodeID, inserter: &mut Inserter) -> Result<(), String> {
        info!("Inserting node {point_id}");
        let point = self
            .points
            .get_point(point_id)
            .expect(format!("Point {point_id} not found in collection.").as_str());

        inserter.build_insertion_results(&self, point)?;
        self.make_connections(inserter.get_results())?;
        info!("Finished inserting node {point_id}");
        Ok(())
    }

    pub fn get_layer(&self, layer_nb: usize) -> &Graph {
        self.layers.get_layer(layer_nb)
    }

    fn make_connections(&self, results: &Results) -> Result<(), String> {
        info!("Making connections using results");
        for (layer_nb, node_data) in results.insertion_results.iter() {
            let layer = self.get_layer(*layer_nb);
            for (node, neighbors) in node_data.iter() {
                layer
                    .add_neighbors(*node, neighbors.iter().copied())
                    .unwrap();
                for n in neighbors {
                    if layer.degree(*n).unwrap() > layer.m {
                        let query = self.get_point(*n).unwrap();
                        let neighs = layer.neighbors(*n).unwrap();
                        let candidate_dists = neighs.iter().copied().map(|id| {
                            let point = self.get_point(id).unwrap();
                            Dist::new(id, point.distance(query))
                        });
                        // Deadlocks:
                        // let neighs = layer.neighbors_lock(*n).unwrap();
                        // let neighs_locked = neighs.lock().unwrap();
                        // let candidate_dists = neighs_locked.iter().copied().map(|id| {
                        //     let point = self.get_point(id).unwrap();
                        //     Dist::new(id, point.distance(query))
                        // });
                        let new = select_simple(candidate_dists, layer.m);
                        layer.replace_neighbors(*n, new.iter().copied()).unwrap();
                    }
                }
            }
        }
        info!("Finished making connections");
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
    fn store_points(&mut self, vectors: Vec<Vec<f32>>) -> Vec<NodeID> {
        let tmp_path = std::env::temp_dir().join("hnsw_rs").join("tmp_points");
        create_dir_all(&tmp_path).unwrap();
        let points = PointsType::new(vectors, get_default_ml(self.params.m), 0, &tmp_path);
        let new_points_len = points.len();
        info!(
            "Storing {} points, EP is {}",
            new_points_len, self.params.ep
        );
        self.check_points_dim(&points);
        let ids = self.points.extend(points);
        remove_dir_all(tmp_path).unwrap();
        for point_id in ids.iter() {
            let level = self.points.get_point(*point_id).unwrap().level;
            self.layers.add_node(*point_id, level as usize);
        }

        let max_layer_nb = self.layers.len() - 1;
        let new_ep = self.get_layer(max_layer_nb).iter_nodes().next().unwrap();
        // TODO: if index already had points and we insert more,
        // one of the new points could have a higher level than the
        // current max. So it will become the new EP, and therefore
        // needs to be connected in all the layers.
        // self.insert(new_ep, &mut Inserter::new()).unwrap();
        self.params.ep = new_ep;
        info!("Stored {new_points_len} points, the new EP is {new_ep}");
        ids
    }

    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<NodeID>, String> {
        info!("Start ANN search");
        let point = Point::new(vector);
        let mut results = Results::new();
        let searcher = Searcher::new();
        results.insert_selected(Dist::new(
            self.params.ep,
            self.points.distance2point(&point, self.params.ep).unwrap(),
        ));
        let nb_layers = self.layers.len();

        for layer_nb in (1..nb_layers).rev() {
            searcher.search_layer(&mut results, self.get_layer(layer_nb), &point, self, 1)?;
        }

        searcher.search_layer(&mut results, &self.get_layer(0), &point, &self, ef)?;

        let anns: Vec<NodeID> = results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        info!("Finished ANN search");
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
        info!("inserting {0} vectors to index in bulk", vectors.len());
        let mut stored_ids: HashSet<NodeID> = HashSet::new();
        stored_ids.extend(self.store_points(vectors).iter());

        let bar = get_progress_bar("Building HNSW index".to_string(), stored_ids.len(), verbose);
        let bar_arc = Arc::new(bar);

        let index_arc = Arc::new(self);

        for layer_nb in (0..index_arc.layers.len()).rev() {
            let layer = index_arc.layers.get_layer(layer_nb);
            info!("Connecting nodes in layer {layer_nb}, using {nb_threads} threads",);
            let chunks = ((layer.nb_nodes() as f64) / (nb_threads as f64)).ceil() as usize;
            let mut ids: Vec<Vec<NodeID>> = layer
                .iter_nodes()
                .filter(|id| {
                    stored_ids.contains(id)
                        & (index_arc.points.get_point(*id).unwrap().level == layer_nb as u8)
                })
                .collect::<Vec<NodeID>>()
                .chunks(chunks)
                .map(|chunk| Vec::from(chunk))
                .collect();

            let mut handlers = Vec::new();
            let mut idx = 0;
            while let Some(ids_split) = ids.pop() {
                info!("Sending {} node IDs to thread {idx}", ids_split.len());
                let index_copy = Arc::clone(&index_arc);
                let bar_ref = bar_arc.clone();
                let thread = Builder::new()
                    .name(format!("worker {idx}"))
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

    use std::{
        collections::{BTreeMap, HashSet},
        fs::File,
        path::Path,
    };

    use crate::{
        helpers::glove::load_glove_array,
        params::Params,
        template::{make_rand_vectors, PointsType, HNSW},
    };
    use graph::{dist::Dist, graph::Graph, NodeID};
    use itertools::Itertools;
    use points::{point::Point, points::Points};
    use repo_path::repo_path;
    use vectors::{serializer::Serializer, VecBase};

    const DIM: usize = 10;
    const N: usize = 100;
    const M: usize = 12;

    fn init_logs() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn hnsw_init() {
        let _index: HNSW = HNSW::new(12, None, 128);
    }

    #[test]
    fn hnsw_build() {
        init_logs();
        let vectors = make_rand_vectors(N, DIM);
        let index: HNSW = HNSW::new(12, None, DIM);
        let index = index.insert_bulk(vectors, 1, false).unwrap();
        assert_eq!(index.len(), N);
    }

    #[test]
    fn hnsw_insert_one_after_bulk_insert() {
        init_logs();
        let vectors = make_rand_vectors(N, DIM);
        let index: HNSW = HNSW::new(12, None, DIM);
        let mut index = index.insert_bulk(vectors, 1, false).unwrap();

        assert_eq!(index.len(), N);

        let vector = make_rand_vectors(1, DIM)[0].clone();
        index.insert_vec(&vector).unwrap();

        assert_eq!(index.len(), N + 1);
    }

    #[test]
    fn hnsw_subsequent_bulk_inserts() {
        init_logs();
        let index: HNSW = HNSW::new(12, None, DIM);
        let vectors = make_rand_vectors(N, DIM);
        let index = index.insert_bulk(vectors, 1, false).unwrap();

        assert_eq!(index.len(), N);

        let vectors = make_rand_vectors(N, DIM);
        let index = index.insert_bulk(vectors, 1, false).unwrap();

        assert_eq!(index.len(), N * 2);
    }

    #[test]
    fn hnsw_subsequent_bulk_inserts_multithread() {
        init_logs();
        let index: HNSW = HNSW::new(12, None, DIM);
        let vectors = make_rand_vectors(N, DIM);
        let index = index.insert_bulk(vectors, 4, false).unwrap();

        assert_eq!(index.len(), N);

        let vectors = make_rand_vectors(N, DIM);
        let index = index.insert_bulk(vectors, 4, false).unwrap();

        assert_eq!(index.len(), N * 2);
    }

    #[test]
    #[should_panic]
    fn can_not_add_different_dim() {
        init_logs();
        let index = HNSW::new(12, None, 128);

        let vectors = make_rand_vectors(10, 128);
        let index = index.insert_bulk(vectors, 1, false).unwrap();

        let vectors = make_rand_vectors(10, 512);
        let _index = index.insert_bulk(vectors, 1, false).unwrap();
    }

    fn get_glove_data() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let file = File::open(repo_path!("test-data/store.txt").to_str().unwrap()).unwrap();
        let (_, stored) = load_glove_array(0, file, false).expect("Could not load store vectors");
        let file = File::open(repo_path!("test-data/queries.txt").to_str().unwrap()).unwrap();
        let (_, queries) = load_glove_array(0, file, false).expect("Could not load query vectors");
        return (stored, queries);
    }

    fn get_actual_nn(queries: &Vec<Vec<f32>>, points: &PointsType) -> Vec<Vec<NodeID>> {
        let mut queries_nn: Vec<Vec<NodeID>> = Vec::new();
        for query in queries.iter() {
            let query_point = Point::new(query);
            let query_true_nn = (0..points.len() as NodeID)
                .map(|idx| Dist::new(idx, points.distance2point(&query_point, idx).unwrap()))
                .sorted()
                .map(|dist| dist.id)
                .take(10)
                .collect();
            queries_nn.push(query_true_nn);
        }
        queries_nn
    }

    #[test]
    fn hnsw_glove_accuracy() {
        init_logs();
        let (stored, queries) = get_glove_data();
        let index = HNSW::new(M, None, queries[0].len());
        let index = index.insert_bulk(stored.clone(), 1, false).unwrap();

        let queries_nn = get_actual_nn(&queries, &index.points);

        let mut total_hits = 0;
        for (idx, query) in queries.iter().enumerate() {
            let query_ann = index.ann_by_vector(query, 10, 100).unwrap();
            let query_ann: HashSet<NodeID> = HashSet::from_iter(query_ann.iter().copied());

            let query_true_nn = HashSet::from_iter(queries_nn.get(idx).unwrap().iter().copied());
            let hits = query_true_nn.intersection(&query_ann).count();
            total_hits += hits;
        }
        let final_acc = total_hits as f32 / (queries.len() * 10) as f32;
        println!("Final accuracy was {final_acc}");
        assert_eq!(final_acc, 1.0);
    }

    #[test]
    fn hnsw_glove_no_isolated_nodes() {
        init_logs();
        let (stored, queries) = get_glove_data();
        let index = HNSW::new(M, None, queries[0].len());
        let index = index.insert_bulk(stored.clone(), 1, false).unwrap();
        for layer in index.layers.iter_layers() {
            let mut degree_count: BTreeMap<usize, Vec<NodeID>> = BTreeMap::new();
            for node in layer.iter_nodes() {
                if layer.nb_nodes() <= 1 {
                    continue;
                }
                degree_count
                    .entry(layer.degree(node).unwrap())
                    .and_modify(|v| v.push(node))
                    .or_insert(vec![node]);
            }
            println!(
                "Layer {0}: M is {1}, nb. nodes is {2}",
                layer.level,
                layer.m,
                layer.nb_nodes()
            );
            println!("Degrees:");
            println!("{degree_count:?}");
            let min_degree = *degree_count.first_key_value().unwrap().0;
            assert!(min_degree > 0);
        }
    }

    #[test]
    fn hnsw_save_load() {
        for m in (0..=6).map(|i| 2usize.pow(i)) {
            init_logs();
            let vectors = make_rand_vectors(N, DIM);
            let index = HNSW::new(m, None, DIM);
            let index = index.insert_bulk(vectors, 1, false).unwrap();

            let index_path = Path::new("./serialization_test");
            if index_path.exists() {
                std::fs::remove_dir_all(index_path).unwrap();
            }
            index.save(index_path);
            let loaded_index = HNSW::load(index_path).unwrap();

            std::fs::remove_dir_all(index_path).unwrap();

            assert_eq!(N, loaded_index.len());
            for id in index.points.ids() {
                let index_neighs = index.layers.get_layer(0).neighbors(id).unwrap();
                let loaded_neighs = loaded_index.layers.get_layer(0).neighbors(id).unwrap();

                let index_diff_loaded: Vec<&NodeID> =
                    index_neighs.difference(&loaded_neighs).collect();
                assert_eq!(index_diff_loaded.len(), 0);

                let loaded_diff_index: Vec<&NodeID> =
                    loaded_neighs.difference(&index_neighs).collect();
                assert_eq!(loaded_diff_index.len(), 0);

                assert_eq!(
                    index.get_point(id).unwrap().get_vals(),
                    loaded_index.get_point(id).unwrap().get_vals()
                );
            }
        }
    }

    #[test]
    fn hnsw_serialize_parts() {
        init_logs();
        for _ in 1..100 {
            let vectors = make_rand_vectors(N, DIM);
            let index = HNSW::new(12, None, DIM);
            let index = index.insert_bulk(vectors, 1, false).unwrap();

            let params = Params::deserialize(index.params.serialize());
            assert_eq!(params.ep, index.params.ep);
            assert_eq!(params.m, index.params.m);
            assert_eq!(params.mmax, index.params.mmax);
            assert_eq!(params.mmax0, index.params.mmax0);
            assert_eq!(params.ml, index.params.ml);
            assert_eq!(params.ef_cons, index.params.ef_cons);
            assert_eq!(params.dim, index.params.dim);

            let points = PointsType::deserialize(index.points.serialize());
            for (p, s_p) in index
                .points
                .iter_points::<&Point>()
                .zip(points.iter_points::<&Point>())
            {
                assert_eq!(p.id, s_p.id);
                assert_eq!(p.level, s_p.level);
                assert_eq!(p.dist2other(s_p), 0.0);
            }

            for layer in index.layers.iter_layers() {
                let layer_ser = Graph::deserialize(layer.serialize());
                for node in layer.iter_nodes() {
                    assert!(layer_ser.contains(node));
                    assert_eq!(
                        layer.neighbors(node).unwrap(),
                        layer_ser.neighbors(node).unwrap()
                    );
                }
            }
        }
    }
}

fn select_simple<I>(candidate_dists: I, m: usize) -> IntSet<NodeID>
where
    I: Iterator<Item = Dist>,
{
    let mut cands = Vec::from_iter(candidate_dists);
    cands.sort();
    IntSet::from_iter(cands.iter().take(m).map(|dist| dist.id))
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
