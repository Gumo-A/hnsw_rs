extern crate hnsw;

use std::sync::{Arc, Mutex};

use hnsw::hnsw::{dist::Dist, graph::Graph};
use indicatif::ProgressBar;
use nohash_hasher::{IntMap, IntSet};
use rand::Rng;

#[test]
fn serialization() {
    let n = 1000;
    let graph = make_rand_graph(n, 5);
    let (_nb_edges, graph_bytes) = graph.to_bytes_el();
    let graph_reconstructed = Graph::from_edge_list_bytes(&graph_bytes, &ProgressBar::hidden());

    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let rand_node = rng.gen_range(0..n) as u32;
        let neigh_1 = *graph.neighbors(rand_node).unwrap().iter().max().unwrap();
        let neigh_2 = *graph_reconstructed
            .neighbors(rand_node)
            .unwrap()
            .iter()
            .max()
            .unwrap();
        assert_eq!(neigh_1, neigh_2);
    }
}

fn make_rand_graph(n: usize, degree: usize) -> Graph {
    let mut rng = rand::thread_rng();
    let nodes =
        IntMap::from_iter((0..n).map(|id| (id as u32, Arc::new(Mutex::new(IntSet::default())))));
    let graph = Graph { nodes };
    for node in 0..n {
        for _ in 0..degree {
            let neighbor = rng.gen_range(0..n) as u32;
            graph
                .add_edge(
                    node as u32,
                    neighbor,
                    Dist::new(rng.gen::<f32>(), neighbor as u32),
                )
                .unwrap();
        }
    }
    graph
}
