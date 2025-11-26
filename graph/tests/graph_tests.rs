use graph::{graph::Graph, nodes::Node};
use nohash_hasher::{IntMap, IntSet};
use rand::seq::IteratorRandom;
use std::sync::{Arc, Mutex};

fn make_rand_graph(n: usize, degree: usize) -> Graph {
    let mut rng = rand::thread_rng();
    let nodes =
        IntMap::from_iter((0..n).map(|id| (id as Node, Arc::new(Mutex::new(IntSet::default())))));
    let graph = Graph { nodes };
    for node in 0..n {
        let neighbors = (0..n).choose_multiple(&mut rng, degree);
        for n in neighbors {
            graph.add_edge(node as Node, n as Node).unwrap();
        }
    }
    graph
}

#[test]
fn build_graph() {
    let graph = make_rand_graph(1_000, 86);
    assert!(graph.nodes.len() == 1_000);
}

#[test]
fn no_one_way_connections() {
    let graph = make_rand_graph(1_000, 86);
    for idx in 0..1_000 {
        let neighbors = graph.neighbors(idx).unwrap();
        for n in neighbors {
            println!("checking if {idx}'s neighbor's ({0}) contains {idx}", n);
            let neighs_neighs = graph.neighbors(n).unwrap();
            print!("{neighs_neighs:?}");
            assert!(neighs_neighs.contains(&idx));
        }
    }
}
