pub mod graph;
pub mod helpers;

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use graph::Graph;
    use ndarray::{Array1, Array2};
    use rand::Rng;

    #[test]
    fn graph_operations() {
        let mut rng = rand::thread_rng();
        let n = 10;
        let mut g = Graph {
            nodes: HashMap::new(),
            node_vectors: HashMap::new(),
            self_connexions: false,
        };

        let nodes = Vec::from_iter(0..n);
        let mut node_vectors: Vec<Array1<f32>> = vec![];
        for _ in 0..n {
            let vector = (0..100).map(|_| rng.gen::<f32>()).collect();
            let vect_arr = Array1::from_vec(vector);
            node_vectors.push(vect_arr);
        }

        for (node, vector) in nodes.iter().zip(node_vectors) {
            g.add_node(*node, vector)
        }

        let mut edges: Vec<(i32, i32)> = (0..n * 2)
            .map(|_| (rng.gen_range(0..n), rng.gen_range(0..n)))
            .collect();

        edges.push((0, 9));

        for edge in edges.iter() {
            g.add_edge(edge.0, edge.1);
        }

        g.remove_edge(9, 0);

        // run test with -- --nocapture to display
        println!("edges: {:?}", edges);
        for node in nodes.iter() {
            println!("Neighbors of {node} {:?}", g.neighbors(*node));
        }
    }
}
