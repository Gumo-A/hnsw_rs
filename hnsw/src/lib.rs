pub mod graph;
pub mod helpers;
pub mod hnsw;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use graph::Graph;
    use helpers::data::load_bf_data;
    use helpers::glove::load_glove_array;
    use ndarray::{Array1, Array2};
    use rand::Rng;

    #[test]
    fn regex() {
        let re = regex::Regex::new(r"\d+").unwrap();
        let vals = ["layer_3.json", "layer_24.json", "layer_842.json"];
        let tests = ["3", "24", "842"];
        for i in vals.iter() {
            let result = re.find(i).unwrap().as_str();
            assert!(tests.contains(&result));
            // println!("{}", result);
        }
    }

    #[test]
    fn check_brute_force() {
        // must be run with -- --nocapture
        let mut rng = rand::thread_rng();
        let (dim, lim) = (100, 10_000);
        let bf_data = load_bf_data(dim.try_into().unwrap(), lim.try_into().unwrap()).unwrap();
        let (words, _): (Vec<String>, Array2<f32>) =
            load_glove_array(dim.try_into().unwrap(), lim.try_into().unwrap(), true, 0)
                .expect("Words loading failed");

        for _ in 0..3 {
            let rnd_idx = rng.gen_range(0..10_000);
            let nns = bf_data[&rnd_idx].clone();
            let nns_words: Vec<String> = nns.iter().map(|i| words[*i].clone()).take(10).collect();
            println!("NNs of '{}':", words[rnd_idx]);
            println!("{:?}", nns_words);
        }
    }

    #[test]
    fn graph_add_nodes() {
        let n = 10;
        let mut rng = rand::thread_rng();
        let mut g = Graph {
            nodes: HashMap::new(),
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
            g.add_node(*node, &vector)
        }

        assert_eq!(g.nb_nodes(), n as usize);
    }

    #[test]
    fn graph_add_remove_edges() {
        let n = 10;
        let mut rng = rand::thread_rng();
        let mut g = Graph {
            nodes: HashMap::new(),
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
            g.add_node(*node, &vector)
        }

        let mut edges: Vec<(usize, usize)> = (0..n * 2)
            .map(|_| (rng.gen_range(0..n), rng.gen_range(0..n)))
            .collect();

        edges.push((0, 9));
        for edge in edges.iter() {
            g.add_edge(edge.0, edge.1);
        }
        assert!(g.neighbors(0).contains(&9));
        assert!(g.neighbors(9).contains(&0));
        g.remove_edge(9, 0);
        assert!(!g.neighbors(0).contains(&9));
        assert!(!g.neighbors(9).contains(&0));
    }

    #[test]
    fn graph_vec_operation() {
        let n = 10;
        let mut rng = rand::thread_rng();
        let mut g = Graph {
            nodes: HashMap::new(),
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
            g.add_node(*node, &vector);
        }

        let mut dist = helpers::distance::v2v_dist(&g.node(0).1, &g.node(0).1);
        dist = (dist * 10000.0).round() / 10000.0;

        assert_eq!(dist, 0.0);
    }
}
