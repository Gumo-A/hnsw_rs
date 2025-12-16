use crate::errors::GraphError;
use crate::nodes::Node;

use nohash_hasher::{IntMap, IntSet};
use rand::seq::IteratorRandom;
use std::sync::{Arc, Mutex};
use vectors::serializer::Serializer;

type Neighbors = Arc<Mutex<IntSet<Node>>>;

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: IntMap<Node, Neighbors>,
    pub level: u8,
    // TODO: for fixed max nb of neighbors.
    // this would help make disk ops by fixing the size of all
    // neighbor lists on disk.
    // pub m: u16
}

impl Graph {
    pub fn new(level: u8) -> Self {
        Graph {
            nodes: IntMap::default(),
            level,
        }
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = Node> {
        self.nodes.keys().copied()
    }

    pub fn add_node(&mut self, point_id: Node) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    pub fn add_edge(&self, node_a: Node, node_b: Node) -> Result<(), GraphError> {
        if node_a == node_b {
            return Err(GraphError::SelfConnection(node_a));
        }

        let (a, b) = self.get_nodes_neighbors(node_a, node_b)?;

        {
            a.lock().unwrap().insert(node_b);
        }
        {
            b.lock().unwrap().insert(node_a);
        }
        Ok(())
    }

    fn get_nodes_neighbors(
        &self,
        node_a: Node,
        node_b: Node,
    ) -> Result<(&Neighbors, &Neighbors), GraphError> {
        match (self.nodes.get(&node_a), self.nodes.get(&node_b)) {
            (Some(a), Some(b)) => Ok((a, b)),
            (Some(_), None) => {
                return Err(GraphError::NodeNotInGraph(node_b));
            }
            _ => {
                return Err(GraphError::NodeNotInGraph(node_a));
            }
        }
    }

    /// Removes an edge from the Graph.
    /// Since the add_edge method won't allow for self-connecting nodes, we don't check that here.
    ///
    /// Returns error if one node doesn't exist or if removing would isolate a node.
    pub fn remove_edge(&self, node_a: Node, node_b: Node) -> Result<(), GraphError> {
        let (a, b) = self.get_nodes_neighbors(node_a, node_b)?;

        if self.degree(node_a).unwrap() == 1 {
            return Err(GraphError::WouldIsolateNode(node_a));
        }
        if self.degree(node_b).unwrap() == 1 {
            return Err(GraphError::WouldIsolateNode(node_b));
        }

        {
            a.lock().unwrap().remove(&node_b);
        }
        {
            b.lock().unwrap().remove(&node_a);
        }

        Ok(())
    }

    pub fn neighbors(&self, node: Node) -> Result<IntSet<Node>, GraphError> {
        match self.nodes.get(&node) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(GraphError::NodeNotInGraph(node)),
        }
    }

    pub fn neighbors_vec(&self, node: Node) -> Result<Vec<Node>, GraphError> {
        let neighbors = match self.nodes.get(&node) {
            Some(neighbors) => neighbors,
            None => {
                return Err(GraphError::NodeNotInGraph(node));
            }
        };

        Ok(neighbors.lock().unwrap().iter().cloned().collect())
    }

    pub fn replace_neighbors<I>(&self, node: Node, new_neighbors: I) -> Result<(), GraphError>
    where
        I: Iterator<Item = Node>,
    {
        if self.degree(node)? == 0 {
            for other in new_neighbors {
                self.add_edge(node, other)?;
            }
            return Ok(());
        }
        let news = IntSet::from_iter(new_neighbors);
        let olds = self.neighbors(node)?;

        let to_remove: Vec<Node> = olds.difference(&news).copied().collect();
        let to_add: Vec<Node> = news.difference(&olds).copied().collect();

        for new_neighbor in to_add {
            self.add_edge(node, new_neighbor)?;
        }

        for ex_neighbor in to_remove {
            if let Err(e) = self.remove_edge(node, ex_neighbor) {
                match e {
                    GraphError::WouldIsolateNode(n) => {
                        // println!(
                        //     "Was going to remove edge {node}-{ex_neighbor}, but didn't because it would leave {n} isolated"
                        // );
                    }
                    _ => return Err(e),
                }
            };
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node: Node) -> Result<(), GraphError> {
        for ex_neighbor in self.neighbors(node)? {
            self.nodes
                .get_mut(&ex_neighbor)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node);
        }
        self.nodes.remove(&node);
        Ok(())
    }

    fn remove_edges_with_node(&mut self, node: Node) -> Result<(), GraphError> {
        for ex_neighbor in self.remove_neighbors(node)? {
            self.nodes
                .get_mut(&ex_neighbor)
                .unwrap()
                .lock()
                .unwrap()
                .remove(&node);
        }
        Ok(())
    }

    fn remove_neighbors(&mut self, node: Node) -> Result<IntSet<Node>, GraphError> {
        let removed = self.nodes.remove(&node);
        self.nodes
            .insert(node, Arc::new(Mutex::new(IntSet::default())));
        match removed {
            Some(neighbors) => Ok(Arc::into_inner(neighbors).unwrap().into_inner().unwrap()),
            None => Err(GraphError::NodeNotInGraph(node)),
        }
    }

    pub fn degree(&self, node: Node) -> Result<usize, GraphError> {
        match self.nodes.get(&node) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().len()),
            None => Err(GraphError::NodeNotInGraph(node)),
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node_id: Node) -> bool {
        self.nodes.contains_key(&node_id)
    }

    fn node_size(&self, node_id: Node) -> usize {
        6 + (self.degree(node_id).unwrap() * 4)
    }

    /// Val          Bytes
    /// Node         4
    /// nb_neighbors 2
    /// neighbors    nb_neighbors * 4
    fn serialize_adj_list(&self, node_id: Node) -> Vec<u8> {
        let neighbors = self.neighbors_vec(node_id).unwrap();
        let mut bytes = Vec::with_capacity(neighbors.len() + 1);
        bytes.extend_from_slice(&node_id.to_be_bytes());
        bytes.extend_from_slice(&(neighbors.len() as u16).to_be_bytes());
        for n in neighbors {
            bytes.extend_from_slice(&n.to_be_bytes());
        }
        bytes
    }

    /// Val          Bytes
    /// neighbors    nb_neighbors * 4
    fn deserialize_neighbors(data: &[u8]) -> Vec<Node> {
        let nb_neighbors = data.len() / 4;
        let mut neighbors = Vec::with_capacity(nb_neighbors);
        let mut i = 0;
        for _ in 0..nb_neighbors {
            let n: Node = u32::from_be_bytes(data[i..i + 4].try_into().unwrap());
            i += 4;
            neighbors.push(n);
        }
        neighbors
    }
}

impl Serializer for Graph {
    fn size(&self) -> usize {
        let mut size = 5;
        for node in self.iter_nodes() {
            size += self.node_size(node);
        }
        size
    }

    /// Val          Bytes
    /// level        1
    /// nb_nodes     4
    /// adj_list     nb_nodes * variable
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.nb_nodes() * 4);
        bytes.extend_from_slice(&[self.level]);
        bytes.extend_from_slice(&(self.nb_nodes() as u32).to_be_bytes());
        for node in self.iter_nodes() {
            bytes.extend(self.serialize_adj_list(node));
        }
        bytes
    }

    /// Val          Bytes
    /// level        1
    /// nb_nodes     4
    /// adj_list     nb_nodes * variable
    fn deserialize(data: Vec<u8>) -> Graph {
        let mut i = 1;
        let level = u8::from_be_bytes(data[..i].try_into().unwrap());
        let nb_nodes = u32::from_be_bytes(data[i..i + 4].try_into().unwrap());
        i += 4;
        let mut nodes = IntMap::default();
        for _ in 0..nb_nodes {
            let node = u32::from_be_bytes(data[i..i + 4].try_into().unwrap());
            i += 4;
            let nb_neighbors = u16::from_be_bytes(data[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            let neighbors = IntSet::from_iter(
                Graph::deserialize_neighbors(&data[i..i + (nb_neighbors * 4)])
                    .iter()
                    .copied(),
            );
            i += nb_neighbors * 4;

            nodes.insert(node, Arc::new(Mutex::new(neighbors)));
        }

        Graph { nodes, level }
    }
}

pub fn make_rand_graph(n: usize, degree: usize) -> Graph {
    let mut rng = rand::thread_rng();
    let nodes =
        IntMap::from_iter((0..n).map(|id| (id as Node, Arc::new(Mutex::new(IntSet::default())))));
    let graph = Graph { nodes, level: 0 };
    for node in 0..n {
        let neighbors = (0..n).choose_multiple(&mut rng, degree);
        for n in neighbors {
            match graph.add_edge(node as Node, n as Node) {
                Err(_) => continue,
                Ok(_) => continue,
            }
        }
    }
    graph
}

pub fn simple_graph() -> Graph {
    let mut g = Graph::new(1);
    for i in 0..5 {
        g.add_node(i as Node);
    }
    g.add_edge(0, 1).unwrap();
    g.add_edge(0, 2).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(3, 4).unwrap();
    g.add_edge(4, 1).unwrap();
    g
}

#[cfg(test)]
mod test {
    use crate::{
        graph::{Graph, make_rand_graph, simple_graph},
        nodes::Node,
    };
    use nohash_hasher::IntSet;
    use std::sync::Arc;
    use vectors::serializer::Serializer;

    #[test]
    fn build_graph() {
        let graph = make_rand_graph(100, 8);
        assert!(graph.nodes.len() == 100);
    }

    #[test]
    fn no_one_way_connections() {
        let graph = make_rand_graph(100, 8);
        for idx in 0..100 {
            let neighbors = graph.neighbors(idx).unwrap();
            for n in neighbors {
                let neighs_neighs = graph.neighbors(n).unwrap();
                assert!(neighs_neighs.contains(&idx));
            }
        }
    }

    #[test]
    fn add_node_and_contains() {
        let mut g = Graph::new(0);
        assert!(!g.contains(42));
        g.add_node(42);
        assert!(g.contains(42));
        assert_eq!(g.nb_nodes(), 1);
        assert_eq!(g.degree(42).unwrap(), 0);
    }

    #[test]
    fn add_edge_rejects_missing_nodes() {
        let mut g = Graph::new(0);
        g.add_node(1);
        assert!(g.add_edge(1, 2).is_err());
        assert!(g.add_edge(999, 1000).is_err());
    }

    #[test]
    fn no_self_loops() {
        let mut g = Graph::new(0);
        g.add_node(5);
        assert!(g.add_edge(5, 5).is_err());
    }

    #[test]
    fn remove_edge_success_and_failure() {
        let g = simple_graph();

        // Existing edge
        g.remove_edge(0, 1).unwrap();
        assert!(!g.neighbors(0).unwrap().contains(&1));
        assert!(!g.neighbors(1).unwrap().contains(&0));

        // Missing node
        assert!(g.remove_edge(0, 999).is_err());
    }

    #[test]
    fn can_not_leave_node_isolated() {
        let mut g = Graph::new(0);
        g.add_node(10);
        g.add_node(20);
        g.add_node(30);

        // Now 20 has degree 2, 10 and 30 have degree 1
        g.add_edge(10, 20).unwrap();
        g.add_edge(20, 30).unwrap();

        // Can't remove edge, it would leave 30 isolated
        assert!(g.remove_edge(20, 30).is_err());
    }

    #[test]
    fn neighbors_and_neighbors_vec() {
        let g = simple_graph();
        let neigh_set = g.neighbors(1).unwrap();
        let neigh_vec = g.neighbors_vec(1).unwrap();

        let expected: IntSet<Node> = [0, 2, 4].into_iter().collect();
        assert_eq!(neigh_set, expected);

        assert_eq!(neigh_vec.len(), 3);
        assert!(neigh_vec.contains(&0));
        assert!(neigh_vec.contains(&2));
        assert!(neigh_vec.contains(&4));
    }

    #[test]
    fn replace_neighbors_full_replace() {
        let mut g = Graph::new(0);
        for i in 0..6 {
            g.add_node(i);
        }

        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();

        // So that nodes are not isolated after replacement
        g.add_edge(1, 3).unwrap();
        g.add_edge(1, 2).unwrap();

        assert_eq!(g.degree(0).unwrap(), 3);

        // Replace with completely different neighbors
        g.replace_neighbors(0, 4..=5).unwrap();

        let new_neigh = g.neighbors(0).unwrap();
        assert_eq!(new_neigh.len(), 2);
        assert!(new_neigh.contains(&4));
        assert!(new_neigh.contains(&5));
        assert!(!new_neigh.contains(&1));

        // Old neighbors should no longer point back (symmetry preserved)
        assert!(!g.neighbors(1).unwrap().contains(&0));
        assert!(!g.neighbors(2).unwrap().contains(&0));
        assert!(g.neighbors(4).unwrap().contains(&0));
        assert!(g.neighbors(5).unwrap().contains(&0));
    }

    #[test]
    fn replace_neighbors_on_isolated_node() {
        let mut g = Graph::new(0);
        g.add_node(100);
        for i in 200..205 {
            g.add_node(i);
        }

        g.replace_neighbors(100, 200..205).unwrap();

        assert_eq!(g.degree(100).unwrap(), 5);
        for i in 200..205 {
            assert!(g.neighbors(i).unwrap().contains(&100));
        }
    }

    #[test]
    fn remove_node_cleans_up_all_edges() {
        let mut g = simple_graph();
        g.remove_node(1).unwrap();

        // Node 1 gone
        assert!(!g.contains(1));
        assert_eq!(g.nb_nodes(), 4);

        // All former neighbors of 1 no longer have it
        for node in [0, 2, 4] {
            assert!(!g.neighbors(node).unwrap().contains(&1));
        }

        // Other edges preserved
        assert!(g.neighbors(0).unwrap().contains(&2));
        assert!(g.neighbors(3).unwrap().contains(&4));
    }

    #[test]
    fn remove_edges_with_node_leaves_isolated_node() {
        let mut g = simple_graph();

        g.remove_edges_with_node(2);

        // Node 2 still exists but isolated
        assert!(g.contains(2));
        assert_eq!(g.degree(2).unwrap(), 0);

        // Neighbors no longer connected to 2
        assert!(!g.neighbors(0).unwrap().contains(&2));
        assert!(!g.neighbors(1).unwrap().contains(&2));
        assert!(!g.neighbors(3).unwrap().contains(&2));
    }

    #[test]
    fn degree_on_missing_node_errors() {
        let g = Graph::new(0);
        assert!(g.degree(999).is_err());
    }

    #[test]
    fn serialization_round_trip() {
        let original = simple_graph();
        let original_bytes = original.serialize();

        assert_eq!(original_bytes.len(), original.size());

        let restored = Graph::deserialize(original_bytes.try_into().unwrap());

        assert_eq!(original.nb_nodes(), restored.nb_nodes());
        assert_eq!(original.level, restored.level);

        for node in original.iter_nodes() {
            assert!(restored.contains(node));
            let orig_neigh: IntSet<Node> = original.neighbors(node).unwrap().into_iter().collect();
            let rest_neigh: IntSet<Node> = restored.neighbors(node).unwrap().into_iter().collect();
            assert_eq!(orig_neigh, rest_neigh);
        }
    }

    #[test]
    fn concurrent_add_edge_from_multiple_threads() {
        // Node 0 starts with 2 neighbors
        let g = Arc::new(simple_graph());
        let mut handles = vec![];

        for i in 1..5 {
            let g_clone = g.clone();
            let handle = std::thread::spawn(move || {
                g_clone.add_edge(0, i).unwrap();
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        // Node 0 should now have connetions the other 4
        let final_neighbors = g.neighbors(0).unwrap();
        assert!(final_neighbors.len() == 4);
        for i in 1..5 {
            assert!(final_neighbors.contains(&i));
        }
    }

    #[test]
    fn remove_neighbors_on_missing_node_panics() {
        let mut g = Graph::new(0);
        assert!(g.remove_neighbors(999).is_err());
    }
}
