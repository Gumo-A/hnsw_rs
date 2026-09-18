use crate::{NodeID, errors::GraphError};

use log::{trace, warn};
use nohash_hasher::{IntMap, IntSet};
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

type Neighbors = Arc<Mutex<IntSet<NodeID>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: IntMap<NodeID, Neighbors>,
    pub level: usize,
    pub m: usize,
}

impl Graph {
    pub fn new(level: usize, m: usize) -> Self {
        Graph {
            nodes: IntMap::default(),
            level,
            m,
        }
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeID> {
        self.nodes.keys().copied()
    }

    pub fn add_node(&mut self, point_id: NodeID) {
        self.nodes
            .entry(point_id)
            .or_insert(Arc::new(Mutex::new(IntSet::default())));
    }

    fn add_edge(&self, node_a: NodeID, node_b: NodeID) -> Result<(), GraphError> {
        if node_a == node_b {
            return Err(GraphError::SelfConnection(node_a));
        }

        let (a, b) = self.get_nodes_neighbors(node_a, node_b)?;

        {
            let mut a = a.lock().unwrap();
            a.insert(node_b);
            if a.len() > self.m {
                warn!("Edge ({node_a}, {node_b}) made {node_a} exceed M");
            }
        }
        {
            let mut b = b.lock().unwrap();
            b.insert(node_a);
            if b.len() > self.m {
                warn!("Edge ({node_a}, {node_b}) made {node_b} exceed M");
            }
        }
        trace!("Added edge ({node_a}, {node_b})");
        Ok(())
    }

    fn get_nodes_neighbors(
        &self,
        node_a: NodeID,
        node_b: NodeID,
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
    fn remove_edge(&self, node_a: NodeID, node_b: NodeID) -> Result<(), GraphError> {
        let (a, b) = self.get_nodes_neighbors(node_a, node_b)?;

        {
            let mut a = a.lock().unwrap();
            a.remove(&node_b);
            if a.len() == 0 {
                a.insert(node_b);
                return Err(GraphError::IsolatedNode(node_a));
            }
        }
        {
            let mut b = b.lock().unwrap();
            b.remove(&node_a);
            if b.len() == 0 {
                b.insert(node_a);
                return Err(GraphError::IsolatedNode(node_b));
            }
        }
        trace!("Removed edge ({node_a}, {node_b})");
        Ok(())
    }

    pub fn neighbors(&self, node: NodeID) -> Result<IntSet<NodeID>, GraphError> {
        match self.nodes.get(&node) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().clone()),
            None => Err(GraphError::NodeNotInGraph(node)),
        }
    }

    pub fn neighbors_lock(&self, node: NodeID) -> Result<Arc<Mutex<IntSet<NodeID>>>, GraphError> {
        trace!("Getting neighbors of {node}");
        let neighbors = match self.nodes.get(&node) {
            Some(neighbors) => neighbors,
            None => {
                return Err(GraphError::NodeNotInGraph(node));
            }
        };
        Ok(neighbors.clone())
    }

    /// Replaces the neighbors of node with new_neighbors.
    pub fn replace_neighbors<I>(&self, node: NodeID, new_neighbors: I) -> Result<(), GraphError>
    where
        I: Iterator<Item = NodeID>,
    {
        trace!("Replacing {node}'s neighbors");
        let new_neighbors = IntSet::from_iter(new_neighbors);
        // Deadlocks:
        // let tmp = self.neighbors_lock(node)?;
        // let current_neighbors = tmp.lock().unwrap();
        let current_neighbors = self.neighbors(node)?;
        let to_remove = current_neighbors.difference(&new_neighbors).copied();
        let to_add = new_neighbors.difference(&current_neighbors).copied();
        self.add_neighbors(node, to_add)?;
        self.remove_neighbors(node, to_remove)?;
        Ok(())
    }

    pub fn add_neighbors<I>(&self, node: NodeID, new_neighbors: I) -> Result<(), GraphError>
    where
        I: Iterator<Item = NodeID>,
    {
        for n in new_neighbors {
            self.add_edge(node, n)?;
        }

        Ok(())
    }

    fn remove_neighbors<I>(&self, node: NodeID, old_neighbors: I) -> Result<(), GraphError>
    where
        I: Iterator<Item = NodeID>,
    {
        for n in old_neighbors {
            match self.remove_edge(node, n) {
                Ok(()) => (),
                Err(e) => match e {
                    GraphError::IsolatedNode(a) => {
                        warn!("Was going to remove edge ({node}, {n}), but {a} would be isolated");
                    }
                    _ => return Err(e),
                },
            };
        }

        Ok(())
    }

    pub fn degree(&self, node: NodeID) -> Result<usize, GraphError> {
        match self.nodes.get(&node) {
            Some(neighbors) => Ok(neighbors.lock().unwrap().len()),
            None => Err(GraphError::NodeNotInGraph(node)),
        }
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, node_id: NodeID) -> bool {
        self.nodes.contains_key(&node_id)
    }
}

pub fn gen_rand_graph(n: usize, degree: usize) -> Graph {
    let mut rng = rand::thread_rng();
    let nodes =
        IntMap::from_iter((0..n).map(|id| (id as NodeID, Arc::new(Mutex::new(IntSet::default())))));
    let graph = Graph {
        nodes,
        level: 0,
        m: 12,
    };
    for node in 0..n {
        let neighbors = (0..n).choose_multiple(&mut rng, degree);
        for n in neighbors {
            match graph.add_edge(node as NodeID, n as NodeID) {
                Err(_) => continue,
                Ok(_) => continue,
            }
        }
    }
    graph
}

pub fn simple_graph() -> Graph {
    let mut g = Graph::new(1, 12);
    for i in 0..5 {
        g.add_node(i as NodeID);
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
        NodeID,
        graph::{Graph, gen_rand_graph, simple_graph},
    };
    use nohash_hasher::IntSet;
    use std::sync::Arc;

    #[test]
    fn build_graph() {
        let graph = gen_rand_graph(100, 8);
        assert!(graph.nodes.len() == 100);
    }

    #[test]
    fn no_one_way_connections() {
        let graph = gen_rand_graph(100, 8);
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
        let mut g = Graph::new(0, 12);
        assert!(!g.contains(42));
        g.add_node(42);
        assert!(g.contains(42));
        assert_eq!(g.nb_nodes(), 1);
        assert_eq!(g.degree(42).unwrap(), 0);
    }

    #[test]
    fn add_edge_rejects_missing_nodes() {
        let mut g = Graph::new(0, 12);
        g.add_node(1);
        assert!(g.add_edge(1, 2).is_err());
        assert!(g.add_edge(999, 1000).is_err());
    }

    #[test]
    fn no_self_loops() {
        let mut g = Graph::new(0, 12);
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
        let mut g = Graph::new(0, 12);
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
    fn replace_neighbors_full_replace() {
        let mut g = Graph::new(0, 12);
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
        g.replace_neighbors(0, [4, 5].iter().copied()).unwrap();

        let new_neigh = g.neighbors(0).unwrap();
        dbg!(&new_neigh);
        assert_eq!(new_neigh.len(), 2);
        assert!(new_neigh.contains(&4));
        assert!(new_neigh.contains(&5));
        assert!(!new_neigh.contains(&1));
        assert!(!new_neigh.contains(&2));
        assert!(!new_neigh.contains(&3));

        // Old neighbors should no longer point back (symmetry preserved)
        assert!(!g.neighbors(1).unwrap().contains(&0));
        assert!(!g.neighbors(2).unwrap().contains(&0));
        assert!(g.neighbors(4).unwrap().contains(&0));
        assert!(g.neighbors(5).unwrap().contains(&0));
    }

    #[test]
    fn replace_neighbors_on_isolated_node() {
        let mut g = Graph::new(0, 12);
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
    fn degree_on_missing_node_errors() {
        let g = Graph::new(0, 12);
        assert!(g.degree(999).is_err());
    }

    #[test]
    fn serialization_round_trip() {
        let original = simple_graph();
        let original_bytes = rmp_serde::encode::to_vec(&original).unwrap();

        let restored: Graph = rmp_serde::decode::from_slice(&original_bytes).unwrap();

        assert_eq!(original.nb_nodes(), restored.nb_nodes());
        assert_eq!(original.level, restored.level);

        for node in original.iter_nodes() {
            assert!(restored.contains(node));
            let orig_neigh: IntSet<NodeID> =
                original.neighbors(node).unwrap().into_iter().collect();
            let rest_neigh: IntSet<NodeID> =
                restored.neighbors(node).unwrap().into_iter().collect();
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
}
