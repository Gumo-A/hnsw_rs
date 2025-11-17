use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub dist: Option<f32>,
    pub id: u32,
}

impl Node {
    pub fn new(id: u32) -> Self {
        Node { dist: None, id }
    }

    pub fn new_with_dist(dist: f32, id: u32) -> Self {
        Node {
            dist: Some(dist),
            id,
        }
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl nohash_hasher::IsEnabled for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Node) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Node {}

mod tests {
    use crate::hnsw::dist::Node;

    #[test]
    fn create_node() {
        let _ = Node::new(0);
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.dist {
            Some(d) => write!(f, "{}", d),
            None => write!(f, "{}", "None"),
        }
    }
}
