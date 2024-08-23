use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Dist {
    pub dist: f32,
    pub id: usize,
}

impl Dist {
    pub fn new(dist: f32, id: usize) -> Self {
        Dist { dist, id }
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Dist) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Dist {
    fn eq(&self, other: &Dist) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Dist {}

impl std::fmt::Display for Dist {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.dist)
    }
}
