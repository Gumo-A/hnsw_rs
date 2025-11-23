pub type Node = u32;

#[derive(Clone, Copy, Debug)]
pub struct Dist {
    pub id: Node,
    pub dist: f32,
}

impl Dist {
    pub fn new(id: Node, dist: f32) -> Self {
        Dist { id, dist }
    }
}

impl PartialEq for Dist {
    fn eq(&self, other: &Dist) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Dist) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

use std::cmp::Ordering;
impl Ord for Dist {
    fn cmp(&self, other: &Dist) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}
