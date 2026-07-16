use crate::NodeID;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub struct Dist {
    pub id: NodeID,
    pub dist: f32,
}

impl Dist {
    pub fn new(id: NodeID, dist: f32) -> Self {
        Dist { id, dist }
    }
}

impl PartialEq for Dist {
    fn eq(&self, other: &Dist) -> bool {
        (self.id == other.id) & (self.dist == other.dist)
    }
}

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Dist) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Dist) -> Ordering {
        match self.dist.partial_cmp(&other.dist).unwrap() {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => self.id.cmp(&other.id),
        }
    }
}
