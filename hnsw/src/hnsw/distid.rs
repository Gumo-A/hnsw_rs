use std::cmp::Ordering;

#[derive(Eq, Clone, Copy)]
pub struct DistId {
    pub id: usize,
    pub dist: isize,
}
impl DistId {
    pub fn deconstruct(&self) -> (isize, usize) {
        (self.dist, self.id)
    }
}

impl Ord for DistId {
    fn cmp(&self, other: &DistId) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl PartialOrd for DistId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DistId {
    fn eq(&self, other: &DistId) -> bool {
        self.dist == other.dist
    }
}
