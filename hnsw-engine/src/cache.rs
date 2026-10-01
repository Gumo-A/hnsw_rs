use std::collections::HashMap;

use points::point::Point;

use crate::ID;

const MAX_ENTRIES: usize = 1024;

pub struct PointCache {
    cache: HashMap<ID, Point>,
}

impl PointCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: ID, point: Point) {
        if self.full() {
            self.remove_last()
        }
        self.cache.insert(id, point);
    }

    pub fn get(&mut self, id: &ID) -> Option<&Point> {
        self.cache.get(id)
    }

    pub fn contains_key(&self, id: &ID) -> bool {
        self.cache.contains_key(id)
    }

    fn full(&self) -> bool {
        false
    }

    fn remove_last(&mut self) {}
}
