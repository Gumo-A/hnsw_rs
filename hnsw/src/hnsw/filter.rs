pub struct FilterVectorHolder {
    pub candidates: FilterVector,
    pub visited: FilterVector,
    pub selected: FilterVector,
}

impl FilterVectorHolder {
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: FilterVector::new(capacity),
            visited: FilterVector::new(capacity),
            selected: FilterVector::new(capacity),
        }
    }
    pub fn set_entry_points(&mut self, entry_points: &Vec<usize>) {
        self.candidates.fill(entry_points);
        self.visited.fill(entry_points);
        self.selected.fill(entry_points);
    }

    pub fn clear(&mut self, node_ids: &Vec<usize>) {
        self.candidates.clear(&node_ids);
        self.visited.clear(&node_ids);
        self.selected.clear(&node_ids);
    }
}

pub struct FilterVector {
    pub bools: Vec<bool>,
    pub counter: usize,
    pub min_idx: Option<usize>,
    pub max_idx: Option<usize>,
}
impl FilterVector {
    fn new(capacity: usize) -> Self {
        Self {
            bools: vec![false; capacity],
            counter: 0,
            min_idx: None,
            max_idx: None,
        }
    }
    fn fill(&mut self, entry_points: &Vec<usize>) {
        self.clear(&vec![0]);
        for ep in entry_points {
            let ep = *ep as usize;
            self.add(ep);
        }
    }
    fn clear(&mut self, node_ids: &Vec<usize>) {
        // for idx in node_ids.iter() {
        //     self.vector[*idx] = false;
        // }
        self.bools.fill(false);
        self.counter = 0;
        assert_eq!(self.bools.iter().filter(|x| **x).count(), 0);
    }
    pub fn add(&mut self, node_id: usize) {
        if self.bools[node_id] {
            self.counter += 0;
        } else {
            self.bools[node_id] = true;
            self.min_idx = match self.min_idx {
                Some(idx) => Some(idx.min(node_id)),
                None => Some(node_id),
            };
            self.max_idx = match self.max_idx {
                Some(idx) => Some(idx.max(node_id)),
                None => Some(node_id),
            };
            self.counter += 1;
        };
    }

    pub fn remove(&mut self, node_id: usize) {
        self.counter -= if self.bools[node_id] {
            self.bools[node_id] = false;
            match self.min_idx {
                Some(idx) => {
                    if idx == node_id {
                        let mut new_min = idx;
                        for i in (idx..self.bools.len()).take_while(|x| !self.bools[*x]) {
                            new_min = i + 1;
                        }
                        self.min_idx = Some(new_min);
                    }
                }
                None => {}
            };
            match self.max_idx {
                Some(idx) => {
                    if idx == node_id {
                        let mut new_max = if idx > 0 { idx - 1 } else { idx };
                        for i in (1..idx).rev().take_while(|x| !self.bools[*x]) {
                            new_max = i - 1;
                        }
                        self.max_idx = Some(new_max);
                    }
                }
                None => {}
            }
            1
        } else {
            0
        }
    }
}
