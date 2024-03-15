use std::collections::{HashMap, HashSet};

struct Graph {
    nodes: HashSet<i32>,
    edges: HashMap<i32, (i32, i32)>,
}
