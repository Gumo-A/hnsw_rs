use graph::nodes::Node;
use rand::rngs::ThreadRng;
use rand::{Rng, thread_rng};

use crate::point::Point;
use vectors::{FullVec, LVQVec, VecTrait};

fn get_new_node_layer(ml: f32, rng: &mut ThreadRng) -> u8 {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    (-rand_nb.log(std::f32::consts::E) * ml).floor() as u8
}

fn compute_means<T: VecTrait>(vectors: &Vec<Point<T>>) -> Option<Vec<f32>> {
    match vectors.first() {
        Some(p) => {
            let mut means = Vec::from_iter((0..p.dim()).map(|_| 0.0));
            for point in vectors.iter() {
                for (idx, val) in point.iter_vals().enumerate() {
                    means[idx] += val
                }
            }
            for idx in 0..means.len() {
                means[idx] /= vectors.len() as f32;
            }
            Some(means)
        }
        None => None,
    }
}

#[derive(Debug, Clone)]
pub struct Points<T: VecTrait> {
    collection: Vec<Point<T>>,
    means: Option<Vec<f32>>,
}

impl Points<FullVec> {
    pub fn new_full(mut vecs: Vec<Vec<f32>>, ml: f32) -> Points<FullVec> {
        let mut collection = Vec::new();
        let mut rng = thread_rng();
        for (idx, v) in vecs.drain(..).enumerate() {
            collection.push(Point::new_full(
                idx as Node,
                get_new_node_layer(ml, &mut rng),
                v,
            ));
        }
        let means = compute_means(&collection);
        Points { collection, means }
    }

    fn quantize(mut self) -> Points<LVQVec> {
        let mut collection = Vec::new();
        let means = self.means.as_ref().unwrap();
        for mut point in self.collection.drain(..) {
            point.center(means);
            collection.push(Point::new_quant(
                point.id,
                point.level,
                point.get_low_vector(),
            ));
        }
        Points {
            collection,
            means: self.means,
        }
    }
}

impl Points<LVQVec> {
    pub fn new_quant(vecs: Vec<Vec<f32>>, ml: f32) -> Points<LVQVec> {
        Points::new_full(vecs, ml).quantize()
    }
}

impl<T: VecTrait> Points<T> {
    pub fn new() -> Points<T> {
        Points {
            collection: Vec::new(),
            means: None,
        }
    }
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    pub fn ids(&self) -> impl Iterator<Item = Node> + '_ {
        self.collection.iter().map(|p| p.id)
    }

    /// Iterator over (ID, Level) pairs of stored Point structs.
    fn ids_levels(&self) -> impl Iterator<Item = (Node, u8)> + '_ {
        self.collection.iter().map(|p| (p.id, p.level))
    }

    /// If you call this function, you can be sure that point IDs correspond
    /// to their positions in the vector of points.
    /// Will change the IDs of each point to correspond to their positions if it
    /// it was not the case before.
    /// Returns whether the IDs were modified as an option
    /// Returns None variant if there are no points.
    fn check_ids(&mut self) -> Option<bool> {
        let mut is_ok = true;

        for (idx, point) in self.collection.iter().enumerate() {
            if !(idx == (point.id as usize)) {
                is_ok = false;
                break;
            }
        }
        if !is_ok {
            for (idx, point) in self.collection.iter_mut().enumerate() {
                point.id = idx as Node;
            }
            Some(true) // changed ids
        } else {
            Some(false) // didnt change anything
        }
    }

    pub fn dim(&self) -> Option<usize> {
        match self.collection.first() {
            Some(p) => Some(p.dim()),
            None => None,
        }
    }

    fn contains(&self, index: usize) -> bool {
        self.collection.len() < index
    }

    pub fn push(&mut self, mut point: Point<T>) {
        point.id = self.len() as Node;
        self.collection.push(point);
    }

    /// Removes a Point from the collection,
    /// returning true if it was removed,
    /// or false if it was already.
    pub fn remove(&mut self, index: Node) -> bool {
        if index >= self.len() as Node {
            false
        } else {
            self.collection
                .get_mut(index as usize)
                .unwrap()
                .set_removed()
        }
    }

    pub fn get_point(&self, index: Node) -> Option<&Point<T>> {
        self.collection.get(index as usize)
    }

    pub fn distance(&self, a_idx: Node, b_idx: Node) -> Option<f32> {
        let point_a = self.get_point(a_idx);
        let point_b = self.get_point(b_idx);
        match (point_a, point_b) {
            (Some(a), Some(b)) => Some(a.dist2other(b)),
            _ => None,
        }
    }

    pub fn distance2point(&self, point: &Point<T>, idx: Node) -> Option<f32> {
        let other = self.get_point(idx);
        match other {
            Some(b) => Some(point.dist2other(b)),
            _ => None,
        }
    }

    pub fn get_points(&self, indices: &Vec<Node>) -> Vec<&Point<T>> {
        indices
            .iter()
            .map(|idx| self.collection.get(*idx as usize).unwrap())
            .collect()
    }

    pub fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point<T>>
    where
        I: Iterator<Item = Node>,
    {
        indices.map(|idx| self.collection.get(idx as usize).unwrap())
    }

    fn get_point_mut(&mut self, index: Node) -> Option<&mut Point<T>> {
        self.collection.get_mut(index as usize)
    }

    pub fn extend(&mut self, other: Points<T>) {
        self.check_ids();
        for point in self.collection.iter_mut() {
            point.decenter(self.means.as_ref().unwrap());
        }
        let mut next_id = self.collection.len();
        for mut point in other.collection {
            point.id = next_id as Node;
            self.collection.push(point);
            next_id += 1;
        }

        self.means = compute_means(&self.collection);
        for point in self.collection.iter_mut() {
            point.center(self.means.as_ref().unwrap());
        }
    }
}

impl<'a, T: VecTrait> Points<T> {
    pub fn iter_points<'b: 'a>(&'b self) -> impl Iterator<Item = &'a Point<T>> {
        self.collection.iter()
    }
    pub fn iter_points_mut<'b: 'a>(&mut self) -> impl Iterator<Item = &mut Point<T>> {
        self.collection.iter_mut()
    }
}
