use graph::nodes::Node;
use rand::rngs::ThreadRng;
use rand::{Rng, thread_rng};
use vectors::serializer::Serializer;

use crate::point::Point;
use vectors::{FullVec, LVQVec, VecBase, VecTrait};

fn get_new_node_layer(ml: f64, rng: &mut ThreadRng) -> u8 {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    (-rand_nb.log(std::f32::consts::E) * ml as f32).floor() as u8
}

#[derive(Debug, Clone)]
pub struct Points<T: VecTrait> {
    pub collection: Vec<Point<T>>,
    pub quantized: bool,
}

impl<T: VecTrait> Points<T> {
    pub fn new(mut vecs: Vec<Vec<f32>>, ml: f64) -> Points<T> {
        let mut collection = Vec::new();
        let mut rng = thread_rng();
        for (idx, v) in vecs.drain(..).enumerate() {
            collection.push(Point::new_with(
                idx as Node,
                get_new_node_layer(ml, &mut rng) as usize,
                &v,
            ));
        }
        Points {
            collection,
            quantized: false,
        }
    }
}

impl Points<FullVec> {
    pub fn new_full(vectors: Vec<Vec<f32>>, ml: f64) -> Points<FullVec> {
        Points::new(vectors, ml)
    }
}

impl Points<LVQVec> {
    pub fn new_quant(vectors: Vec<Vec<f32>>, ml: f64) -> Points<LVQVec> {
        let mut points = Points::new(vectors, ml);
        points.quantized = true;
        points
    }
}

impl<T: VecTrait> Points<T> {
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    pub fn ids(&self) -> impl Iterator<Item = Node> + '_ {
        self.collection.iter().map(|p| p.id)
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

    /// Updates means, adds points with correct IDs, returns vector
    /// with tuples of new IDs and levels.
    pub fn extend(&mut self, other: Points<T>) -> Vec<(Node, usize)> {
        self.check_ids();
        let mut ids_levels = Vec::with_capacity(other.len());
        let mut next_id = self.collection.len();
        for mut point in other.collection {
            point.id = next_id as Node;
            ids_levels.push((point.id, point.level as usize));
            self.collection.push(point);
            next_id += 1;
        }

        ids_levels
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

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    fn get_collection_data<T: VecTrait>(points: &Points<T>) -> (f32, f32, u32, usize) {
        let dist = points
            .get_point(32)
            .unwrap()
            .distance(points.get_point(64).unwrap());
        let val = points.get_point(16).unwrap().iter_vals().next().unwrap();
        (dist, val, points.len() as u32, points.size())
    }

    #[test]
    fn serialization() {
        let rand_vecs = gen_rand_vecs(128, 1024);
        let points = Points::new_full(rand_vecs, 1.0);
        let (dist, val, len, size) = get_collection_data(&points);

        let points_ser = points.serialize();
        let points: Points<FullVec> = Points::deserialize(points_ser);
        let (dist_ser, val_ser, len_ser, size_ser) = get_collection_data(&points);

        println!("dist: {dist} dist_ser: {dist_ser}");
        assert_eq!(dist, dist_ser);
        println!("val: {val} val_ser: {val_ser}");
        assert_eq!(val, val_ser);
        println!("len: {len} len_ser: {len_ser}");
        assert_eq!(len, len_ser);
        println!("size: {size} size_ser: {size_ser}");
        assert_eq!(size, size_ser);
    }

    #[test]
    fn placeholder() {}
}
