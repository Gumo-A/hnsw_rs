use nohash_hasher::IntSet;
use rand::rngs::ThreadRng;
use rand::Rng;

use crate::hnsw::points::point::Point;
use crate::hnsw::vectors::LVQVec;

fn get_new_node_layer(ml: f32, rng: &mut ThreadRng) -> u8 {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.gen::<f32>();
        } else {
            break;
        }
    }

    (-rand_nb.log(std::f32::consts::E) * ml).floor() as u8
}

#[derive(Debug, Clone)]
pub struct Points {
    collection: Vec<Point>,
    means: Option<Vec<f32>>,
}

pub trait Storage {
    fn new() -> Points {
        Points {
            collection: Vec::new(),
            means: None,
        }
    }

    fn contains(&self, index: usize) -> bool {
        self.len() < index
    }

    fn empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;

    fn ids(&self) -> impl Iterator<Item = u32> + '_;

    /// Iterator over (ID, Level) pairs of stored Point structs.
    fn ids_levels(&self) -> impl Iterator<Item = (u32, u8)> + '_;

    /// If you call this function, you can be sure that point IDs correspond
    /// to their positions in the vector of points.
    /// Will change the IDs of each point to correspond to their positions if it
    /// it was not the case before.
    /// Returns whether the IDs were modified as an option
    /// Returns None variant if there are no points.
    fn check_ids(&mut self) -> Option<bool>;

    fn dim(&self) -> Option<usize>;

    fn recompute_means(&mut self);

    fn push(&mut self, point: Point);

    fn remove(&mut self, index: u32) -> Option<Point>;

    fn get_point(&self, index: u32) -> &Point;

    fn get_points(&self, indices: &IntSet<u32>) -> Vec<&Point>;

    fn get_point_mut(&mut self, index: u32) -> Option<&mut Point>;

    fn extend_concrete(&mut self, other: Points);

    fn extend(&mut self, other: Points) {
        self.check_ids();
        self.extend_concrete(other);
    }
}

impl Storage for Points {
    fn len(&self) -> usize {
        self.collection.len()
    }

    fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.collection.iter().map(|p| p.id)
    }

    /// Iterator over (ID, Level) pairs of stored Point structs.
    fn ids_levels(&self) -> impl Iterator<Item = (u32, u8)> + '_ {
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
                point.id = idx as u32;
            }
            Some(true) // changed ids
        } else {
            Some(false) // didnt change anything
        }
    }

    fn dim(&self) -> Option<usize> {
        match self.collection.first() {
            Some(p) => Some(p.dim()),
            None => None,
        }
    }

    fn contains(&self, index: usize) -> bool {
        self.collection.len() < index
    }

    fn recompute_means(&mut self) {
        self.means = match self.dim() {
            Some(dim) => {
                let mut means = Vec::from_iter((0..dim).map(|_| 0.0));
                for point in self.collection.iter() {
                    for (idx, val) in point.iter_vals().enumerate() {
                        means[idx] += val
                    }
                }
                for idx in 0..means.len() {
                    means[idx] /= self.collection.len() as f32;
                }
                Some(means)
            }
            None => None,
        }
    }

    fn push(&mut self, mut point: Point) {
        point.id = self.len() as u32;
        self.collection.push(point);
    }

    fn remove(&mut self, index: u32) -> Option<Point> {
        if index >= self.len() as u32 {
            None
        } else {
            Some(self.collection.remove(index as usize))
        }
    }

    fn get_point(&self, index: u32) -> &Point {
        match self.collection.get(index as usize) {
            None => panic!("tried to get point {index}, but it is out of range"),
            Some(p) => p,
        }
    }

    fn get_points(&self, indices: &IntSet<u32>) -> Vec<&Point> {
        indices
            .iter()
            .map(|idx| self.collection.get(*idx as usize).unwrap())
            .collect()
    }

    fn get_point_mut(&mut self, index: u32) -> Option<&mut Point> {
        self.collection.get_mut(index as usize)
    }

    fn extend_concrete(&mut self, other: Points) {
        let old_means = match &self.means {
            None => return,
            Some(m) => m.clone(),
        };

        let mut next_id = self.collection.len();
        for mut point in other.collection {
            point.id = next_id as u32;
            point.quantize();
            self.collection.push(point);
            next_id += 1;
        }

        self.recompute_means();
        let new_means = match &self.means {
            None => return,
            Some(m) => m,
        };

        let means_diff: Vec<f32> = old_means
            .iter()
            .zip(new_means)
            .map(|(o, n)| o - n)
            .collect();

        for point in self.collection.iter_mut() {
            point.set_quantized(LVQVec::new(
                &point
                    .iter_vals()
                    .zip(&means_diff)
                    .map(|(v, m)| v + m)
                    .collect(),
            ));
        }
    }
}

impl<'a> Points {
    pub fn iter_points<'b: 'a>(&'b self) -> impl Iterator<Item = &'a Point> {
        self.collection.iter()
    }
    pub fn iter_points_mut<'b: 'a>(&'b mut self) -> impl Iterator<Item = &mut Point> {
        self.collection.iter_mut()
    }
}
