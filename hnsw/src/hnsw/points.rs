use std::collections::{HashMap, HashSet};

use nohash_hasher::BuildNoHashHasher;
use serde::{Deserialize, Serialize};

use super::{dist::Dist, lvq::LVQVec};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Vector {
    Compressed(LVQVec),
    Full(Vec<f32>),
}

impl Vector {
    pub fn get_quantized(&self) -> LVQVec {
        match self {
            Self::Full(full) => LVQVec::new(full, 8),
            Self::Compressed(quant) => quant.clone(),
        }
    }

    pub fn get_full(&self) -> Vec<f32> {
        match self {
            Self::Full(full) => full.clone(),
            Self::Compressed(quant) => quant.reconstruct(),
        }
    }

    pub fn to_full(&mut self) {
        match self {
            Self::Full(_) => (),
            Self::Compressed(quant) => {
                let full = quant.reconstruct();
                *self = Self::Full(full);
            }
        }
    }
    pub fn quantize(&mut self) {
        match self {
            Self::Compressed(_) => (),
            Self::Full(full) => {
                *self = Self::Compressed(LVQVec::new(full, 8));
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Compressed(quant) => quant.dim(),
            Self::Full(full) => full.len(),
        }
    }

    pub fn get_vec(&self) -> Vec<f32> {
        match self {
            Self::Full(full) => full.clone(),
            Self::Compressed(quant) => quant.reconstruct(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: usize,
    pub level: Option<usize>,
    pub vector: Vector,
}

impl Point {
    pub fn new_full(id: usize, level: Option<usize>, vector: Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Full(vector),
        }
    }

    pub fn new_quantized(id: usize, level: Option<usize>, vector: &Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Compressed(LVQVec::new(vector, 8)),
        }
    }

    pub fn from_vector(id: usize, vector: Vector) -> Point {
        Point {
            id,
            level: None,
            vector,
        }
    }

    pub fn dist2other(&self, other: &Point) -> Dist {
        self.dist2vec(&other.vector)
    }

    pub fn dist2vec(&self, other_vec: &Vector) -> Dist {
        let dist = match &self.vector {
            Vector::Compressed(compressed_self) => match other_vec {
                Vector::Compressed(compressed_other) => {
                    compressed_self.dist2other(&compressed_other)
                }
                Vector::Full(full_other) => compressed_self.dist2vec(&full_other),
            },
            Vector::Full(full_self) => match other_vec {
                Vector::Compressed(compressed_other) => compressed_other.dist2vec(&full_self),
                Vector::Full(full_other) => Dist::new(
                    full_self
                        .iter()
                        .zip(full_other.iter())
                        .fold(0.0, |acc, e| acc + (e.0 - e.1).powi(2))
                        .sqrt(),
                ),
            },
        };
        dist
    }

    pub fn quantize(&mut self) {
        self.vector.quantize();
    }
    pub fn to_full(&mut self) {
        self.vector.to_full();
    }

    pub fn get_full_precision(&self) -> Vec<f32> {
        self.vector.get_full()
    }
    pub fn get_quantized(&self) -> LVQVec {
        self.vector.get_quantized()
    }

    pub fn dim(&self) -> usize {
        self.vector.dim()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Points {
    Empty,
    Collection(HashMap<usize, Point, BuildNoHashHasher<usize>>),
}

impl Points {
    pub fn ids(&self) -> std::collections::hash_map::Keys<'_, usize, Point> {
        match self {
            Points::Empty => {
                panic!("Tried to get ids, but there are no stored vectors in the index.");
            }
            Points::Collection(points) => points.keys(),
        }
    }

    pub fn ids_levels(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        match self {
            Points::Empty => {
                panic!("Tried to get ids, but there are no stored vectors in the index.");
            }
            Points::Collection(points) => points
                .values()
                .map(|point| (point.id, point.level.unwrap())),
        }
    }

    pub fn insert(&mut self, point: Point) {
        match self {
            Points::Empty => {
                let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
                collection.insert(point.id, point);
                *self = Points::Collection(collection);
            }
            Points::Collection(points) => {
                points.insert(point.id, point);
            }
        };
    }

    pub fn dim(&self) -> usize {
        match self {
            Points::Empty => 0,
            Points::Collection(points) => points.iter().next().unwrap().1.vector.dim(),
        }
    }

    /// Removes and returns some point from the collection.
    pub fn pop_rand(&mut self) -> Option<Point> {
        match self {
            Points::Empty => None,
            Points::Collection(points) => {
                let key = points.keys().next().unwrap().clone();
                points.remove(&key)
            }
        }
    }

    pub fn pop(&mut self, index: usize) -> Option<Point> {
        match self {
            Points::Empty => None,
            Points::Collection(points) => points.remove(&index),
        }
    }

    /// Removes all the points with the ids and returns them in a new Points struct.
    pub fn pop_multiple(&mut self, ids: &Vec<usize>) -> Option<Self> {
        match self {
            Points::Empty => None,
            Points::Collection(points) => {
                let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
                for key in ids {
                    let point = points
                        .remove(&key)
                        .expect(format!("Could not find {key} in collection.").as_str());
                    collection.insert(*key, point);
                }
                Some(Self::Collection(collection))
            }
        }
    }

    pub fn contains(&self, index: &usize) -> bool {
        match self {
            Points::Empty => false,
            Points::Collection(points) => points.contains_key(index),
        }
    }

    pub fn get_point(&self, index: usize) -> Option<&Point> {
        match self {
            Points::Empty => {
                panic!(
                    "Tried to get point with index {index}, but there are no stored vectors in the index."
                );
            }
            Points::Collection(points) => points.get(&index),
        }
    }

    pub fn get_points(&self, indices: &HashSet<usize, BuildNoHashHasher<usize>>) -> Vec<&Point> {
        let points = match self {
            Points::Empty => {
                panic!("Tried to get points, but there are no stored vectors in the index.");
            }
            Points::Collection(points) => {
                indices.iter().map(|idx| points.get(idx).unwrap()).collect()
            }
        };
        points
    }
    pub fn get_point_mut(&mut self, index: usize) -> &mut Point {
        let point: &mut Point = match self {
            Points::Empty => {
                panic!(
                    "Tried to get point with index {index}, but there are no stored vectors in the index."
                );
            }
            Points::Collection(points) => points.get_mut(&index).unwrap(),
        };

        point
    }

    /// Extends the Collection variant with the provided vector,
    /// or, in the case it is the Empty variant, fills it and changes the variant to Collection.
    pub fn extend_or_fill(&mut self, other: Points) {
        let points = match other {
            Self::Empty => return (),
            Self::Collection(points_map) => points_map,
        };

        match self {
            Points::Empty => {
                *self = Points::Collection(points);
            }
            Points::Collection(points_map) => {
                let old_ids: HashSet<usize> = points_map.keys().cloned().collect();
                let new_ids: HashSet<usize> = points.keys().cloned().collect();
                let intersection_len = new_ids.intersection(&old_ids).count();

                if intersection_len > 0 {
                    println!("At least one id in the new points are present in the old ids.");
                    println!("Cannot insert the new points to the collection.");
                    return ();
                }

                for (id, point) in points {
                    points_map.insert(id, point);
                }
            }
        }
    }

    pub fn iterate(&self) -> std::collections::hash_map::Iter<'_, usize, Point> {
        match self {
            Points::Empty => {
                panic!("Tried to iterate over empty collection of Points.");
            }
            Points::Collection(points) => points.iter(),
        }
    }
    pub fn iterate_mut(&mut self) -> std::collections::hash_map::IterMut<'_, usize, Point> {
        match self {
            Points::Empty => {
                panic!("Tried to iterate over empty collection of Points.");
            }
            Points::Collection(points) => points.iter_mut(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Points::Empty => 0,
            Points::Collection(points) => points.len(),
        }
    }

    pub fn quantize(&mut self) {
        for (_, point) in self.iterate_mut() {
            point.quantize();
        }
    }

    pub fn to_full(&mut self) {
        for (_, point) in self.iterate_mut() {
            point.to_full();
        }
    }
}
