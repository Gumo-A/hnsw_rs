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
    pub vector: Vector,
}

impl Point {
    pub fn new(id: usize, vector: Vec<f32>, quantize: bool) -> Point {
        let vector_stored = if quantize {
            Vector::Compressed(LVQVec::new(&vector, 8))
        } else {
            Vector::Full(vector)
        };
        Point {
            id,
            vector: vector_stored,
        }
    }

    pub fn from_vector(id: usize, vector: Vector) -> Point {
        Point { id, vector }
    }

    pub fn dist2vec(&self, other_vec: &Vector) -> Dist {
        let dist = match &self.vector {
            Vector::Compressed(compressed_self) => match other_vec {
                Vector::Compressed(compressed_other) => {
                    // println!("Q 2 Q");
                    compressed_self.dist2other(&compressed_other)
                }
                Vector::Full(full_other) => {
                    // println!("Q 2 F");
                    compressed_self.dist2vec(&full_other)
                }
            },
            Vector::Full(full_self) => match other_vec {
                Vector::Compressed(compressed_other) => {
                    // println!("F 2 Q");
                    compressed_other.dist2vec(&full_self)
                }
                Vector::Full(full_other) => {
                    // println!("F 2 F");
                    let mut result = 0.0;
                    for (x, y) in full_self.iter().zip(full_other) {
                        result += (x - y).powi(2);
                    }
                    Dist::new(result.sqrt())
                }
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

    pub fn pop(&mut self, index: &usize) -> Option<Point> {
        match self {
            Points::Empty => None,
            Points::Collection(points) => points.remove(index),
        }
    }

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

    pub fn get_point(&self, index: usize) -> &Point {
        let point: &Point = match self {
            Points::Empty => {
                panic!(
                    "Tried to get point with index {index}, but there are no stored vectors in the index."
                );
            }
            Points::Collection(points) => points
                .get(&index)
                .expect(format!("Could not get point {index} from collection.").as_str()),
        };

        point
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
    pub fn extend_or_fill(&mut self, points: Vec<Point>) {
        match self {
            Points::Empty => {
                let mut collection =
                    HashMap::with_capacity_and_hasher(points.len(), BuildNoHashHasher::default());
                let mut idx = 0;
                for point in points {
                    collection.insert(idx, point);
                    idx += 1;
                }
                *self = Points::Collection(collection);
            }
            Points::Collection(points_map) => {
                let old_max = points_map.keys().max().unwrap_or(&0) + 1;
                let mut idx = 1;
                for point in points {
                    points_map.insert(old_max + idx, point);
                    idx += 1;
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
