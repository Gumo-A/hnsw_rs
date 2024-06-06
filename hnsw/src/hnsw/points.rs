// use crate::helpers::distance::norm_vector;
use super::lvq::LVQVec;

#[derive(Debug, Clone)]
pub enum Vector {
    Compressed(LVQVec),
    Full(Vec<f32>),
}

#[derive(Debug, Clone)]
pub enum Points {
    Empty,
    Collection(Vec<Point>),
}

impl Points {
    pub fn get_point(&self, index: usize) -> &Point {
        let point: &Point = match self {
            Points::Empty => {
                panic!(
                    "Tried to get point with index {index}, but there are no stored vectors in the index."
                );
            }
            Points::Collection(points) => points.get(index).unwrap(),
        };

        point
    }

    /// Extends the Collection variant with the provided vector,
    /// or, in the case it is the Empty variant, fills it and changes the variant to Collection.
    pub fn extend_or_fill(&mut self, points: Vec<Point>) {
        match self {
            Points::Empty => {
                *self = Points::Collection(points);
            }
            Points::Collection(points_vec) => {
                points_vec.extend(points);
            }
        }
    }

    pub fn iterate(&self) -> std::slice::Iter<'_, Point> {
        match self {
            Points::Empty => {
                panic!("Tried to iterate over empty collection of Points.");
            }
            Points::Collection(points) => points.iter(),
        }
    }
    pub fn iterate_mut(&mut self) -> std::slice::IterMut<'_, Point> {
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
}

#[derive(Debug, Clone)]
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

    pub fn dist2vec(&self, other_vec: &Vector) -> f32 {
        match &self.vector {
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
                    result.sqrt()
                }
            },
        }
    }

    pub fn quantize(&mut self) {
        let centered: &Vec<f32> = match &self.vector {
            Vector::Full(full) => full,
            Vector::Compressed(_) => {
                return ();
            }
        };
        self.vector = Vector::Compressed(LVQVec::new(centered, 8));
    }
}
