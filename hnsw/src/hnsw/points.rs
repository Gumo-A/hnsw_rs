use nohash_hasher::IntSet;
use rand::rngs::ThreadRng;
use rand::Rng;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: u32,
    pub level: u8,
    pub vector: Vector,
}

impl Point {
    pub fn new_full(id: u32, level: u8, vector: Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Full(vector),
        }
    }

    pub fn new_quantized(id: u32, level: u8, vector: &Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Compressed(LVQVec::new(vector, 8)),
        }
    }

    pub fn from_vector(id: u32, level: u8, vector: Vector) -> Point {
        Point { id, level, vector }
    }

    pub fn from_bytes_quant(bytes: &Vec<u8>) -> Point {
        let mut id_bytes = [0u8; 4];
        for i in 0..4 {
            id_bytes[i] = bytes[i];
        }
        let id = u32::from_be_bytes(id_bytes) as u32;

        let level = bytes[4];

        let offset = 5;

        let mut delta_bytes = [0u8; 4];
        let mut lower_bytes = [0u8; 4];
        for i in 0..4 {
            delta_bytes[i] = bytes[offset + i];
        }

        let offset = 9;
        for i in 0..4 {
            lower_bytes[i] = bytes[offset + i];
        }

        let quantized_vec = bytes[13..].to_vec();

        let vector = Vector::Compressed(LVQVec::from_quantized(
            quantized_vec,
            f32::from_be_bytes(delta_bytes),
            f32::from_be_bytes(lower_bytes),
        ));

        Point { id, level, vector }
    }

    pub fn dist2other(&self, other: &Point) -> Dist {
        self.dist2vec(&other.vector, other.id)
    }

    pub fn dist2vec(&self, other_vec: &Vector, id: u32) -> Dist {
        let dist = match &self.vector {
            Vector::Compressed(compressed_self) => match other_vec {
                Vector::Compressed(compressed_other) => {
                    compressed_self.dist2other(compressed_other, id)
                }
                Vector::Full(full_other) => compressed_self.dist2vec(full_other, id),
            },
            Vector::Full(full_self) => match other_vec {
                Vector::Compressed(compressed_other) => compressed_other.dist2vec(full_self, id),
                Vector::Full(full_other) => Dist::new(
                    full_self
                        .iter()
                        .zip(full_other.iter())
                        .fold(0.0, |acc, e| acc + (e.0 - e.1).powi(2))
                        .sqrt(),
                    id,
                ),
            },
        };
        dist
    }

    pub fn dist2others<'a, I>(&'a self, others: I) -> impl Iterator<Item = Dist> + 'a
    where
        I: Iterator<Item = &'a Point> + 'a,
    {
        match &self.vector {
            Vector::Compressed(lvq) => lvq.dist2many(others.map(|p| match &p.vector {
                Vector::Compressed(lvq_other) => (lvq_other, p.id),
                _ => panic!("nope!"),
            })),
            _ => panic!("nope!"),
        }
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
    Collection((Vec<Point>, Vec<f32>)),
}

impl Points {
    pub fn from_vecs_quant(mut vectors: Vec<Vec<f32>>, ml: f32) -> Self {
        let mut rng = rand::thread_rng();

        let mut means = Vec::from_iter((0..vectors[0].len()).map(|_| 0.0));
        for vector in vectors.iter() {
            for (idx, val) in vector.iter().enumerate() {
                means[idx] += val
            }
        }
        for idx in 0..means.len() {
            means[idx] /= vectors.len() as f32;
        }

        vectors.iter_mut().for_each(|v| {
            v.iter_mut()
                .enumerate()
                .for_each(|(idx, x)| *x -= means[idx])
        });

        let collection =
            Vec::from_iter(vectors.iter().enumerate().map(|(id, v)| {
                Point::new_quantized(id as u32, get_new_node_layer(ml, &mut rng), v)
            }));

        Self::Collection((collection, means))
    }

    pub fn from_vecs_full(mut vectors: Vec<Vec<f32>>, ml: f32) -> Self {
        let mut rng = rand::thread_rng();

        let mut means = Vec::from_iter((0..vectors[0].len()).map(|_| 0.0));
        for vector in vectors.iter() {
            for (idx, val) in vector.iter().enumerate() {
                means[idx] += val
            }
        }
        for idx in 0..means.len() {
            means[idx] /= vectors.len() as f32;
        }

        vectors.iter_mut().for_each(|v| {
            v.iter_mut()
                .enumerate()
                .for_each(|(idx, x)| *x -= means[idx])
        });

        let collection = Vec::from_iter(vectors.iter().enumerate().map(|(id, v)| {
            Point::new_full(id as u32, get_new_node_layer(ml, &mut rng), v.clone())
        }));

        Self::Collection((collection, means))
    }

    pub fn get_means(&self) -> Result<&Vec<f32>, String> {
        match self {
            Self::Empty => Err("There are no points in this struct.".to_string()),
            Self::Collection(col) => Ok(&col.1),
        }
    }

    /// If you call this function, you can be sure that point IDs correspond
    /// to their positions in the vector of points.
    /// Will change the IDs of each point to correspond to their positions if it
    /// it was not the case before.
    /// Returns whether the IDs were modified.
    pub fn assert_ids(&mut self) -> bool {
        let points = match self {
            Self::Empty => return false,
            Self::Collection(ps) => ps,
        };
        let mut is_ok = true;
        for (idx, point) in points.0.iter().enumerate() {
            if !(idx == (point.id as usize)) {
                is_ok = false;
                break;
            }
        }
        if is_ok {
            false
        } else {
            for (idx, point) in points.0.iter_mut().enumerate() {
                point.id = idx as u32;
            }
            true
        }
    }

    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        match self {
            Self::Empty => {
                panic!("Tried to get ids, but there are no stored vectors in the index.");
            }
            Self::Collection(points) => points.0.iter().map(|p| p.id),
        }
    }

    /// Iterator over (ID, Level) pairs of stored Point structs.
    pub fn ids_levels(&self) -> impl Iterator<Item = (u32, u8)> + '_ {
        match self {
            Self::Empty => {
                panic!("Tried to get ids, but there are no stored vectors in the index.");
            }
            Self::Collection(points) => points.0.iter().map(|point| (point.id, point.level)),
        }
    }

    pub fn insert(&mut self, point: Point) {
        match self {
            Self::Empty => {
                *self = Self::Collection((
                    Vec::from([point.clone()]),
                    Vec::from_iter(point.get_full_precision()),
                ));
            }
            Self::Collection(points) => {
                points.0.insert(point.id as usize, point);
            }
        };
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Collection(points) => points.0.first().unwrap().vector.dim(),
        }
    }

    pub fn remove(&mut self, index: u32) -> Option<Point> {
        match self {
            Self::Empty => None,
            Self::Collection(points) => Some(points.0.remove(index as usize)),
        }
    }

    pub fn contains(&self, index: &u32) -> bool {
        match self {
            Self::Empty => false,
            Self::Collection(points) => points.0.get(*index as usize).is_some(),
        }
    }

    pub fn get_point(&self, index: u32) -> Option<&Point> {
        match self {
            Self::Empty => None,
            Self::Collection(points) => points.0.get(index as usize),
        }
    }

    pub fn get_points(&self, indices: &IntSet<u32>) -> Vec<&Point> {
        let points = match self {
            Self::Empty => {
                panic!("Tried to get points, but there are no stored vectors in the index.");
            }
            Self::Collection(points) => indices
                .iter()
                .map(|idx| points.0.get(*idx as usize).unwrap())
                .collect(),
        };
        points
    }
    pub fn get_point_mut(&mut self, index: u32) -> &mut Point {
        let point: &mut Point = match self {
            Self::Empty => {
                panic!(
                    "Tried to get point with index {index}, but there are no stored vectors in the index."
                );
            }
            Self::Collection(points) => points.0.get_mut(index as usize).unwrap(),
        };

        point
    }

    /// Extends the Collection variant with the provided Points struct,
    /// or, in the case it is the Empty variant, fills it and changes the variant to Collection.
    pub fn extend_or_fill(&mut self, other: Self) {
        self.assert_ids();

        let points = match other {
            Self::Empty => return,
            Self::Collection(points_other) => points_other,
        };

        match self {
            Self::Empty => {
                *self = Self::Collection(points);
            }
            Self::Collection(points_self) => {
                let mut last_id = points_self.0.last().unwrap().id;
                for mut point in points.0 {
                    point.id = last_id + 1;
                    points_self.0.push(point);
                    last_id += 1;
                }
            }
        }
    }

    pub fn iterate(&self) -> impl Iterator<Item = (u32, &Point)> + '_ {
        match self {
            Self::Empty => {
                panic!("Tried to iterate over empty collection of Points.");
            }
            Self::Collection(points) => points.0.iter().map(|p| (p.id, p)),
        }
    }
    pub fn iterate_mut(&mut self) -> impl Iterator<Item = (u32, &mut Point)> + '_ {
        match self {
            Self::Empty => {
                panic!("Tried to iterate over empty collection of Points.");
            }
            Self::Collection(points) => points.0.iter_mut().map(|p| (p.id, p)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Collection(points) => points.0.len(),
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

pub fn get_new_node_layer(ml: f32, rng: &mut ThreadRng) -> u8 {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.gen::<f32>();
        } else {
            break;
        }
    }

    // TODO
    // casting to u8 should only be done if we are
    // confident the value will fall in the range,
    // which depends entirely on ml
    (-rand_nb.log(std::f32::consts::E) * ml).floor() as u8
}
