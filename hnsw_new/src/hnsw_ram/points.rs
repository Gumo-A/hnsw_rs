use core::panic;
use nohash_hasher::IntSet;
use rand::rngs::ThreadRng;
use rand::Rng;
use crate::lvq::LVQVec;
use crate::dist::Dist;

#[derive(Debug, Clone)]
pub struct Point {
    pub id: u32,
    pub layer: u8,
    pub vector: LVQVec,
}

impl Point {
    pub fn new(id: u32, layer: u8, vector: &Vec<f32>) -> Self {
        Point {
            id,
            layer,
            vector: LVQVec::new(vector),
        }
    }

    pub fn dist2other(&self, other: &Point) -> Dist {
        Dist::new(self.vector.dist2other(&other.vector), other.id)
    }

    pub fn dist2others<'a, I>(&'a self, others: I) -> impl Iterator<Item = f32> + 'a
    where
        I: Iterator<Item = &'a Point> + 'a,
    {
        self.vector.dist2many(others.map(|p| &p.vector))
    }

    pub fn to_full(&mut self) {
        self.vector.reconstruct();
    }

    pub fn dim(&self) -> usize {
        self.vector.dim()
    }
}

#[derive(Debug, Clone)]
pub struct Points {
    collection : Vec<Point>,
    means: Vec<f32>,
}

impl Points {
    pub fn from_vecs(mut vectors: Vec<Vec<f32>>, ml: f32) -> Self {
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
                Point::new(id as u32, compute_point_layer(ml, &mut rng), v)
        }));

        Self {
            collection,
            means
        }
    }
    
    pub fn get_means(&self) -> &Vec<f32> {
        &self.means
    }

    /// If you call this function, you can be sure that point IDs correspond
    /// to their positions in the vector of points.
    /// Will change the IDs of each point to correspond to their positions if it
    /// it was not the case before.
    /// Returns whether the IDs were modified.
    pub fn assert_ids(&mut self) -> bool {
        let points = &mut self.collection;

        let mut is_ok = true;
        for (idx, point) in points.iter().enumerate() {
            if !(idx == (point.id as usize)) {
                is_ok = false;
                break;
            }
        }
        if is_ok {
            false
        } else {
            for (idx, point) in points.iter_mut().enumerate() {
                point.id = idx as u32;
            }
            true
        }
    }

    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.collection.iter().map(|p| p.id)
    }

    /// Iterator over (ID, Level) pairs of stored Point structs.
    pub fn ids_levels(&self) -> impl Iterator<Item = (u32, u8)> + '_ {
        self.collection.iter().map(|point| (point.id, point.layer))
    }

    pub fn insert(&mut self, point: Point) {
        self.collection.push(point);
    }

    pub fn dim(&self) -> usize {
        self.collection.first().unwrap().vector.dim()
    }

    pub fn remove(&mut self, index: u32) -> Option<Point> {
        if index < self.collection.len() as u32 {
            let deleted_point = self.collection.swap_remove(index as usize);
            self.collection.get_mut(index as usize).unwrap().id = index;
            Some(deleted_point)
        } else {
            None
        }
    }

    pub fn contains(&self, index: &u32) -> bool {
        self.collection.get(*index as usize).is_some()
    }

    pub fn get_point(&self, index: u32) -> Option<&Point> {
        self.collection.get(index as usize)
    }

    pub fn get_points(&self, indices: &IntSet<u32>) -> Vec<&Point> {
        indices
            .iter()
            .map(|idx| self.collection.get(*idx as usize).unwrap())
            .collect()
    }
    pub fn get_point_mut(&mut self, index: u32) -> Option<&mut Point> {
        self.collection.get_mut(index as usize)
    }

    /// Extends the collection with the provided Points struct,
    pub fn extend_or_fill(&mut self, other: Self) {
        self.assert_ids();

        let new_id = self.collection.last().unwrap().id + 1;
        for (idx, mut point) in other.collection.iter().enumerate() {
            point.id = new_id + idx;
            self.collection.push(point);
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

pub fn compute_point_layer(ml: f32, rng: &mut ThreadRng) -> u8 {
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

