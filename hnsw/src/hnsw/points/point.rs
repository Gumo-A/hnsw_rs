use crate::hnsw::{
    dist::Node,
    vectors::{ByteVec, FullVec, LVQVec, VecTrait, Vector},
};

#[derive(Debug, Clone)]
pub struct Point {
    pub id: u32,
    pub level: u8,
    vector: Vector,
}

impl Point {
    pub fn new_full(id: u32, level: u8, vector: Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Full(FullVec::new(vector)),
        }
    }
    pub fn new_quant(id: u32, level: u8, vector: Vec<f32>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Quant(LVQVec::new(&vector)),
        }
    }
    pub fn new_byte(id: u32, level: u8, vector: Vec<u8>) -> Point {
        Point {
            id,
            level,
            vector: Vector::Byte(ByteVec::new(vector)),
        }
    }

    pub fn quantize(&mut self) {
        self.vector = Vector::Quant(self.vector.quantize());
    }

    pub fn set_quantized(&mut self, quant: LVQVec) {
        self.vector = Vector::Quant(quant)
    }

    pub fn iter_vals(&self) -> impl Iterator<Item = f32> + '_ {
        self.vector.iter_vals()
    }

    pub fn dim(&self) -> usize {
        self.vector.dim()
    }

    pub fn distance(&self, other: &Point) -> f32 {
        self.vector.distance(&other.vector)
    }

    pub fn dist2many<'a, I>(&'a self, others: I) -> impl Iterator<Item = Node> + 'a
    where
        I: Iterator<Item = &'a Point> + 'a,
    {
        others.map(|other| Node::new_with_dist(self.vector.distance(&other.vector), other.id))
    }
}
