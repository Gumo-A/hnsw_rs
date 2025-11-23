use vectors::{FullVec, LVQVec, VecTrait};

#[derive(Debug, Clone)]
pub struct Point<T: VecTrait> {
    pub id: u32,
    pub level: u8,
    removed: bool,
    vector: T,
}

impl<T: VecTrait> Point<T> {
    /// Marks the point as removed,
    /// returns true if the toggle was made,
    /// false if the point was already marked.
    pub fn set_removed(&mut self) -> bool {
        if self.removed {
            false
        } else {
            self.removed = true;
            true
        }
    }
}

impl<T: VecTrait> VecTrait for Point<T> {
    fn distance(&self, other: &impl VecTrait) -> f32 {
        self.vector.distance(other)
    }
    fn dist2other(&self, other: &Self) -> f32 {
        self.vector.dist2other(&other.vector)
    }
    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vector.iter_vals()
    }
    fn dim(&self) -> usize {
        self.vector.dim()
    }
    fn center(&mut self, means: &Vec<f32>) {
        self.vector.center(means);
    }
    fn decenter(&mut self, means: &Vec<f32>) {
        self.vector.decenter(means);
    }
}

impl Point<LVQVec> {
    pub fn new_quant(id: u32, level: u8, vector: &Vec<f32>) -> Point<LVQVec> {
        Point {
            id,
            level,
            removed: false,
            vector: LVQVec::new(vector),
        }
    }
}

impl Point<FullVec> {
    pub fn new_full(id: u32, level: u8, vector: Vec<f32>) -> Point<FullVec> {
        Point {
            id,
            level,
            removed: false,
            vector: FullVec::new(vector),
        }
    }

    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vector.iter_vals_mut()
    }

    pub fn get_low_vector(&self) -> &Vec<f32> {
        &self.vector.vals
    }

    pub fn quantized(&self) -> Point<LVQVec> {
        Point::new_quant(self.id, self.level, self.get_low_vector())
    }
}
