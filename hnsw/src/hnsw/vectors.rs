mod bytes;
mod full;
mod lvq;

pub use bytes::ByteVec;
pub use full::FullVec;
pub use lvq::LVQVec;

pub trait VecTrait {
    fn iter_vals(&self) -> Box<dyn Iterator<Item = f32>>;

    fn dim(&self) -> usize;

    fn distance(&self, other: &impl VecTrait) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .fold(0.0, |acc, e| acc + (e.0 - e.1).powi(2))
            .sqrt()
    }

    fn get_vals(&self) -> Vec<f32> {
        self.iter_vals().collect()
    }

    fn quantize(&self) -> LVQVec {
        LVQVec::new(&self.get_vals())
    }
}

#[derive(Debug, Clone)]
pub enum Vector {
    Full(FullVec),
    Quant(LVQVec),
    Byte(ByteVec),
}

impl VecTrait for Vector {
    fn iter_vals(&self) -> Box<dyn Iterator<Item = f32>> {
        match self {
            Self::Full(v) => v.iter_vals(),
            Self::Quant(v) => v.iter_vals(),
            Self::Byte(v) => v.iter_vals(),
        }
    }
    fn dim(&self) -> usize {
        match self {
            Self::Full(v) => v.dim(),
            Self::Quant(v) => v.dim(),
            Self::Byte(v) => v.dim(),
        }
    }
}

impl VecTrait for FullVec {
    fn iter_vals(&self) -> Box<dyn Iterator<Item = f32>> {
        Box::new(self.vals.iter().copied())
    }
    fn dim(&self) -> usize {
        self.vals.len()
    }
    fn quantize(&self) -> LVQVec {
        LVQVec::new(&self.vals)
    }
}

impl VecTrait for LVQVec {
    fn iter_vals(&self) -> Box<dyn Iterator<Item = f32>> {
        Box::new(self.iter_full())
    }

    fn dim(&self) -> usize {
        self.quantized_vec.len()
    }

    fn quantize(&self) -> LVQVec {
        self.clone()
    }
}

impl VecTrait for ByteVec {
    fn iter_vals(&self) -> Box<dyn Iterator<Item = f32>> {
        Box::new(self.data.iter().map(|x| *x as f32))
    }

    fn dim(&self) -> usize {
        self.data.len()
    }
}
