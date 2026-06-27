const BITS: i32 = 8;
const CHUNK_SIZE: usize = 8;

use crate::{VecBase, VecTrait, serializer::Serializer};

#[derive(Debug, Clone)]
pub struct QuantVec {
    delta: f32,
    min: f32,
    codes: Vec<u8>,
}

impl QuantVec {
    fn distance_unrolled(&self, other: &QuantVec) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];

        let self_chunks = self.codes.chunks_exact(CHUNK_SIZE);
        let other_chunks = other.codes.chunks_exact(CHUNK_SIZE);

        let self_rem = self_chunks.remainder();
        let other_rem = other_chunks.remainder();

        for (chunkx, chunky) in self_chunks.zip(other_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                let x_f32 = ((*x as f32) * self.delta) + self.min;
                let y_f32 = ((*y as f32) * other.delta) + other.min;
                acc[idx] += (x_f32 - y_f32).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            let x_f32 = (*x as f32) * self.delta + self.min;
            let y_f32 = (*y as f32) * other.delta + other.min;
            acc[0] += (x_f32 - y_f32).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }
}

impl VecBase for QuantVec {
    fn new(vector: &Vec<f32>) -> QuantVec {
        let upper_bound: f32 = *vector
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let lower_bound: f32 = *vector
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let delta: f32 = (upper_bound - lower_bound) / (2.0f32.powi(BITS) - 1.0);

        let quantized: Vec<u8> = vector
            .iter()
            .map(|x| {
                let mut buffer: f32 = (x - lower_bound) / delta;
                buffer += 0.5f32;
                buffer.floor() as u8
            })
            .collect();

        QuantVec {
            delta,
            min: lower_bound,
            codes: quantized,
        }
    }
    fn distance(&self, other: &impl VecBase) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn dist2other(&self, other: &Self) -> f32 {
        self.distance_unrolled(other)
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.codes
            .iter()
            .map(|x| ((*x as f32) * self.delta) + self.min)
    }

    fn dim(&self) -> usize {
        self.codes.len()
    }
}

impl Serializer for QuantVec {
    fn size(&self) -> usize {
        8 + self.dim()
    }

    /// 4 for low, 4 for delta
    /// followed by dim bytes
    ///
    /// Val       Bytes
    /// low       4
    /// delta     4
    /// quant_vec dim
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.min.to_be_bytes());
        bytes.extend_from_slice(&self.delta.to_be_bytes());
        for byte in self.codes.iter() {
            bytes.push(*byte);
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let lower = f32::from_be_bytes(data[..4].try_into().unwrap());
        let delta = f32::from_be_bytes(data[4..8].try_into().unwrap());
        let mut quantized_vec: Vec<u8> = Vec::new();
        for bytes_arr in data.iter().skip(8) {
            quantized_vec.push(*bytes_arr);
        }
        QuantVec {
            delta,
            min: lower,
            codes: quantized_vec,
        }
    }
}

impl VecTrait for QuantVec {}

#[cfg(test)]
mod tests {
    use crate::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let a = QuantVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let ser_a = a.serialize();
        let b = QuantVec::deserialize(ser_a);
        for (a_val, b_val) in a.iter_vals().zip(b.iter_vals()) {
            assert_eq!(a_val, b_val);
        }
    }

    #[test]
    fn distance() {
        let mut others = Vec::new();
        for _ in 0..100 {
            let a = QuantVec::new(&gen_rand_vecs(128, 1)[0].clone());
            others.push(a);
        }
        let a = QuantVec::new(&gen_rand_vecs(128, 1)[0].clone());
        a.dist2many(others.iter())
            .for_each(|dist| assert!(dist >= 0.0));

        let a = QuantVec::new(&vec![0.5]);
        let b = QuantVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.25);
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&vec![0.75]);
        let b = QuantVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.5);
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&vec![0.0, 0.0]);
        let b = QuantVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 1.0);
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&vec![1.0, 0.0]);
        let b = QuantVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&vec![-1.0, 0.0]);
        let b = QuantVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&vec![1.0, 0.0]);
        let b = QuantVec::new(&vec![0.0, -1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = QuantVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let b = a.clone();
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.0);
        assert_eq!(dist, dist2other);
    }
}
