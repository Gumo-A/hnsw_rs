use crate::{VecBase, VecTrait, serializer::Serializer};

const CHUNK_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct FullVec {
    pub vector: Vec<f32>,
}

impl FullVec {
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vector.iter_mut()
    }
    pub fn iter_vals(&mut self) -> impl Iterator<Item = &f32> {
        self.vector.iter()
    }
}

impl VecBase for FullVec {
    fn new(vector: &Vec<f32>) -> Self {
        FullVec {
            vector: vector.clone(),
        }
    }
    fn distance(&self, other: &impl VecBase) -> f32 {
        let other = other.get_vals();
        let mut acc = [0.0f32; CHUNK_SIZE];
        let vector_chunks = other.chunks_exact(CHUNK_SIZE);
        let chunks_iter = self.vector.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();

        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (x - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (x - y).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }

    fn dist2other(&self, other: &Self) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let chunks_iter = self.vector.chunks_exact(CHUNK_SIZE);
        let vector_chunks = other.vector.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();
        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (x - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (x - y).powi(2);
        }
        let dist = acc.iter().sum::<f32>().sqrt();
        dist
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vector.iter().copied()
    }

    fn dim(&self) -> usize {
        self.vector.len()
    }

    fn center(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x -= means[idx]);
    }

    fn decenter(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x += means[idx]);
    }
}

impl Serializer for FullVec {
    fn size(&self) -> usize {
        self.dim() * 4
    }

    /// blocks of 4 bytes
    /// for the floats (big endian)
    ///
    /// Val    Bytes
    /// vector dim * 4
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for float in self.vector.iter() {
            bytes.extend_from_slice(&float.to_be_bytes());
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let bytes = data.chunks_exact(4);
        let mut vals: Vec<f32> = Vec::new();
        for bytes_arr in bytes {
            vals.push(f32::from_be_bytes(bytes_arr.try_into().unwrap()))
        }
        FullVec::new(&vals)
    }
}

impl VecTrait for FullVec {}

#[cfg(test)]
mod tests {
    use crate::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let ser_a = a.serialize();
        let b = FullVec::deserialize(ser_a);
        for (a_val, b_val) in a.iter_vals().zip(b.iter_vals()) {
            assert_eq!(a_val, b_val);
        }
    }

    #[test]
    fn distance() {
        let mut others = Vec::new();
        for _ in 0..100 {
            let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
            others.push(a);
        }
        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        a.dist2many(others.iter())
            .for_each(|dist| assert!(dist >= 0.0));

        let a = FullVec::new(&vec![0.5]);
        let b = FullVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.25);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![0.75]);
        let b = FullVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.5);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![0.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 1.0);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![-1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, -1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let b = a.clone();
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.0);
        assert_eq!(dist, dist2other);
    }

    #[test]
    fn center_decenter() {
        let n = 128;
        let means = gen_rand_vecs(n, 1)[0].clone();
        let mut vectors: Vec<FullVec> = gen_rand_vecs(n, 4)
            .iter()
            .map(|v| FullVec::new(&v.clone()))
            .collect();
        let vectors_clone = vectors.clone();
        for (v, vc) in vectors.iter_mut().zip(vectors_clone.iter()) {
            v.center(&means);
            for (idx, (v_val, vc_val)) in v.iter_vals().zip(vc.iter_vals()).enumerate() {
                let err = (v_val - (vc_val - means[idx])).abs() / (vc_val - means[idx]);
                assert!(err < 0.0001);
            }
        }

        for (v, vc) in vectors.iter_mut().zip(vectors_clone.iter()) {
            v.decenter(&means);
            for (v_val, vc_val) in v.iter_vals().zip(vc.iter_vals()) {
                let err = (v_val - vc_val).abs() / vc_val;
                assert!(err < 0.0001);
            }
        }
    }
}
