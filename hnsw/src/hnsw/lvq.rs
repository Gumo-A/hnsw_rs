// Only usable with bits = 8 for now
#[derive(Debug)]
pub struct LVQVec {
    upper: f32,
    lower: f32,
    quantized_vec: Vec<u8>,
}

impl LVQVec {
    pub fn reconstruct(&self) -> Vec<f32> {
        let recontructed: Vec<f32> = self
            .quantized_vec
            .iter()
            .map(|x| self.lower + ((*x as f32) * (self.upper - self.lower) / 255.0f32))
            .collect();
        recontructed
    }
}

// Scalar quantization as defined in the paper
pub fn Q(vector: &Vec<f32>, bits: usize) -> LVQVec {
    let upper_bound: f32 = *vector
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let lower_bound: f32 = *vector
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let delta: f32 = (upper_bound - lower_bound) / (2.0f32.powi(bits as i32) - 1.0);

    let quantized_inter: Vec<f32> = vector
        .iter()
        .map(|x| {
            let mut buffer: f32 = (x - lower_bound) / delta;
            buffer += 0.5f32;
            buffer = buffer.floor() * delta;
            buffer += lower_bound;
            buffer
        })
        .collect();

    let quantized: Vec<u8> = quantized_inter
        .iter()
        .map(|x| (255.0 * (x - lower_bound) / (upper_bound - lower_bound)) as u8)
        .collect();

    LVQVec {
        upper: upper_bound,
        lower: lower_bound,
        quantized_vec: quantized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn quantization() {
        let mut rng = rand::thread_rng();
        let test_vec = (0..100).map(|_| rng.gen::<f32>()).collect();
        let quantized = Q(&test_vec, 8);
        let q = &quantized.quantized_vec;
        let recontructed = quantized.reconstruct();

        println!("{:?}", test_vec);
        println!("{:?}", quantized);
        println!("{:?}", recontructed);

        println!("{}", std::mem::size_of_val(&*test_vec));
        println!("{}", std::mem::size_of_val(&*q));
        println!("{}", std::mem::size_of_val(&*recontructed));
    }
}
