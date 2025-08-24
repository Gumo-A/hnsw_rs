#[derive(Debug, Clone)]
pub struct Config {
    pub m: u8,
    pub mmax: u8,
    pub mmax0: u8,
    pub ml: f32,
    pub ef_cons: usize,
    pub dim: u32,
}

impl Config {
    pub fn new(m: u8, dim: u32) -> Config {
        Config {
            m,
            mmax: m,
            mmax0: (m * 2) as u8,
            ml: 1.0 / (m as f32).ln(),
            ef_cons: (m as usize) * 2,
            dim,
        }
    }
}
