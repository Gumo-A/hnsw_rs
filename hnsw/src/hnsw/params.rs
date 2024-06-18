use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Params {
    pub m: usize,
    pub mmax: usize,
    pub mmax0: usize,
    pub ml: f32,
    pub ef_cons: usize,
    pub dim: usize,
}

impl Params {
    pub fn from_m(m: usize, dim: usize) -> Params {
        Params {
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons: m * 2,
            dim,
        }
    }

    pub fn from_m_efcons(m: usize, ef_cons: usize, dim: usize) -> Params {
        Params {
            m,
            mmax: m + m / 2,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).log(std::f32::consts::E),
            ef_cons,
            dim,
        }
    }

    pub fn from(
        m: usize,
        ef_cons: Option<usize>,
        mmax: Option<usize>,
        mmax0: Option<usize>,
        ml: Option<f32>,
        dim: usize,
    ) -> Params {
        Params {
            m,
            mmax: mmax.unwrap_or(m + m / 2),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).log(std::f32::consts::E)),
            ef_cons: ef_cons.unwrap_or(m * 2),
            dim,
        }
    }
}
