use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Params {
    pub m: u8,
    pub mmax: u8,
    pub mmax0: u8,
    pub ml: f32,
    pub ef_cons: u32,
    pub dim: u32,
}

impl Params {
    pub fn from_m(m: u8, dim: u32) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: (m * 2) as u8,
            ml: 1.0 / (m as f32).ln(),
            ef_cons: (m as u32) * 2,
            dim,
        }
    }

    pub fn from_m_efcons(m: u8, ef_cons: u32, dim: u32) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).ln(),
            ef_cons,
            dim,
        }
    }

    pub fn from(
        m: u8,
        ef_cons: Option<u32>,
        mmax: Option<u8>,
        mmax0: Option<u8>,
        ml: Option<f32>,
        dim: u32,
    ) -> Params {
        Params {
            m,
            mmax: mmax.unwrap_or(m),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f32).ln()),
            ef_cons: ef_cons.unwrap_or((m as u32) * 2),
            dim,
        }
    }
}
