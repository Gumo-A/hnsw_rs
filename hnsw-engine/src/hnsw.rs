mod results;
pub mod traverser;

use graph::NodeID;
use serde::{Deserialize, Serialize};

use crate::engine::INT;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSW {
    pub ep: NodeID,
    pub m: INT,
    pub mmax: INT,
    pub mmax0: INT,
    pub ml: f32,
    pub ef_cons: INT,
    pub cache_cap: INT,
    pub dim: Option<INT>,
}

const CACHE_SIZE_BYTES: INT = 134217728; // 128MiB
const DEFAULT_CACHE_CAP: INT = 130056; // 128MiB worth of dim-1024 QuantVecs
fn get_cache_cap(dim: INT) -> INT {
    // CACHE_SIZE_BYTES / (8 + dim)
    DEFAULT_CACHE_CAP
}

impl HNSW {
    pub fn new(m: INT) -> HNSW {
        HNSW {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).ln(),
            ef_cons: m * 2,
            dim: None,
            ep: 0,
            cache_cap: DEFAULT_CACHE_CAP,
        }
    }

    pub fn set_dim(&mut self, dim: INT) {
        self.dim = Some(dim);
        // self.cache_cap = get_cache_cap(dim);
    }
}
