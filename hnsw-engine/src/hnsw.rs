mod results;
pub mod traverser;

use std::sync::{Arc, RwLock, RwLockReadGuard};

use log::debug;
use serde::{Deserialize, Serialize};

use crate::engine::INT;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSW {
    pub m: INT,
    pub mmax: INT,
    pub mmax0: INT,
    pub ml: f32,
    pub ef_cons: INT,
    pub cache_cap: INT,
    pub dim: Arc<RwLock<Option<INT>>>,
    pub ep: Arc<RwLock<INT>>,
}

const CACHE_SIZE_BYTES: INT = 134217728; // 128MiB
const DEFAULT_CACHE_CAP: INT = 130056; // 128MiB worth of dim-1024 QuantVecs
fn get_cache_cap(dim: INT) -> INT {
    CACHE_SIZE_BYTES / (8 + dim)
}

impl HNSW {
    pub fn new(m: INT) -> HNSW {
        HNSW {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: 1.0 / (m as f32).ln(),
            ef_cons: m * 2,
            dim: Arc::new(RwLock::new(None)),
            ep: Arc::new(RwLock::new(0)),
            cache_cap: DEFAULT_CACHE_CAP,
        }
    }

    pub fn set_dim(&self, dim: INT) {
        debug!("Setting DIM to {dim}");
        self.dim.write().unwrap().replace(dim);
        // self.cache_cap = get_cache_cap(dim);
    }

    pub fn set_ep(&self, ep: INT) {
        debug!("Setting EP to {ep}");
        *self.ep.write().unwrap() = ep;
    }

    pub fn dim_is_none(&self) -> bool {
        self.dim.read().unwrap().is_none()
    }

    pub fn get_dim(&self) -> RwLockReadGuard<Option<INT>> {
        self.dim.read().unwrap()
    }

    pub fn get_ep(&self) -> RwLockReadGuard<INT> {
        self.ep.read().unwrap()
    }
}
