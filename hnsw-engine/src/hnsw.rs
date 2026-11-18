pub mod traverser;

use std::{
    env,
    sync::{Arc, RwLock, RwLockReadGuard},
};

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
    pub block_size: INT,
    pub dim: Arc<RwLock<Option<INT>>>,
    pub ep: Arc<RwLock<INT>>,
}

const DEFAULT_BLOCK_SIZE: INT = 16384; // QuantVecs per file

impl HNSW {
    pub fn new(m: INT) -> HNSW {
        HNSW {
            m,
            mmax: env::var("HNSW_MMAX")
                .ok()
                .and_then(|v| v.parse::<INT>().ok())
                .unwrap_or(m),
            mmax0: env::var("HNSW_MMAX0")
                .ok()
                .and_then(|v| v.parse::<INT>().ok())
                .unwrap_or(m * 2),
            ml: env::var("HNSW_ML")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(1.0 / (m as f32).ln()),
            ef_cons: env::var("HNSW_EF_CONS")
                .ok()
                .and_then(|v| v.parse::<INT>().ok())
                .unwrap_or(m * 2),
            dim: Arc::new(RwLock::new(None)),
            ep: Arc::new(RwLock::new(0)),
            block_size: env::var("HNSW_BLOCK_SIZE")
                .ok()
                .and_then(|t| t.parse::<INT>().ok())
                .unwrap_or(DEFAULT_BLOCK_SIZE),
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

    pub fn get_dim(&self) -> RwLockReadGuard<'_, Option<INT>> {
        self.dim.read().unwrap()
    }

    pub fn get_ep(&self) -> RwLockReadGuard<'_, INT> {
        self.ep.read().unwrap()
    }
}
