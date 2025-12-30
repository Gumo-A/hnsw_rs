use graph::nodes::NodeID;
use points::points::block::BlockID;
use vectors::serializer::Serializer;

#[derive(Debug, Clone)]
pub struct Params {
    pub ep: NodeID,
    pub m: usize,
    pub mmax: usize,
    pub mmax0: usize,
    pub ml: f64,
    pub ef_cons: usize,
    pub dim: usize,
    pub max_per_block: BlockID,
}

pub fn get_default_ml(m: usize) -> f64 {
    1.0 / (m as f64).ln()
}

impl Params {
    pub fn from_m(m: usize, dim: usize, max_per_block: BlockID) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: get_default_ml(m),
            ef_cons: m * 2,
            dim,
            ep: 0,
            max_per_block,
        }
    }

    pub fn from_m_efcons(m: usize, ef_cons: usize, dim: usize, max_per_block: BlockID) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_cons,
            dim,
            ep: 0,
            max_per_block,
        }
    }

    pub fn from(
        m: usize,
        ef_cons: Option<usize>,
        mmax: Option<usize>,
        mmax0: Option<usize>,
        ml: Option<f64>,
        dim: usize,
        max_per_block: BlockID,
    ) -> Params {
        Params {
            m,
            mmax: mmax.unwrap_or(m),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f64).ln()),
            ef_cons: ef_cons.unwrap_or(m * 2),
            dim,
            ep: 0,
            max_per_block,
        }
    }
}

impl Serializer for Params {
    /// Val            Bytes
    /// M              8
    /// Mmax           8
    /// Mmax0          8
    /// ml             8
    /// ef_cons        8
    /// dim            8
    /// EP             8
    /// max_per_block  2
    fn size(&self) -> usize {
        58
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.m.to_be_bytes());
        bytes.extend_from_slice(&self.mmax.to_be_bytes());
        bytes.extend_from_slice(&self.mmax0.to_be_bytes());
        bytes.extend_from_slice(&self.ml.to_be_bytes());
        bytes.extend_from_slice(&self.ef_cons.to_be_bytes());
        bytes.extend_from_slice(&self.dim.to_be_bytes());
        bytes.extend_from_slice(&(self.ep as usize).to_be_bytes());
        bytes.extend_from_slice(&(self.max_per_block).to_be_bytes());
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let mut i = 0;
        let m = usize::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let mmax = usize::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let mmax0 = usize::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let ml = f64::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let ef_cons = usize::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let dim = usize::from_be_bytes(data[i..i + 8].try_into().unwrap());
        i += 8;
        let ep = usize::from_be_bytes(data[i..i + 8].try_into().unwrap()) as NodeID;
        i += 8;
        let max_per_block = BlockID::from_be_bytes(data[i..i + 2].try_into().unwrap());
        Params {
            ep,
            m,
            mmax,
            mmax0,
            ml,
            ef_cons,
            dim,
            max_per_block,
        }
    }
}
