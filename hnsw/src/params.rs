use graph::nodes::Node;
use vectors::serializer::Serializer;

#[derive(Debug, Clone)]
pub struct Params {
    pub ep: Node,
    pub m: usize,
    pub mmax: usize,
    pub mmax0: usize,
    pub ml: f64,
    pub ef_cons: usize,
    pub dim: usize,
}

pub fn get_default_ml(m: usize) -> f64 {
    1.0 / (m as f64).ln()
}

impl Params {
    pub fn from_m(m: usize, dim: usize) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: get_default_ml(m),
            ef_cons: m * 2,
            dim,
            ep: 0,
        }
    }

    pub fn from_m_efcons(m: usize, ef_cons: usize, dim: usize) -> Params {
        Params {
            m,
            mmax: m,
            mmax0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_cons,
            dim,
            ep: 0,
        }
    }

    pub fn from(
        m: usize,
        ef_cons: Option<usize>,
        mmax: Option<usize>,
        mmax0: Option<usize>,
        ml: Option<f64>,
        dim: usize,
    ) -> Params {
        Params {
            m,
            mmax: mmax.unwrap_or(m),
            mmax0: mmax0.unwrap_or(m * 2),
            ml: ml.unwrap_or(1.0 / (m as f64).ln()),
            ef_cons: ef_cons.unwrap_or(m * 2),
            dim,
            ep: 0,
        }
    }
}

impl Serializer for Params {
    fn size(&self) -> usize {
        56
    }
    /// Val        Bytes
    /// M          8
    /// Mmax       8
    /// Mmax0      8
    /// ml         8
    /// ef_cons    8
    /// dim        8
    /// EP         8
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.m.to_be_bytes());
        bytes.extend_from_slice(&self.mmax.to_be_bytes());
        bytes.extend_from_slice(&self.mmax0.to_be_bytes());
        bytes.extend_from_slice(&self.ml.to_be_bytes());
        bytes.extend_from_slice(&self.ef_cons.to_be_bytes());
        bytes.extend_from_slice(&self.dim.to_be_bytes());
        bytes.extend_from_slice(&(self.ep as usize).to_be_bytes());
        bytes
    }

    /// Val        Bytes
    /// M          8
    /// Mmax       8
    /// Mmax0      8
    /// ml         8
    /// ef_cons    8
    /// dim        8
    /// EP         8
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
        let ep = usize::from_be_bytes(data[i..i + 8].try_into().unwrap()) as Node;
        Params {
            ep,
            m,
            mmax,
            mmax0,
            ml,
            ef_cons,
            dim,
        }
    }
}
