use std::{collections::HashSet, io, sync::Arc};

use points::point::Point;
use serde::{Deserialize, Serialize};

use crate::engine::INT;

#[derive(Debug)]
pub enum HNSWErr {
    Other,
    IDNotFound(INT),
    IDPairNotFound(INT, INT),
    IOError(io::Error),
    JSONErr(serde_json::Error),
    NothingToInsert,
    InvalidDimension,
    CacheMiss,
}

pub fn map_io_err(e: io::Error) -> HNSWErr {
    HNSWErr::IOError(e)
}

pub fn map_json_err(e: serde_json::Error) -> HNSWErr {
    HNSWErr::JSONErr(e)
}

pub trait Index {
    fn distance(&mut self, a: INT, b: INT) -> Result<f32, HNSWErr>;
    fn distance2point(&mut self, id: INT, point: &Point) -> Result<f32, HNSWErr>;
    fn get_point(&mut self, id: INT) -> Result<Arc<Point>, HNSWErr>;
    fn get_entry(&mut self, id: INT) -> Result<Arc<IndexEntry>, HNSWErr>;
    fn get_ep(&self) -> INT;
    fn get_m(&self) -> INT;
    fn get_m_at_level(&self, level: INT) -> INT;
    fn get_ef_cons(&self) -> INT;
    fn get_nb_levels(&mut self) -> Result<INT, HNSWErr>;
    fn degree(&mut self, id: INT, level: INT) -> Result<INT, HNSWErr>;
    fn add_node(&mut self, id: INT, level: INT) -> Result<(), HNSWErr>;
    fn add_edge(&mut self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr>;
    fn remove_edge(&mut self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr>;
    fn neighbors(&mut self, id: INT, level: INT) -> Result<HashSet<INT>, HNSWErr>;

    fn get_level(&mut self, id: INT) -> Result<INT, HNSWErr> {
        Ok(self.get_point(id)?.level.into())
    }

    fn add_neighbors<I>(&mut self, id: INT, new_neighbors: I, level: INT) -> Result<(), HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        for i in new_neighbors {
            self.add_edge(id, i, level)?;
        }

        Ok(())
    }

    fn remove_neighbors<I>(&mut self, id: INT, old_neighbors: I, level: INT) -> Result<(), HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        for n in old_neighbors {
            match self.remove_edge(id, n, level) {
                Ok(()) => (),
                Err(e) => match e {
                    // GraphError::IsolatedNode(a) => {
                    //     warn!("Was going to remove edge ({node}, {n}), but {a} would be isolated");
                    // }
                    _ => return Err(e),
                },
            };
        }

        Ok(())
    }

    fn replace_neighbors<I>(&mut self, id: INT, new_neighbors: I, level: INT) -> Result<(), HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        let new_neighbors = HashSet::from_iter(new_neighbors);
        let current_neighbors = self.neighbors(id, level)?;
        let to_remove = current_neighbors.difference(&new_neighbors).copied();
        let to_add = new_neighbors.difference(&current_neighbors).copied();
        self.add_neighbors(id, to_add, level)?;
        self.remove_neighbors(id, to_remove, level)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: INT,
    pub levels: Vec<HashSet<INT>>,
}

impl IndexEntry {
    pub fn new(id: INT, level: INT) -> Self {
        Self {
            id,
            levels: Vec::from_iter((0..level + 1).map(|_| HashSet::new())),
        }
    }
}
