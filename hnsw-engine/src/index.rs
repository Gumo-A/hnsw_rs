use std::{
    collections::HashSet,
    io,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use graph::dist::Dist;
use itertools::Itertools;
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
    NodeNotInLevel(INT, INT),
}

pub fn map_io_err(e: io::Error) -> HNSWErr {
    HNSWErr::IOError(e)
}

pub fn map_json_err(e: serde_json::Error) -> HNSWErr {
    HNSWErr::JSONErr(e)
}

pub type CachedPoint = Arc<Point>;
pub type CachedEntry = Arc<IndexEntry>;
pub trait Index {
    fn distance(&mut self, a: INT, b: INT) -> Result<f32, HNSWErr>;
    fn distance2point(&mut self, id: INT, point: &Point) -> Result<f32, HNSWErr>;
    fn get_point(&mut self, id: INT) -> Result<CachedPoint, HNSWErr>;
    fn get_entry(&mut self, id: INT) -> Result<CachedEntry, HNSWErr>;
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
    fn commit_results(&self) -> Result<(), HNSWErr>;
    fn len(&self) -> Result<INT, HNSWErr>;
    fn ann(&mut self, point: &Point, n: usize, ef: usize) -> Result<Vec<INT>, HNSWErr>;

    fn cap_neighbors(&mut self, id: INT, level: INT, m: INT) -> Result<(), HNSWErr> {
        let new_neighbors = self
            .get_entry(id)?
            .read_levels()?
            .get(level)
            .ok_or(HNSWErr::NodeNotInLevel(id, level))?
            .iter()
            .map(|n| Dist::new(*n, self.distance(*n, id).unwrap()))
            .sorted()
            .take(m)
            .map(|dist| dist.id);
        self.replace_neighbors(id, new_neighbors, level)
    }

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
pub struct Levels {
    levels: Vec<HashSet<INT>>,
    is_modified: bool,
}

impl Levels {
    pub fn new(level: INT) -> Self {
        Self {
            levels: Vec::from_iter((0..level + 1).map(|_| HashSet::new())),
            is_modified: false,
        }
    }

    pub fn get_levels_cloned(&self) -> Vec<HashSet<INT>> {
        self.levels.clone()
    }

    pub fn replace_levels(&mut self, levels: Vec<HashSet<INT>>) {
        self.levels = levels;
    }

    pub fn len(&self) -> INT {
        self.levels.len()
    }

    pub fn get(&self, level: INT) -> Option<&HashSet<INT>> {
        self.levels.get(level)
    }

    pub fn get_mut(&mut self, level: INT) -> Option<&mut HashSet<INT>> {
        self.levels.get_mut(level)
    }

    pub fn push(&mut self) {
        self.levels.push(HashSet::new())
    }

    pub fn set_is_not_modified(&mut self) {
        self.is_modified = false;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexEntry {
    id: INT,
    levels: RwLock<Levels>,
}

impl Clone for IndexEntry {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            levels: RwLock::new(self.read_levels().unwrap().clone()),
        }
    }
}

impl IndexEntry {
    pub fn new(id: INT, level: INT) -> Self {
        Self {
            id,
            levels: RwLock::new(Levels::new(level)),
        }
    }

    pub fn get_id(&self) -> INT {
        self.id
    }

    pub fn read_levels(&self) -> Result<RwLockReadGuard<'_, Levels>, HNSWErr> {
        self.levels.read().map_err(|_| HNSWErr::Other)
    }

    fn write_levels(&self) -> Result<RwLockWriteGuard<'_, Levels>, HNSWErr> {
        let mut guard = self.levels.write().map_err(|_| HNSWErr::Other)?;
        guard.is_modified = true;
        Ok(guard)
    }

    pub fn degree(&self, level: INT) -> Result<INT, HNSWErr> {
        Ok(self
            .read_levels()?
            .get(level)
            .ok_or(HNSWErr::NodeNotInLevel(self.id, level))?
            .len())
    }

    pub fn add_level(&self) -> Result<(), HNSWErr> {
        self.write_levels()?.push();
        Ok(())
    }

    pub fn add_neighbor(&self, id: INT, level: INT) -> Result<(), HNSWErr> {
        let self_id = self.id;
        self.write_levels()?
            .get_mut(level)
            .ok_or(HNSWErr::NodeNotInLevel(self_id, level))?
            .insert(id);
        Ok(())
    }

    pub fn remove_neighbor(&self, id: INT, level: INT) -> Result<bool, HNSWErr> {
        let self_id = self.id;
        self.write_levels()?
            .get_mut(level)
            .ok_or(HNSWErr::NodeNotInLevel(self_id, level))?
            .remove(&id);
        Ok(true)
    }

    pub fn nb_levels(&self) -> Result<INT, HNSWErr> {
        Ok(self.read_levels()?.len())
    }

    pub fn get_neighbors_cloned(&self, level: INT) -> Result<HashSet<INT>, HNSWErr> {
        Ok(self
            .read_levels()?
            .get(level)
            .ok_or(HNSWErr::NodeNotInLevel(self.id, level))?
            .clone())
    }

    pub fn is_modified(&self) -> Result<bool, HNSWErr> {
        Ok(self.read_levels()?.is_modified)
    }

    pub fn set_is_not_modified(&self) -> Result<(), HNSWErr> {
        Ok(self.write_levels()?.set_is_not_modified())
    }

    pub fn get_levels_cloned(&self) -> Result<Vec<HashSet<INT>>, HNSWErr> {
        Ok(self.read_levels()?.get_levels_cloned())
    }

    pub fn replace_levels(&self, levels: Vec<HashSet<INT>>) -> Result<(), HNSWErr> {
        self.write_levels()?.replace_levels(levels);
        Ok(())
    }
}
