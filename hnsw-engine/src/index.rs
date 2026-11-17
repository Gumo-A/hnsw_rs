use std::{
    io,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use graph::dist::Dist;
use itertools::Itertools;
use log::{debug, trace};
use nohash_hasher::IntSet;
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
    PointCacheMiss(INT),
    EntryCacheMiss(INT),
    NodeNotInLevel(INT, INT),
    SelfConnection,
    WouldIsolate(INT),
    NoMatchingManifest(INT),
}

pub fn map_io_err(e: io::Error) -> HNSWErr {
    HNSWErr::IOError(e)
}

pub fn map_json_err(e: serde_json::Error) -> HNSWErr {
    HNSWErr::JSONErr(e)
}

pub type CachedPoint = Arc<Point>;
pub type CachedEntry = Arc<IndexEntry>;
pub trait Index: IndexFacade + Clone + Send + Sync {}
pub trait IndexFacade {
    fn distance(&self, a: INT, b: INT) -> Result<f32, HNSWErr>;
    fn distance2point(&self, id: INT, point: &Point) -> Result<f32, HNSWErr>;
    fn get_point(&self, id: INT) -> Result<CachedPoint, HNSWErr>;
    fn get_entry(&self, id: INT) -> Result<CachedEntry, HNSWErr>;
    fn get_ep(&self) -> INT;
    fn set_ep(&self, id: INT) -> Result<(), HNSWErr>;
    fn get_m(&self) -> INT;
    fn get_m_at_level(&self, level: INT) -> INT;
    fn get_ef_cons(&self) -> INT;
    fn get_nb_levels(&self) -> Result<INT, HNSWErr>;
    fn degree(&self, id: INT, level: INT) -> Result<INT, HNSWErr>;
    fn add_node(&self, id: INT, level: INT) -> Result<(), HNSWErr>;
    fn add_edge(&self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr>;
    fn remove_edge(&self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr>;
    fn neighbors(&self, id: INT, level: INT) -> Result<IntSet<INT>, HNSWErr>;
    fn commit_results(&self) -> Result<(), HNSWErr>;
    fn len(&self) -> Result<INT, HNSWErr>;
    fn ann(&self, point: &Point, n: usize, ef: usize) -> Result<Vec<INT>, HNSWErr>;

    fn cap_neighbors(&self, id: INT, level: INT, m: INT) -> Result<isize, HNSWErr> {
        trace!("Capping neighbors for {id} at level {level} to max of {m}");
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

    fn get_level(&self, id: INT) -> Result<INT, HNSWErr> {
        Ok(self.get_entry(id)?.nb_levels()? - 1)
    }

    fn add_neighbors<I>(&self, id: INT, new_neighbors: I, level: INT) -> Result<INT, HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        let mut added = 0;
        for i in new_neighbors {
            let _ = match self.add_edge(id, i, level) {
                Ok(()) => {
                    added += 1;
                    Ok(())
                }
                Err(e) => match e {
                    HNSWErr::SelfConnection => {
                        trace!("Attempted to connect {id}-{i}, ignoring request.");
                        Ok(())
                    }
                    _ => Err(e),
                },
            };
        }

        Ok(added)
    }

    fn remove_neighbors<I>(&self, id: INT, old_neighbors: I, level: INT) -> Result<INT, HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        let mut removed = 0;
        for n in old_neighbors {
            match self.remove_edge(id, n, level) {
                Ok(()) => removed += 1,
                Err(e) => match e {
                    HNSWErr::WouldIsolate(a) => {
                        debug!(
                            "Was going to remove edge {id}-{n}, but {a} would be isolated, ignoring request"
                        );
                    }
                    _ => return Err(e),
                },
            };
        }
        Ok(removed)
    }

    fn replace_neighbors<I>(&self, id: INT, new_neighbors: I, level: INT) -> Result<isize, HNSWErr>
    where
        I: Iterator<Item = INT>,
    {
        trace!("Replacing neighbors for {id} at level {level}");
        let new_neighbors = IntSet::from_iter(new_neighbors);
        let current_neighbors = self.neighbors(id, level)?;
        trace!(
            "Replacing neighbors: {id} at level {level} has {} neighbors",
            current_neighbors.len()
        );
        trace!(
            "Replacing neighbors: {id} at level {level} will have {} neighbors",
            new_neighbors.len()
        );
        let to_remove = current_neighbors.difference(&new_neighbors).copied();
        let to_add = new_neighbors.difference(&current_neighbors).copied();
        let mut delta: isize = 0;
        delta += self.add_neighbors(id, to_add, level)? as isize;
        delta -= self.remove_neighbors(id, to_remove, level)? as isize;
        Ok(delta)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Levels {
    levels: Vec<IntSet<INT>>,
    #[serde(skip)]
    is_modified: bool,
}

impl Levels {
    pub fn new(level: INT) -> Self {
        Self {
            levels: Vec::from_iter((0..level + 1).map(|_| IntSet::default())),
            is_modified: false,
        }
    }

    pub fn get_levels_cloned(&self) -> Vec<IntSet<INT>> {
        self.levels.clone()
    }

    pub fn len(&self) -> INT {
        self.levels.len()
    }

    pub fn get(&self, level: INT) -> Option<&IntSet<INT>> {
        self.levels.get(level)
    }

    pub fn replace_levels(&mut self, levels: Vec<IntSet<INT>>) {
        self.levels = levels;
        self.is_modified = true;
    }

    pub fn add_neighbor(&mut self, id: INT, level: INT) -> Option<()> {
        self.levels.get_mut(level)?.insert(id);
        self.is_modified = true;
        Some(())
    }

    pub fn remove_neighbor(&mut self, id: INT, level: INT) -> Option<bool> {
        let removed = self.levels.get_mut(level)?.remove(&id);
        self.is_modified = true;
        Some(removed)
    }

    pub fn push(&mut self) {
        self.is_modified = true;
        self.levels.push(IntSet::default())
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
            .add_neighbor(id, level)
            .ok_or(HNSWErr::NodeNotInLevel(self_id, level))?;
        Ok(())
    }

    pub fn remove_neighbor(&self, id: INT, level: INT) -> Result<bool, HNSWErr> {
        let self_id = self.id;
        self.write_levels()?
            .remove_neighbor(id, level)
            .ok_or(HNSWErr::NodeNotInLevel(self_id, level))
    }

    pub fn nb_levels(&self) -> Result<INT, HNSWErr> {
        Ok(self.read_levels()?.len())
    }

    pub fn get_neighbors_cloned(&self, level: INT) -> Result<IntSet<INT>, HNSWErr> {
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

    pub fn get_levels_cloned(&self) -> Result<Vec<IntSet<INT>>, HNSWErr> {
        Ok(self.read_levels()?.get_levels_cloned())
    }

    pub fn replace_levels(&self, levels: Vec<IntSet<INT>>) -> Result<(), HNSWErr> {
        self.write_levels()?.replace_levels(levels);
        Ok(())
    }
}
