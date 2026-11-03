use std::{collections::HashSet, io, sync::Arc};

use points::point::Point;
use serde::{Deserialize, Serialize};

use crate::engine::INT;

pub trait Index {
    fn distance(&mut self, a: INT, b: INT) -> io::Result<Option<f32>>;
    fn distance2point(&mut self, id: INT, point: &Point) -> io::Result<Option<f32>>;
    fn get_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>>;
    fn get_ep(&self) -> INT;
    fn get_m(&self) -> INT;
    fn get_m_at_level(&self, level: INT) -> INT;
    fn get_ef_cons(&self) -> INT;
    fn get_nb_levels(&self) -> io::Result<INT>;
    fn degree(&self, id: INT, level: INT) -> io::Result<INT>;
    fn add_node(&self, id: INT, level: INT) -> io::Result<()>;
    fn add_edge(&self, a: INT, b: INT, level: INT) -> io::Result<()>;
    fn remove_edge(&self, a: INT, b: INT, level: INT) -> io::Result<()>;
    fn neighbors(&self, id: INT, level: INT) -> io::Result<HashSet<INT>>;

    fn get_level(&mut self, id: INT) -> io::Result<Option<INT>> {
        match self.get_point(id)? {
            Some(point) => Ok(Some(point.level.into())),
            None => Ok(None),
        }
    }

    fn add_neighbors<I>(&self, id: INT, new_neighbors: I, level: INT) -> io::Result<()>
    where
        I: Iterator<Item = INT>,
    {
        for i in new_neighbors {
            self.add_edge(id, i, level)?;
        }

        Ok(())
    }

    fn remove_neighbors<I>(&self, id: INT, old_neighbors: I, level: INT) -> io::Result<()>
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

    fn replace_neighbors<I>(&self, id: INT, new_neighbors: I, level: INT) -> io::Result<()>
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
    pub levels: Vec<Vec<INT>>,
}

impl IndexEntry {
    pub fn new(id: INT, level: INT) -> Self {
        Self {
            id,
            levels: Vec::from_iter((0..level + 1).map(|_| Vec::new())),
        }
    }
}
