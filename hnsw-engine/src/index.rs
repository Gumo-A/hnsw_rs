use std::{io, sync::Arc};

use graph::graph::Graph;
use points::point::Point;

use crate::engine::INT;

pub trait Index {
    fn distance(&mut self, a: INT, b: INT) -> io::Result<Option<f32>>;
    fn distance2point(&mut self, id: INT, point: &Point) -> io::Result<Option<f32>>;
    fn get_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>>;
    fn get_layer(&mut self, id: INT) -> io::Result<Option<Arc<Graph>>>;
    fn get_nb_layers(&self) -> io::Result<INT>;
    fn get_ep(&self) -> INT;
    fn get_m(&self) -> INT;
    fn get_ef_cons(&self) -> INT;
    fn layers_at_level(&mut self, level: INT) -> impl Iterator<Item = INT>;
}
