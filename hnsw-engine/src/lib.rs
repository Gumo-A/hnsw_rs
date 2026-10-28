#[allow(unused)]
mod engine;
mod hnsw;
mod manifest;

use std::path::PathBuf;
use uuid::Uuid;

pub type INT = usize;
pub type FileName = String;

fn path_to_f_name_str(loc: &PathBuf) -> FileName {
    loc.file_name().unwrap().to_string_lossy().to_string()
}

fn get_rand_file_name() -> String {
    Uuid::new_v4().as_hyphenated().to_string()
}
