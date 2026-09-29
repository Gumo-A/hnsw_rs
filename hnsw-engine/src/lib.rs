mod engine;
mod manifest;

use std::path::PathBuf;
use uuid::Uuid;

pub type ID = u64;
pub type AddedPoints = u64;
pub type FileName = String;

fn path_to_f_name_str(loc: &PathBuf) -> String {
    loc.file_name().unwrap().to_string_lossy().to_string()
}

fn read_layer(loc: &PathBuf) -> std::fs::ReadDir {
    match loc.read_dir() {
        Ok(d) => d,
        Err(e) => panic!("Could not read {loc:?}: {e}"),
    }
}

fn get_rand_file_name() -> String {
    Uuid::new_v4().as_hyphenated().to_string()
}
