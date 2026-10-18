use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{AddedPoints, FileName, ID, path_to_f_name_str, read_layer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub loc: PathBuf,
    pub id_ranges: Vec<(ID, ID)>,
}

impl Manifest {
    pub fn new(loc: &PathBuf, first_id: ID, last_id: ID) -> Self {
        let manifest = Manifest {
            loc: loc.clone(),
            id_ranges: vec![(first_id, last_id)],
        };
        manifest
    }

    pub fn with_ranges(loc: &PathBuf, id_ranges: Vec<(ID, ID)>) -> Self {
        let manifest = Manifest {
            loc: loc.clone(),
            id_ranges,
        };
        manifest
    }

    pub fn get_nb_points(&self) -> ID {
        self.id_ranges
            .iter()
            .map(|(first, last)| 1 + last - first)
            .sum()
    }

    pub fn id_in_range(&self, id: ID) -> bool {
        self.id_ranges
            .iter()
            .any(|(first, last)| (id >= *first) & (id <= *last))
    }

    pub fn ids(&self) -> impl Iterator<Item = ID> {
        self.id_ranges
            .iter()
            .flat_map(|(first, last)| (*first..=*last))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestList {
    pub manifests: Vec<Manifest>,
    pub loc: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> Self {
        let metadata_dir = loc.join("metadata-layer");
        match Self::read_current_manifest_list(&metadata_dir) {
            Some(s) => s,
            None => Self::create_first_manifest_list(metadata_dir),
        }
    }

    pub fn from_manifests(loc: &PathBuf, manifests: Vec<Manifest>) -> Self {
        let metadata_dir = loc.join("metadata-layer");
        let loc = Self::get_next_manifest_list_path(&metadata_dir);
        let writer = BufWriter::new(
            File::create_new(&loc).expect("Could not create next manifest list file"),
        );
        let s = Self { manifests, loc };
        serde_json::to_writer_pretty(writer, &s)
            .expect("Could not write contents to new manifest list file");
        s
    }

    fn create_first_manifest_list(metadata_dir: PathBuf) -> Self {
        let loc = metadata_dir.join("0-manifest-list.json");
        let writer = BufWriter::new(
            File::create_new(&loc).expect("Could not create first manifest list file"),
        );
        let manifest_list = ManifestList {
            loc,
            manifests: Vec::new(),
        };
        serde_json::to_writer_pretty(writer, &manifest_list)
            .expect("Could not write empty vec to first manifest list file");
        manifest_list
    }

    fn read_current_manifest_list(metadata_dir: &PathBuf) -> Option<Self> {
        match Self::get_latest_manifest_list_file(metadata_dir) {
            None => None,
            Some(f) => Some(Self::load_manifest_list(&f)),
        }
    }

    pub fn get_prev_manifest_list_paths(&self) -> Vec<PathBuf> {
        let current_nb = self.get_manifest_list_number();
        let dir = read_layer(
            &self
                .loc
                .parent()
                .expect("Could not read metadata dir form manifest list file")
                .to_path_buf(),
        );
        let to_remove = dir
            .map(|e| {
                e.expect("Could not read directory entry in metadata-layer")
                    .path()
            })
            .filter(|e| path_to_f_name_str(&e).ends_with("-manifest-list.json"))
            .filter(|e| Self::parse_int_from_file_name(&path_to_f_name_str(&e)) < current_nb)
            .map(|e| e.to_path_buf())
            .collect();
        to_remove
    }

    pub fn get_manifest_list_number(&self) -> usize {
        Self::parse_int_from_file_name(&path_to_f_name_str(&self.loc))
    }

    pub fn get_manifest_from_id(&self, id: ID) -> Option<&Manifest> {
        for manifest in self.manifests.iter() {
            if manifest.id_in_range(id) {
                return Some(manifest);
            }
        }
        None
    }

    fn get_latest_manifest_list_file(metadata_dir: &PathBuf) -> Option<PathBuf> {
        let dir = read_layer(metadata_dir);
        let latest = dir
            .map(|e| {
                let e = e
                    .expect("Could not read directory entry in metadata-layer")
                    .path();
                (path_to_f_name_str(&e), e)
            })
            .filter(|(s, _)| s.ends_with("-manifest-list.json"))
            .max_by_key(|(s, _)| Self::parse_int_from_file_name(s));
        match latest {
            None => None,
            Some((_, p)) => Some(p),
        }
    }

    fn parse_int_from_file_name(f_name: &FileName) -> usize {
        f_name
            .split_once("-")
            .expect("Could not split file name by '-'")
            .0
            .parse::<usize>()
            .expect("Could not parse integer from file name")
    }

    fn load_manifest_list(loc: &PathBuf) -> Self {
        let f = File::open(loc).unwrap();
        let reader = BufReader::new(f);
        let r = serde_json::from_reader::<BufReader<File>, ManifestList>(reader)
            .expect("serde_json could not parse Manifest List");
        r
    }

    /// we assume we have the write lock here
    pub fn update(&mut self, new_manifest: &Manifest) {
        let new_path = Self::get_next_manifest_list_path(
            &self
                .loc
                .parent()
                .expect("Could not get metadata dir from manifest list path")
                .to_path_buf(),
        );
        let writer = BufWriter::new(
            File::create_new(&new_path).expect("Could not create next manifest list file"),
        );
        self.loc = new_path;
        self.manifests.push(new_manifest.clone());
        serde_json::to_writer_pretty(writer, &self)
            .expect("Could not write contents to new manifest list file");
    }

    pub fn get_next_manifest_list_path(metadata_dir: &PathBuf) -> PathBuf {
        let latest = match Self::get_latest_manifest_list_file(metadata_dir) {
            Some(p) => p,
            None => panic!("Was expecting a manifest list file to exist, but didn't find it"),
        };

        let next_idx = Self::parse_int_from_file_name(
            &latest.file_name().unwrap().to_string_lossy().to_string(),
        ) + 1;
        let next_file_name = format!("{next_idx}-manifest-list.json");
        metadata_dir.join(next_file_name)
    }

    pub fn get_nb_points(&self) -> ID {
        self.manifests
            .iter()
            .map(|manifest| manifest.get_nb_points())
            .sum()
    }

    pub fn get_optimization_plan(&self, optimal_size: ID) -> Vec<Vec<Manifest>> {
        let mut plan = Vec::new();
        let mut manifests = Vec::new();
        let mut to_combine: Vec<_> = self.manifests.iter().collect();
        to_combine.sort_by_key(|m| m.get_nb_points());
        for manif in to_combine.iter() {
            let n: ID = manifests.iter().map(|m: &Manifest| m.get_nb_points()).sum();
            if (n <= optimal_size) & ((n + manif.get_nb_points()) > optimal_size) {
                plan.push(manifests.clone());
                manifests.clear();
            }
            manifests.push((*manif).clone());
        }
        if manifests.len() > 0 {
            plan.push(manifests.clone());
        }
        plan
    }
}
