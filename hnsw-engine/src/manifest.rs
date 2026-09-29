use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};

use crate::{AddedPoints, FileName, ID, path_to_f_name_str, read_layer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    loc: PathBuf,
    id_offset: ID,
    nb_points: ID,
}

impl Manifest {
    pub fn new(loc: &PathBuf, id_offset: ID, nb_points: AddedPoints) -> Self {
        let manifest = Manifest {
            loc: loc.clone(),
            id_offset,
            nb_points,
        };
        manifest
    }

    pub fn get_nb_points(&self) -> ID {
        self.nb_points
    }

    pub fn id_in_range(&self, id: ID) -> bool {
        (id >= self.id_offset) & (id <= self.max_id())
    }

    fn max_id(&self) -> ID {
        self.id_offset + self.nb_points - 1
    }

    pub fn ids(&self) -> impl Iterator<Item = ID> {
        self.id_offset..self.id_offset + self.nb_points
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestList {
    manifests: Vec<Manifest>,
    nb_points: ID,
    metadata_dir: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> Self {
        let metadata_dir = loc.join("metadata-layer");
        match Self::read_current_manifest_list(&metadata_dir) {
            Some(s) => s,
            None => Self::create_first_manifest_list(metadata_dir),
        }
    }

    pub fn get_file_and_offset_from_id(&self, id: ID) -> Option<(PathBuf, ID)> {
        for manifest in self.manifests.iter() {
            if manifest.id_in_range(id) {
                return Some((manifest.loc.clone(), manifest.id_offset));
            }
        }
        None
    }

    fn create_first_manifest_list(metadata_dir: PathBuf) -> Self {
        let file_loc = metadata_dir.join("0-manifest-list.json");
        let writer = BufWriter::new(
            File::create_new(&file_loc).expect("Could not create first manifest list file"),
        );
        let manifest_list = ManifestList {
            metadata_dir,
            manifests: Vec::new(),
            nb_points: 0,
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
        serde_json::from_reader::<BufReader<File>, ManifestList>(reader)
            .expect("serde_json could not parse Manifest List")
    }

    /// we assume we have the write lock here
    pub fn update(&mut self, new_manifest: &Manifest) {
        let new_path = self.get_next_manifest_file_path();
        let writer = BufWriter::new(
            File::create_new(&new_path).expect("Could not create next manifest list file"),
        );
        self.manifests.push(new_manifest.clone());
        self.nb_points += new_manifest.get_nb_points();
        serde_json::to_writer_pretty(writer, &self)
            .expect("Could not write contents to new manifest list file");
    }

    fn get_next_manifest_file_path(&self) -> PathBuf {
        let latest = match Self::get_latest_manifest_list_file(&self.metadata_dir) {
            Some(p) => p,
            None => panic!("Was expecting a manifest list file to exist, but didn't find it"),
        };

        let next_idx = Self::parse_int_from_file_name(
            &latest.file_name().unwrap().to_string_lossy().to_string(),
        ) + 1;
        let next_file_name = format!("{next_idx}-manifest-list.json");
        self.metadata_dir.join(next_file_name)
    }

    pub fn get_nb_points(&self) -> ID {
        self.nb_points
    }
}
