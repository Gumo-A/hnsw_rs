use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufReader, BufWriter},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};

use crate::{FileName, INT, path_to_f_name_str};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub loc: PathBuf,
    pub id_ranges: Vec<(INT, INT)>,
    #[serde(skip)]
    pub ids: HashSet<INT>,
}

impl Manifest {
    pub fn new(loc: &PathBuf, first_id: INT, last_id: INT) -> Self {
        let manifest = Manifest {
            loc: loc.clone(),
            ids: HashSet::from_iter(first_id..=last_id),
            id_ranges: vec![(first_id, last_id)],
        };
        manifest
    }

    pub fn with_ranges(loc: &PathBuf, id_ranges: Vec<(INT, INT)>) -> Self {
        let mut manifest = Manifest {
            loc: loc.clone(),
            ids: HashSet::new(),
            id_ranges,
        };
        manifest.build_ids();
        manifest
    }

    fn build_ids(&mut self) {
        self.ids = HashSet::from_iter(
            self.id_ranges
                .iter()
                .flat_map(|(first, last)| *first..=*last),
        );
    }

    pub fn get_nb_points(&self) -> INT {
        self.ids.len() as INT
    }

    pub fn contains_id(&self, id: INT) -> bool {
        self.ids.contains(&id)
    }

    pub fn ids(&self) -> impl Iterator<Item = INT> {
        self.ids.iter().copied()
    }

    pub fn ordered_ids(&self) -> impl Iterator<Item = INT> {
        self.id_ranges
            .iter()
            .flat_map(|(first, last)| *first..=*last)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestList {
    pub manifests: Vec<Manifest>,
    pub loc: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> io::Result<Self> {
        let metadata_dir = loc.join("metadata-layer");
        match Self::read_current_manifest_list(&metadata_dir) {
            Ok(s) => Ok(s),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => Self::create_first_manifest_list(metadata_dir),
                _ => panic!("{e}"),
            },
        }
    }

    pub fn from_manifests(loc: &PathBuf, manifests: Vec<Manifest>) -> io::Result<Self> {
        let metadata_dir = loc.join("metadata-layer");
        let loc = Self::get_next_manifest_list_path(&metadata_dir)?;
        let writer = BufWriter::new(
            File::create_new(&loc).expect("Could not create next manifest list file"),
        );
        let s = Self { manifests, loc };
        serde_json::to_writer_pretty(writer, &s)
            .expect("Could not write contents to new manifest list file");
        Ok(s)
    }

    fn create_first_manifest_list(metadata_dir: PathBuf) -> io::Result<Self> {
        let first_manifest_list_path = metadata_dir.join("0-manifest-list.json");
        let writer = BufWriter::new(
            File::create_new(&first_manifest_list_path)
                .expect("Could not create first manifest list file"),
        );
        let manifest_list = ManifestList {
            loc: first_manifest_list_path,
            manifests: Vec::new(),
        };
        serde_json::to_writer_pretty(writer, &manifest_list)?;
        Ok(manifest_list)
    }

    fn read_current_manifest_list(metadata_dir: &PathBuf) -> io::Result<Self> {
        Self::load_from_path(Self::get_latest_manifest_list_path(metadata_dir)?)
    }

    pub fn get_prev_manifest_list_paths(&self) -> io::Result<Vec<PathBuf>> {
        let current_nb = self.get_manifest_list_number();
        let to_remove = self
            .loc
            .parent()
            .expect("Could not read meradata dir")
            .read_dir()?
            .map(|e| {
                e.expect("Could not read directory entry in metadata-layer")
                    .path()
            })
            .filter(|e| path_to_f_name_str(&e).ends_with("-manifest-list.json"))
            .filter(|e| Self::parse_int_from_file_name(&path_to_f_name_str(&e)) < current_nb)
            .map(|e| e.to_path_buf())
            .collect();
        Ok(to_remove)
    }

    pub fn get_manifest_list_number(&self) -> usize {
        Self::parse_int_from_file_name(&path_to_f_name_str(&self.loc))
    }

    pub fn get_manifest_from_id(&self, id: INT) -> Option<&Manifest> {
        for manifest in self.manifests.iter() {
            if manifest.contains_id(id) {
                return Some(manifest);
            }
        }
        None
    }

    fn get_latest_manifest_list_path(metadata_dir: &PathBuf) -> io::Result<PathBuf> {
        let latest = metadata_dir
            .read_dir()?
            .map(|e| {
                let e = e
                    .expect("Could not read directory entry in metadata-layer")
                    .path();
                (path_to_f_name_str(&e), e)
            })
            .filter(|(s, _)| s.ends_with("-manifest-list.json"))
            .max_by_key(|(s, _)| Self::parse_int_from_file_name(s));
        match latest {
            None => Err(io::ErrorKind::NotFound.into()),
            Some((_, p)) => Ok(p),
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

    fn load_from_path(loc: PathBuf) -> io::Result<Self> {
        let f = File::open(loc)?;
        let reader = BufReader::new(f);
        let mut deserialized = serde_json::from_reader::<BufReader<File>, ManifestList>(reader)?;
        deserialized.rebuild_ids();
        Ok(deserialized)
    }

    fn rebuild_ids(&mut self) {
        self.manifests.iter_mut().for_each(|m| m.build_ids());
    }

    /// we assume we have the write lock here
    pub fn update(&mut self, new_manifest: &Manifest) -> io::Result<()> {
        let new_path = Self::get_next_manifest_list_path(
            &self
                .loc
                .parent()
                .expect("Could not get metadata dir from manifest list path")
                .to_path_buf(),
        )?;
        let writer = BufWriter::new(
            File::create_new(&new_path).expect("Could not create next manifest list file"),
        );
        self.loc = new_path;
        self.manifests.push(new_manifest.clone());
        serde_json::to_writer_pretty(writer, &self)
            .expect("Could not write contents to new manifest list file");
        Ok(())
    }

    pub fn get_next_manifest_list_path(metadata_dir: &PathBuf) -> io::Result<PathBuf> {
        let latest = Self::get_latest_manifest_list_path(metadata_dir)?;
        let next_idx = Self::parse_int_from_file_name(
            &latest.file_name().unwrap().to_string_lossy().to_string(),
        ) + 1;
        let next_file_name = format!("{next_idx}-manifest-list.json");
        Ok(metadata_dir.join(next_file_name))
    }

    pub fn get_nb_points(&self) -> INT {
        self.manifests
            .iter()
            .map(|manifest| manifest.get_nb_points())
            .sum()
    }

    pub fn get_optimization_plan(&self, optimal_size: INT) -> Vec<Vec<Manifest>> {
        let mut plan = Vec::new();
        let mut manifests = Vec::new();
        let mut to_combine: Vec<_> = self.manifests.iter().collect();
        to_combine.sort_by_key(|m| m.get_nb_points());
        for manif in to_combine.iter() {
            let n: INT = manifests.iter().map(|m: &Manifest| m.get_nb_points()).sum();
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
