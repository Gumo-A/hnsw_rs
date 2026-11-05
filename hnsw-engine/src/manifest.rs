use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{File, remove_file},
    io::{self, BufReader, BufWriter, Read, Write},
    ops::Deref,
    path::PathBuf,
};

use itertools::Itertools;
use points::point::{Point, VecType};
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::{
    FileName,
    engine::{DATA_LAYER, INDEX_LAYER, INT, METADATA_LAYER, get_level},
    get_rand_file_name,
    index::{CachedEntry, IndexEntry},
    path_to_f_name_str,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub data_loc: PathBuf,
    pub index_loc: PathBuf,
    pub id_ranges: Vec<(INT, INT)>,
    #[serde(skip)]
    pub ids: HashSet<INT>,
}

impl Manifest {
    pub fn new(data_loc: PathBuf, index_loc: PathBuf, first_id: INT, last_id: INT) -> Self {
        let manifest = Manifest {
            data_loc,
            index_loc,
            ids: HashSet::from_iter(first_id..=last_id),
            id_ranges: vec![(first_id, last_id)],
        };
        manifest
    }

    pub fn with_ranges(data_loc: PathBuf, index_loc: PathBuf, id_ranges: Vec<(INT, INT)>) -> Self {
        let mut manifest = Manifest {
            data_loc,
            index_loc,
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

    pub fn read_entries(&self) -> Vec<IndexEntry> {
        let mut bytes = vec![];
        File::open(&self.index_loc)
            .expect("Could not open entry file")
            .read_to_end(&mut bytes)
            .expect("Could not read entry file");
        if env::var("HNSW_DEBUG").is_ok() {
            match self.deserialize_entries_debug(&bytes) {
                Ok(r) => r,
                Err(_) => self.deserialize_entries(&bytes),
            }
        } else {
            self.deserialize_entries(&bytes)
        }
    }

    pub fn read_data(&self) -> Vec<Point> {
        let mut bytes = vec![];
        File::open(&self.data_loc)
            .expect("Could not open data file")
            .read_to_end(&mut bytes)
            .expect("Could not read data file");
        if env::var("HNSW_DEBUG").is_ok() {
            match self.deserialize_data_debug(&bytes) {
                Ok(r) => r,
                Err(_) => self.deserialize_data(&bytes),
            }
        } else {
            self.deserialize_data(&bytes)
        }
    }

    fn write_data(&self, data: &[Point]) -> INT {
        let mut file = match File::create_new(&self.data_loc) {
            Ok(f) => f,
            Err(e) => panic!("Could not create data file at {:?}: {e}", self.data_loc),
        };
        let decoded_data = if env::var("HNSW_DEBUG").is_ok() {
            match self.serialize_data_debug(&data) {
                Ok(r) => r,
                Err(_) => self.serialize_data(&data),
            }
        } else {
            self.serialize_data(&data)
        };
        match file.write(&decoded_data) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write data to file {:?}: {e}", self.data_loc),
        }
    }

    fn write_entries(&self, entries: &[IndexEntry]) -> INT {
        let mut file = match File::create(&self.index_loc) {
            Ok(f) => f,
            Err(e) => panic!("Could not create index file at {:?}: {e}", self.index_loc),
        };
        let encoded_entries = if env::var("HNSW_DEBUG").is_ok() {
            match self.serialize_entries_debug(&entries) {
                Ok(r) => r,
                Err(_) => self.serialize_entries(&entries),
            }
        } else {
            self.serialize_entries(&entries)
        };
        match file.write(&encoded_entries) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write index to file {:?}: {e}", self.index_loc),
        }
    }

    fn serialize_entries(&self, data: &[IndexEntry]) -> Vec<u8> {
        rmp_serde::encode::to_vec(data).expect("Could not encode index entries")
    }

    fn serialize_data(&self, data: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = data.iter().map(|p| &p.vector).collect();
        rmp_serde::encode::to_vec(&p).expect("Could not encode data")
    }

    fn serialize_data_debug(&self, data: &[Point]) -> io::Result<Vec<u8>> {
        let p: Vec<&VecType> = data.iter().map(|p| &p.vector).collect();
        Ok(serde_json::to_vec_pretty(&p)?)
    }

    fn serialize_entries_debug(&self, data: &[IndexEntry]) -> io::Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(&data)?)
    }

    fn deserialize_data(&self, bytes: &Vec<u8>) -> Vec<Point> {
        let vecs: Vec<VecType> =
            rmp_serde::decode::from_slice(&bytes).expect("Could not decode data");
        Vec::from_iter(
            vecs.iter()
                .zip_eq(self.ordered_ids())
                .map(|(v, id)| Point::with_vec_and_id(v, id)),
        )
    }

    fn deserialize_data_debug(&self, bytes: &Vec<u8>) -> io::Result<Vec<Point>> {
        let vecs: Vec<VecType> = serde_json::from_slice(bytes)?;
        Ok(Vec::from_iter(
            vecs.iter()
                .zip_eq(self.ordered_ids())
                .map(|(v, id)| Point::with_vec_and_id(v, id)),
        ))
    }

    fn deserialize_entries(&self, bytes: &Vec<u8>) -> Vec<IndexEntry> {
        rmp_serde::decode::from_slice(&bytes).expect("Could not decode entries")
    }

    fn deserialize_entries_debug(&self, bytes: &Vec<u8>) -> io::Result<Vec<IndexEntry>> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestList {
    pub manifests: Vec<Manifest>,
    pub loc: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> io::Result<Self> {
        let metadata_dir = loc.join(METADATA_LAYER);
        match Self::read_current_manifest_list(&metadata_dir) {
            Ok(s) => Ok(s),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => Self::create_first_manifest_list(metadata_dir),
                _ => panic!("{e}"),
            },
        }
    }

    pub fn from_manifests(loc: &PathBuf, manifests: Vec<Manifest>) -> io::Result<Self> {
        let s = Self {
            manifests,
            loc: Self::get_next_manifest_list_path(&loc.join(METADATA_LAYER))?,
        };
        Self::write_self(&s)?;
        Ok(s)
    }

    pub fn add_data(&mut self, mut data: &[Point], ml: f32) -> io::Result<Manifest> {
        let new_manifest = self.write_new_data(&mut data, ml);
        self.add_manifest(&new_manifest)?;
        Ok(new_manifest)
    }

    fn write_new_data(&self, data: &[Point], ml: f32) -> Manifest {
        let first = self.get_nb_points();
        let last = first + data.len() as INT - 1;
        let data_path = self.get_data_layer().join(get_rand_file_name());
        let index_path = self.get_index_layer().join(get_rand_file_name());
        let manifest = Manifest::new(data_path, index_path, first, last);
        manifest.write_data(data);
        manifest.write_entries(&self.gen_entries(data, ml));
        manifest
    }

    fn gen_entries(&self, data: &[Point], ml: f32) -> Vec<IndexEntry> {
        Vec::from_iter(
            data.iter()
                .map(|p| IndexEntry::new(p.id, get_level(ml, &mut thread_rng()))),
        )
    }

    fn create_first_manifest_list(metadata_dir: PathBuf) -> io::Result<Self> {
        let manifest_list = ManifestList {
            loc: metadata_dir.join("0-manifest-list.json"),
            manifests: Vec::new(),
        };
        Self::write_self(&manifest_list)?;
        Ok(manifest_list)
    }

    fn write_self(manifest_list: &ManifestList) -> io::Result<()> {
        let writer = BufWriter::new(File::create_new(&manifest_list.loc)?);
        if env::var("HNSW_DEBUG").is_ok() {
            serde_json::to_writer_pretty(writer, &manifest_list)?;
        } else {
            serde_json::to_writer(writer, &manifest_list)?;
        }
        Ok(())
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
    fn add_manifest(&mut self, new_manifest: &Manifest) -> io::Result<()> {
        self.loc = Self::get_next_manifest_list_path(
            &self
                .loc
                .parent()
                .expect("Could not get metadata dir from manifest list path")
                .to_path_buf(),
        )?;
        self.manifests.push(new_manifest.clone());
        Self::write_self(&self)?;
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

    pub fn optimize(&mut self, cache_cap: INT) -> io::Result<()> {
        let op_plan = self.get_optimization_plan(cache_cap);
        let combined_manifests = Vec::from_iter(
            op_plan
                .iter()
                .map(|to_combine| self.combine_manifests(to_combine)),
        );
        let new_self = ManifestList::from_manifests(&self.get_root(), combined_manifests)?;
        remove_file(&self.loc)?;
        self.remove_prev_manifest_lists()?;
        self.remove_prev_manifest_list_data()?;
        self.remove_prev_manifest_list_entries()?;
        *self = new_self;
        Ok(())
    }

    fn remove_prev_manifest_lists(&self) -> io::Result<()> {
        for old_manifest_list in self.get_prev_manifest_list_paths()?.iter() {
            remove_file(&old_manifest_list)?;
        }
        Ok(())
    }

    fn remove_prev_manifest_list_data(&self) -> io::Result<()> {
        for manifest in self.manifests.iter() {
            remove_file(&manifest.data_loc)?;
        }
        Ok(())
    }

    fn remove_prev_manifest_list_entries(&self) -> io::Result<()> {
        for manifest in self.manifests.iter() {
            remove_file(&manifest.index_loc)?;
        }
        Ok(())
    }

    fn combine_manifests(&self, to_combine: &Vec<Manifest>) -> Manifest {
        let mut combined_points = Vec::new();
        let mut combined_entries = Vec::new();
        let mut id_ranges = Vec::new();
        for manifest in to_combine.iter() {
            combined_points.extend(manifest.read_data());
            combined_entries.extend(manifest.read_entries());
            id_ranges.extend(manifest.id_ranges.clone());
        }
        let data_path = self.get_data_layer().join(get_rand_file_name());
        let index_path = self.get_index_layer().join(get_rand_file_name());
        let manifest = Manifest::with_ranges(data_path, index_path, id_ranges);
        manifest.write_data(&combined_points);
        manifest.write_entries(&combined_entries);
        manifest
    }

    fn get_optimization_plan(&self, optimal_size: INT) -> Vec<Vec<Manifest>> {
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

    fn get_root(&self) -> PathBuf {
        self.loc
            .parent()
            .expect("Could not get metadata dir")
            .parent()
            .expect("Could not get root dir")
            .to_path_buf()
    }

    fn get_data_layer(&self) -> PathBuf {
        self.get_root().join(DATA_LAYER)
    }

    fn get_index_layer(&self) -> PathBuf {
        self.get_root().join(INDEX_LAYER)
    }

    pub fn write_entries(&mut self, entries: Vec<CachedEntry>) -> io::Result<()> {
        for e in entries {
            let mod_manifest = self.get_manifest_from_id(e.get_id()).unwrap();
            let mut old_entries = mod_manifest.read_entries();
            for i in old_entries.iter_mut() {
                if i.get_id() == e.get_id() {
                    i.replace_levels(e.get_levels_cloned().unwrap()).unwrap()
                }
                i.set_is_not_modified().unwrap();
            }
            mod_manifest.write_entries(&old_entries);
        }
        Ok(())
    }
}
