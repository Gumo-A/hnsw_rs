use std::{
    collections::HashSet,
    fs::{File, remove_file},
    io::{self, BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

use itertools::Itertools;
use log::debug;
use points::point::{Point, VecType};
use serde::{Deserialize, Serialize};

use crate::{
    FileName,
    engine::{DATA_LAYER, INDEX_LAYER, INT},
    get_rand_file_name,
    index::IndexEntry,
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
    pub fn new(data_loc: &PathBuf, index_loc: &PathBuf, first_id: INT, last_id: INT) -> Self {
        let manifest = Manifest {
            data_loc: data_loc.clone(),
            index_loc: index_loc.clone(),
            ids: HashSet::from_iter(first_id..=last_id),
            id_ranges: vec![(first_id, last_id)],
        };
        manifest
    }

    pub fn with_ranges(
        data_loc: &PathBuf,
        index_loc: &PathBuf,
        id_ranges: Vec<(INT, INT)>,
    ) -> Self {
        let mut manifest = Manifest {
            data_loc: data_loc.clone(),
            index_loc: index_loc.clone(),
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

    pub fn load_entries(&self) -> Vec<IndexEntry> {
        let mut bytes = vec![];
        File::open(&self.index_loc)
            .expect("Could not open entry file")
            .read_to_end(&mut bytes)
            .expect("Could not read entry file");
        self.deserialize_entries(&bytes)
    }

    pub fn load_data(&self) -> Vec<Point> {
        let mut bytes = vec![];
        File::open(&self.data_loc)
            .expect("Could not open data file")
            .read_to_end(&mut bytes)
            .expect("Could not read data file");
        self.deserialize_data(&bytes)
    }

    fn deserialize_data_uncompressed(&self, bytes: &Vec<u8>) -> Vec<Point> {
        let vecs: Vec<VecType> =
            rmp_serde::decode::from_slice(&bytes).expect("Could not decode data");
        let mut points = vec![];
        for (v, id) in vecs.iter().zip_eq(self.ordered_ids()) {
            points.push(Point::with_vec_and_id(v, id));
        }
        points
    }

    fn deserialize_data_debug(&self, bytes: &Vec<u8>) -> Vec<Point> {
        let vecs: Vec<VecType> = serde_json::from_slice(bytes).expect("Could not decode data");
        Vec::from_iter(
            vecs.iter()
                .zip_eq(self.ids())
                .map(|(v, id)| Point::with_vec_and_id(v, id)),
        )
    }

    fn deserialize_entries(&self, bytes: &Vec<u8>) -> Vec<IndexEntry> {
        // self.deserialize_entries_debug(bytes)
        self.deserialize_entries_uncompressed(bytes)
    }

    fn deserialize_data(&self, bytes: &Vec<u8>) -> Vec<Point> {
        // self.deserialize_data_debug(bytes)
        self.deserialize_data_uncompressed(bytes)
    }

    fn deserialize_entries_uncompressed(&self, bytes: &Vec<u8>) -> Vec<IndexEntry> {
        rmp_serde::decode::from_slice(&bytes).expect("Could not decode entries")
    }

    fn deserialize_entries_debug(&self, bytes: &Vec<u8>) -> Vec<IndexEntry> {
        serde_json::from_slice(bytes).expect("Could not decode entries")
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

    pub fn add_data(&mut self, mut data: &[Point]) -> io::Result<Manifest> {
        let new_manifest = self.write_new_data(&mut data);
        self.update(&new_manifest)?;
        Ok(new_manifest)
    }

    fn write_new_data(&self, data: &[Point]) -> Manifest {
        let first = self.get_nb_points();
        let last = first + data.len() as INT - 1;
        let data_path = self.get_data_layer().join(get_rand_file_name());
        let index_path = self.get_index_layer().join(get_rand_file_name());
        self.save_serialized_data(&data_path, data);
        self.save_serialized_entries(&index_path, &self.gen_entries(data));
        Manifest::new(&data_path, &index_path, first, last)
    }

    fn gen_entries(&self, data: &[Point]) -> Vec<IndexEntry> {
        Vec::from_iter(data.iter().map(|p| IndexEntry::new(p.id, 0)))
    }

    fn save_serialized_data(&self, path: &PathBuf, data: &[Point]) -> INT {
        let mut file = match File::create_new(path) {
            Ok(f) => f,
            Err(e) => panic!("Could not create data file at {path:?}: {e}"),
        };
        match file.write(&self.serialize_data(&data)) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write data to file {path:?}: {e}"),
        }
    }

    fn save_serialized_entries(&self, path: &PathBuf, data: &[IndexEntry]) -> INT {
        let mut file = match File::create_new(path) {
            Ok(f) => f,
            Err(e) => panic!("Could not create index file at {path:?}: {e}"),
        };
        match file.write(&self.serialize_index_entries(&data)) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write index to file {path:?}: {e}"),
        }
    }

    fn serialize_index_entries_uncompressed(&self, data: &[IndexEntry]) -> Vec<u8> {
        rmp_serde::encode::to_vec(data).expect("Could not serialize index entries")
    }

    fn serialize_index_entries(&self, data: &[IndexEntry]) -> Vec<u8> {
        // self.serialize_index_entries_debug(data)
        self.serialize_index_entries_uncompressed(data)
    }

    fn serialize_data(&self, data: &[Point]) -> Vec<u8> {
        // self.serialize_data_debug(data)
        self.serialize_data_uncompressed(data)
    }

    fn serialize_data_uncompressed(&self, data: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = data.iter().map(|p| &p.vector).collect();
        rmp_serde::encode::to_vec(&p).expect("Could not serialize data")
    }

    fn serialize_data_debug(&self, data: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = data.iter().map(|p| &p.vector).collect();
        serde_json::to_vec_pretty(&p).expect("Could not encode data")
    }

    fn serialize_index_entries_debug(&self, data: &[IndexEntry]) -> Vec<u8> {
        let p: Vec<IndexEntry> = Vec::from_iter((0..data.len()).map(|id| IndexEntry::new(id, 0)));
        serde_json::to_vec_pretty(&p).expect("Could not encode index entries")
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
            combined_points.extend(manifest.load_data());
            combined_entries.extend(manifest.load_entries());
            id_ranges.extend(manifest.id_ranges.clone());
        }
        let data_path = self.get_data_layer().join(get_rand_file_name());
        let index_path = self.get_index_layer().join(get_rand_file_name());
        self.save_serialized_data(&data_path, &combined_points);
        self.save_serialized_entries(&index_path, &combined_entries);
        Manifest::with_ranges(&data_path, &index_path, id_ranges)
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
}
