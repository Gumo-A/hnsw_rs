use std::{
    collections::HashMap,
    fs::{File, ReadDir, create_dir, create_dir_all, remove_file},
    io::{self, BufReader, Read, Write},
    path::PathBuf,
    thread::sleep,
    time::Duration,
};

use points::point::Point;
use serde::{Deserialize, Serialize};

type ID = u64;
type IDOffset = ID;
type AddedPoints = u64;
type FileName = String;
type NBBytes = usize;
type SaveInfo = (FileName, IDOffset, AddedPoints, NBBytes);

pub struct Engine {
    loc: PathBuf,
}

impl Engine {
    pub fn new(loc: PathBuf) -> io::Result<Self> {
        let engine = Self { loc: loc.clone() };
        if loc.exists() {
            if loc.is_dir() {
                Self::check_can_use_dir(&loc)?;
            } else {
                return Err(io::ErrorKind::NotADirectory.into());
            }
        } else {
            match create_dir_all(&loc) {
                Ok(()) => (),
                Err(e) => match e.kind() {
                    io::ErrorKind::AlreadyExists => (),
                    _ => panic!("Could not create {loc:?}: {e}"),
                },
            };
        }
        Self::check_or_create_dir_structure(&loc);
        Ok(engine)
    }

    fn check_can_use_dir(loc: &PathBuf) -> io::Result<()> {
        let check = |p| loc.join(p).exists();
        let empty = loc.read_dir()?.count() == 0;
        if empty {
            Ok(())
        } else if check("metadata-layer") & check("data-layer") & check("index-layer") {
            Ok(())
        } else {
            Err(io::ErrorKind::DirectoryNotEmpty.into())
        }
    }

    fn try_create_dir(path: PathBuf) {
        match create_dir(&path) {
            Ok(()) => (),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => (),
                _ => panic!("Could not create dir {path:?}: {e}"),
            },
        }
    }

    fn check_or_create_dir_structure(loc: &PathBuf) {
        Self::try_create_dir(loc.join("metadata-layer"));
        Self::try_create_dir(loc.join("data-layer"));
        Self::try_create_dir(loc.join("index-layer"));
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn write_lock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("lock-file");
        while lock_file.exists() {
            self.wait(100);
        }
        self.create_lock_file(&lock_file);
    }

    fn create_lock_file(&self, lock_file: &PathBuf) {
        match File::create_new(lock_file) {
            Ok(_) => (),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => {
                    self.write_lock();
                }
                _ => panic!("Could not create {lock_file:?}: {e}"),
            },
        }
    }

    fn write_unlock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("write-lock-file");
        if lock_file.exists() {
            match remove_file(&lock_file) {
                Ok(()) => (),
                Err(e) => match e.kind() {
                    io::ErrorKind::NotFound => (),
                    _ => panic!("Could not delete {lock_file:?}: {e}"),
                },
            };
        }
    }

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<SaveInfo> {
        self.write_lock();
        let manifest = self.get_current_manifest();
        let save_info = self.add_payloads(&manifest, payloads);
        self.index_id_range(&save_info);
        self.write_unlock();
        Ok(save_info)
    }

    fn index_id_range(&self, _added_id_range: &SaveInfo) {
        // payloads were added to the data layer
        // their IDs are all IDs in the given range
        // insert these IDs in the index
        todo!("insert IDs in the index")
    }

    fn get_current_manifest(&self) -> ManifestList {
        ManifestList::new(&self.loc)
    }

    fn add_payloads(&self, manifest: &ManifestList, mut payloads: Vec<Point>) -> SaveInfo {
        let save_info: SaveInfo = self.save_payloads(manifest, &mut payloads);
        manifest.update(&save_info);
        save_info
    }

    fn save_payloads(&self, manifest: &ManifestList, payloads: &Vec<Point>) -> SaveInfo {
        let id_offset = manifest.get_nb_points();
        let len = payloads.len() as AddedPoints;
        let f_name = format!("{id_offset}-{len}-payload");
        let path = self.loc.join("data-layer").join(&f_name);
        let written_bytes = self.save_serialized_payloads(&path, payloads);
        (f_name, id_offset, len, written_bytes)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &Vec<Point>) -> NBBytes {
        let mut file = match File::create_new(path) {
            Ok(f) => f,
            Err(e) => panic!("Could not create payload file at {path:?}: {e}"),
        };
        let payloads_ser = self.serialize_payloads(&payloads);
        match file.write(&payloads_ser) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write payload to file {path:?}: {e}"),
        }
    }

    fn serialize_payloads(&self, payloads: &Vec<Point>) -> Vec<u8> {
        // let mut buf = GzEncoder::new(vec![], Compression::default());
        // buf.write(&serde_json::to_vec(payloads).unwrap())
        // buf.write(&rmp_serde::encode::to_vec(payloads).unwrap())
        //     .unwrap();
        // buf.finish().unwrap()
        // match rmp_serde::encode::to_vec(payloads) {
        match serde_json::to_vec(payloads) {
            Ok(bytes) => bytes,
            Err(e) => panic!("Error while serializing {} payloads: {e}", payloads.len()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    file_name: FileName,
    id_offset: ID,
    nb_points: ID,
}

impl Manifest {
    pub fn new(loc: &PathBuf, save_info: &SaveInfo) -> Self {
        let manifest = Manifest {
            file_name: save_info.0.clone(),
            id_offset: save_info.1,
            nb_points: save_info.2,
        };
        let manifest_json = serde_json::to_vec_pretty(&manifest).unwrap();
        File::create_new(loc.join("metadata-layer").join("0000-manifest"))
            .unwrap()
            .write(&manifest_json)
            .unwrap();
        manifest
    }

    /// Reads manifest at `file_loc`,
    /// returning a Manifest that can tell if an
    /// ID is in one of its files
    pub fn load(file_loc: &PathBuf) -> Self {
        todo!("load manifest content")
    }

    pub fn get_nb_points(&self) -> ID {
        self.nb_points
    }

    pub fn ids(&self) -> impl Iterator<Item = ID> {
        (self.id_offset..self.id_offset + self.nb_points)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestListContent {
    list: Vec<FileName>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestList {
    manifests: Vec<Manifest>,
    nb_points: ID,
    loc: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> Self {
        match Self::read_current_manifest_list(loc) {
            Some(s) => s,
            None => ManifestList {
                manifests: Vec::new(),
                nb_points: 0,
                loc: loc.join("metadata-layer").join("0000-manifest-list.json"),
            },
        }
    }

    fn read_current_manifest_list(loc: &PathBuf) -> Option<Self> {
        match Self::get_latest_manifest_list_file(loc) {
            None => None,
            Some(f) => Some(Self::load_manifest_list(&f)),
        }
    }

    fn get_latest_manifest_list_file(loc: &PathBuf) -> Option<PathBuf> {
        let dir = read_layer(&loc.join("metadata-layer"));
        let latest = dir
            .map(|e| {
                let e = e.unwrap().path();
                (path_to_f_name_str(e.clone()), e)
            })
            .filter(|(s, _)| s.contains("manifest-list"))
            .max_by_key(|(s, _)| s.split_once("-").unwrap().0.parse::<usize>().unwrap());
        match latest {
            None => None,
            Some((_, p)) => Some(p),
        }
    }

    fn load_manifest_list(loc: &PathBuf) -> Self {
        let f = File::open(loc).unwrap();
        let reader = BufReader::new(f);
        let manifest_list: ManifestListContent = serde_json::from_reader(reader).unwrap().list;
        let mut manifests = Vec::new();
        let mut nb_points = 0;
        for f_name in manifest_list {
            let manifest = Manifest::load(&loc.join("metadata-layer").join(f_name));
            nb_points += manifest.get_nb_points();
            manifests.push(manifest)
        }
        ManifestList {
            loc: loc.clone(),
            manifests,
            nb_points,
        }
    }

    fn create_new_manifest_file(&self, save_info: &SaveInfo) -> Manifest {
        Manifest::new(&self.loc, save_info)
    }

    fn update_manifest_list(&self, new_manifest: &Manifest) {
        todo!("create new manifest list. Use all previous manifests plus the new one")
    }

    pub fn update(&self, save_info: &SaveInfo) {
        let manifest = self.create_new_manifest_file(save_info);
        self.update_manifest_list(&manifest);
    }

    pub fn get_nb_points(&self) -> ID {
        self.nb_points
    }
}

fn path_to_f_name_str(loc: PathBuf) -> String {
    loc.file_name().unwrap().to_string_lossy().to_string()
}

fn read_layer(loc: &PathBuf) -> ReadDir {
    match loc.read_dir() {
        Ok(d) => d,
        Err(e) => panic!("Could not read {loc:?}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use points::point::Point;
    use vectors::gen_rand_vecs;

    use crate::Engine;

    fn gen_rand_points(n: usize) -> Vec<Point> {
        let vs = gen_rand_vecs(1024, n);
        let mut points = vec![];
        for v in vs {
            points.push(Point::with_level_and_id(&v, 0, 0));
        }
        points
    }

    #[test]
    fn add_payloads() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..16 {
            let p = gen_rand_points(128);
            e.add_payloads(p);
        }
    }

    #[test]
    fn add_one_payload() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..8 {
            let p = gen_rand_points(1);
            e.add_payloads(p);
        }
    }
}
