use std::{
    fs::{File, ReadDir, create_dir, create_dir_all, remove_file},
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
    thread::sleep,
    time::Duration,
};

use points::point::Point;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

type ID = u64;
type AddedPoints = u64;
type FileName = String;

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
        let lock_file = self.loc.join("metadata-layer").join("write-lock-file");
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

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<Manifest> {
        self.write_lock();
        let mut manifest_list = self.get_current_manifest_list();
        let manifest = self.add_payloads(&mut manifest_list, payloads);
        // self.index_id_range(&manifest);
        self.write_unlock();
        Ok(manifest)
    }

    fn index_id_range(&self, _manifest: &Manifest) {
        // payloads were added to the data layer
        // their IDs are all IDs in the given range
        // insert these IDs in the index
        todo!("insert IDs in the index")
    }

    fn get_current_manifest_list(&self) -> ManifestList {
        ManifestList::new(&self.loc)
    }

    fn add_payloads(&self, manifest: &mut ManifestList, mut payloads: Vec<Point>) -> Manifest {
        let new_manifest = self.save_payloads(manifest, &mut payloads);
        manifest.update(&new_manifest);
        new_manifest
    }

    fn save_payloads(&self, manifest: &ManifestList, payloads: &Vec<Point>) -> Manifest {
        let id_offset = manifest.get_nb_points();
        let len = payloads.len() as AddedPoints;
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, payloads);
        Manifest::new(&path, id_offset, len)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &Vec<Point>) -> usize {
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
        match serde_json::to_vec_pretty(payloads) {
            Ok(bytes) => bytes,
            Err(e) => panic!("Error while serializing {} payloads: {e}", payloads.len()),
        }
    }
}

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
        // let manifest_json = serde_json::to_vec_pretty(&manifest).unwrap();
        // let p = loc
        //     .parent()
        //     .expect("Could not get data layer dir")
        //     .parent()
        //     .expect("Could not get index base location")
        //     .join("metadata-layer")
        //     .join(get_rand_file_name());
        // File::create_new(p)
        //     .expect("Could not create new manifest file")
        //     .write(&manifest_json)
        //     .expect("Could not write manifest content to manifest file");
        manifest
    }

    /// Reads manifest at `file_loc`,
    pub fn load(file_loc: &PathBuf) -> Self {
        let f = File::open(file_loc).expect("Could not open manifest file");
        let reader = BufReader::new(f);
        serde_json::from_reader(reader).unwrap()
    }

    pub fn get_nb_points(&self) -> ID {
        self.nb_points
    }

    pub fn ids(&self) -> impl Iterator<Item = ID> {
        self.id_offset..self.id_offset + self.nb_points
    }
}

type ManifestListContent = Vec<FileName>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestList {
    manifests: Vec<Manifest>,
    nb_points: ID,
    file_loc: PathBuf,
}

impl ManifestList {
    pub fn new(loc: &PathBuf) -> Self {
        let metadata_dir = loc.join("metadata-layer");
        match Self::read_current_manifest_list(&metadata_dir) {
            Some(s) => s,
            None => Self::create_first_manifest_list(&metadata_dir),
        }
    }

    fn create_first_manifest_list(metadata_dir: &PathBuf) -> Self {
        let file_loc = metadata_dir.join("0-manifest-list.json");
        let writer = BufWriter::new(
            File::create_new(&file_loc).expect("Could not create first manifest list file"),
        );
        let manifest_list = ManifestList {
            file_loc,
            manifests: Vec::new(),
            nb_points: 0,
        };
        serde_json::to_writer(writer, &manifest_list)
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
        self.file_loc = new_path;
        serde_json::to_writer_pretty(writer, &self)
            .expect("Could not write contents to new manifest list file");
    }

    fn get_next_manifest_file_path(&self) -> PathBuf {
        let metadata_dir = self
            .file_loc
            .parent()
            .expect("Could not get metadata layer from manifest list file path")
            .to_path_buf();
        let latest = match Self::get_latest_manifest_list_file(&metadata_dir) {
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
        self.nb_points
    }
}

fn path_to_f_name_str(loc: &PathBuf) -> String {
    loc.file_name().unwrap().to_string_lossy().to_string()
}

fn read_layer(loc: &PathBuf) -> ReadDir {
    println!("Reading dir {loc:?}");
    match loc.read_dir() {
        Ok(d) => d,
        Err(e) => panic!("Could not read {loc:?}: {e}"),
    }
}

fn get_rand_file_name() -> String {
    Uuid::new_v4().as_hyphenated().to_string()
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

    const N: u64 = 128;

    #[test]
    fn add_payloads() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..16 {
            let p = gen_rand_points(N as usize);
            let r = e.insert(p);
            assert!(r.is_ok());
            assert_eq!(r.unwrap().nb_points, N);
        }
    }

    #[test]
    fn add_one_payload() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..8 {
            let p = gen_rand_points(1);
            let r = e.insert(p);
            assert!(r.is_ok());
        }
    }
}
