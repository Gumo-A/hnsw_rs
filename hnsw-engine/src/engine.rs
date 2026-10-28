use std::{
    env,
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, ErrorKind, Read, Write},
    path::PathBuf,
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use hnsw::template::inserter::Inserter;
use itertools::Itertools;
use moka::{policy::EvictionPolicy, sync::Cache};
use points::point::{Point, VecType};
use vectors::VecBase;

use crate::{
    INT, get_rand_file_name,
    manifest::{Manifest, ManifestList},
};

macro_rules! timeit {
    ($f:expr) => {
        let s = Instant::now();
        $f;
        let e = s.elapsed();
        println!("Executed {} in {e:?}", stringify!($f));
    };
    (let $x:ident = $f:expr) => {
        let s = Instant::now();
        let $x = $f;
        let e = s.elapsed();
        println!("Executed {} in {e:?}", stringify!($f));
    };
}

const CACHE_SIZE_BYTES: INT = 134217728; // 128MiB
const LAYERS: [&str; 3] = ["metadata-layer", "data-layer", "index-layer"];

fn get_cache_cap(dim: INT) -> INT {
    CACHE_SIZE_BYTES / (8 + dim)
}

enum Error {
    IncorrectDimension,
}

pub struct Engine {
    loc: PathBuf,
    cache: Cache<INT, Arc<Point>>,
    debug: bool,
    cache_cap: INT,
    dim: INT,
}

impl Engine {
    pub fn new(loc: PathBuf, dim: INT) -> io::Result<Self> {
        let cap = get_cache_cap(dim);
        let cache = Cache::builder()
            .max_capacity(cap as u64)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let engine = Self {
            loc: loc.clone(),
            cache,
            debug: env::var("HNSW_DEBUG").is_ok(),
            cache_cap: cap,
            dim,
        };
        Self::init_at_loc(&loc, dim)?;
        Ok(engine)
    }

    pub fn distance(&self, a: INT, b: INT) -> io::Result<f32> {
        let point_a = self
            .get_point(a)?
            .expect("could not get point while computing distance");
        let point_b = self
            .get_point(b)?
            .expect("could not get point while computing distance");
        Ok(point_a.dist2other(&point_b))
    }

    fn init_at_loc(loc: &PathBuf, dim: INT) -> io::Result<()> {
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
        Self::check_or_create_dir_structure(&loc, dim)?;
        Ok(())
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

    fn try_create_dir(path: PathBuf) -> io::Result<()> {
        match create_dir(&path) {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => Ok(()),
                _ => Err(e),
            },
        }
    }

    fn check_or_create_dir_structure(loc: &PathBuf, dim: INT) -> io::Result<()> {
        for layer in LAYERS {
            Self::try_create_dir(loc.join(layer))?;
        }
        Self::check_or_create_dim_file(loc.join("metadata-layer"), dim)
    }

    fn check_or_create_dim_file(metadata_dir: PathBuf, dim: INT) -> io::Result<()> {
        let dim_file = metadata_dir.join("dim.txt");
        match File::create_new(&dim_file) {
            Ok(mut f) => {
                let dim_bytes: Vec<u8> = format!("{dim}").bytes().collect();
                f.write_all(&dim_bytes)
            }
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => {
                    let existing_dim = Self::read_dim_file(&dim_file)?;
                    if existing_dim == dim {
                        Ok(())
                    } else {
                        Err(ErrorKind::InvalidInput.into())
                    }
                }
                _ => Err(e),
            },
        }
    }
    fn read_dim_file(path: &PathBuf) -> io::Result<INT> {
        match File::open(path) {
            Ok(mut f) => {
                let mut dim_str = String::new();
                f.read_to_string(&mut dim_str)?;
                Ok(dim_str.parse::<INT>().unwrap())
            }
            Err(e) => Err(e),
        }
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

    pub fn get_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>> {
        let result = if let Some(p) = self.cache.get(&id) {
            Some(p)
        } else {
            self.load_point(id)?
        };
        Ok(result)
    }

    /// Also caches points stored in the same data file
    fn load_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>> {
        let manifest_list = self.get_current_manifest_list()?;
        if manifest_list.get_nb_points() <= id {
            Ok(None)
        } else {
            if let Some(manifest) = manifest_list.get_manifest_from_id(id) {
                let points = self.load_points_from_manifest(manifest);
                for p in points {
                    if !self.cache.contains_key(&(p.id as INT)) {
                        self.cache.insert(p.id as INT, Arc::new(p));
                    }
                }
                Ok(self.cache.get(&id))
            } else {
                Ok(None)
            }
        }
    }

    pub fn insert(&self, payloads: &[Point]) -> io::Result<Manifest> {
        if payloads.len() > self.cache_cap as INT {
            self.divide_and_insert(payloads)
        } else {
            self.write_lock();
            let mut manifest_list = self.get_current_manifest_list()?;
            let manifest = self.add_payloads_to_store(&mut manifest_list, payloads);
            self.add_payloads_to_index(&manifest);
            self.write_unlock();
            Ok(manifest)
        }
    }

    fn divide_and_insert(&self, payloads: &[Point]) -> io::Result<Manifest> {
        let mid = payloads.len() / 2;
        let p1 = &payloads[..mid];
        let p2 = &payloads[mid..];
        self.insert(p1)?;
        self.insert(p2)
    }

    fn add_payloads_to_index(&self, manifest: &Manifest) {
        Inserter::new().insert(index, point);
    }

    fn load_points_from_manifest(&self, manifest: &Manifest) -> Vec<Point> {
        let mut bytes = vec![];
        File::open(&manifest.loc)
            .expect("Could not open payload file")
            .read_to_end(&mut bytes)
            .expect("Could not read payload file");
        self.deserialize_payloads(&bytes, manifest)
    }

    pub fn get_current_manifest_list(&self) -> io::Result<ManifestList> {
        ManifestList::new(&self.loc)
    }

    pub fn get_number_of_manifests(&self) -> io::Result<INT> {
        Ok(self.get_current_manifest_list()?.manifests.len())
    }

    pub fn get_nb_points(&self) -> io::Result<INT> {
        Ok(self.get_current_manifest_list()?.get_nb_points())
    }

    fn add_payloads_to_store(
        &self,
        manifest_list: &mut ManifestList,
        mut payloads: &[Point],
    ) -> Manifest {
        let new_manifest = self.save_payloads(manifest_list, &mut payloads);
        manifest_list.update(&new_manifest);
        new_manifest
    }

    fn save_payloads(&self, manifest_list: &ManifestList, payloads: &[Point]) -> Manifest {
        let first = manifest_list.get_nb_points();
        let last = first + payloads.len() as INT - 1;
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, payloads);
        Manifest::new(&path, first, last)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &[Point]) -> INT {
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

    fn serialize_payloads(&self, payloads: &[Point]) -> Vec<u8> {
        if self.debug {
            self.serialize_payloads_debug(payloads)
        } else {
            self.serialize_payloads_uncompressed(payloads)
        }
    }

    fn serialize_payloads_uncompressed(&self, payloads: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        let ser = rmp_serde::encode::to_vec(&p).expect("Could not serialize payload");
        ser
    }

    fn deserialize_payloads(&self, bytes: &Vec<u8>, manifest: &Manifest) -> Vec<Point> {
        if self.debug {
            self.deserialize_payloads_debug(bytes, manifest)
        } else {
            self.deserialize_payloads_uncompressed(bytes, manifest)
        }
    }

    fn deserialize_payloads_uncompressed(
        &self,
        bytes: &Vec<u8>,
        manifest: &Manifest,
    ) -> Vec<Point> {
        let vecs: Vec<VecType> =
            rmp_serde::decode::from_slice(&bytes).expect("Could not decode payload");
        let mut points = vec![];
        for (v, id) in vecs.iter().zip_eq(manifest.ordered_ids()) {
            points.push(Point::with_vec_and_id(v, id));
        }
        points
    }

    fn serialize_payloads_debug(&self, payloads: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        serde_json::to_vec_pretty(&p).expect("Could not encode payload")
    }

    fn deserialize_payloads_debug(&self, bytes: &Vec<u8>, manifest: &Manifest) -> Vec<Point> {
        let vecs: Vec<VecType> = serde_json::from_slice(bytes).expect("Could not decode payload");
        let ids = manifest.ids();
        let mut points = vec![];
        for (v, id) in vecs.iter().zip_eq(ids) {
            points.push(Point::with_vec_and_id(v, id));
        }
        points
    }

    pub fn optimize(&self) -> io::Result<ManifestList> {
        let old_manifest_list = self.get_current_manifest_list()?;
        let op_plan = old_manifest_list.get_optimization_plan(self.cache_cap);

        let mut optimized_manifest_list = Vec::new();
        for to_combine in op_plan.iter() {
            optimized_manifest_list.push(self.combine_manifests(to_combine));
        }
        let new_manifest_list = ManifestList::from_manifests(&self.loc, optimized_manifest_list)?;
        remove_file(&old_manifest_list.loc)?;
        self.remove_old_manifest_lists(&old_manifest_list)?;
        self.purge_old_manifest_list_data(&old_manifest_list)?;
        Ok(new_manifest_list)
    }

    fn remove_old_manifest_lists(&self, manifest_list: &ManifestList) -> io::Result<()> {
        for old_manifest_list in manifest_list.get_prev_manifest_list_paths()?.iter() {
            remove_file(&old_manifest_list)?;
        }
        Ok(())
    }

    fn purge_old_manifest_list_data(&self, manifest_list: &ManifestList) -> io::Result<()> {
        for manifest in manifest_list.manifests.iter() {
            remove_file(&manifest.loc)?;
        }
        Ok(())
    }

    fn combine_manifests(&self, to_combine: &Vec<Manifest>) -> Manifest {
        let mut combined_points = Vec::new();
        let mut id_ranges = Vec::new();
        for manifest in to_combine.iter() {
            combined_points.extend(self.load_points_from_manifest(manifest));
            id_ranges.extend(manifest.id_ranges.clone());
        }
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, &combined_points);
        Manifest::with_ranges(&path, id_ranges)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::remove_dir_all,
        path::{Path, PathBuf},
        time::Instant,
        u64,
    };

    use points::point::Point;
    use rand::{
        Rng,
        distributions::{Distribution, Uniform, uniform::UniformSampler},
    };
    use vectors::{VecBase, gen_rand_vecs};

    use crate::{INT, engine::Engine};

    const DIM: INT = 1024;
    const IOMSG: &str = "IO error ocurred";

    fn remove_path_if_exists(path: &str) -> PathBuf {
        let path = Path::new(path).to_path_buf();
        if path.exists() {
            remove_dir_all(&path).unwrap();
        }
        path
    }

    fn init_engine(path: &str) -> Engine {
        let path = remove_path_if_exists(path);
        Engine::new(path, DIM).unwrap()
    }

    fn init_engine_dim(path: &str, dim: INT, remove: bool) -> Result<Engine, std::io::Error> {
        let path = if remove {
            remove_path_if_exists(path)
        } else {
            Path::new(path).to_path_buf()
        };
        Engine::new(path, dim)
    }

    fn gen_rand_points(n: INT) -> Vec<Point> {
        let vs = gen_rand_vecs(DIM, n);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx));
        }
        points
    }

    #[test]
    fn different_dimension_not_allowed() {
        let path = "./diff-dim-test";
        let r = init_engine_dim(path, 4, true);
        assert!(r.is_ok());
        assert!(init_engine_dim(path, 8, false).is_err());
        remove_dir_all(path).unwrap()
    }

    #[test]
    fn add_payloads() {
        let path = "./add-payloads-test";
        let e = init_engine(path);
        let n = 8;
        for _ in 0..4 {
            let p = gen_rand_points(n);
            let r = e.insert(&p);
            assert!(r.is_ok());
            assert_eq!(r.unwrap().get_nb_points(), n as INT);
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn add_one_payload() {
        let path = "./add-one-payload-test";
        let e = init_engine(path);
        for _ in 0..4 {
            let p = gen_rand_points(1);
            let r = e.insert(&p);
            assert!(r.is_ok());
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_simple() {
        let path = "./get-points-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(&points[..4]).unwrap();
        let _ = e.insert(&points[4..]).unwrap();

        assert_eq!(e.get_current_manifest_list().unwrap().get_nb_points(), 8);

        for i in 0..8 {
            let a = e
                .get_point(i)
                .expect(IOMSG)
                .expect("The point was not found");
            let b = points[i as INT].clone();
            assert_eq!(a.get_vals(), b.get_vals());
            assert_eq!(a.id, b.id);
            assert_eq!(a.level, b.level);
        }

        assert!(e.get_point(8).expect(IOMSG).is_none());
        assert!(e.get_point(16).expect(IOMSG).is_none());
        assert!(e.get_point(1024).expect(IOMSG).is_none());
        assert!(e.get_point(INT::MAX).expect(IOMSG).is_none());

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_uses_cache() {
        let path = "./get-points-cache-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(&points[..4]).unwrap();
        let _ = e.insert(&points[4..]).unwrap();

        assert_eq!(e.get_current_manifest_list().unwrap().get_nb_points(), 8);

        let mut check_cache_updates = |idx| {
            assert_eq!(e.cache.contains_key(&idx), false);
            assert_eq!(e.cache.contains_key(&(idx + 1)), false);
            assert_eq!(e.cache.contains_key(&(idx + 2)), false);
            assert_eq!(e.cache.contains_key(&(idx + 3)), false);
            e.get_point(idx);
            assert_eq!(e.cache.contains_key(&idx), true);
            assert_eq!(e.cache.contains_key(&(idx + 1)), true);
            assert_eq!(e.cache.contains_key(&(idx + 2)), true);
            assert_eq!(e.cache.contains_key(&(idx + 3)), true);
        };

        check_cache_updates(0);
        check_cache_updates(4);

        remove_dir_all(path).unwrap();
    }

    #[test]
    /// Inserts a number of points in the store
    /// via a random number of insertions until all
    /// points are inserted. This creates many manifests
    /// and data files. After optimize() is called,
    /// it should reduce the number of manifests
    /// and the number of data files to 1
    fn optimize_rand_additions() {
        let path = "./optimize-rand-test";
        let mut e = init_engine(path);

        let mut rng = rand::thread_rng();
        let n = 2_usize.pow(16);
        timeit!(let points = gen_rand_points(n));
        let sample: Vec<INT> = Uniform::new(0, points.len())
            .sample_iter(rng.clone())
            .take(1000)
            .collect();
        let mut cursor = 0;
        let mut insertions = 0;
        let mut insert_loop = |e: &mut Engine| {
            loop {
                insertions += 1;
                let jump = rng.gen_range(1..n);
                if cursor + jump >= points.len() {
                    e.insert(&points[cursor..]).unwrap();
                    break;
                } else {
                    e.insert(&points[cursor..cursor + jump]).unwrap();
                }
                cursor += jump;
            }
        };
        let mut check_sample = |e: &mut Engine| {
            for id in sample.iter().copied() {
                let p = points.get(id).unwrap();
                assert_eq!(id, p.id as INT);

                let a = e
                    .get_point(id as INT)
                    .expect(IOMSG)
                    .expect("The point was not found");

                assert_eq!(a.id, p.id);
                assert_eq!(a.get_vals(), p.get_vals());
            }
        };

        timeit!(insert_loop(&mut e));

        assert_eq!(e.get_number_of_manifests().expect(IOMSG), insertions);
        timeit!(e.optimize().unwrap());
        assert_eq!(e.get_number_of_manifests().expect(IOMSG), 1);

        timeit!(check_sample(&mut e));

        // remove_dir_all(path).unwrap();
    }
}
