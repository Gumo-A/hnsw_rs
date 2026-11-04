use std::{
    collections::HashSet,
    env,
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, BufReader, ErrorKind, Read, Write},
    path::PathBuf,
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use graph::graph::Graph;
use itertools::Itertools;
use moka::{policy::EvictionPolicy, sync::Cache};
use points::point::{Point, VecType};
use rand::{Rng, rngs::ThreadRng, thread_rng};
use vectors::VecBase;

use crate::{
    get_rand_file_name,
    hnsw::{HNSW, traverser::Traverser},
    index::{HNSWErr, Index, IndexEntry, map_io_err, map_json_err},
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

pub type INT = usize;
const PARAMS_FILE_NAME: &str = "params.json";
pub const METADATA_LAYER: &str = "metadata-layer";
pub const DATA_LAYER: &str = "data-layer";
pub const INDEX_LAYER: &str = "index-layer";
const LAYERS: [&str; 3] = [METADATA_LAYER, DATA_LAYER, INDEX_LAYER];

pub struct Engine {
    loc: PathBuf,
    point_cache: Cache<INT, Arc<Point>>,
    entry_cache: Cache<INT, Arc<IndexEntry>>,
    params: HNSW,
}

impl Engine {
    pub fn new(loc: PathBuf, m: INT) -> Result<Self, HNSWErr> {
        let params = Self::init_at_loc(&loc, m)?;
        let point_cache = Cache::builder()
            .max_capacity(params.cache_cap as u64)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let entry_cache = Cache::builder()
            .max_capacity(params.cache_cap as u64)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let engine = Self {
            loc: loc.clone(),
            point_cache,
            entry_cache,
            params,
        };
        Ok(engine)
    }

    fn init_at_loc(loc: &PathBuf, dim: INT) -> Result<HNSW, HNSWErr> {
        if loc.exists() {
            if loc.is_dir() {
                Self::check_can_use_dir(&loc)?;
            } else {
                return Err(HNSWErr::IOError(io::ErrorKind::NotADirectory.into()));
            }
        } else {
            match create_dir_all(&loc) {
                Ok(()) => (),
                Err(e) => match e.kind() {
                    io::ErrorKind::AlreadyExists => (),
                    _ => return Err(HNSWErr::IOError(e)),
                },
            };
        }
        Self::check_or_create_dir_structure(&loc, dim)
    }

    fn check_can_use_dir(loc: &PathBuf) -> Result<(), HNSWErr> {
        let check = |p| loc.join(p).exists();
        let empty = loc.read_dir().map_err(|e| HNSWErr::IOError(e))?.count() == 0;
        if empty {
            Ok(())
        } else if check(METADATA_LAYER) & check(DATA_LAYER) & check(INDEX_LAYER) {
            Ok(())
        } else {
            Err(HNSWErr::IOError(io::ErrorKind::DirectoryNotEmpty.into()))
        }
    }

    fn try_create_dir(path: PathBuf) -> Result<(), HNSWErr> {
        match create_dir(&path) {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => Ok(()),
                _ => Err(HNSWErr::IOError(e)),
            },
        }
    }

    fn check_or_create_dir_structure(loc: &PathBuf, m: INT) -> Result<HNSW, HNSWErr> {
        for layer in LAYERS {
            Self::try_create_dir(loc.join(layer))?;
        }
        Self::load_or_create_param_file(loc.join(METADATA_LAYER), m)
    }

    fn load_or_create_param_file(metadata_dir: PathBuf, m: INT) -> Result<HNSW, HNSWErr> {
        let params = HNSW::new(m);
        match Self::write_params(&metadata_dir, &params, false) {
            Ok(()) => Ok(params),
            Err(e) => match e {
                HNSWErr::IOError(e) => match e.kind() {
                    ErrorKind::AlreadyExists => Self::read_params_file(&metadata_dir),
                    _ => Err(HNSWErr::IOError(e)),
                },
                _ => Err(e),
            },
        }
    }

    fn ser_params(params: &HNSW) -> Result<Vec<u8>, HNSWErr> {
        if env::var("HNSW_DEBUG").is_ok() {
            serde_json::to_vec_pretty(&params).map_err(map_json_err)
        } else {
            serde_json::to_vec(&params).map_err(map_json_err)
        }
    }

    fn read_params_file(metadata_dir: &PathBuf) -> Result<HNSW, HNSWErr> {
        let f = File::open(Self::get_param_path(metadata_dir)).map_err(map_io_err)?;
        let reader = BufReader::new(f);
        serde_json::from_reader(reader).map_err(map_json_err)
    }

    fn get_param_path(metadata_dir: &PathBuf) -> PathBuf {
        metadata_dir.join(PARAMS_FILE_NAME)
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn write_lock(&self) {
        let lock_file = self.loc.join(METADATA_LAYER).join("write-lock-file");
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
        let lock_file = self.loc.join(METADATA_LAYER).join("write-lock-file");
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

    /// Also caches points stored in the same data file
    fn load_point(&mut self, id: INT) -> Result<Arc<Point>, HNSWErr> {
        let manifest_list = self.get_current_manifest_list()?;
        let points = manifest_list
            .get_manifest_from_id(id)
            .ok_or(HNSWErr::IDNotFound(id))?
            .read_data();
        for p in points {
            if !self.point_cache.contains_key(&(p.id as INT)) {
                self.point_cache.insert(p.id as INT, Arc::new(p));
            }
        }
        self.point_cache.get(&id).ok_or(HNSWErr::CacheMiss)
    }

    fn load_entry(&mut self, id: INT) -> Result<Arc<IndexEntry>, HNSWErr> {
        let manifest_entries = self
            .get_current_manifest_list()?
            .get_manifest_from_id(id)
            .unwrap()
            .read_entries();
        for entry in manifest_entries {
            self.entry_cache.insert(entry.id, Arc::new(entry));
        }
        self.entry_cache.get(&id).ok_or(HNSWErr::CacheMiss)
    }

    pub fn insert(&mut self, points: &[Point]) -> Result<Manifest, HNSWErr> {
        self.check_points(points)?;
        if points.len() > self.params.cache_cap as INT {
            self.divide_and_insert(points)
        } else {
            self.write_lock();
            let manifest = self
                .get_current_manifest_list()?
                .add_data(points, self.params.ml)
                .map_err(map_io_err)?;
            self.add_points_to_index(&manifest);
            self.write_unlock();
            Ok(manifest)
        }
    }

    pub fn optimize(&mut self) -> Result<(), HNSWErr> {
        self.get_current_manifest_list()?
            .optimize(self.params.cache_cap)
            .map_err(map_io_err)
    }

    fn check_points(&mut self, points: &[Point]) -> Result<(), HNSWErr> {
        if points.len() == 0 {
            return Err(HNSWErr::NothingToInsert);
        } else if self.params.dim.is_none() {
            self.update_dim(points)?;
        }

        let dim = self.params.dim.unwrap();
        let nb_valid = points.iter().filter(|p| p.dim() == dim).count();
        if nb_valid != points.len() {
            Err(HNSWErr::InvalidDimension)
        } else {
            Ok(())
        }
    }

    fn update_dim(&mut self, points: &[Point]) -> Result<(), HNSWErr> {
        self.params.set_dim(points[0].dim());
        Self::write_params(&self.loc.join(METADATA_LAYER), &self.params, true)
    }

    fn write_params(metadata_dir: &PathBuf, params: &HNSW, overwrite: bool) -> Result<(), HNSWErr> {
        let param_file = Self::get_param_path(metadata_dir);
        let params_ser = Self::ser_params(params)?;
        let mut f = if overwrite {
            File::create(&param_file).map_err(map_io_err)?
        } else {
            File::create_new(&param_file).map_err(map_io_err)?
        };
        f.write_all(&params_ser);
        Ok(())
    }

    fn divide_and_insert(&mut self, points: &[Point]) -> Result<Manifest, HNSWErr> {
        let mid = points.len() / 2;
        let p1 = &points[..mid];
        let p2 = &points[mid..];
        self.insert(p1)?;
        self.insert(p2)
    }

    fn add_points_to_index(&mut self, manifest: &Manifest) {
        let mut inserter = Traverser::new();
        for id in manifest.ids() {
            inserter.insert(self, id);
        }
    }

    pub fn get_current_manifest_list(&self) -> Result<ManifestList, HNSWErr> {
        ManifestList::new(&self.loc).map_err(map_io_err)
    }

    pub fn get_number_of_manifests(&self) -> Result<INT, HNSWErr> {
        Ok(self.get_current_manifest_list()?.manifests.len())
    }

    pub fn get_nb_points(&self) -> Result<INT, HNSWErr> {
        Ok(self.get_current_manifest_list()?.get_nb_points())
    }

    pub fn write_entries(&self) -> Result<(), HNSWErr> {
        todo!("persist changes to index entries to disk")
    }
}

impl Index for Engine {
    fn get_point(&mut self, id: INT) -> Result<Arc<Point>, HNSWErr> {
        match self.point_cache.get(&id) {
            Some(p) => Ok(p),
            None => self.load_point(id),
        }
    }

    fn get_entry(&mut self, id: INT) -> Result<Arc<IndexEntry>, HNSWErr> {
        match self.entry_cache.get(&id) {
            Some(p) => Ok(p),
            None => self.load_entry(id),
        }
    }

    fn distance(&mut self, a: INT, b: INT) -> Result<f32, HNSWErr> {
        let p_a = self.get_point(a)?;
        let p_b = self.get_point(b)?;
        Ok(p_a.dist2other(&p_b))
    }

    fn distance2point(&mut self, id: INT, point: &Point) -> Result<f32, HNSWErr> {
        Ok(self.get_point(id)?.dist2other(point))
    }

    fn get_ep(&self) -> INT {
        self.params.ep
    }

    fn get_m(&self) -> INT {
        self.params.m
    }

    fn get_m_at_level(&self, level: INT) -> INT {
        if level == 0 {
            self.params.mmax0
        } else {
            self.params.mmax
        }
    }

    fn get_ef_cons(&self) -> INT {
        self.params.ef_cons
    }

    fn degree(&mut self, id: INT, level: INT) -> Result<INT, HNSWErr> {
        Ok(self
            .get_entry(id)?
            .levels
            .get(level)
            .expect("Could not get level while getting a degree")
            .len())
    }

    fn add_node(&mut self, id: INT, level: INT) -> Result<(), HNSWErr> {
        let mut entry = self.get_entry(id)?;
        while entry.levels.len() <= level {
            entry.levels.push(HashSet::new());
        }
        Ok(())
    }

    fn add_edge(&mut self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr> {
        let mut a_entry = self.get_entry(a)?;
        let a_neighbors = a_entry
            .levels
            .get_mut(level)
            .expect("Could not get level of `a` while adding edge");
        let mut b_entry = self.get_entry(b)?;
        let b_neighbors = b_entry
            .levels
            .get_mut(level)
            .expect("Could not get level of `b` while adding edge");
        a_neighbors.insert(b);
        b_neighbors.insert(a);
        Ok(())
    }

    fn remove_edge(&mut self, a: INT, b: INT, level: INT) -> Result<(), HNSWErr> {
        let mut a_entry = self.get_entry(a)?;
        let a_neighbors = a_entry
            .levels
            .get_mut(level)
            .expect("Could not get level of `a` while removing edge");
        let mut b_entry = self.get_entry(b)?;
        let b_neighbors = b_entry
            .levels
            .get_mut(level)
            .expect("Could not get level of `b` while removing edge");
        a_neighbors.remove(&b);
        b_neighbors.remove(&a);
        Ok(())
    }

    fn neighbors(
        &mut self,
        id: INT,
        level: INT,
    ) -> Result<std::collections::HashSet<INT>, HNSWErr> {
        let entry = self.get_entry(self.get_ep())?;
        let neighs_vec = entry.levels.get(level).unwrap();
        let neighbors = HashSet::from_iter(neighs_vec.iter().cloned());
        Ok(neighbors)
    }

    fn get_nb_levels(&mut self) -> Result<INT, HNSWErr> {
        Ok(self.get_entry(self.get_ep())?.levels.len())
    }
}

pub fn get_level(ml: f32, rng: &mut ThreadRng) -> INT {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    let level = (-rand_nb.ln() * ml).floor() as usize;
    level
}

#[cfg(test)]
mod tests {
    use std::{
        fs::remove_dir_all,
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
        u64,
    };

    use env_logger::init;
    use log::trace;
    use points::point::Point;
    use rand::{
        Rng,
        distributions::{Distribution, Uniform, uniform::UniformSampler},
    };
    use vectors::{VecBase, gen_rand_vecs};

    use crate::{
        engine::{Engine, INT},
        index::{HNSWErr, Index},
    };

    const DIM: INT = 128;
    const IOMSG: &str = "IO error ocurred";

    fn remove_path_if_exists(path: &str) -> PathBuf {
        let path = Path::new(path).to_path_buf();
        if path.exists() {
            remove_dir_all(&path).unwrap();
        }
        path
    }

    fn init_engine(path: &str) -> Engine {
        // env_logger::init();
        let path = remove_path_if_exists(path);
        Engine::new(path, DIM).unwrap()
    }

    fn gen_rand_points(n: INT) -> Vec<Point> {
        let vs = gen_rand_vecs(DIM, n);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx));
        }
        points
    }

    fn gen_rand_point(dim: INT) -> Vec<Point> {
        let vs = gen_rand_vecs(dim, 1);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx));
        }
        points
    }

    #[test]
    fn different_dimension_not_allowed() {
        let path = "./diff-dim-test";
        let mut e = init_engine(path);
        assert!(e.params.dim.is_none());

        let r = e.insert(&vec![gen_rand_point(4)[0].clone()]);
        assert!(r.is_ok());
        assert_eq!(e.params.dim, Some(4));

        let r = e.insert(&vec![gen_rand_point(8)[0].clone()]);
        assert!(r.is_err());
        remove_dir_all(path).unwrap()
    }

    #[test]
    fn add_payloads() {
        let path = "./add-payloads-test";
        let mut e = init_engine(path);
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
        let mut e = init_engine(path);
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
            let a = e.get_point(i).expect("Error getting point");
            let b = points[i as INT].clone();
            assert_eq!(a.get_vals(), b.get_vals());
            assert_eq!(a.id, b.id);
            assert_eq!(a.level, b.level);
        }

        let mut f = |e: &mut Engine, id: INT| -> bool {
            match e.get_point(id) {
                Err(e) => match e {
                    HNSWErr::IDNotFound(other_id) => id == other_id,
                    _ => false,
                },
                Ok(_) => false,
            }
        };

        assert!(f(&mut e, 8));
        assert!(f(&mut e, 16));
        assert!(f(&mut e, 1024));
        assert!(f(&mut e, INT::MAX));

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

        e.point_cache.invalidate_all();

        let mut check_cache_updates = |idx| {
            assert_eq!(e.point_cache.contains_key(&idx), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 1)), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 2)), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 3)), false);
            e.get_point(idx);
            assert_eq!(e.point_cache.contains_key(&idx), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 1)), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 2)), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 3)), true);
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
        let n = 2_usize.pow(12);
        timeit!(let points = gen_rand_points(n));
        timeit!(let sample = Uniform::new(0, points.len())
            .sample_iter(rng.clone())
            .take(1000)
            .collect::<Vec<INT>>());
        let mut cursor = 0;
        let mut len = points.len();
        let mut insertions = 0;
        let mut insert_loop = |e: &mut Engine| {
            loop {
                insertions += 1;
                let jump = (len / 2) - rng.gen_range(0..(len / 2).max(1)) + 1;
                e.insert(&points[cursor..cursor + jump]).unwrap();
                len -= jump;
                cursor += jump;
                if cursor == points.len() {
                    break;
                }
            }
        };
        let mut check_sample = |e: &mut Engine| {
            for id in sample.iter().copied() {
                let p = points.get(id).unwrap();
                assert_eq!(id, p.id as INT);

                let a = e.get_point(id as INT).expect("The point was not found");

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
