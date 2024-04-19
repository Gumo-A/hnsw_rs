use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug)]
pub struct Bencher {
    pub counters: HashMap<String, usize>,
    timers: HashMap<String, Instant>,
    records: HashMap<String, Vec<f32>>,
}

impl Bencher {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            timers: HashMap::new(),
            records: HashMap::new(),
        }
    }

    pub fn count(&mut self, key: &str) {
        if self.counters.contains_key(key) {
            let counter = self.counters.get_mut(key).unwrap();
            *counter += 1;
        } else {
            self.counters.insert(key.to_string(), 1);
        }
    }

    pub fn start_timer(&mut self, record_name: &str) {
        self.timers.insert(record_name.to_string(), Instant::now());
        if !self.records.contains_key(record_name) {
            self.records.insert(record_name.to_string(), Vec::new());
        }
    }

    pub fn end_timer(&mut self, record_name: &str) {
        self.records
            .get_mut(&record_name.to_string())
            .unwrap()
            .push(
                self.timers
                    .get(&record_name.to_string())
                    .unwrap()
                    .elapsed()
                    .as_secs_f32(),
            );
    }

    pub fn get_means(&self) -> HashMap<String, f32> {
        let mut means = HashMap::new();
        for (key, val) in self.records.iter() {
            let mut mean = 0.0;
            let mut counter = 0.0;
            for record in val.iter() {
                counter += 1.0;
                mean += record;
            }
            mean /= counter;
            means.insert(key.to_string(), mean);
        }
        means
    }

    pub fn get_sums(&self) -> HashMap<String, f32> {
        let mut sums = HashMap::new();
        for (key, val) in self.records.iter() {
            sums.insert(key.to_string(), val.iter().sum());
        }
        sums
    }

    pub fn get_frac_of(&self, base: &str, exclude: Vec<&str>) -> HashMap<String, f32> {
        let records: Vec<f32> = self.records.get(base).unwrap().clone();
        let base_sum: f32 = records.iter().sum::<f32>();
        let sums = self.get_sums();
        let mut fracs = HashMap::new();
        for (key, sum) in sums.iter() {
            if (key == base) | (exclude.contains(&key.as_str())) {
                continue;
            }
            fracs.insert(key.to_string(), sum / base_sum);
        }
        fracs
    }
}
