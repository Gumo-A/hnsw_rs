use vectors::{FullVec, LVQVec, VecTrait, gen_rand_vecs};

#[test]
fn distance_computation() {
    for _ in 0..100 {
        let rand_vecs = gen_rand_vecs(128, 2);
        let a = FullVec::new(rand_vecs[0].clone());
        let b = LVQVec::new(&rand_vecs[1].clone());
        assert!(a.distance(&b) >= 0.0);
    }
}

#[test]
fn distance_precision() {
    let a = FullVec::new(vec![0.75, 0.75]);
    let b = LVQVec::new(&vec![0.25, 0.25]);
    let dist = a.distance(&b);
    print!("{dist}");
    assert!(dist == (1.0 / (2.0f32).sqrt()));
}
