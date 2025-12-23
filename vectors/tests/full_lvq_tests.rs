use vectors::{FullVec, LVQVec, VecBase, gen_rand_vecs};

#[test]
fn dist_err_lt_one_percent() {
    for _ in 0..1000 {
        let rand_a = gen_rand_vecs(128, 1)[0].clone();
        let rand_b = gen_rand_vecs(128, 1)[0].clone();

        let a = FullVec::new(&rand_a.clone());
        let b = FullVec::new(&rand_b.clone());

        let full2full_dist = a.distance(&b);

        let a = LVQVec::new(&rand_a);
        let quant2full_dist = a.distance(&b);

        let b = LVQVec::new(&rand_b);
        let quant2quant_dist = a.distance(&b);

        let quant2full_err = (full2full_dist - quant2full_dist).abs() / full2full_dist;
        let quant2quant_err = (full2full_dist - quant2quant_dist).abs() / full2full_dist;
        println!("quant2full_diff {quant2full_err}");
        println!("quant2quant_diff {quant2quant_err}");
        assert!(quant2full_err < 0.01);
        assert!(quant2quant_err < 0.01);
    }
}
