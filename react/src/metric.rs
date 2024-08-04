use crate::dataset::{State, TestResult};

pub fn tp_tn_fp_fn(results: &[TestResult]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    for r in results {
        match (r.result, r.test.ground_truth) {
            (State::Vuln, State::Vuln) => tp += 1,
            (State::Patch, State::Patch) => tn += 1,
            (State::Vuln, State::Patch) => fp += 1,
            (State::Patch, State::Vuln) => fn_ += 1,
        }
    }
    (tp, tn, fp, fn_)
}

pub fn precision_recall_f1(results: &[TestResult]) -> (f64, f64, f64) {
    let (tp, _tn, fp, fn_) = tp_tn_fp_fn(results);
    let p = tp as f64 / (tp + fp) as f64;
    let r = tp as f64 / (tp + fn_) as f64;
    let f1 = 2.0 * p * r / (p + r);
    (p, r, f1)
}
