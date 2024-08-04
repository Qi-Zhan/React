use std::env::current_dir;

use lazy_static::lazy_static;

lazy_static! {
    pub static ref DATASET_DIR: String = current_dir().unwrap().to_str().unwrap().to_string() + "/dataset";
    pub static ref CVE_INFO: String = DATASET_DIR.to_owned() + "/CVE_info.jsonl";
    pub static ref TEST: String = DATASET_DIR.to_owned() + "/test.jsonl";
    pub static ref DIFF_DIR: String = DATASET_DIR.to_owned() + "/diff";
    pub static ref BITCODE_DIR: String = DATASET_DIR.to_owned() + "/bitcodes";
    pub static ref BINARIES_DIR: String = DATASET_DIR.to_owned() + "/binary";
}
