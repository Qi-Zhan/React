//! Main Program

use clap::Parser;
use ir_analysis::IRState;
use ir_analysis::{smt::init_ctx, IRAnalysis2, Smt};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    target_path: PathBuf,
    vuln_path: PathBuf,
    patch_path: PathBuf,
    diff_path: PathBuf,
}

fn main() {
    let args = Args::parse();
    let mut sovler: Smt = init_ctx().unwrap();
    let mut ir_analysis = IRAnalysis2::new(
        args.vuln_path.to_str().unwrap(),
        args.patch_path.to_str().unwrap(),
        args.diff_path.to_str().unwrap(),
    );
    match ir_analysis.test(args.target_path.to_str().unwrap(), &mut sovler) {
        IRState::Patch => println!("Patch"),
        IRState::Vuln => println!("Vuln"),
    }
}
