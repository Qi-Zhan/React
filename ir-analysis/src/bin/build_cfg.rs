//! Draw CFG for a specific bitcode and function
//!
//! Usage: cfg: <bitcode> <func_name> [output_file]

use std::{env, fs, io::Write, process};

use ir_analysis::{analysis::FunctionAnalysis, module::KModule};

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args.len() < 3 {
        panic!("Usage: cfg: <bitcode> <func_name> [output_file]");
    }
    let kmodule = KModule::from_bc_path(&args[1]).unwrap();
    let dot_file = if args.len() == 4 {
        args[3].clone()
    } else {
        args[2].clone() + ".dot"
    };
    let func = kmodule
        .get_function(&args[2])
        .expect("Cannot find function named");
    for instr in func.insts() {
        println!("{}", instr)
    }
    let function_analysis = FunctionAnalysis::new(func);
    let cfg = function_analysis.control_flow_graph();
    let mut file = fs::File::create(&dot_file).unwrap();
    file.write_all(cfg.to_dot(func).as_bytes()).unwrap();
    // generate png
    process::Command::new("dot")
        .arg("-Tpng")
        .arg(&dot_file)
        .arg("-o")
        .arg(&dot_file.replace(".dot", ".png"))
        .output()
        .unwrap();
    // remove the dot file
    if dot_file.ends_with(".dot") {
        fs::remove_file(&dot_file).unwrap();
    }
}
