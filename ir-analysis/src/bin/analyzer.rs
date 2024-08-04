use std::collections::HashSet;

use ir_analysis::{emulator::Generator, module::KModule};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        println!("Usage: {} <bitcode1> <bitcode2> <function>", args[0]);
        return;
    }
    let path1 = &args[1];
    let path2 = &args[2];
    let function = &args[3];
    let kmodule1 = KModule::from_bc_path(path1).expect("failed to load bitcode");
    let kmodule2 = KModule::from_bc_path(path2).expect("failed to load bitcode");
    let kfunction1 = kmodule1
        .get_function(function)
        .expect("failed to get function");
    let generator = Generator::new(kfunction1, &kmodule1);
    let sigs1 = generator.execute_function();
    let kfunction2 = kmodule2
        .get_function(function)
        .expect("failed to get function");
    let generator = Generator::new(kfunction2, &kmodule2);
    let sigs2 = generator.execute_function();
    // println!("{} signatures in {}", sigs1.0.len(), path1);
    // for sig in sigs1.0.iter() {
    //     println!("{sig}");
    // }
    // println!("{} signatures in {}", sigs2.0.len(), path2);
    // for sig in sigs2.0.iter() {
    //     println!("{sig}");
    // }
    let a: HashSet<_> = sigs1.0.difference(&sigs2.0).cloned().collect();
    let b: HashSet<_> = sigs2.1.difference(&sigs1.1).cloned().collect();
    println!("{} signatures in {} but not in {}", a.len(), path1, path2);
    for sig in a {
        println!("{sig}");
    }
    for sig in b {
        println!("{sig}");
    }
}
