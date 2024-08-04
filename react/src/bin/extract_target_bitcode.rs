use std::collections::HashMap;
use std::{fs, path, process};

use react::config::*;
use react::dataset::*;

fn main() {
    let cves = read_cves(CVE_INFO.as_str()).unwrap();
    let tests = read_tests(TEST.as_str()).unwrap();

    // map from cve id to function names
    let mut cve_function_map: HashMap<String, Vec<String>> = HashMap::new();
    for cve in cves.values() {
        let source_diff_path = &format!(
            "{}/{}_{}.diff",
            DIFF_DIR.as_str(),
            cve.id,
            &cve.commit[0..6]
        );
        let source_diff = source_analysis::SourceDiff::from_path(source_diff_path)
            .unwrap_or_else(|_| panic!("{} not found", source_diff_path));
        // we only need the function names
        let functions = source_diff
            .function_add
            .keys()
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        cve_function_map.insert(cve.id.clone(), functions);
    }

    let mut test_file2project = HashMap::new();
    for test in tests.iter().map(|(_, t)| t) {
        for t in test {
            test_file2project.insert(t.file.clone(), t.project.clone());
        }
    }

    // map from test_file to function names
    let mut test_for_cve: HashMap<&String, Vec<&String>> = HashMap::new();
    for (id, test) in &tests {
        let functions = cve_function_map.get(id).unwrap();
        for t in test {
            let entry = test_for_cve.entry(&t.file).or_default();
            entry.extend(functions);
        }
    }

    let temp_dir = "./.cache";
    if !path::Path::new(temp_dir).exists() {
        fs::create_dir(temp_dir).unwrap();
    }

    for (test_file, function_list) in test_for_cve {
        let project = test_file2project.get(test_file).unwrap();
        let binary = format!("{}/{}/{}", BINARIES_DIR.as_str(), project, test_file);
        let output = temp_dir.to_owned() + "/" + test_file + ".c";

        // get .bc file and .ll file
        let bc_path = format!("{}/{}.bc", temp_dir, test_file);
        let ll_path = format!("{}/{}.ll", temp_dir, test_file);
        let bc_name = format!("{}/{}/{}.bc", BITCODE_DIR.as_str(), project, test_file);
        let ll_name = format!("{}/{}/{}.ll", BITCODE_DIR.as_str(), project, test_file);

        if path::Path::new(&bc_name).exists() && path::Path::new(&ll_name).exists() {
            continue;
        }

        let function_list = function_list
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let command = format!(
            "retdec-decompiler --select-decode-only --select-functions {} {} -o {}",
            function_list, binary, output
        );
        println!("{}", command);
        let _ = process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .unwrap();
        let _ = fs::copy(&bc_path, bc_name).unwrap_or_else(|_| panic!("{} not found", &bc_path));
        let _ = fs::copy(&ll_path, ll_name).unwrap_or_else(|_| panic!("{} not found", &ll_path));
    }
}
