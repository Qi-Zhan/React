//! Main Program

use ir_analysis::IRState;

use std::collections::HashMap;
use std::{fs, io::Write};

use clap::Parser;

use ir_analysis::{smt::init_ctx, IRAnalysis2, Smt};
use react::config::DIFF_DIR;
use react::config::*;
use react::dataset::*;
use react::metric::precision_recall_f1;

const LOG_FILE: &str = "log.txt";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// CVE Pattern, None for all
    #[arg(short, long)]
    cve: Option<String>,
    /// number of test cases to run, None for all
    #[arg(short, long)]
    test: Option<usize>,
    /// specify which binary to run
    #[arg(short, long)]
    binary: Option<String>,
    /// specify which cve not to run
    #[arg(short, long)]
    exclude: Option<String>,
}

fn test_one_target(
    testcase: &TestCase,
    ir_analysis: &mut IRAnalysis2,
    solver: &mut Smt,
) -> TestResult {
    println!("testing {} ...", testcase);
    let test_bitcode_path = format!(
        "{}/{}/{}.bc",
        BITCODE_DIR.as_str(),
        testcase.project,
        testcase.file
    );
    let result = match ir_analysis.test(&test_bitcode_path, solver) {
        IRState::Patch => State::Patch,
        IRState::Vuln => State::Vuln,
    };
    // write to file log.txt
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .unwrap();
    file.write_fmt(format_args!("{}  tested: {}\n", testcase, result))
        .unwrap();
    TestResult {
        test: testcase.clone(),
        result,
    }
}

fn test_each_cve(
    cve: &Cve,
    tests: &[TestCase],
    number: Option<usize>,
    case: &Option<String>,
    solver: &mut Smt,
) -> Vec<TestResult> {
    println!("testing {} ...", cve.id);
    if cfg!(debug_assertions) {
        for test in tests {
            assert_eq!(test.cve, cve.id);
        }
    }
    let source_diff_path = &format!(
        "{}/{}_{}.diff",
        DIFF_DIR.as_str(),
        cve.id,
        &cve.commit[0..6]
    );
    let prefix = format!("{}/{}/{}", BITCODE_DIR.as_str(), cve.project, cve.id);
    let (bitcode_path1, bitcode_path2) = (prefix.clone() + "_vuln.bc", prefix + "_patch.bc");
    let mut ir_analysis = IRAnalysis2::new(&bitcode_path1, &bitcode_path2, source_diff_path);
    tests
        .iter()
        .filter(|test| case.is_none() || test.file.contains(case.as_ref().unwrap()))
        .take(number.unwrap_or(tests.len()))
        .map(|test| test_one_target(test, &mut ir_analysis, solver))
        .collect()
}

fn main() {
    if fs::metadata(LOG_FILE).is_ok() {
        fs::remove_file(LOG_FILE).unwrap();
    }
    let args = Args::parse();
    let cve_infos = read_cves(&CVE_INFO).unwrap();
    let mut tests = read_tests(&TEST).unwrap();
    tests.sort_by(|a, b| a.0.cmp(&b.0));
    let mut sovler: Smt = init_ctx().unwrap();
    let test_results = tests.iter();
    let test_results = test_results
        .filter_map(|(cve, tests)| {
            if let Some(exclude) = &args.exclude {
                if cve.contains(exclude) {
                    return None;
                }
            }
            if args.cve.is_none() || cve.contains(args.cve.as_ref().unwrap()) {
                let cve_info = cve_infos.get(cve).unwrap();
                let results = test_each_cve(cve_info, tests, args.test, &args.binary, &mut sovler);
                let prf = precision_recall_f1(&results);
                println!("{}: {:?}", cve, prf);
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(LOG_FILE)
                    .unwrap();
                file.write_fmt(format_args!("{} {}: {:?}\n", cve_info.project, cve, prf))
                    .unwrap();
                Some((cve, results))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    // calculate metrics for the whole dataset
    let results = test_results
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect::<Vec<_>>();
    let prf = precision_recall_f1(&results);
    println!("all: {:?}", prf);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .unwrap();
    file.write_fmt(format_args!("all: {:?}\n", prf)).unwrap();
    // calculate for each compile: gcc, clang and O0, O1, O2, O3 combination
    let mut rq2 = HashMap::new();
    for compiler in ["gcc", "clang"] {
        for opt in ["O0", "O1", "O2", "O3"] {
            rq2.insert((compiler.to_string(), opt.to_string()), Vec::new());
        }
    }
    for test in results {
        let (compiler, opt) = test.compiler_opt();
        rq2.get_mut(&(compiler, opt)).unwrap().push(test);
    }
    for ((compiler, opt), tests) in rq2 {
        let prf = precision_recall_f1(&tests);
        println!("{} {}: {:?}", compiler, opt, prf);
    }
}
