//! build dataset
//!
//! - build bitcode

use std::{fs, path, process};

use anyhow::{Ok, Result};
use react::config::*;
use react::dataset::{read_cves, Cve, Project, State};

const GCLANG: &str = "gclang";
const GETBC: &str = "get-bc";
const MULTITHREAD: usize = 8;

trait CompileScript {
    /* Variable */
    fn repo_path(&self) -> String;
    fn generate_files(&self) -> Vec<&str>;
    fn file_name_in_repo(&self, file: &str) -> String;
    fn file_name_in_bitcode(&self, state: &State) -> String;

    /* Script */
    fn checkout_script(&self, stat: &State) -> String;
    fn configure_script(&self) -> String;
    fn extract_script(&self, state: &State) -> String;
    fn build_script(&self) -> String {
        format!("make -j {}", MULTITHREAD)
    }
    fn clean_script(&self) -> String {
        "git clean -f".to_string()
    }

    /* Command */
    fn run_command(&self, cmd: &str) -> Result<()> {
        let output = process::Command::new("bash")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.repo_path())
            .output()?;
        if output.status.success() {
            Ok(())
        } else {
            println!("{}", String::from_utf8_lossy(&output.stderr));
            anyhow::bail!(format!("{} failed", cmd));
        }
    }

    fn run_commands(&self, cmds: Vec<String>) -> Result<()> {
        for cmd in cmds {
            println!("{}", cmd);
            self.run_command(&cmd)?;
        }
        Ok(())
    }

    fn compile(&self, state: State) -> Result<()> {
        self.run_commands(vec![
            self.clean_script(),
            self.checkout_script(&state),
            self.configure_script(),
            self.build_script(),
            self.extract_script(&state),
            self.clean_script(),
        ])
    }
}

fn check_or_create(path: &str) -> Result<()> {
    if !path::Path::new(path).exists() {
        fs::create_dir(path)?;
    }
    Ok(())
}

fn prepare_dirs() -> Result<()> {
    // check if dataset/bitcodes/ exists
    check_or_create(&DATASET_DIR)?;
    check_or_create(&BITCODE_DIR)?;
    let projects = vec![
        Project::FFmpeg,
        Project::OpenSSL,
        Project::LibXml2,
        Project::Tcpdump,
    ];
    for project in projects {
        let project_dir = BITCODE_DIR.clone() + "/" + &project.to_string();
        check_or_create(&project_dir)?;
    }
    Ok(())
}

impl CompileScript for Cve {
    fn repo_path(&self) -> String {
        format!("{}/repos/{}", *DATASET_DIR, self.project)
    }

    fn file_name_in_repo(&self, file: &str) -> String {
        match self.project {
            Project::FFmpeg => format!("{file}_g"),
            #[cfg(target_os = "linux")]
            Project::OpenSSL => {
                if self.vuln.starts_with('1') {
                    format!("{file}.so.1.1")
                } else {
                    format!("{file}.so3")
                }
            }
            #[cfg(target_os = "macos")]
            Project::OpenSSL => {
                if self.vuln.starts_with('1') {
                    format!("{file}.1.1.dylib")
                } else {
                    format!("{file}.3.dylib")
                }
            }
            Project::Tcpdump => file.to_string(),
            #[cfg(target_os = "linux")]
            Project::LibXml2 => format!("./.libs/{file}.so.{}", &self.vuln[0..1]),
            #[cfg(target_os = "macos")]
            Project::LibXml2 => format!("./.libs/{file}.dylib.{}", &self.vuln[0..1]),
        }
    }

    fn file_name_in_bitcode(&self, state: &State) -> String {
        format!("{}_{}", self.id, state)
    }

    fn generate_files(&self) -> Vec<&str> {
        match self.project {
            Project::FFmpeg => vec!["ffmpeg"],
            Project::OpenSSL => vec!["libssl", "libcrypto"],
            Project::Tcpdump => vec!["tcpdump"],
            Project::LibXml2 => vec!["libxml2"],
        }
    }

    fn checkout_script(&self, state: &State) -> String {
        let commit = match state {
            State::Patch => self.commit.clone(),
            State::Vuln => self.commit.clone() + "~1",
        };
        format!("git checkout {}", commit)
    }

    fn extract_script(&self, state: &State) -> String {
        for file in self.generate_files() {
            if self.file.starts_with(file) {
                return format!(
                    "{} {} -o ../bitcodes/{}/{}",
                    GETBC,
                    self.file_name_in_repo(file),
                    self.project,
                    self.file_name_in_bitcode(state)
                );
            }
        }
        panic!("{} extract bitcode not found", self.id);
    }

    fn configure_script(&self) -> String {
        match self.project {
            Project::FFmpeg => format!("./configure --disable-optimizations --cc={GCLANG}"),
            Project::OpenSSL => {
                if self.vuln.starts_with("1.1.1") {
                    format!("CC={GCLANG} ./config --debug")
                } else {
                    format!("CC={GCLANG} ./Configure --debug")
                }
            }
            Project::Tcpdump | Project::LibXml2 => r#"./configure CFLAGS="-O0 -g"#.to_string(),
        }
    }
}

fn main() {
    prepare_dirs().unwrap();
    let cves = read_cves(&CVE_INFO).unwrap();
    for cve in cves.values() {
        if cve.project != Project::OpenSSL {
            continue;
        }
        cve.compile(State::Vuln).unwrap();
        cve.compile(State::Patch).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffmpeg_script() {
        let cve = Cve {
            id: "CVE-2020-22019".to_string(),
            func: "ff_vmafmotion_init".to_string(),
            vuln: "4.4".to_string(),
            patch: "4.4.1".to_string(),
            file: "ffmpeg".to_string(),
            commit: "cea03683b93c1569b33611d71233235933b3cbce".to_string(),
            project: Project::FFmpeg,
        };
        let commit = cve.checkout_script(&State::Vuln);
        assert_eq!(
            commit,
            "git checkout cea03683b93c1569b33611d71233235933b3cbce~1"
        );
        let commit = cve.checkout_script(&State::Patch);
        assert_eq!(
            commit,
            "git checkout cea03683b93c1569b33611d71233235933b3cbce"
        );
        let configure = cve.configure_script();
        assert_eq!(configure, "./configure --disable-optimizations --cc=gclang");
        let extract = cve.extract_script(&State::Vuln);
        assert_eq!(
            extract,
            "get-bc ffmpeg_g -o CVE-2020-22019_vuln".to_string(),
        );
    }

    #[test]
    fn test_openssl_script() {
        let cve = Cve {
            id: "CVE-2021-3711".to_string(),
            commit: "f6b9b7e".to_string(),
            vuln: "1.1.1".to_string(),
            project: Project::OpenSSL,
            func: "EVP_DigestSignInit".to_string(),
            patch: "openssl-1.1.1k".to_string(),
            file: "libssl".to_string(),
        };
        let configure = cve.configure_script();
        assert_eq!(configure, "CC=gclang ./config --debug");
        let extract = cve.extract_script(&State::Vuln);
        assert_eq!(
            extract,
            "get-bc libssl.1.1.dylib -o CVE-2021-3711_vuln".to_string(),
        );
    }
}
