use std::collections::HashMap;

use any_lexer::{CLexer, CToken};
use anyhow::Result;
use unidiff::PatchSet;

/// try to find the function name in the diff line
fn find_function_name(line: &str) -> Option<String> {
    let lexer = CLexer::new(line);
    let tokens = lexer
        .into_iter()
        .filter(|token| token.0 != CToken::Space)
        .collect::<Vec<_>>();
    let mut deep = 0; // record match deepth
    for (i, token) in tokens.iter().rev().enumerate() {
        match token.1.as_str() {
            ")" => deep += 1,
            "(" => deep -= 1,
            _ => {}
        }
        if deep <= 0 && token.1.as_str() == "(" {
            let name = &tokens[tokens.len() - i - 2];
            assert_eq!(name.0, CToken::Ident);
            return Some(name.1.to_string());
        }
    }
    None
}

type AddLine = Vec<usize>;
type DelLine = Vec<usize>;

#[derive(Debug, Clone)]
pub struct SourceDiff {
    /// function name -> added lines
    pub function_add: HashMap<String, AddLine>,
    /// function name -> deleted lines
    pub function_del: HashMap<String, DelLine>,
    pub pure_deletion: bool,
    pub pure_addition: bool,
}

impl SourceDiff {
    /// Generate source-level diff based on path
    pub fn from_path(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        SourceDiff::parse(&content)
    }

    /// Generate source-level diff based on content
    pub fn parse(content: &str) -> Result<Self> {
        let mut patch_set = PatchSet::new();
        patch_set.parse(content)?;
        Self::new(patch_set)
    }

    fn new(patch: PatchSet) -> Result<Self> {
        let mut pure_deletion = true;
        let mut pure_addition = true;
        let mut function_add = HashMap::new();
        let mut function_del = HashMap::new();
        for file in patch.files() {
            if !file.source_file.ends_with(".c") {
                continue;
            }
            for hunk in file.hunks() {
                if let Some(name) = find_function_name(&hunk.section_header) {
                    function_add.entry(name.clone()).or_insert_with(Vec::new);
                    function_del.entry(name.clone()).or_insert_with(Vec::new);
                    let mut add_line = Vec::new();
                    let mut del_line = Vec::new();
                    for line in hunk.lines() {
                        if line.is_added() {
                            pure_deletion = false;
                            add_line.push(line.target_line_no.unwrap());
                        } else if line.is_removed() {
                            pure_addition = false;
                            del_line.push(line.source_line_no.unwrap());
                        }
                    }
                    function_add.get_mut(&name).unwrap().extend(add_line);
                    function_del.get_mut(&name).unwrap().extend(del_line);
                } else {
                    eprintln!("cannot find function name in `{}`", hunk.section_header)
                }
            }
        }
        Ok(Self {
            function_add,
            function_del,
            pure_deletion,
            pure_addition,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_function() {
        let line = "static int find_next_start_code(const uint8_t *buf, const uint8_t *next_avc)";
        let name = find_function_name(line).unwrap();
        assert_eq!(name, "find_next_start_code");
        let line = "int find(const (uint8_t *)buf, const uint8_t *next_avc)";
        let name = find_function_name(line).unwrap();
        assert_eq!(name, "find");
    }

    #[test]
    fn test_diff_parse() {
        let path = "../dataset/diff/CVE-2022-3602_fe3b63.diff";
        let diff_results = SourceDiff::from_path(path).unwrap();
        assert_eq!(diff_results.function_add.len(), 1);
        assert_eq!(diff_results.function_del.len(), 1);
    }
}
