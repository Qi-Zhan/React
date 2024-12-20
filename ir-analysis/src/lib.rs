#![allow(clippy::type_complexity)]

pub mod analysis;
mod effect;
pub mod emulator;
mod expr;
mod maxweight;
pub mod module;
pub mod smt;

use std::collections::{HashMap, HashSet};

use effect::Effect;
use emulator::Generator;
use maxweight::matching;
use module::{KFunction, KModule};
use smt::is_always_true;
use source_analysis::SourceDiff;

pub type Smt = (easy_smt::Context, HashMap<String, easy_smt::SExpr>);

pub struct IRAnalysis2 {
    vuln: String,
    patch: String,
    diff_path: String,
    standard: IRAnalysis,
    opt: Option<IRAnalysis>,
}

impl IRAnalysis2 {
    pub fn new(vuln: &str, patch: &str, diff_path: &str) -> Self {
        let mut standard = IRAnalysis::new(vuln, patch, diff_path);
        standard.generate();
        let opt = None;
        Self {
            vuln: vuln.to_string(),
            patch: patch.to_string(),
            diff_path: diff_path.to_string(),
            standard,
            opt,
        }
    }

    pub fn test(&mut self, target: &str, ctx: &mut Smt) -> IRState {
        let result = self.standard.test(target, ctx);
        match result {
            Ok(result) => result,
            Err(functions_effects) => {
                if self.opt.is_none() {
                    let new_vuln = self.vuln.replace(".bc", "_O3.bc");
                    let new_patch = self.patch.replace(".bc", "_O3.bc");
                    if std::fs::metadata(&new_vuln).is_err()
                        || std::fs::metadata(&new_patch).is_err()
                    {
                        return IRState::Vuln;
                    }
                    let mut opt = IRAnalysis::new(&new_vuln, &new_patch, &self.diff_path);
                    opt.generate();
                    self.opt = Some(opt);
                }
                self.opt.as_mut().unwrap().test2(functions_effects, ctx)
            }
        }
    }
}

struct IRAnalysis {
    vuln: KModule,
    patch: KModule,
    source_diff: SourceDiff,
    /// a **sorted** list of functions and effects to distinguish
    effects: Vec<(String, Option<Effect>, Option<Effect>)>,
    /// strings in vuln and patch functions
    strings: HashMap<String, (HashSet<String>, HashSet<String>)>,
}

impl IRAnalysis {
    pub fn new(vuln: &str, patch: &str, diff_path: &str) -> Self {
        let patch = KModule::from_bc_path(patch).expect("failed to load patch bitcode");
        let vuln = KModule::from_bc_path(vuln).expect("failed to load vuln bitcode");
        let source_diff = SourceDiff::from_path(diff_path).expect("failed to load source diff");
        Self {
            vuln,
            patch,
            source_diff,
            effects: vec![],
            strings: HashMap::new(),
        }
    }

    /// generate signature to distinguish two functions
    pub fn generate(&mut self) {
        let added_functions = self.source_diff.function_add.keys().collect::<HashSet<_>>();
        let deleted_functions = self.source_diff.function_del.keys().collect::<HashSet<_>>();
        let mut effects = HashMap::new();
        if cfg!(debug_assertions) {
            assert_eq!(added_functions, deleted_functions);
        }
        for name in added_functions {
            let func1 = self.vuln.get_function(name);
            let func2 = self.patch.get_function(name);
            match (func1, func2) {
                (Some(func1), Some(func2)) => {
                    let (
                        vuln_effects,
                        patch_effects,
                        vuln_all,
                        patch_all,
                        vuln_strings,
                        patch_strings,
                    ) = self.generate_modify(func1, func2);
                    self.strings
                        .insert(name.clone(), (vuln_strings, patch_strings));
                    if cfg!(debug_assertions) {
                        println!("{}: ", name);
                        println!("vuln: ");
                        for effect in &vuln_effects {
                            println!("{}, ", effect);
                        }
                        println!("patch: ");
                        for effect in &patch_effects {
                            println!("{}, ", effect);
                        }
                    }
                    effects.insert(
                        name.clone(),
                        (vuln_effects, patch_effects, vuln_all, patch_all),
                    );
                }
                _ => {
                    println!("function {} not found", name);
                }
            }
        }
        let effects = self.refine(effects);
        if cfg!(debug_assertions) {
            println!("Refined effects: ");
            for (name, (vuln, patch)) in &effects {
                println!("{}: ", name);
                println!("vuln: ");
                for effect in vuln {
                    println!("{}, ", effect);
                }
                println!("patch: ");
                for effect in patch {
                    println!("{}, ", effect);
                }
            }
        }
        let effects = self.rank(effects);
        if cfg!(debug_assertions) {
            println!("Ranked effects: ");
            for (name, vuln, patch) in &effects {
                let vuln_string = vuln
                    .as_ref()
                    .map(|effect| effect.to_string())
                    .unwrap_or_default();
                let patch_string = patch
                    .as_ref()
                    .map(|effect| effect.to_string())
                    .unwrap_or_default();
                println!("{} {} {}", name, vuln_string, patch_string);
            }
        }
        self.effects = effects;
    }

    fn refine(
        &self,
        effects: HashMap<String, (Vec<Effect>, Vec<Effect>, Vec<Effect>, Vec<Effect>)>,
    ) -> HashMap<String, (Vec<Effect>, Vec<Effect>)> {
        #[cfg(feature = "wrefine")]
        return effects
            .into_iter()
            .map(|(name, (vuln, patch, _, _))| (name, (vuln, patch)))
            .collect();
        #[cfg(not(feature = "wrefine"))]
        effects
            .into_iter()
            .map(|(name, (vuln, patch, vuln_all, patch_all))| {
                let mut vuln_new = vuln.clone();
                let mut patch_new = patch.clone();
                vuln_new = vuln_new
                    .into_iter()
                    .map(|effect| effect.refine(&patch_all))
                    .collect();
                patch_new = patch_new
                    .into_iter()
                    .map(|effect| effect.refine(&vuln_all))
                    .collect();
                (name, (vuln_new, patch_new))
            })
            .collect()
    }

    fn match_one_function(
        &self,
        vuln: Vec<Effect>,
        patch: Vec<Effect>,
    ) -> Vec<(Option<Effect>, Option<Effect>)> {
        if vuln.is_empty() {
            patch
                .into_iter()
                .map(|effect| (None, Some(effect)))
                .collect()
        } else if patch.is_empty() {
            vuln.into_iter()
                .map(|effect| (Some(effect), None))
                .collect()
        } else {
            let (vuln_len, patch_len) = (vuln.len(), patch.len());
            let matrix = if vuln_len < patch_len {
                // construct matrix Vec<Vec<i32>> with similarity
                vuln.iter()
                    .map(|vuln_effect| {
                        patch
                            .iter()
                            .map(|patch_effect| vuln_effect.similarity(patch_effect))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            } else {
                patch
                    .iter()
                    .map(|patch_effect| {
                        vuln.iter()
                            .map(|vuln_effect| patch_effect.similarity(vuln_effect))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            // for every row, match_[row] is the column index of the matched element
            let match_ = matching(matrix);
            // construct result
            let mut result = vec![];
            let mut exists = HashSet::new();
            for (row, col) in match_.iter().enumerate() {
                if vuln_len < patch_len {
                    result.push((Some(vuln[row].clone()), Some(patch[*col].clone())));
                    exists.insert(*col);
                } else {
                    result.push((Some(vuln[*col].clone()), Some(patch[row].clone())));
                    exists.insert(row);
                }
            }
            if vuln_len < patch_len {
                if cfg!(debug_assertions) {
                    assert_eq!(vuln_len, result.len());
                }
                for (i, p) in patch.into_iter().enumerate() {
                    if !exists.contains(&i) {
                        result.push((None, Some(p)));
                    }
                }
            } else {
                if cfg!(debug_assertions) {
                    assert_eq!(patch_len, result.len());
                }
                for (i, v) in vuln.into_iter().enumerate() {
                    if !exists.contains(&i) {
                        result.push((Some(v), None));
                    }
                }
            }
            result
        }
    }

    fn rank(
        &self,
        effects: HashMap<String, (Vec<Effect>, Vec<Effect>)>,
    ) -> Vec<(String, Option<Effect>, Option<Effect>)> {
        #[cfg(feature = "wrank")]
        return effects
            .into_iter()
            .flat_map(|(name, (vuln, patch))| {
                let vuln_len = vuln.len();
                let patch_len = patch.len();
                let mut vec = vec![];
                if vuln_len < patch_len {
                    for i in patch.iter().skip(vuln_len) {
                        vec.push((name.clone(), None, Some(i.clone())));
                    }
                } else {
                    for i in vuln.iter().skip(patch_len) {
                        vec.push((name.clone(), Some(i.clone()), None));
                    }
                }
                for i in vuln.iter().zip(patch.iter()) {
                    vec.push((name.clone(), Some(i.0.clone()), Some(i.1.clone())));
                }
                vec
            })
            .collect();
        let mut new_effects = vec![];
        for (name, (vuln, patch)) in effects {
            let match_result = self.match_one_function(vuln, patch);
            for (vuln, patch) in match_result {
                new_effects.push((name.clone(), vuln, patch));
            }
        }
        new_effects.sort_by(|(_, vuln, patch), (_, vuln2, patch2)| {
            // if one is two some and another is none, the former is smaller
            if vuln.is_some() && patch.is_some() && (vuln2.is_none() || patch2.is_none()) {
                std::cmp::Ordering::Less
            } else if vuln2.is_some() && patch2.is_some() && (vuln.is_none() || patch.is_none()) {
                std::cmp::Ordering::Greater
            } else {
                let vuln = vuln.as_ref().map(|effect| effect.complexity()).unwrap_or(0);
                let patch = patch
                    .as_ref()
                    .map(|effect| effect.complexity())
                    .unwrap_or(0);
                let vuln2 = vuln2
                    .as_ref()
                    .map(|effect| effect.complexity())
                    .unwrap_or(0);
                let patch2 = patch2
                    .as_ref()
                    .map(|effect| effect.complexity())
                    .unwrap_or(0);
                (vuln + patch).cmp(&(vuln2 + patch2))
            }
        });
        new_effects
    }

    fn test_effect(
        &self,
        vuln: Option<&Effect>,
        patch: Option<&Effect>,
        target: &[Effect],
        ctx: &mut Smt,
    ) -> Option<IRState> {
        if cfg!(debug_assertions) {
            for effect in target {
                if let Effect::Condition(..) = effect {
                    // println!("target: {}, ", effect);
                }
            }
        }
        match (vuln, patch) {
            (Some(vuln), Some(patch)) => {
                let (vuln_r, patch_r) = (target.contains(vuln), target.contains(patch));
                if cfg!(debug_assertions) {
                    println!("vuln: {}, patch: {} {vuln} {patch}", vuln_r, patch_r);
                }
                match (vuln_r, patch_r) {
                    (true, true) => None,
                    (true, false) => Some(IRState::Vuln),
                    (false, true) => Some(IRState::Patch),
                    (false, false) => {
                        #[cfg(feature = "wsmt")]
                        return None;
                        // ** costly slow path **
                        let vuln_r = smt::contains(vuln, target, &mut ctx.0, &mut ctx.1);
                        let patch_r = smt::contains(patch, target, &mut ctx.0, &mut ctx.1);
                        if cfg!(debug_assertions) {
                            println!(
                                "vuln-smt: {}, patch-smt: {} {vuln} {patch}",
                                vuln_r, patch_r
                            );
                        }
                        match (vuln_r, patch_r) {
                            (true, true) => None,
                            (true, false) => Some(IRState::Vuln),
                            (false, true) => Some(IRState::Patch),
                            (false, false) => None,
                        }
                    }
                }
            }
            (Some(vuln), None) => match target.contains(vuln) {
                true => Some(IRState::Vuln),
                false => {
                    #[cfg(feature = "wsmt")]
                    return None;
                    if is_always_true(vuln, &mut ctx.0, &mut ctx.1) {
                        return None;
                    }
                    // ** costly slow path **
                    match smt::contains(vuln, target, &mut ctx.0, &mut ctx.1) {
                        true => Some(IRState::Vuln),
                        false => None,
                    }
                }
            },
            (None, Some(patch)) => {
                #[cfg(feature = "wsmt")]
                return None;
                let result = target.contains(patch);
                if cfg!(debug_assertions) {
                    println!("patch: {} {patch}", result);
                }
                if is_always_true(patch, &mut ctx.0, &mut ctx.1) {
                    return None;
                }
                match result {
                    true => Some(IRState::Patch),
                    false => {
                        // ** costly slow path **
                        let result = smt::contains(patch, target, &mut ctx.0, &mut ctx.1);
                        if cfg!(debug_assertions) {
                            println!("patch-smt: {} {patch}", result);
                        }
                        if result {
                            Some(IRState::Patch)
                        } else {
                            None
                        }
                    }
                }
            }
            (None, None) => unreachable!("no effect to test"),
        }
    }

    fn test_strings(
        &self,
        strings: &HashSet<String>,
        vuln_strings: &HashSet<String>,
        patch_strings: &HashSet<String>,
    ) -> Option<IRState> {
        if vuln_strings.is_empty() && patch_strings.is_empty() {
            return None;
        }
        let vuln = strings.intersection(vuln_strings).collect::<HashSet<_>>();
        let patch = strings.intersection(patch_strings).collect::<HashSet<_>>();
        match (vuln.is_empty(), patch.is_empty()) {
            (true, true) => {
                if !vuln_strings.is_empty() && patch_strings.is_empty() {
                    Some(IRState::Patch)
                } else if !patch_strings.is_empty() && vuln_strings.is_empty() {
                    Some(IRState::Vuln)
                } else {
                    None
                }
            }
            (true, false) => Some(IRState::Patch),
            (false, true) => Some(IRState::Vuln),
            (false, false) => None,
        }
    }

    pub fn test(
        &self,
        target: &str,
        ctx: &mut Smt,
    ) -> Result<IRState, HashMap<String, Vec<Effect>>> {
        let target_mod = KModule::from_bc_path(target).expect("failed to load target bitcode");
        let functions = self
            .effects
            .iter()
            .map(|(name, _, _)| name)
            .collect::<Vec<_>>();
        if functions.is_empty() {
            return Ok(IRState::Vuln);
        }
        let exist_functions = functions
            .iter()
            .flat_map(|name| target_mod.get_function(name).map(|func| (name, func)))
            .collect::<HashMap<_, _>>();
        if exist_functions.is_empty() {
            panic!("no function in test {}", target);
        }
        let mut functions_effects = HashMap::new();
        let mut functions_strings = HashMap::new();
        let mut strings_tested = HashSet::new();
        for (name, vuln, patch) in &self.effects {
            if let Some(kfunc) = exist_functions.get(&name) {
                if !functions_effects.contains_key(name) {
                    let (effects, strings) = Generator::new(kfunc, &target_mod).execute_function();
                    functions_effects.insert(name.clone(), effects.into_iter().collect::<Vec<_>>());
                    functions_strings.insert(name.clone(), strings);
                }
                let (vuln_strings, patch_strings) = &self.strings[name];
                let function_string = &functions_strings[name];
                if !strings_tested.contains(name) {
                    if let Some(result) =
                        self.test_strings(function_string, vuln_strings, patch_strings)
                    {
                        return Ok(result);
                    }
                }
                strings_tested.insert(name);
                let function_effect = &functions_effects[name];
                if let Some(result) =
                    self.test_effect(vuln.as_ref(), patch.as_ref(), function_effect, ctx)
                {
                    return Ok(result);
                }
            }
        }
        println!("no effects useful");
        let all_patch_none = self.effects.iter().all(|(_, _, patch)| patch.is_none());
        let all_vuln_none = self.effects.iter().all(|(_, vuln, _)| vuln.is_none());
        // double check
        if self.source_diff.pure_deletion && all_patch_none {
            return Ok(IRState::Patch);
        }
        if self.source_diff.pure_addition && all_vuln_none {
            return Ok(IRState::Vuln);
        }
        Err(functions_effects)
    }

    /// opt test2
    pub fn test2(&self, functions_effects: HashMap<String, Vec<Effect>>, ctx: &mut Smt) -> IRState {
        for (name, vuln, patch) in &self.effects {
            if functions_effects.contains_key(name) {
                let function_effect = &functions_effects[name];
                if let Some(result) =
                    self.test_effect(vuln.as_ref(), patch.as_ref(), function_effect, ctx)
                {
                    return result;
                }
            }
        }
        IRState::Vuln
    }

    fn generate_modify(
        &self,
        fun1: &KFunction,
        fun2: &KFunction,
    ) -> (
        // effects in vuln but not in patch
        Vec<Effect>,
        // effects in patch but not in vuln
        Vec<Effect>,
        // effects in vuln
        Vec<Effect>,
        // effects in patch
        Vec<Effect>,
        // strings in vuln but not in patch
        HashSet<String>,
        // strings in patch but not in vuln
        HashSet<String>,
    ) {
        let generator = Generator::new(fun1, &self.vuln);
        let (vuln_effects, vuln_strings) = generator.execute_function();
        let generator = Generator::new(fun2, &self.patch);
        let (patch_effects, patch_strings) = generator.execute_function();
        (
            vuln_effects.difference(&patch_effects).cloned().collect(),
            patch_effects.difference(&vuln_effects).cloned().collect(),
            vuln_effects.into_iter().collect(),
            patch_effects.into_iter().collect(),
            vuln_strings.difference(&patch_strings).cloned().collect(),
            patch_strings.difference(&vuln_strings).cloned().collect(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IRState {
    Patch,
    Vuln,
}
