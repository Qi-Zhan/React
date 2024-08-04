use std::fmt;

use llvm_ir::{Instruction, Terminator};

#[derive(PartialEq, Clone, Debug)]
pub struct KInstruction {
    pub inst: Instruction,
}

impl From<&Instruction> for KInstruction {
    fn from(inst: &Instruction) -> Self {
        Self { inst: inst.clone() }
    }
}

impl fmt::Display for KInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inst)
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct KTerminator {
    pub term: Terminator,
    pub mark: bool,
}

impl From<&Terminator> for KTerminator {
    fn from(term: &Terminator) -> Self {
        Self {
            term: term.clone(),
            mark: false,
        }
    }
}

impl fmt::Display for KTerminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.term)
    }
}
