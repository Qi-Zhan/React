use llvm_ir::Name;

use super::{KFunction, KInstruction};

#[derive(PartialEq, Clone, Debug)]
pub struct KBlock {
    pub index: usize,
    pub name: Name,
    pub insts: Vec<usize>,
}

impl KBlock {
    pub fn new(index: usize, name: &Name, instrs: Vec<usize>) -> Self {
        Self {
            index,
            name: name.clone(),
            insts: instrs,
        }
    }

    pub fn insts<'a>(&'a self, function: &'a KFunction) -> impl Iterator<Item = &KInstruction> {
        self.insts.iter().map(move |i| function.get_inst(*i))
    }

    pub fn index_insts<'a>(&'a self, function: &'a KFunction) -> impl Iterator<Item = (usize, &KInstruction)> {
        self.insts.iter().map(move |i| (*i, function.get_inst(*i)))
    }
}
