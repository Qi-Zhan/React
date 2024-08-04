use std::collections::HashMap;

use llvm_ir::{function::Parameter, Function, Name, TypeRef};

use super::{
    kblock::KBlock,
    kinstruction::{KInstruction, KTerminator},
};

#[derive(PartialEq, Clone, Debug)]
pub struct KFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: TypeRef,
    kblocks: Vec<KBlock>,
    insts: Vec<KInstruction>,
    /// terminator instructions
    ///
    /// index is basic block index
    terms: Vec<KTerminator>,
    /// basic block name -> index
    pub bb2index: HashMap<Name, usize>,
}

impl KFunction {
    pub fn insts(&self) -> impl Iterator<Item = &KInstruction> {
        self.insts.iter()
    }

    pub fn blocks(&self) -> impl Iterator<Item = &KBlock> {
        self.kblocks.iter()
    }

    pub fn terms(&self) -> impl Iterator<Item = &KTerminator> {
        self.terms.iter()
    }

    pub fn get_inst(&self, index: usize) -> &KInstruction {
        &self.insts[index]
    }

    pub fn get_block(&self, index: usize) -> &KBlock {
        &self.kblocks[index]
    }

    pub fn get_term(&self, index: usize) -> &KTerminator {
        &self.terms[index]
    }

    pub fn get_term_mut(&mut self, index: usize) -> &mut KTerminator {
        &mut self.terms[index]
    }

    pub fn get_inst_mut(&mut self, index: usize) -> &mut KInstruction {
        &mut self.insts[index]
    }

    pub fn get_block_by_name(&self, name: &Name) -> Option<&KBlock> {
        if let Some(index) = self.bb2index.get(name) {
            Some(&self.kblocks[*index])
        } else {
            None
        }
    }

    pub fn block_len(&self) -> usize {
        self.kblocks.len()
    }
}

impl From<&Function> for KFunction {
    fn from(func: &Function) -> Self {
        let mut kblocks = vec![];
        let mut index = 0;
        let mut instructions = vec![];
        let mut terminators = vec![];

        for (bb_index, basic_block) in func.basic_blocks.iter().enumerate() {
            let mut bb_instrs = vec![];
            // add instructions
            for instr in &basic_block.instrs {
                bb_instrs.push(index);
                index += 1;
                instructions.push(KInstruction::from(instr));
            }
            kblocks.push(KBlock::new(bb_index, &basic_block.name, bb_instrs));
            let kterm = KTerminator::from(&basic_block.term);
            terminators.push(kterm);
        }

        let name2index = func
            .basic_blocks
            .iter()
            .enumerate()
            .map(|(i, bb)| (bb.name.clone(), i))
            .collect();

        // assign index to each instruction
        Self {
            name: func.name.clone(),
            parameters: func.parameters.clone(),
            return_type: func.return_type.clone(),
            kblocks,
            insts: instructions,
            terms: terminators,
            bb2index: name2index,
        }
    }
}
