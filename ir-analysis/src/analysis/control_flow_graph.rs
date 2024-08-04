use llvm_ir::{ConstantRef, Name, Operand, Terminator};
use petgraph::prelude::{DiGraphMap, Direction};
use std::{collections::HashMap, fmt};

use crate::module::KFunction;

/// The control flow graph for a particular function.
#[derive(Debug)]
pub struct ControlFlowGraph<'m> {
    pub(crate) graph: DiGraphMap<CFGNode<'m>, JumpKind<'m>>,

    /// Entry node for the function
    pub(crate) entry_node: CFGNode<'m>,

    /// Map from basic block name to basic block index
    pub(crate) name2index: HashMap<&'m Name, usize>,
}

/// A CFGNode represents a basic block, or the special node `Return`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum CFGNode<'m> {
    /// The block with the given `Name`
    Block(usize, &'m Name),
    /// The special `Return` node indicating function return
    Return,
}

#[derive(Clone, PartialEq, Debug)]
pub enum JumpKind<'m> {
    True(&'m Operand),
    False(&'m Operand),
    Address(&'m Operand),
    Switch(ConstantRef),
    SwitchDefault,
    Uncond,
}

impl<'m> CFGNode<'m> {
    fn bb_name(&self) -> String {
        match self {
            CFGNode::Block(_, name) => {
                "BB".to_string()
                    + &name
                        .to_string()
                        .strip_prefix('%')
                        .unwrap()
                        .to_string()
                        .replace('.', "")
            }
            CFGNode::Return => "Return".to_string(),
        }
    }
}
impl<'m> fmt::Display for CFGNode<'m> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CFGNode::Block(_, block) => write!(f, "{}", block),
            CFGNode::Return => write!(f, "Return"),
        }
    }
}

impl<'m> ControlFlowGraph<'m> {
    pub(crate) fn new(function: &'m KFunction) -> Self {
        let mut graph: DiGraphMap<CFGNode<'m>, JumpKind> = DiGraphMap::with_capacity(
            function.block_len() + 1,
            2 * function.block_len(), // arbitrary guess
        );

        let name2index = function
            .blocks()
            .map(|bb| (&bb.name, bb.index))
            .collect::<std::collections::HashMap<_, _>>();

        for bb in function.blocks() {
            let kt = function.get_term(bb.index);
            let term = &kt.term;
            match term {
                Terminator::Br(br) => {
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Block(name2index[&br.dest], &br.dest),
                        JumpKind::Uncond,
                    );
                }
                Terminator::CondBr(condbr) => {
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Block(name2index[&condbr.true_dest], &condbr.true_dest),
                        JumpKind::True(&condbr.condition),
                    );
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Block(name2index[&condbr.false_dest], &condbr.false_dest),
                        JumpKind::False(&condbr.condition),
                    );
                }
                Terminator::IndirectBr(ibr) => {
                    for dest in &ibr.possible_dests {
                        graph.add_edge(
                            CFGNode::Block(bb.index, &bb.name),
                            CFGNode::Block(name2index[dest], dest),
                            JumpKind::Address(&ibr.operand),
                        );
                    }
                }
                Terminator::Switch(switch) => {
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Block(name2index[&switch.default_dest], &switch.default_dest),
                        JumpKind::SwitchDefault,
                    );
                    for (value, dest) in &switch.dests {
                        graph.add_edge(
                            CFGNode::Block(bb.index, &bb.name),
                            CFGNode::Block(name2index[dest], dest),
                            JumpKind::Switch(value.clone()),
                        );
                    }
                }
                Terminator::Ret(_) | Terminator::Resume(_) => {
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Return,
                        JumpKind::Uncond,
                    );
                }
                Terminator::Invoke(invoke) => {
                    graph.add_edge(
                        CFGNode::Block(bb.index, &bb.name),
                        CFGNode::Block(name2index[&invoke.return_label], &invoke.return_label),
                        JumpKind::Uncond,
                    );
                }
                Terminator::CleanupRet(_cleanupret) => {
                    todo!("CleanupRet instruction in CFG construction");
                    // if let Some(dest) = &cleanupret.unwind_dest {
                    //     graph.add_edge(CFGNode::Block(&bb.name), CFGNode::Block(dest), ());
                    // } else {
                    //     graph.add_edge(CFGNode::Block(&bb.name), CFGNode::Return, ());
                    // }
                }
                Terminator::CatchRet(_catchret) => {
                    todo!("CatchRet instruction in CFG construction");
                    // Despite its name, my reading of the LLVM 10 LangRef indicates that CatchRet cannot directly return from the function
                    // graph.add_edge(
                    //     CFGNode::Block(&bb.name),
                    //     CFGNode::Block(&catchret.successor),
                    //     (),
                    // );
                }
                Terminator::CatchSwitch(_catchswitch) => {
                    todo!("CatchSwitch instruction in CFG construction")
                    // if let Some(dest) = &catchswitch.default_unwind_dest {
                    //     graph.add_edge(CFGNode::Block(&bb.name), CFGNode::Block(dest), ());
                    // } else {
                    //     graph.add_edge(CFGNode::Block(&bb.name), CFGNode::Return, ());
                    // }
                    // for handler in &catchswitch.catch_handlers {
                    //     graph.add_edge(CFGNode::Block(&bb.name), CFGNode::Block(handler), ());
                    // }
                }
                Terminator::CallBr(_) => unimplemented!("CallBr instruction"),
                Terminator::Unreachable(_) => {
                    // no successors
                }
            }
        }

        Self {
            graph,
            entry_node: CFGNode::Block(0, &function.get_block(0).name),
            name2index,
        }
    }

    /// Get the predecessors of the basic block with the given `Name`
    pub fn preds<'s>(&'s self, block: &'m Name) -> impl Iterator<Item = usize> + 's {
        self.preds_of_cfgnode(CFGNode::Block(self.name2index[block], block))
    }

    /// Get the predecessors of the special `Return` node, i.e., get all blocks
    /// which may directly return
    pub fn preds_of_return(&self) -> impl Iterator<Item = usize> + '_ {
        self.preds_of_cfgnode(CFGNode::Return)
    }

    pub(crate) fn preds_of_cfgnode<'s>(
        &'s self,
        node: CFGNode<'m>,
    ) -> impl Iterator<Item = usize> + 's {
        self.preds_as_nodes(node).map(|cfgnode| match cfgnode {
            CFGNode::Block(index, _) => index,
            CFGNode::Return => panic!("Shouldn't have CFGNode::Return as a predecessor"), // perhaps you tried to call this on a reversed CFG? In-crate users can use `preds_as_nodes()` if they need to account for the possibility of a reversed CFG
        })
    }

    pub(crate) fn preds_as_nodes<'s>(
        &'s self,
        node: CFGNode<'m>,
    ) -> impl Iterator<Item = CFGNode<'m>> + 's {
        self.graph.neighbors_directed(node, Direction::Incoming)
    }

    /// Get the successors of the basic block with the given `Name`.
    /// Here, `CFGNode::Return` indicates that the function may directly return
    /// from this basic block.
    pub fn succs<'s>(&'s self, block: &'m Name) -> impl Iterator<Item = CFGNode<'m>> + 's {
        self.graph.neighbors_directed(
            CFGNode::Block(self.name2index[block], block),
            Direction::Outgoing,
        )
    }

    /// Get the `Name` of the entry block for the function
    pub fn entry(&self) -> &'m Name {
        match self.entry_node {
            CFGNode::Block(_, block) => block,
            CFGNode::Return => panic!("Return node should not be entry"), // perhaps you tried to call this on a reversed CFG? In-crate users can use the `entry_node` field directly if they need to account for the possibility of a reversed CFG
        }
    }

    pub fn to_dot(&self, function: &'m KFunction) -> String {
        let mut dot = String::new();
        dot.push_str("digraph {\n");
        // add a point pointing to the entry node
        dot.push_str(&format!("    entry -> {};\n", self.entry_node.bb_name()));
        // edges
        for (a, b, e) in self.graph.all_edges() {
            match e {
                JumpKind::True(cond) => {
                    dot.push_str(&format!(
                        "    {} -> {} [label=\"true({})\"];\n",
                        a.bb_name(),
                        b.bb_name(),
                        cond
                    ));
                }
                JumpKind::False(cond) => {
                    dot.push_str(&format!(
                        "    {} -> {} [label=\"false({})\"];\n",
                        a.bb_name(),
                        b.bb_name(),
                        cond
                    ));
                }
                JumpKind::Address(_) => {
                    dot.push_str(&format!(
                        "    {} -> {} [label=\"address\"];\n",
                        a.bb_name(),
                        b.bb_name()
                    ));
                }
                JumpKind::Uncond => {
                    dot.push_str(&format!("    {} -> {};\n", a.bb_name(), b.bb_name()));
                }
                JumpKind::Switch(value) => {
                    dot.push_str(&format!(
                        "    {} -> {} [label=\"switch({})\"];\n",
                        a.bb_name(),
                        b.bb_name(),
                        value
                    ));
                }
                JumpKind::SwitchDefault => {
                    dot.push_str(&format!(
                        "    {} -> {} [label=\"default\"];\n",
                        a.bb_name(),
                        b.bb_name()
                    ));
                }
            }
        }

        // nodes
        for node in self.graph.nodes() {
            match node {
                CFGNode::Block(index, _) => {
                    let insts = function
                        .get_block(index)
                        .insts(function)
                        .map(|ki| ki.to_string().replace(['<', '>'], ""))
                        .collect::<Vec<String>>()
                        .join("\\l\n    ");
                    dot.push_str(&format!(
                        "    {} [shape=record, label=\"{{{}:\\l\\l  {}\\l}}\"];\n",
                        node.bb_name(),
                        node.bb_name(),
                        insts
                    ));
                }
                CFGNode::Return => {}
            }
        }
        dot.push_str("}\n");
        dot
    }
}
