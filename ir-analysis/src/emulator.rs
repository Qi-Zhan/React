use llvm_ir::types::Typed;
use llvm_ir::{Instruction, Name, Operand, Terminator, TypeRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::analysis::{CFGNode, FunctionAnalysis, JumpKind};
use crate::module::{KBlock, KFunction, KInstruction, KModule, KTerminator};
use crate::{effect::Effect, expr::Expr};

#[derive(Debug, Clone)]
struct State {
    locals: HashMap<Name, Arc<Expr>>,
    memory: HashMap<Arc<Expr>, Arc<Expr>>,
    constraints: Vec<Arc<Expr>>,
    effects: HashSet<Effect>,
    strings: HashSet<String>,
}

impl State {
    fn empty(_kfunc: &KFunction) -> Self {
        Self {
            locals: HashMap::new(),
            memory: HashMap::new(),
            constraints: Vec::new(),
            effects: HashSet::new(),
            strings: HashSet::new(),
        }
    }

    fn get_local(&self, name: &Name) -> Option<&Arc<Expr>> {
        self.locals.get(name)
    }

    fn set_local(&mut self, name: Name, expr: Arc<Expr>) {
        self.locals.insert(name, expr);
    }

    fn get_mem(&mut self, name: &Arc<Expr>) -> &Arc<Expr> {
        // if do not exist, create a new memory name -> Mem(name)
        self.memory
            .entry(name.clone())
            .or_insert_with(|| Arc::new(Expr::Mem(name.clone())))
    }

    fn set_mem(&mut self, name: Arc<Expr>, expr: Arc<Expr>) {
        self.memory.insert(name, expr);
    }

    fn add_constraint(&mut self, expr: Arc<Expr>) {
        self.constraints.push(expr);
    }

    fn add_effect(&mut self, effect: Effect) {
        if let Effect::Return(x) = &effect {
            if let Some(0) = x.as_int() {
                return;
            }
        }
        self.effects.insert(effect);
    }

    fn get_effects(&self) -> &HashSet<Effect> {
        &self.effects
    }

    fn add_string(&mut self, string: String) {
        self.strings.insert(string);
    }

    fn get_strings(&self) -> &HashSet<String> {
        &self.strings
    }

    fn fork(&self) -> Self {
        Self {
            locals: self.locals.clone(),
            memory: self.memory.clone(),
            constraints: self.constraints.clone(),
            effects: HashSet::new(),
            strings: HashSet::new(),
        }
    }
}

pub struct Generator<'a> {
    analysis: FunctionAnalysis<'a>,
    kmod: &'a KModule,
    /// if we use `load %ret, %6` and `return (load %ret)`, then we also need to record store %6 as ret
    ret: Option<&'a Name>,
    /// if we use `%ret %x`, then we need to record assign %x as ret
    ret2: Option<&'a Name>,
}

impl<'a> Generator<'a> {
    pub fn new(kfunc: &'a KFunction, kmod: &'a KModule) -> Self {
        let function_analysis = FunctionAnalysis::new(kfunc);
        Self {
            analysis: function_analysis,
            kmod,
            ret: Generator::set_ret(kfunc),
            ret2: Generator::set_ret2(kfunc),
        }
    }

    fn set_ret2(kfunc: &'a KFunction) -> Option<&'a Name> {
        let len = kfunc.block_len();
        let term = kfunc.get_term(len - 1);
        if let Terminator::Ret(ret) = &term.term {
            if let Some(Operand::LocalOperand { name: ret_name, .. }) = &ret.return_operand {
                return Some(ret_name);
            }
        }
        None
    }

    fn set_ret(kfunc: &'a KFunction) -> Option<&'a Name> {
        let len = kfunc.block_len();
        let bb = kfunc.get_block(len - 1);
        let term = kfunc.get_term(len - 1);
        let last_inst = bb.insts(kfunc).last()?;
        if let Terminator::Ret(ret) = &term.term {
            if let Some(Operand::LocalOperand { name: ret_name, .. }) = &ret.return_operand {
                if let Instruction::Load(load) = &last_inst.inst {
                    if let Operand::LocalOperand { name: address, .. } = &load.address {
                        let dest_name = &load.dest;
                        // load, ret pattern
                        assert_eq!(dest_name, ret_name);
                        return Some(address);
                    }
                }
            }
        }
        None
    }

    fn get_visit_num(&self, kfunc: &'a KFunction) -> HashMap<usize, usize> {
        let mut phi_nodes = HashMap::new();
        for (index, block) in kfunc.blocks().enumerate() {
            if block.insts.is_empty() {
                phi_nodes.insert(index, 1);
            } else if let Instruction::Phi(phi) = &block.insts(kfunc).next().unwrap().inst {
                phi_nodes.insert(index, phi.incoming_values.len());
            } else {
                phi_nodes.insert(index, 1);
            }
        }
        phi_nodes
    }

    pub fn execute_function(&self) -> (HashSet<Effect>, HashSet<String>) {
        // println!("{}", self.ret.unwrap());
        let cfg = self.analysis.control_flow_graph();
        let kfunc = self.analysis.function();
        // all basic block entry
        let length = kfunc.block_len();
        let visited_expected = self.get_visit_num(kfunc);
        let mut states = vec![State::empty(kfunc); length];
        let mut worklist: Vec<_> = vec![(cfg.entry_node, None)];
        let mut visited = HashMap::new();
        let mut effects = HashSet::new();
        let mut strings = HashSet::new();
        while let Some((node, prev)) = worklist.pop() {
            let visit_num = visited.entry(node).or_insert_with(|| 0);
            *visit_num += 1;
            match node {
                CFGNode::Block(index, bb) => {
                    if index == 0 {
                        for (i, parameter) in kfunc.parameters.iter().enumerate() {
                            let expr = Expr::Parameter(i);
                            states[index].set_local(parameter.name.clone(), Arc::new(expr));
                        }
                    }
                    let state = &mut states[index];
                    let block = kfunc.get_block(index);
                    self.transfer_block(state, block, prev);
                    let constaint = self.transfer_term(state, kfunc.get_term(index));
                    effects.extend(state.get_effects().clone());
                    strings.extend(state.get_strings().clone());
                    let new_state = state.fork();
                    for succ in cfg.succs(bb) {
                        let visit_num = visited.get(&succ).unwrap_or(&0);
                        let expected_num = if let CFGNode::Block(succ_index, _) = succ {
                            visited_expected.get(&succ_index).unwrap()
                        } else {
                            &1
                        };
                        if *visit_num < *expected_num {
                            if let CFGNode::Block(succ_index, _) = succ {
                                let mut new_state = new_state.fork();
                                if let Some(ref expr) = constaint {
                                    match cfg.graph.edge_weight(node, succ).unwrap() {
                                        JumpKind::True(_) => {
                                            new_state.add_constraint(expr.clone());
                                        }
                                        JumpKind::False(_) => {
                                            new_state
                                                .add_constraint(Arc::new(Expr::Not(expr.clone())));
                                        }
                                        JumpKind::Switch(value) => {
                                            let switch_expr = Arc::new(Expr::ICmp(
                                                llvm_ir::IntPredicate::EQ,
                                                expr.clone(),
                                                Arc::new(Expr::Const(value.clone())),
                                            ));
                                            new_state.add_constraint(switch_expr);
                                        }
                                        JumpKind::SwitchDefault => {}
                                        _ => unreachable!(
                                            "{bb} unexpected jumpkind {:?}",
                                            cfg.graph.edge_weight(node, succ)
                                        ),
                                    }
                                }
                                states[succ_index] = new_state;
                                worklist.push((succ, Some(node)));
                            }
                        }
                    }
                }
                CFGNode::Return => continue,
            }
        }
        (effects, strings)
    }

    fn transfer_block(&self, state: &mut State, block: &KBlock, prev: Option<CFGNode>) {
        for inst in block.insts(self.analysis.function()) {
            self.transfer_inst(state, inst, prev);
        }
    }

    fn transfer_inst(&self, state: &mut State, ki: &KInstruction, prev: Option<CFGNode>) {
        let inst = &ki.inst;
        use Instruction::*;
        match inst {
            /* Arithmetric Operation */
            Add(add) => {
                let op1 = self.eval_operand(&add.operand0, state);
                let op2 = self.eval_operand(&add.operand1, state);
                let expr = if let Some(v1) = op1.as_int() {
                    if let Some(v2) = op2.as_int() {
                        if v1 == 0 {
                            op2
                        } else if v2 == 0 {
                            op1
                        } else {
                            Arc::new(Expr::build_constant_int(v1.wrapping_add(v2)))
                        }
                    } else if v1 == 0 {
                        op2.clone()
                    } else {
                        Arc::new(Expr::Add(op1, op2))
                    }
                } else if op2.is_zero() {
                    op1
                } else {
                    Arc::new(Expr::Add(op1, op2))
                };
                let dest = &add.dest;
                state.set_local(dest.clone(), expr);
            }
            Sub(sub) => {
                let op1 = self.eval_operand(&sub.operand0, state);
                let op2 = self.eval_operand(&sub.operand1, state);
                let expr = if let Some(v1) = op1.as_int() {
                    if let Some(v2) = op2.as_int() {
                        Arc::new(Expr::build_constant_int(v1.wrapping_sub(v2)))
                    } else {
                        Arc::new(Expr::Sub(op1, op2))
                    }
                } else if op2.is_zero() {
                    op1
                } else {
                    Arc::new(Expr::Sub(op1, op2))
                };
                let dest = &sub.dest;
                state.set_local(dest.clone(), expr);
            }
            Mul(mul) => {
                let op1 = self.eval_operand(&mul.operand0, state);
                let op2 = self.eval_operand(&mul.operand1, state);
                // if zero
                let expr = if op1.is_zero() || op2.is_zero() {
                    Arc::new(Expr::build_constant_int(0))
                } else {
                    Arc::new(Expr::Mul(op1, op2))
                };
                let dest = &mul.dest;
                state.set_local(dest.clone(), expr);
            }
            UDiv(udiv) => {
                let op1 = self.eval_operand(&udiv.operand0, state);
                let op2 = self.eval_operand(&udiv.operand1, state);
                let expr = if let Some(v1) = op1.as_int() {
                    if let Some(v2) = op2.as_int() {
                        Arc::new(Expr::build_constant_int(v1 / v2))
                    } else {
                        Arc::new(Expr::UDiv(op1, op2))
                    }
                } else {
                    Arc::new(Expr::UDiv(op1, op2))
                };
                let dest = &udiv.dest;
                state.set_local(dest.clone(), expr);
            }
            SDiv(sdiv) => {
                let op1 = self.eval_operand(&sdiv.operand0, state);
                let op2 = self.eval_operand(&sdiv.operand1, state);
                let expr = if let Some(v1) = op1.as_int() {
                    if let Some(v2) = op2.as_int() {
                        Arc::new(Expr::build_constant_int(v1 / v2))
                    } else {
                        Arc::new(Expr::SDiv(op1, op2))
                    }
                } else {
                    Arc::new(Expr::SDiv(op1, op2))
                };
                let dest = &sdiv.dest;
                state.set_local(dest.clone(), expr);
            }
            // rem seems useless
            URem(urem) => {
                let op1 = self.eval_operand(&urem.operand0, state);
                let op2 = self.eval_operand(&urem.operand1, state);
                let expr = Arc::new(Expr::URem(op1, op2));
                let dest = &urem.dest;
                state.set_local(dest.clone(), expr);
            }
            SRem(srem) => {
                let op1 = self.eval_operand(&srem.operand0, state);
                let op2 = self.eval_operand(&srem.operand1, state);
                let expr = Arc::new(Expr::SRem(op1, op2));
                let dest = &srem.dest;
                state.set_local(dest.clone(), expr);
            }
            And(and) => {
                let op1 = self.eval_operand(&and.operand0, state);
                let op2 = self.eval_operand(&and.operand1, state);
                let expr = Arc::new(Expr::And(op1, op2));
                let dest = &and.dest;
                state.set_local(dest.clone(), expr);
            }
            Or(or) => {
                let op1 = self.eval_operand(&or.operand0, state);
                let op2 = self.eval_operand(&or.operand1, state);
                let expr = Arc::new(Expr::Or(op1, op2));
                let dest = &or.dest;
                state.set_local(dest.clone(), expr);
            }
            Xor(xor) => {
                let op1 = self.eval_operand(&xor.operand0, state);
                let op2 = self.eval_operand(&xor.operand1, state);
                let expr = Arc::new(Expr::Xor(op1, op2));
                let dest = &xor.dest;
                state.set_local(dest.clone(), expr);
            }
            Shl(shl) => {
                let op1 = self.eval_operand(&shl.operand0, state);
                let op2 = self.eval_operand(&shl.operand1, state);
                let expr = if let Some(value) = op2.as_int() {
                    let op2 = Expr::build_constant_int(2u64.wrapping_pow(value as u32));
                    Arc::new(Expr::Mul(op1, Arc::new(op2)))
                } else {
                    Arc::new(Expr::Shl(op1, op2))
                };
                let dest = &shl.dest;
                state.set_local(dest.clone(), expr);
            }
            LShr(lshr) => {
                let op1 = self.eval_operand(&lshr.operand0, state);
                let op2 = self.eval_operand(&lshr.operand1, state);
                let expr = if let Some(value) = op2.as_int() {
                    let op2 = Expr::build_constant_int(2u64.wrapping_pow(value as u32));
                    Arc::new(Expr::UDiv(op1, Arc::new(op2)))
                } else {
                    Arc::new(Expr::LShr(op1, op2))
                };
                let dest = &lshr.dest;
                state.set_local(dest.clone(), expr);
            }
            AShr(ashr) => {
                let op1 = self.eval_operand(&ashr.operand0, state);
                let op2 = self.eval_operand(&ashr.operand1, state);
                let expr = if let Some(value) = op2.as_int() {
                    let op2 = Expr::build_constant_int(2u64.wrapping_pow(value as u32));
                    Arc::new(Expr::UDiv(op1, Arc::new(op2)))
                } else {
                    Arc::new(Expr::AShr(op1, op2))
                };
                let dest = &ashr.dest;
                state.set_local(dest.clone(), expr);
            }

            /* Float Arithmetic */
            FAdd(fadd) => {
                let op1 = self.eval_operand(&fadd.operand0, state);
                let op2 = self.eval_operand(&fadd.operand1, state);
                let expr = Arc::new(Expr::FAdd(op1, op2));
                let dest = &fadd.dest;
                state.set_local(dest.clone(), expr);
            }
            FSub(fsub) => {
                let op1 = self.eval_operand(&fsub.operand0, state);
                let op2 = self.eval_operand(&fsub.operand1, state);
                let expr = Arc::new(Expr::FSub(op1, op2));
                let dest = &fsub.dest;
                state.set_local(dest.clone(), expr);
            }
            FMul(fmul) => {
                let op1 = self.eval_operand(&fmul.operand0, state);
                let op2 = self.eval_operand(&fmul.operand1, state);
                let expr = Arc::new(Expr::FMul(op1, op2));
                let dest = &fmul.dest;
                state.set_local(dest.clone(), expr);
            }
            FDiv(fdiv) => {
                let op1 = self.eval_operand(&fdiv.operand0, state);
                let op2 = self.eval_operand(&fdiv.operand1, state);
                let expr = Arc::new(Expr::FDiv(op1, op2));
                let dest = &fdiv.dest;
                state.set_local(dest.clone(), expr);
            }
            FRem(frem) => {
                let op1 = self.eval_operand(&frem.operand0, state);
                let op2 = self.eval_operand(&frem.operand1, state);
                let expr = Arc::new(Expr::FRem(op1, op2));
                let dest = &frem.dest;
                state.set_local(dest.clone(), expr);
            }
            FNeg(fneg) => {
                let op = self.eval_operand(&fneg.operand, state);
                let expr = Arc::new(Expr::FNeg(op));
                let dest = &fneg.dest;
                state.set_local(dest.clone(), expr);
            }

            InsertValue(insertvalue) => {
                let dest = &insertvalue.dest;
                state.set_local(dest.clone(), Arc::new(Expr::build_constant_int(0)));
            }
            ExtractValue(extractvalue) => {
                let dest = &extractvalue.dest;
                state.set_local(dest.clone(), Arc::new(Expr::build_constant_int(0)));
            }

            Alloca(alloc) => state.set_local(
                alloc.dest.clone(),
                Arc::new(Expr::Alloc(alloc.dest.clone())),
            ),
            GetElementPtr(gep) => {
                let mut ptr_type = gep.address.get_type(&self.kmod.moudle.types);
                let mut address = self.eval_operand(&gep.address, state);
                for index in gep.indices.iter() {
                    let offset = self.eval_operand(index, state);
                    let offset = match offset.as_ref() {
                        Expr::Const(constant) => match constant.as_ref() {
                            llvm_ir::Constant::Int { bits: _, value } => *value,
                            _ => 0,
                        },
                        _ => 0,
                    };
                    if offset != 0 {
                        let size = ptr_type.up_to(offset, self.kmod);
                        // if addr is add + k, then we can simplify it to add + (k + offset)
                        // it is useful in struct field offset
                        if let Expr::Add(base, offset_expr) = &address.as_ref() {
                            if let Expr::Const(constant) = offset_expr.as_ref() {
                                if let llvm_ir::Constant::Int { value, .. } = constant.as_ref() {
                                    let new_offset = value.wrapping_add(size);
                                    address = Arc::new(Expr::Add(
                                        base.clone(),
                                        Arc::new(Expr::build_constant_int(new_offset)),
                                    ));
                                }
                            }
                        } else {
                            address = Arc::new(Expr::Add(
                                address.clone(),
                                Arc::new(Expr::build_constant_int(size)),
                            ));
                        }
                    }
                    ptr_type = ptr_type.subtype(offset as usize, self.kmod);
                    state.get_mem(&address);
                }
                state.set_local(gep.dest.clone(), address);
            }
            Load(load) => {
                let dest = &load.dest;
                let address = self.eval_operand(&load.address, state);
                let value = state.get_mem(&address).clone();
                state.set_local(dest.clone(), value);
            }
            Store(store) => {
                let address = self.eval_operand(&store.address, state);
                let value = self.eval_operand(&store.value, state);
                state.set_mem(address.clone(), value.clone());
                if let Some(ret) = self.ret {
                    if let Operand::LocalOperand { name, .. } = &store.address {
                        if name == ret {
                            state.add_effect(Effect::Return(value.clone()));
                        }
                    }
                }
                if address.contain_parameter() {
                    state.add_effect(Effect::ParameterWrite(address, value));
                }
            }
            Trunc(truc) => {
                let value = self.eval_operand(&truc.operand, state);
                let dest = &truc.dest;
                state.locals.insert(dest.clone(), value);
            }
            ZExt(zext) => {
                let value = self.eval_operand(&zext.operand, state);
                let dest = &zext.dest;
                state.locals.insert(dest.clone(), value);
            }
            SExt(sext) => {
                let value = self.eval_operand(&sext.operand, state);
                let dest = &sext.dest;
                state.locals.insert(dest.clone(), value);
            }
            FPTrunc(fptrunc) => {
                let value = self.eval_operand(&fptrunc.operand, state);
                let dest = &fptrunc.dest;
                state.locals.insert(dest.clone(), value);
            }
            FPExt(pfext) => {
                let value = self.eval_operand(&pfext.operand, state);
                let dest = &pfext.dest;
                state.locals.insert(dest.clone(), value);
            }
            FPToUI(ftoui) => {
                let value = self.eval_operand(&ftoui.operand, state);
                let dest = &ftoui.dest;
                state.locals.insert(dest.clone(), value);
            }
            FPToSI(ftosi) => {
                let value = self.eval_operand(&ftosi.operand, state);
                let dest = &ftosi.dest;
                state.locals.insert(dest.clone(), value);
            }
            UIToFP(utfp) => {
                let value = self.eval_operand(&utfp.operand, state);
                let dest = &utfp.dest;
                state.locals.insert(dest.clone(), value);
            }
            SIToFP(stfp) => {
                let value = self.eval_operand(&stfp.operand, state);
                let dest = &stfp.dest;
                state.locals.insert(dest.clone(), value);
            }
            PtrToInt(pti) => {
                let value = self.eval_operand(&pti.operand, state);
                let dest = &pti.dest;
                state.locals.insert(dest.clone(), value);
            }
            IntToPtr(itp) => {
                let value = self.eval_operand(&itp.operand, state);
                let dest = &itp.dest;
                state.locals.insert(dest.clone(), value);
            }
            BitCast(bc) => {
                let value = self.eval_operand(&bc.operand, state);
                let dest = &bc.dest;
                state.locals.insert(dest.clone(), value);
            }
            AddrSpaceCast(_asc) => todo!(),

            ICmp(icmp) => {
                let op1 = self.eval_operand(&icmp.operand0, state);
                let op2 = self.eval_operand(&icmp.operand1, state);
                let predicate = match icmp.predicate {
                    llvm_ir::IntPredicate::NE | llvm_ir::IntPredicate::EQ => {
                        llvm_ir::IntPredicate::EQ
                    }
                    llvm_ir::IntPredicate::UGT
                    | llvm_ir::IntPredicate::SGT
                    | llvm_ir::IntPredicate::SLE
                    | llvm_ir::IntPredicate::ULE => llvm_ir::IntPredicate::SLE,
                    llvm_ir::IntPredicate::UGE
                    | llvm_ir::IntPredicate::SGE
                    | llvm_ir::IntPredicate::SLT
                    | llvm_ir::IntPredicate::ULT => llvm_ir::IntPredicate::SLT,
                };
                let expr = Arc::new(Expr::ICmp(predicate, op1, op2));
                let dest = &icmp.dest;
                state.add_effect(Effect::Condition(expr.clone()));
                state.set_local(dest.clone(), expr);
            }
            FCmp(fcmp) => {
                let op1 = self.eval_operand(&fcmp.operand0, state);
                let op2 = self.eval_operand(&fcmp.operand1, state);
                let predicate = match fcmp.predicate {
                    llvm_ir::FPPredicate::ONE => llvm_ir::FPPredicate::OEQ,
                    _ => fcmp.predicate,
                };
                let expr = Arc::new(Expr::FCmp(predicate, op1, op2));
                state.add_effect(Effect::Condition(expr.clone()));
                let dest = &fcmp.dest;
                state.set_local(dest.clone(), expr);
            }
            Phi(phi) => {
                let prev_name = match prev.unwrap() {
                    CFGNode::Block(_, name) => name,
                    CFGNode::Return => todo!(),
                };
                for (dest, block) in &phi.incoming_values {
                    if block == prev_name {
                        let dest_value = self.eval_operand(dest, state);
                        if let Some(ret) = self.ret2 {
                            if &phi.dest == ret {
                                state.add_effect(Effect::Return(dest_value.clone()));
                            }
                        }
                        state.set_local(phi.dest.clone(), dest_value);
                        return;
                    }
                }
                unreachable!("phi node has no incoming value")
            }
            Select(select) => {
                let cond = self.eval_operand(&select.condition, state);
                let op1 = self.eval_operand(&select.true_value, state);
                let op2 = self.eval_operand(&select.false_value, state);
                let expr = Arc::new(Expr::If(cond, op1, op2));
                let dest = &select.dest;
                state.locals.insert(dest.clone(), expr);
            }
            Call(call) => {
                let call_str = call.to_string();
                if call_str.contains("llvm.dbg") {
                    return;
                }
                let args = call
                    .arguments
                    .iter()
                    .map(|(arg, _)| self.eval_operand(arg, state))
                    .collect::<Vec<_>>();
                let function_name = call.function.as_ref().either(
                    |_| Arc::new(Expr::String("inline".to_string())),
                    |op| self.eval_operand(op, state),
                );
                let effect = Effect::Call(function_name.clone(), args.clone());
                state.add_effect(effect);
                if let Some(dest) = &call.dest {
                    let expr = Arc::new(Expr::Call(function_name, args));
                    state.locals.insert(dest.clone(), expr);
                }
            }

            /* Vector Instructions */
            ShuffleVector(shufflevector) => {
                state.set_local(
                    shufflevector.dest.clone(),
                    Arc::new(Expr::build_constant_int(0)),
                );
            }
            InsertElement(insertelement) => {
                state.set_local(
                    insertelement.dest.clone(),
                    Arc::new(Expr::build_constant_int(0)),
                );
            }
            ExtractElement(extractelement) => {
                state.set_local(
                    extractelement.dest.clone(),
                    Arc::new(Expr::build_constant_int(0)),
                );
            }
            Freeze(freeze) => {
                let expr = self.eval_operand(&freeze.operand, state);
                let dest = &freeze.dest;
                state.set_local(dest.clone(), expr);
            }
            AtomicRMW(atomicrmw) => {
                let expr = self.eval_operand(&atomicrmw.value, state);
                let dest = &atomicrmw.dest;
                state.set_local(dest.clone(), expr);
            }
            /* Other Instrctions, do nothing */
            CmpXchg(..) | Fence(..) | LandingPad(..) | CatchPad(..) | CleanupPad(..)
            | VAArg(..) => {}
        }
    }

    fn transfer_term(&self, state: &mut State, kt: &KTerminator) -> Option<Arc<Expr>> {
        use Terminator::*;
        let term = &kt.term;
        let mut constraint = None;
        match term {
            Ret(ret) => {
                if let Some(operand) = &ret.return_operand {
                    let expr = self.eval_operand(operand, state);
                    let effect = Effect::Return(expr);
                    state.add_effect(effect);
                }
            }
            CondBr(condbr) => {
                constraint = Some(self.eval_operand(&condbr.condition, state));
            }
            Switch(switch) => constraint = Some(self.eval_operand(&switch.operand, state)),
            IndirectBr(_indirectbr) => todo!(),
            Invoke(_invoke) => todo!(),
            Resume(_resume) => todo!(),
            CatchSwitch(_catchswitch) => todo!(),
            CatchRet(_catchret) => todo!(),
            CleanupRet(_cleanupret) => todo!(),
            CallBr(_) => todo!(),
            Br(..) | Unreachable(..) => {}
        }
        constraint
    }

    fn eval_operand(&self, op: &Operand, state: &mut State) -> Arc<Expr> {
        match op {
            Operand::LocalOperand { name, .. } => state.get_local(name).unwrap().clone(),
            Operand::ConstantOperand(constant) => {
                // if get element ptr to string, we find string
                if let llvm_ir::Constant::GetElementPtr(gep) = constant.as_ref() {
                    // if type is [i8 x 4], we need to find the string
                    let ty = gep.address.get_type(&self.kmod.moudle.types);
                    if let llvm_ir::Type::PointerType {
                        pointee_type,
                        addr_space: _,
                    } = ty.as_ref()
                    {
                        if let llvm_ir::Type::ArrayType {
                            element_type,
                            num_elements: _,
                        } = pointee_type.as_ref()
                        {
                            if let llvm_ir::Type::IntegerType { bits } = element_type.as_ref() {
                                // char type
                                if *bits == 8 {
                                    if let llvm_ir::Constant::GlobalReference { name, ty: _ } =
                                        gep.address.as_ref()
                                    {
                                        let global =
                                            self.kmod.moudle.get_global_var_by_name(name).unwrap();
                                        let init = global.initializer.as_ref();
                                        return match init {
                                            Some(c) => match c.as_ref() {
                                                llvm_ir::Constant::Array {
                                                    element_type: _,
                                                    elements,
                                                } => {
                                                    let mut string = String::new();
                                                    for element in elements {
                                                        if let llvm_ir::Constant::Int {
                                                            bits: _,
                                                            value,
                                                        } = element.as_ref()
                                                        {
                                                            // if not newline
                                                            if *value != 10 {
                                                                string.push(*value as u8 as char);
                                                            }
                                                        }
                                                    }
                                                    state.add_string(string.clone());
                                                    Arc::new(Expr::String(string))
                                                }
                                                _ => Arc::new(Expr::String(name.to_string())),
                                            },
                                            _ => Arc::new(Expr::String(name.to_string())),
                                        };
                                    }
                                }
                            }
                        }
                    }
                } else if let llvm_ir::Constant::PtrToInt(ptr) = constant.as_ref() {
                    if let llvm_ir::Constant::GlobalReference { name, ty: _ } = ptr.operand.as_ref()
                    {
                        let global = self.kmod.moudle.get_global_var_by_name(name).unwrap();
                        if global.initializer.is_some() {
                            // if [4 x i8]*
                            if let llvm_ir::Type::PointerType {
                                pointee_type,
                                addr_space: _,
                            } = global.ty.as_ref()
                            {
                                if let llvm_ir::Type::ArrayType {
                                    element_type,
                                    num_elements: _,
                                } = pointee_type.as_ref()
                                {
                                    if let llvm_ir::Type::IntegerType { bits } =
                                        element_type.as_ref()
                                    {
                                        if *bits == 8 {
                                            // if [4 x i8]*, we need to find the string
                                            let init = global.initializer.as_ref().unwrap();
                                            if let llvm_ir::Constant::Array {
                                                element_type: _,
                                                elements,
                                            } = init.as_ref()
                                            {
                                                let mut string = String::new();
                                                for element in elements {
                                                    if let llvm_ir::Constant::Int {
                                                        bits: _,
                                                        value,
                                                    } = element.as_ref()
                                                    {
                                                        // if not newline
                                                        if *value != 10 {
                                                            string.push(*value as u8 as char);
                                                        }
                                                    }
                                                }
                                                state.add_string(string.clone());
                                                return Arc::new(Expr::String(string));
                                            }
                                        }
                                    }
                                }
                            }
                            return Arc::new(Expr::Const(global.initializer.clone().unwrap()));
                        }
                    }
                } else if let llvm_ir::Constant::GlobalReference { name, ty } = constant.as_ref() {
                    if let llvm_ir::Type::FuncType { .. } = ty.as_ref() {
                        return Arc::new(Expr::Const(constant.clone()));
                    } else {
                        // use @ to identify global variable
                        // it is LLVM lifter's bad, they cannot always get the correct value
                        if name.to_string().contains("global_var") {
                            return Arc::new(Expr::Any);
                        }
                        return Arc::new(Expr::String(name.to_string().replace('%', "@")));
                    }
                } else if let llvm_ir::Constant::Null(..) = constant.as_ref() {
                    return Arc::new(Expr::build_constant_int(0));
                }
                Arc::new(Expr::Const(constant.clone()))
            }
            Operand::MetadataOperand => todo!(),
        }
    }
}

trait Size {
    fn size(&self, kmod: &KModule) -> usize;
    fn up_to(&self, index: u64, kmod: &KModule) -> u64;
    fn subtype(&self, offset: usize, kmod: &KModule) -> Self
    where
        Self: Sized;
}

impl Size for TypeRef {
    fn size(&self, kmod: &KModule) -> usize {
        match self.as_ref() {
            llvm_ir::Type::VoidType => unreachable!("void type has no size"),
            llvm_ir::Type::IntegerType { bits } => *bits as usize / 8,
            llvm_ir::Type::PointerType { .. } => 8,
            llvm_ir::Type::FPType(fp) => match fp {
                llvm_ir::types::FPType::Half => 2,
                llvm_ir::types::FPType::BFloat => 2,
                llvm_ir::types::FPType::Single => 4,
                llvm_ir::types::FPType::Double => 8,
                llvm_ir::types::FPType::FP128 => 16,
                llvm_ir::types::FPType::X86_FP80 => 10,
                llvm_ir::types::FPType::PPC_FP128 => 16,
            },
            llvm_ir::Type::FuncType { .. } => unreachable!("function type has no size"),
            llvm_ir::Type::VectorType { .. } => todo!(),
            llvm_ir::Type::ArrayType {
                element_type,
                num_elements,
            } => element_type.size(kmod) * *num_elements,
            llvm_ir::Type::StructType {
                element_types,
                is_packed: _,
            } => {
                let max_size = element_types.iter().map(|ty| ty.size(kmod)).max().unwrap();
                let mut size = 0;
                for ty in element_types {
                    let ty_size = ty.size(kmod);
                    if ty_size % 8 == 0 && size % ty_size == 4 {
                        size += ty_size - size % ty_size;
                    }
                    size += ty_size;
                }
                // struct need to be aligned 8 unless it is packed
                if size % 8 == 4 && max_size > 4 {
                    size += 8 - size % 8;
                }
                size
            }
            llvm_ir::Type::NamedStructType { name } => {
                let struct_ty = kmod.moudle.types.named_struct_def(name).unwrap();
                match struct_ty {
                    llvm_ir::types::NamedStructDef::Opaque => {
                        todo!("opaque struct type {} size", name)
                    }
                    llvm_ir::types::NamedStructDef::Defined(field) => field.size(kmod),
                }
            }
            llvm_ir::Type::X86_MMXType => todo!(),
            llvm_ir::Type::X86_AMXType => todo!(),
            llvm_ir::Type::MetadataType => todo!(),
            llvm_ir::Type::LabelType => todo!(),
            llvm_ir::Type::TokenType => todo!(),
        }
    }

    fn up_to(&self, index: u64, kmod: &KModule) -> u64 {
        match self.as_ref() {
            llvm_ir::Type::IntegerType { bits } => *bits as u64 / 8 * index,
            llvm_ir::Type::PointerType { .. } => index.wrapping_mul(8),
            llvm_ir::Type::ArrayType {
                element_type,
                num_elements: _,
            } => element_type.size(kmod) as u64 * index,
            llvm_ir::Type::StructType { element_types, .. } => {
                let mut i = 0;
                for ty in element_types.iter().take(index as usize) {
                    let size = ty.size(kmod) as u64;
                    if size % 8 == 0 && i % size == 4 {
                        i += size - i % size;
                    }
                    i += size;
                }
                // padding for the last element
                let last_ty = element_types.get(index as usize).unwrap();
                let last_size = last_ty.size(kmod) as u64;
                if last_size % 8 == 0 && (i % 8 == 4 || i % 8 == 2 || i % 8 == 1) {
                    i + 8 - i % 8
                } else {
                    i
                }
            }
            llvm_ir::Type::NamedStructType { name } => {
                let struct_ty = kmod.moudle.types.named_struct_def(name).unwrap();
                match struct_ty {
                    llvm_ir::types::NamedStructDef::Opaque => {
                        todo!("opaque struct type {} size", name)
                    }
                    llvm_ir::types::NamedStructDef::Defined(field) => field.up_to(index, kmod),
                }
            }
            llvm_ir::Type::VectorType {
                element_type,
                num_elements,
                scalable: _,
            } => element_type.size(kmod) as u64 * index * *num_elements as u64,
            _ => unreachable!("{} is not supported in gep", self),
        }
    }

    fn subtype(&self, offset: usize, kmod: &KModule) -> Self
    where
        Self: Sized,
    {
        match self.as_ref() {
            llvm_ir::Type::PointerType {
                pointee_type,
                addr_space: _,
            } => pointee_type.clone(),
            llvm_ir::Type::ArrayType {
                element_type,
                num_elements: _,
            } => element_type.clone(),
            llvm_ir::Type::StructType {
                element_types,
                is_packed: _,
            } => element_types[offset].clone(),
            llvm_ir::Type::NamedStructType { name } => {
                let struct_ty = kmod.moudle.types.named_struct_def(name).unwrap();
                match struct_ty {
                    llvm_ir::types::NamedStructDef::Opaque => {
                        todo!("opaque struct type {} size", name)
                    }
                    llvm_ir::types::NamedStructDef::Defined(field) => field.subtype(offset, kmod),
                }
            }
            llvm_ir::Type::VectorType { element_type, .. } => element_type.clone(),
            _ => unreachable!("{} is not supported in gep", self),
        }
    }
}
