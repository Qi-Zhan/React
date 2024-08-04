use std::collections::HashMap;
use std::fmt;

use anyhow::{bail, Result};
use easy_smt::{Context, ContextBuilder, Response, SExpr};

use crate::effect::Effect;

use super::expr::Expr;

#[derive(Debug, PartialEq)]
pub enum SolverResult {
    Equal,
    NotEqual,
}

pub fn contains(
    a: &Effect,
    vec: &[Effect],
    ctx: &mut Context,
    dec: &mut HashMap<String, SExpr>,
) -> bool {
    for b in vec {
        if let Ok(SolverResult::Equal) = equal(a, b, ctx, dec) {
            return true;
        }
    }
    false
}

pub fn init_ctx() -> Result<(Context, HashMap<String, SExpr>)> {
    match ContextBuilder::new()
        .solver("z3", ["-smt2", "-in"])
        // .replay_file(Some(std::fs::File::create("replay.smt2")?))
        .build()
    {
        Ok(mut ctx) => {
            let mut dec = HashMap::new();
            init_declaration(&mut dec, &mut ctx);
            Ok((ctx, dec))
        }
        Err(e) => {
            bail!("Failed to create context: {}", e)
        }
    }
}

fn init_declaration(dec: &mut HashMap<String, SExpr>, ctx: &mut Context) {
    let sexpr = ctx
        .declare_fun("mem", vec![ctx.int_sort()], ctx.int_sort())
        .unwrap();
    dec.insert("mem".to_string(), sexpr);
}

pub fn is_always_true(a: &Effect, ctx: &mut Context, dec: &mut HashMap<String, SExpr>) -> bool {
    match a {
        Effect::Condition(c) => {
            if c.contain_any() {
                return false;
            }
        }
        _ => return false,
    }
    ctx.push().unwrap();
    let result = always_true(a, ctx, dec);
    ctx.pop().unwrap();
    dec.retain(|k, _| k == "mem");
    result.unwrap_or(false)
}

/// Check if a condition is always true
fn always_true(a: &Effect, ctx: &mut Context, dec: &mut HashMap<String, SExpr>) -> Result<bool> {
    match a {
        Effect::Condition(cond) => {
            let a_expr = cond.expr_to_z3(ctx, dec)?;
            let const_1 = ctx.numeral(1);
            let not_expr = ctx.not(ctx.eq(a_expr, const_1));
            ctx.assert(not_expr)?;
            match ctx.check()? {
                Response::Sat | Response::Unknown => Ok(false),
                Response::Unsat => Ok(true),
            }
        }
        _ => Ok(false),
    }
}

pub fn equal(
    a: &Effect,
    b: &Effect,
    ctx: &mut Context,
    dec: &mut HashMap<String, SExpr>,
) -> Result<SolverResult> {
    let mut a_exprs = vec![];
    let mut b_exprs = vec![];
    match (a, b) {
        (Effect::Call(name, args), Effect::Call(name2, args2)) => {
            if name.to_string() != name2.to_string() {
                return Ok(SolverResult::NotEqual);
            }
            if args.len() != args2.len() {
                return Ok(SolverResult::NotEqual);
            }
            for (a, b) in args.iter().zip(args2.iter()) {
                a_exprs.push(a.clone());
                b_exprs.push(b.clone());
            }
        }
        (Effect::Return(a1), Effect::Return(b1))
        | (Effect::Condition(a1), Effect::Condition(b1)) => {
            a_exprs.push(a1.clone());
            b_exprs.push(b1.clone());
        }
        (Effect::ParameterWrite(a1, b1), Effect::ParameterWrite(a2, b2)) => {
            a_exprs.push(a1.clone());
            a_exprs.push(b1.clone());
            b_exprs.push(a2.clone());
            b_exprs.push(b2.clone());
        }
        _ => return Ok(SolverResult::NotEqual),
    }
    for (a, b) in a_exprs.iter().zip(b_exprs.iter()) {
        if a.is_any() || b.is_any() {
            continue;
        }
        if a.contain_any() || b.contain_any() {
            return Ok(SolverResult::NotEqual);
        }
        ctx.push().unwrap();
        let result = equal_expr(a, b, ctx, dec);
        ctx.pop().unwrap();
        // remove all declaration in dec unless it is mem
        dec.retain(|k, _| k == "mem");
        let result = result?;
        if let Response::Sat | Response::Unknown = result {
            return Ok(SolverResult::NotEqual);
        }
    }
    Ok(SolverResult::Equal)
}

fn equal_expr(
    a: &Expr,
    b: &Expr,
    ctx: &mut Context,
    dec: &mut HashMap<String, SExpr>,
) -> Result<Response> {
    let a_expr = a.expr_to_z3(ctx, dec)?;
    // // println!("a: {}", ctx.display(a_expr));
    let b_expr = b.expr_to_z3(ctx, dec)?;
    // // println!("b: {}", ctx.display(b_expr));
    // // let a_neq_b = ctx.not(ctx.eq(a_expr, b_expr));
    // // println!("a != b: {}", ctx.display(a_neq_b));
    ctx.assert(ctx.not(ctx.eq(a_expr, b_expr)))?;
    Ok(ctx.check()?)
}

impl Expr {
    fn expr_to_z3(&self, ctx: &mut Context, dec: &mut HashMap<String, SExpr>) -> Result<SExpr> {
        match self {
            Expr::Any => unreachable!("Any should not be used in solver"),
            Expr::Parameter(i) => Ok(*dec.entry(format!("arg{}", i)).or_insert_with(|| {
                ctx.declare_const(&format!("arg{}", i), ctx.int_sort())
                    .unwrap()
            })),
            Expr::Alloc(i) => Ok(*dec.entry(format!("alloc{}", i)).or_insert_with(|| {
                ctx.declare_const(&format!("alloc{}", i), ctx.int_sort())
                    .unwrap()
            })),
            // mem is a function
            Expr::Mem(i) => {
                let f = *dec.get("mem").unwrap();
                let arg = i.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![f, arg]))
            }
            Expr::Call(name, args) => {
                let call_name = format!("{}", name).replace(['@', '_'], "");
                let f = *dec.entry(call_name.clone()).or_insert_with(|| {
                    ctx.declare_fun(
                        call_name.clone(),
                        args.iter().map(|_| ctx.int_sort()).collect(),
                        ctx.int_sort(),
                    )
                    .unwrap()
                });
                let mut args = args
                    .iter()
                    .map(|arg| arg.expr_to_z3(ctx, dec))
                    .collect::<Result<Vec<_>>>()?;
                args.insert(0, f);
                Ok(ctx.list(args))
            }
            Expr::Const(i) => {
                if let llvm_ir::Constant::Int { value, .. } = i.as_ref() {
                    Ok(ctx.numeral(*value as i64))
                } else {
                    bail!("Only support Int")
                }
            }
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("+"), lhs, rhs]))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("-"), lhs, rhs]))
            }
            Expr::UDiv(lhs, rhs) | Expr::SDiv(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("div"), lhs, rhs]))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("*"), lhs, rhs]))
            }
            Expr::And(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                if let Expr::Const(c) = rhs.as_ref() {
                    if let llvm_ir::Constant::Int { value, .. } = c.as_ref() {
                        // if rhs is 0xffffffff, then lhs is the result
                        if *value == 0xffffffff {
                            return Ok(lhs);
                        }
                    }
                }
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("and"), lhs, rhs]))
            }
            Expr::Or(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("or"), lhs, rhs]))
            }
            Expr::Xor(lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                Ok(ctx.list(vec![ctx.atom("xor"), lhs, rhs]))
            }
            Expr::ICmp(pred, lhs, rhs) => {
                let lhs = lhs.expr_to_z3(ctx, dec)?;
                let rhs = rhs.expr_to_z3(ctx, dec)?;
                let pred = match pred {
                    llvm_ir::IntPredicate::EQ => ctx.atom("="),
                    llvm_ir::IntPredicate::NE => ctx.atom("distinct"),
                    llvm_ir::IntPredicate::UGT => ctx.atom(">"),
                    llvm_ir::IntPredicate::UGE => ctx.atom(">="),
                    llvm_ir::IntPredicate::ULT => ctx.atom("<"),
                    llvm_ir::IntPredicate::ULE => ctx.atom("<="),
                    llvm_ir::IntPredicate::SGT => ctx.atom(">"),
                    llvm_ir::IntPredicate::SGE => ctx.atom(">="),
                    llvm_ir::IntPredicate::SLT => ctx.atom("<"),
                    llvm_ir::IntPredicate::SLE => ctx.atom("<="),
                };
                // use if then else to represent icmp
                let ite = ctx.list(vec![
                    ctx.atom("ite"),
                    ctx.list(vec![pred, lhs, rhs]),
                    ctx.numeral(1),
                    ctx.numeral(0),
                ]);
                Ok(ite)
            }
            _ => bail!("Not supported"),
        }
    }
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverResult::Equal => write!(f, "Equal"),
            SolverResult::NotEqual => write!(f, "NotEqual"),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_name() {
        let a = "@BN_CTX_get".to_string();
        let a = a.replace(['@', '_'], "");
        assert_eq!(a, "BNCTXget");
    }
}
