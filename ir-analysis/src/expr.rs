#![allow(clippy::rc_clone_in_vec_init)]

use std::sync::Arc;
use std::{collections::HashSet, iter};

use llvm_ir::{predicates::*, Constant, ConstantRef, Name};

#[derive(Debug, Clone)]
pub enum Expr {
    /// for refined usage
    Any,
    /// A parameter.
    String(String),
    Parameter(usize),
    Alloc(Name),
    Mem(Arc<Expr>),
    Call(Arc<Expr>, Vec<Arc<Expr>>),
    If(Arc<Expr>, Arc<Expr>, Arc<Expr>),
    Const(ConstantRef),
    Not(Arc<Expr>),
    /// A binary operation.
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    UDiv(Arc<Expr>, Arc<Expr>),
    SDiv(Arc<Expr>, Arc<Expr>),
    URem(Arc<Expr>, Arc<Expr>),
    SRem(Arc<Expr>, Arc<Expr>),
    /// A bitwise binary operation.
    And(Arc<Expr>, Arc<Expr>),
    Or(Arc<Expr>, Arc<Expr>),
    Xor(Arc<Expr>, Arc<Expr>),
    Shl(Arc<Expr>, Arc<Expr>),
    LShr(Arc<Expr>, Arc<Expr>),
    AShr(Arc<Expr>, Arc<Expr>),

    /// A floating-point binary operation.
    FAdd(Arc<Expr>, Arc<Expr>),
    FSub(Arc<Expr>, Arc<Expr>),
    FMul(Arc<Expr>, Arc<Expr>),
    FDiv(Arc<Expr>, Arc<Expr>),
    FRem(Arc<Expr>, Arc<Expr>),
    FNeg(Arc<Expr>),

    // A Comparison
    ICmp(IntPredicate, Arc<Expr>, Arc<Expr>),
    FCmp(FPPredicate, Arc<Expr>, Arc<Expr>),
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (_, Self::Any) | (Self::Any, _) => true,
            (Self::Parameter(l0), Self::Parameter(r0)) => l0 == r0,
            (Self::Alloc(l0), Self::Alloc(r0)) => l0 == r0,
            (Self::Const(l0), Self::Const(r0)) => match (l0.as_ref(), r0.as_ref()) {
                (
                    Constant::Int {
                        bits: _l1,
                        value: l2,
                    },
                    Constant::Int {
                        bits: _r1,
                        value: r2,
                    },
                ) => l2 == r2,
                (Constant::Null(_), Constant::Null(_)) => true,
                // inttoptr is limitation of LLVM lifter
                // so we recognize inttoptr as any possible int
                (Constant::IntToPtr(..), Constant::Int { .. })
                | (Constant::Int { .. }, Constant::IntToPtr(..)) => true,
                _ => l0 == r0,
            },
            (Self::Mem(l0), Self::Mem(r0)) => l0 == r0,
            (Self::Call(l0, l1), Self::Call(r0, r1)) => {
                let min_length = l1.len().min(r1.len());
                // something like abc.a, we need to cut the string after the first dot
                l0.to_string().split('.').next().unwrap()
                    == r0.to_string().split('.').next().unwrap()
                    && l1[..min_length] == r1[..min_length]
            }
            (Self::If(l0, l1, l2), Self::If(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Not(l0), Self::Not(r0)) => l0 == r0,
            (Self::Add(l0, l1), Self::Add(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Sub(l0, l1), Self::Sub(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Mul(l0, l1), Self::Mul(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::UDiv(l0, l1), Self::UDiv(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::SDiv(l0, l1), Self::SDiv(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::URem(l0, l1), Self::URem(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::SRem(l0, l1), Self::SRem(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::And(l0, l1), Self::And(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Or(l0, l1), Self::Or(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Xor(l0, l1), Self::Xor(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Shl(l0, l1), Self::Shl(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::LShr(l0, l1), Self::LShr(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::AShr(l0, l1), Self::AShr(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FAdd(l0, l1), Self::FAdd(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FSub(l0, l1), Self::FSub(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FMul(l0, l1), Self::FMul(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FDiv(l0, l1), Self::FDiv(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FRem(l0, l1), Self::FRem(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::FNeg(l0), Self::FNeg(r0)) => l0 == r0,
            (Self::ICmp(l0, l1, l2), Self::ICmp(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::FCmp(l0, l1, l2), Self::FCmp(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Eq for Expr {}

macro_rules! op_possible_subs {
    ($op:ident, $lhs:ident, $rhs:ident) => {
        Box::new($lhs.possible_subs().flat_map(move |l| {
            $rhs.possible_subs()
                .map(move |r| Arc::new(Self::$op(l.clone(), r.clone())))
        }))
    };
    ($op: ident, $op1: ident, $op2: ident, $op3: ident) => {
        Box::new(
            $op1.possible_subs()
                .zip($op2.possible_subs())
                .zip($op3.possible_subs())
                .map(|((l, r), r2)| Arc::new(Self::$op(l.clone(), r.clone(), r2.clone()))),
        )
    };
}

macro_rules! binop_folding {
    ($op:ident, $lhs:expr, $rhs:expr, $op_fn:expr) => {
        if let (Self::Const(c1), Self::Const(c2)) = ($lhs.as_ref(), $rhs.as_ref()) {
            if let (Constant::Int { value: v1, .. }, Constant::Int { value: v2, .. }) =
                (c1.as_ref(), c2.as_ref())
            {
                return Self::build_constant_int($op_fn(*v1, *v2) as u64);
            }
        }
    };
}

impl Expr {
    pub fn build_constant_int(value: u64) -> Self {
        Self::Const(ConstantRef::new(Constant::Int { bits: 64, value }))
    }

    pub fn keys(&self, keys: &mut HashSet<String>) {
        match self {
            Self::Any | Self::String(..) => {}
            Self::If(cond, then_expr, else_expr) => {
                cond.keys(keys);
                then_expr.keys(keys);
                else_expr.keys(keys);
            }
            Self::Not(expr) | Self::FNeg(expr) => expr.keys(keys),
            Self::Parameter(_) => {}
            Self::Alloc(name) => {
                keys.insert(name.to_string());
            }
            Self::Mem(expr) => expr.keys(keys),
            Self::Call(name, args) => {
                name.keys(keys);
                for arg in args {
                    arg.keys(keys);
                }
            }
            Self::Const(_) => {}
            Self::ICmp(_, lhs, rhs) | Self::FCmp(_, lhs, rhs) => {
                lhs.keys(keys);
                rhs.keys(keys);
            }
            Self::Add(lhs, rhs)
            | Self::Sub(lhs, rhs)
            | Self::Mul(lhs, rhs)
            | Self::UDiv(lhs, rhs)
            | Self::SDiv(lhs, rhs)
            | Self::URem(lhs, rhs)
            | Self::SRem(lhs, rhs)
            | Self::And(lhs, rhs)
            | Self::Or(lhs, rhs)
            | Self::Xor(lhs, rhs)
            | Self::Shl(lhs, rhs)
            | Self::LShr(lhs, rhs)
            | Self::AShr(lhs, rhs)
            | Self::FAdd(lhs, rhs)
            | Self::FSub(lhs, rhs)
            | Self::FMul(lhs, rhs)
            | Self::FDiv(lhs, rhs)
            | Self::FRem(lhs, rhs) => {
                lhs.keys(keys);
                rhs.keys(keys);
            }
        }
    }

    pub fn as_int(&self) -> Option<u64> {
        if let Self::Const(c) = self {
            if let Constant::Int { value, .. } = c.as_ref() {
                return Some(*value);
            }
        }
        None
    }

    pub fn is_zero(&self) -> bool {
        if let Some(v) = self.as_int() {
            return v == 0;
        }
        false
    }

    pub fn is_any(&self) -> bool {
        matches!(self, Self::Any)
    }

    /// Normalize the expression, e.g., sort the arguments of compare operations, constant folding.
    pub fn normalize(&self) -> Self {
        match self {
            Self::Any => Self::Any,
            Self::If(cond, then_expr, else_expr) => Self::If(
                Arc::new(cond.normalize()),
                Arc::new(then_expr.normalize()),
                Arc::new(else_expr.normalize()),
            ),
            Self::Not(expr) => Self::Not(Arc::new(expr.normalize())),
            Self::Parameter(index) => Self::Parameter(*index),
            Self::Alloc(name) => Self::Alloc(name.clone()),
            Self::Mem(expr) => Self::Mem(Arc::new(expr.normalize())),
            Self::Call(name, args) => {
                let new_args = args.iter().map(|arg| Arc::new(arg.normalize())).collect();
                Self::Call(name.clone(), new_args)
            }
            Self::Const(c) => Self::Const(c.clone()),
            Self::ICmp(pred, lhs, rhs) => match pred {
                // ne to eq
                IntPredicate::EQ | IntPredicate::NE => Self::ICmp(
                    IntPredicate::EQ,
                    Arc::new(lhs.normalize()),
                    Arc::new(rhs.normalize()),
                ),
                // > is the same
                IntPredicate::UGT | IntPredicate::SGT | IntPredicate::ULE | IntPredicate::SLE => {
                    Self::ICmp(
                        IntPredicate::UGT,
                        Arc::new(lhs.normalize()),
                        Arc::new(rhs.normalize()),
                    )
                }
                IntPredicate::ULT | IntPredicate::SLT | IntPredicate::UGE | IntPredicate::SGE => {
                    Self::ICmp(
                        IntPredicate::UGT,
                        Arc::new(lhs.normalize()),
                        Arc::new(rhs.normalize()),
                    )
                }
            },
            Self::FCmp(pred, lhs, rhs) => {
                Self::FCmp(*pred, Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Add(lhs, rhs) => {
                binop_folding!(Add, lhs, rhs, |l: u64, r| l.wrapping_add(r));
                Self::Add(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Sub(lhs, rhs) => {
                binop_folding!(Sub, lhs, rhs, |l: u64, r| l.wrapping_sub(r));
                Self::Sub(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Mul(lhs, rhs) => {
                binop_folding!(Mul, lhs, rhs, |l: u64, r| l.wrapping_mul(r));
                Self::Mul(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::UDiv(lhs, rhs) => {
                binop_folding!(UDiv, lhs, rhs, |l: u64, r| l.wrapping_div(r));
                Self::UDiv(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::SDiv(lhs, rhs) => {
                binop_folding!(SDiv, lhs, rhs, |l: u64, r| l.wrapping_div(r));
                Self::SDiv(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::URem(lhs, rhs) => {
                binop_folding!(URem, lhs, rhs, |l: u64, r| l.wrapping_rem(r));
                Self::URem(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::SRem(lhs, rhs) => {
                binop_folding!(SRem, lhs, rhs, |l: u64, r| l.wrapping_rem(r));
                Self::SRem(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::And(lhs, rhs) => {
                binop_folding!(And, lhs, rhs, |l: u64, r| l & r);
                Self::And(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Or(lhs, rhs) => {
                binop_folding!(Or, lhs, rhs, |l: u64, r| l | r);
                Self::Or(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Xor(lhs, rhs) => {
                binop_folding!(Xor, lhs, rhs, |l: u64, r| l ^ r);
                Self::Xor(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::Shl(lhs, rhs) => {
                binop_folding!(Shl, lhs, rhs, |l: u64, r| l << r);
                Self::Shl(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::LShr(lhs, rhs) => {
                binop_folding!(LShr, lhs, rhs, |l: u64, r| l >> r);
                Self::LShr(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::AShr(lhs, rhs) => {
                binop_folding!(AShr, lhs, rhs, |l: u64, r| (l as i64) >> r);
                Self::AShr(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FAdd(lhs, rhs) => {
                Self::FAdd(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FSub(lhs, rhs) => {
                Self::FSub(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FMul(lhs, rhs) => {
                Self::FMul(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FDiv(lhs, rhs) => {
                Self::FDiv(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FRem(lhs, rhs) => {
                Self::FRem(Arc::new(lhs.normalize()), Arc::new(rhs.normalize()))
            }
            Self::FNeg(expr) => Self::FNeg(Arc::new(expr.normalize())),
            Self::String(s) => Self::String(s.clone()),
        }
    }

    pub fn contain_parameter(&self) -> bool {
        match self {
            Self::Any | Self::String(..) => false,
            Self::If(cond, then_expr, else_expr) => {
                cond.contain_parameter()
                    || then_expr.contain_parameter()
                    || else_expr.contain_parameter()
            }
            Self::Not(expr) => expr.contain_parameter(),
            Self::Parameter(_) => true,
            Self::Alloc(_) => false,
            Self::Call(_, args) => args.iter().any(|arg| arg.contain_parameter()),
            Self::Const(_) => false,
            Self::Mem(expr) | Self::FNeg(expr) => expr.contain_parameter(),
            Self::ICmp(_, lhs, rhs) => lhs.contain_parameter() || rhs.contain_parameter(),
            Self::FCmp(_, lhs, rhs) => lhs.contain_parameter() || rhs.contain_parameter(),
            Self::Add(lhs, rhs)
            | Self::Sub(lhs, rhs)
            | Self::Mul(lhs, rhs)
            | Self::UDiv(lhs, rhs)
            | Self::SDiv(lhs, rhs)
            | Self::URem(lhs, rhs)
            | Self::SRem(lhs, rhs)
            | Self::And(lhs, rhs)
            | Self::Or(lhs, rhs)
            | Self::Xor(lhs, rhs)
            | Self::Shl(lhs, rhs)
            | Self::LShr(lhs, rhs)
            | Self::AShr(lhs, rhs)
            | Self::FAdd(lhs, rhs)
            | Self::FSub(lhs, rhs)
            | Self::FMul(lhs, rhs)
            | Self::FDiv(lhs, rhs)
            | Self::FRem(lhs, rhs) => lhs.contain_parameter() || rhs.contain_parameter(),
        }
    }

    pub fn contain_any(&self) -> bool {
        match self {
            Self::Any => true,
            Self::String(..) => false,
            Self::If(cond, then_expr, else_expr) => {
                cond.contain_any() || then_expr.contain_any() || else_expr.contain_any()
            }
            Self::Not(expr) => expr.contain_any(),
            Self::Parameter(_) => false,
            Self::Alloc(_) => false,
            Self::Call(_, args) => args.iter().any(|arg| arg.contain_any()),
            Self::Const(_) => false,
            Self::Mem(expr) => expr.contain_any(),
            Self::ICmp(_, lhs, rhs) => lhs.contain_any() || rhs.contain_any(),
            Self::FCmp(_, lhs, rhs) => lhs.contain_any() || rhs.contain_any(),
            Self::Add(lhs, rhs)
            | Self::Sub(lhs, rhs)
            | Self::Mul(lhs, rhs)
            | Self::UDiv(lhs, rhs)
            | Self::SDiv(lhs, rhs)
            | Self::URem(lhs, rhs)
            | Self::SRem(lhs, rhs)
            | Self::And(lhs, rhs)
            | Self::Or(lhs, rhs)
            | Self::Xor(lhs, rhs)
            | Self::Shl(lhs, rhs)
            | Self::LShr(lhs, rhs)
            | Self::AShr(lhs, rhs)
            | Self::FAdd(lhs, rhs)
            | Self::FSub(lhs, rhs)
            | Self::FMul(lhs, rhs)
            | Self::FDiv(lhs, rhs)
            | Self::FRem(lhs, rhs) => lhs.contain_any() || rhs.contain_any(),
            Self::FNeg(expr) => expr.contain_any(),
        }
    }

    #[inline]
    pub fn complexity(&self) -> i32 {
        match self {
            Self::Any | Self::String(_) => 0,
            Self::If(cond, then_expr, else_expr) => {
                1 + cond.complexity() + then_expr.complexity() + else_expr.complexity()
            }
            Self::Not(expr) => 1 + expr.complexity(),
            Self::Parameter(_) => 1,
            Self::Alloc(_) => 1,
            Self::Call(_, args) => 1 + args.iter().map(|arg| arg.complexity()).sum::<i32>(),
            Self::Const(_) => 1,
            Self::Mem(expr) => 1 + expr.complexity(),
            Self::ICmp(_, lhs, rhs)
            | Self::FCmp(_, lhs, rhs)
            | Self::Add(lhs, rhs)
            | Self::Sub(lhs, rhs)
            | Self::Mul(lhs, rhs)
            | Self::UDiv(lhs, rhs)
            | Self::SDiv(lhs, rhs)
            | Self::URem(lhs, rhs)
            | Self::SRem(lhs, rhs)
            | Self::And(lhs, rhs)
            | Self::Or(lhs, rhs)
            | Self::Xor(lhs, rhs)
            | Self::Shl(lhs, rhs)
            | Self::LShr(lhs, rhs)
            | Self::AShr(lhs, rhs)
            | Self::FAdd(lhs, rhs)
            | Self::FSub(lhs, rhs)
            | Self::FMul(lhs, rhs)
            | Self::FDiv(lhs, rhs)
            | Self::FRem(lhs, rhs) => 1 + lhs.complexity() + rhs.complexity(),
            Self::FNeg(expr) => 1 + expr.complexity(),
        }
    }

    pub fn similarity(&self, other: &Self) -> i32 {
        match (self, other) {
            (Self::Any, _) | (_, Self::Any) => 1,
            (Self::If(cond1, then1, else1), Self::If(cond2, then2, else2)) => {
                1 + cond1.similarity(cond2) + then1.similarity(then2) + else1.similarity(else2)
            }
            (Self::Not(expr1), Self::Not(expr2)) | (Self::FNeg(expr1), Self::FNeg(expr2)) => {
                1 + expr1.similarity(expr2)
            }
            (Self::Parameter(i1), Self::Parameter(i2)) => 1 + (i1 == i2) as i32,
            (Self::Alloc(name1), Self::Alloc(name2)) => 1 + (name1 == name2) as i32,
            (Self::Mem(expr1), Self::Mem(expr2)) => 1 + expr1.similarity(expr2),
            (Self::Call(name1, args1), Self::Call(name2, args2)) => {
                1 + (name1 == name2) as i32
                    + args1
                        .iter()
                        .zip(args2.iter())
                        .map(|(a1, a2)| a1.similarity(a2))
                        .sum::<i32>()
            }
            (Self::Const(c1), Self::Const(c2)) => 1 + (c1 == c2) as i32,
            (Self::Add(lhs1, rhs1), Self::Add(lhs2, rhs2))
            | (Self::Mul(lhs1, rhs1), Self::Mul(lhs2, rhs2))
            | (Self::UDiv(lhs1, rhs1), Self::UDiv(lhs2, rhs2))
            | (Self::SDiv(lhs1, rhs1), Self::SDiv(lhs2, rhs2))
            | (Self::URem(lhs1, rhs1), Self::URem(lhs2, rhs2))
            | (Self::SRem(lhs1, rhs1), Self::SRem(lhs2, rhs2))
            | (Self::And(lhs1, rhs1), Self::And(lhs2, rhs2))
            | (Self::Or(lhs1, rhs1), Self::Or(lhs2, rhs2))
            | (Self::Xor(lhs1, rhs1), Self::Xor(lhs2, rhs2))
            | (Self::Sub(lhs1, rhs1), Self::Sub(lhs2, rhs2))
            | (Self::Shl(lhs1, rhs1), Self::Shl(lhs2, rhs2))
            | (Self::LShr(lhs1, rhs1), Self::LShr(lhs2, rhs2))
            | (Self::AShr(lhs1, rhs1), Self::AShr(lhs2, rhs2))
            | (Self::FAdd(lhs1, rhs1), Self::FAdd(lhs2, rhs2))
            | (Self::FSub(lhs1, rhs1), Self::FSub(lhs2, rhs2))
            | (Self::FMul(lhs1, rhs1), Self::FMul(lhs2, rhs2))
            | (Self::FDiv(lhs1, rhs1), Self::FDiv(lhs2, rhs2))
            | (Self::FRem(lhs1, rhs1), Self::FRem(lhs2, rhs2)) => {
                1 + lhs1.similarity(lhs2) + rhs1.similarity(rhs2)
            }
            (Self::ICmp(pred1, lhs1, rhs1), Self::ICmp(pred2, lhs2, rhs2)) => {
                1 + (pred1 == pred2) as i32 + lhs1.similarity(lhs2) + rhs1.similarity(rhs2)
            }
            (Self::FCmp(pred1, lhs1, rhs1), Self::FCmp(pred2, lhs2, rhs2)) => {
                1 + (pred1 == pred2) as i32 + lhs1.similarity(lhs2) + rhs1.similarity(rhs2)
            }
            (Self::String(s1), Self::String(s2)) => 1 + (s1 == s2) as i32,
            _ => 0,
        }
    }

    pub fn possible_subs(&self) -> Box<dyn Iterator<Item = Arc<Self>>> {
        let mut add_any = true;
        let any_iter = iter::once(Arc::new(Self::Any));
        let result: Box<dyn Iterator<Item = Arc<Self>>> = match self.clone() {
            Self::Any => Box::new(iter::empty()),
            Self::Const(c) => {
                if let Constant::Int { .. } = c.as_ref() {
                    add_any = false;
                }
                Box::new(iter::once(Arc::new(self.clone())))
            }
            Self::Parameter(_) | Self::Alloc(_) => Box::new(iter::once(Arc::new(self.clone()))),
            Self::Mem(expr) => {
                let expr_iter = expr.possible_subs();
                Box::new(expr_iter.map(|e| Arc::new(Self::Mem(e.clone()))))
            }
            Self::Not(expr) => {
                let expr_iter = expr.possible_subs();
                Box::new(expr_iter.map(|e| Arc::new(Self::Not(e.clone()))))
            }
            Self::FNeg(expr) => {
                let expr_iter = expr.possible_subs();
                Box::new(expr_iter.map(|e| Arc::new(Self::FNeg(e.clone()))))
            }
            // only use name/any and every args/any once, do not call them recursively
            // at least call name cannot be replaced by any
            Self::Call(name, args) => {
                let mut subs = vec![Arc::new(Self::Call(
                    name.clone(),
                    vec![Arc::new(Self::Any); args.len()],
                ))];
                for i in 0..args.len() {
                    let mut new_args = args.clone();
                    new_args[i] = Arc::new(Self::Any);
                    subs.push(Arc::new(Self::Call(name.clone(), new_args.clone())));
                }
                add_any = false;
                Box::new(subs.into_iter())
            }
            Self::If(cond, then_expr, else_expr) => {
                op_possible_subs!(If, cond, then_expr, else_expr)
            }
            Self::ICmp(pred, lhs, rhs) => Box::new(lhs.possible_subs().flat_map(move |l| {
                rhs.possible_subs()
                    .map(move |r| Arc::new(Self::ICmp(pred, l.clone(), r.clone())))
            })),
            Self::FCmp(pred, lhs, rhs) => Box::new(rhs.possible_subs().flat_map(move |r| {
                lhs.possible_subs()
                    .map(move |l| Arc::new(Self::FCmp(pred, l.clone(), r.clone())))
            })),
            Self::Add(lhs, rhs) => op_possible_subs!(Add, lhs, rhs),
            Self::Sub(lhs, rhs) => op_possible_subs!(Sub, lhs, rhs),
            Self::Mul(lhs, rhs) => op_possible_subs!(Mul, lhs, rhs),
            Self::UDiv(lhs, rhs) => op_possible_subs!(UDiv, lhs, rhs),
            Self::SDiv(lhs, rhs) => op_possible_subs!(SDiv, lhs, rhs),
            Self::URem(lhs, rhs) => op_possible_subs!(URem, lhs, rhs),
            Self::SRem(lhs, rhs) => op_possible_subs!(SRem, lhs, rhs),
            Self::And(lhs, rhs) => op_possible_subs!(And, lhs, rhs),
            Self::Or(lhs, rhs) => op_possible_subs!(Or, lhs, rhs),
            Self::Xor(lhs, rhs) => op_possible_subs!(Xor, lhs, rhs),
            Self::Shl(lhs, rhs) => op_possible_subs!(Shl, lhs, rhs),
            Self::LShr(lhs, rhs) => op_possible_subs!(LShr, lhs, rhs),
            Self::AShr(lhs, rhs) => op_possible_subs!(AShr, lhs, rhs),
            Self::FAdd(lhs, rhs) => op_possible_subs!(FAdd, lhs, rhs),
            Self::FSub(lhs, rhs) => op_possible_subs!(FSub, lhs, rhs),
            Self::FMul(lhs, rhs) => op_possible_subs!(FMul, lhs, rhs),
            Self::FDiv(lhs, rhs) => op_possible_subs!(FDiv, lhs, rhs),
            Self::FRem(lhs, rhs) => op_possible_subs!(FRem, lhs, rhs),
            Self::String(_) => Box::new(iter::once(Arc::new(self.clone()))),
        };
        if self.contain_parameter() {
            result
        } else if add_any {
            Box::new(any_iter.chain(result))
        } else {
            result
        }
    }
}

impl std::hash::Hash for Expr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Any => state.write_u8(0),
            Self::If(..) => state.write_u8(0),
            Self::Not(..) => state.write_u8(1),
            Self::Parameter(..) => state.write_u8(2),
            Self::Alloc(..) => state.write_u8(3),
            Self::Mem(..) => state.write_u8(4),
            Self::Call(..) => state.write_u8(5),
            Self::Const(c) => match c.as_ref() {
                Constant::Int { .. } => {
                    state.write_u8(6);
                }
                Constant::Null(_) => state.write_u8(7),
                _ => state.write_u8(8),
            },
            Self::ICmp(..) => state.write_u8(9),
            Self::FCmp(..) => state.write_u8(10),
            Self::Add(..) => state.write_u8(11),
            Self::Sub(..) => state.write_u8(12),
            Self::Mul(..) => state.write_u8(13),
            Self::UDiv(..) => state.write_u8(14),
            Self::SDiv(..) => state.write_u8(15),
            Self::URem(..) => state.write_u8(16),
            Self::SRem(..) => state.write_u8(17),
            Self::And(..) => state.write_u8(18),
            Self::Or(..) => state.write_u8(19),
            Self::Xor(..) => state.write_u8(20),
            Self::Shl(..) => state.write_u8(21),
            Self::LShr(..) => state.write_u8(22),
            Self::AShr(..) => state.write_u8(23),
            Self::FAdd(..) => state.write_u8(24),
            Self::FSub(..) => state.write_u8(25),
            Self::FMul(..) => state.write_u8(26),
            Self::FDiv(..) => state.write_u8(27),
            Self::FRem(..) => state.write_u8(28),
            Self::FNeg(..) => state.write_u8(29),
            Self::String(..) => state.write_u8(30),
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "T"),
            Self::If(cond, then_expr, else_expr) => {
                write!(f, "if {} then {} else {}", cond, then_expr, else_expr)
            }
            Self::Parameter(index) => write!(f, "arg_{}", index),
            Self::Alloc(name) => write!(f, "alloc_{}", name),
            Self::Mem(expr) => write!(f, "mem({})", expr),
            Self::Call(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Self::Not(expr) => write!(f, "!{}", expr),
            Self::ICmp(pred, lhs, rhs) => write!(f, "{} {} {}", lhs, pred, rhs),
            Self::FCmp(pred, lhs, rhs) => write!(f, "{} {} {}", lhs, pred, rhs),
            Self::Const(c) => write!(f, "{}", c),
            Self::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            Self::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            Self::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
            Self::UDiv(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
            Self::SDiv(lhs, rhs) => write!(f, "({} /s {})", lhs, rhs),
            Self::URem(lhs, rhs) => write!(f, "({} % {})", lhs, rhs),
            Self::SRem(lhs, rhs) => write!(f, "({} %s {})", lhs, rhs),
            Self::And(lhs, rhs) => write!(f, "({} & {})", lhs, rhs),
            Self::Or(lhs, rhs) => write!(f, "({} | {})", lhs, rhs),
            Self::Xor(lhs, rhs) => write!(f, "({} ^ {})", lhs, rhs),
            Self::Shl(lhs, rhs) => write!(f, "({} << {})", lhs, rhs),
            Self::LShr(lhs, rhs) => write!(f, "({} >>u {})", lhs, rhs),
            Self::AShr(lhs, rhs) => write!(f, "({} >>s {})", lhs, rhs),
            Self::FAdd(lhs, rhs) => write!(f, "({} +f {})", lhs, rhs),
            Self::FSub(lhs, rhs) => write!(f, "({} -f {})", lhs, rhs),
            Self::FMul(lhs, rhs) => write!(f, "({} *f {})", lhs, rhs),
            Self::FDiv(lhs, rhs) => write!(f, "({} /f {})", lhs, rhs),
            Self::FRem(lhs, rhs) => write!(f, "({} %f {})", lhs, rhs),
            Self::FNeg(expr) => write!(f, "-{}", expr),
            Self::String(s) => {
                write!(f, "\"{s}\"")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none() {
        let a = Expr::Add(Arc::new(Expr::Any), Arc::new(Expr::Any));
        let b = Expr::Add(Arc::new(Expr::Any), Arc::new(Expr::Any));
        assert_eq!(a, b);
    }
}
