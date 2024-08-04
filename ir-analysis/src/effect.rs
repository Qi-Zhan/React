#![allow(clippy::rc_clone_in_vec_init)]
#![allow(clippy::derived_hash_with_manual_eq)]
use std::{collections::HashSet, fmt, sync::Arc};

use crate::expr::Expr;

#[derive(Debug, Clone, Eq, Hash)]
pub enum Effect {
    /// function call
    Call(Arc<Expr>, Vec<Arc<Expr>>),
    /// function return
    Return(Arc<Expr>),
    /// argument pointer modification
    ParameterWrite(Arc<Expr>, Arc<Expr>),
    /// global variable modification
    GlobalWrite(Arc<Expr>),
    /// condition
    Condition(Arc<Expr>),
}

impl PartialEq for Effect {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Call(l0, l1), Self::Call(r0, r1)) => {
                // CALL SITE is the same if the function name is the same
                let min_length = l1.len().min(r1.len());
                // something like abc.a, we need to cut the string after the first dot
                l0.to_string().split('.').next().unwrap()
                    == r0.to_string().split('.').next().unwrap()
                    && l1[..min_length] == r1[..min_length]
            }
            (Self::Return(l0), Self::Return(r0)) => l0 == r0,
            (Self::ParameterWrite(l0, l1), Self::ParameterWrite(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::GlobalWrite(l0), Self::GlobalWrite(r0)) => l0 == r0,
            (Self::Condition(l0), Self::Condition(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Effect {
    pub fn normalize(&self) -> Self {
        match self {
            Self::Call(name, args) => {
                let args = args.iter().map(|arg| Arc::new(arg.normalize())).collect();
                Self::Call(Arc::new(name.normalize()), args)
            }
            Self::Return(value) => Self::Return(Arc::new(value.normalize())),
            Self::ParameterWrite(index, value) => {
                Self::ParameterWrite(Arc::new(index.normalize()), Arc::new(value.normalize()))
            }
            Self::GlobalWrite(value) => Self::GlobalWrite(Arc::new(value.normalize())),
            Self::Condition(cond) => Self::Condition(Arc::new(cond.normalize())),
        }
    }

    pub fn keys(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        match self {
            Self::Call(name, args) => {
                name.keys(&mut set);
                for arg in args {
                    arg.keys(&mut set);
                }
            }
            Self::Return(value) | Self::GlobalWrite(value) | Self::Condition(value) => {
                value.keys(&mut set)
            }
            Self::ParameterWrite(index, value) => {
                index.keys(&mut set);
                value.keys(&mut set)
            }
        }
        set
    }

    pub fn complexity(&self) -> i32 {
        match self {
            Self::Call(name, args) => {
                // empty args if preferred
                if args.is_empty() {
                    name.complexity()
                } else {
                    1 + name.complexity() + args.iter().map(|arg| arg.complexity()).sum::<i32>()
                }
            }
            Self::Return(value) | Self::GlobalWrite(value) | Self::Condition(value) => {
                1 + value.complexity()
            }
            Self::ParameterWrite(index, value) => 1 + index.complexity() + value.complexity(),
        }
    }

    pub fn similarity(&self, other: &Self) -> i32 {
        match (self, other) {
            (Effect::Call(name, args), Effect::Call(name2, args2)) => {
                1 + name.similarity(name2)
                    + args
                        .iter()
                        .zip(args2)
                        .map(|(a, b)| a.similarity(b))
                        .sum::<i32>()
            }
            (Effect::GlobalWrite(value), Effect::GlobalWrite(value2))
            | (Effect::Condition(value), Effect::Condition(value2))
            | (Effect::Return(value), Effect::Return(value2)) => 1 + value.similarity(value2),
            (Effect::ParameterWrite(index, value), Effect::ParameterWrite(index2, value2)) => {
                1 + index.similarity(index2) + value.similarity(value2)
            }
            _ => 0,
        }
    }

    pub fn refine(self, other: &[Self]) -> Self {
        if cfg!(debug_assertions) {
            assert!(!other.contains(&self));
        }
        match self {
            Self::Return(value) => {
                for sub in value.possible_subs() {
                    let ret = Self::Return(sub.clone());
                    if !other.contains(&ret) {
                        return ret;
                    }
                }
                Self::Return(value)
            }
            Self::GlobalWrite(value) => {
                for sub in value.possible_subs() {
                    let ret = Self::GlobalWrite(sub.clone());
                    if !other.contains(&ret) {
                        return ret;
                    }
                }
                Self::GlobalWrite(value)
            }
            Self::Condition(cond) => {
                for sub in cond.possible_subs() {
                    let ret = Self::Condition(sub.clone());
                    if !other.contains(&ret) {
                        return ret;
                    }
                }
                Self::Condition(cond)
            }
            Self::ParameterWrite(index, value) => {
                for sub1 in index.possible_subs() {
                    for sub2 in value.possible_subs() {
                        let ret = Self::ParameterWrite(sub1.clone(), sub2);
                        if !other.contains(&ret) {
                            return ret;
                        }
                    }
                }
                Self::ParameterWrite(index, value)
            }
            Self::Call(name, args) => {
                let empty = vec![Arc::new(Expr::Any); args.len()];
                let empty_arg = Self::Call(name.clone(), empty.clone());
                if !other.contains(&empty_arg) {
                    return empty_arg;
                }
                for (i, arg) in args.iter().enumerate() {
                    let mut new_args = empty.clone();
                    new_args[i] = arg.clone();
                    let new_call = Self::Call(name.clone(), new_args);
                    if !other.contains(&new_call) {
                        return new_call;
                    }
                }
                Self::Call(name, args)
            }
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call(name, args) => {
                write!(f, "CALL {}(", name)?;
                for arg in args {
                    write!(f, "{}, ", arg)?;
                }
                write!(f, ")")
            }
            Self::Return(value) => write!(f, "RETURN {}", value),
            Self::ParameterWrite(index, value) => write!(f, "UPDATE {} = {}", index, value),
            Self::GlobalWrite(value) => write!(f, "GLOBAL = {}", value),
            Self::Condition(cond) => write!(f, "IF {}", cond),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_eq() {
        let call1 = Effect::Call(Arc::new(Expr::Any), vec![Arc::new(Expr::Parameter(4))]);
        let call2 = Effect::Call(Arc::new(Expr::Any), vec![Arc::new(Expr::Any)]);
        assert_eq!(call1, call2);

        let call1 = Effect::Condition(Arc::new(Expr::Any));
        let call2 = Effect::Condition(Arc::new(Expr::Parameter(4)));
        assert_eq!(call1, call2);
    }
}
