mod control_flow_graph;

pub use control_flow_graph::{CFGNode, ControlFlowGraph, JumpKind};

use std::cell::{Ref, RefCell};
use std::collections::HashMap;

use crate::{KFunction, KModule};

/// Computes (and caches the results of) various analyses on a given `Module`
pub struct ModuleAnalysis<'m> {
    /// Reference to the `llvm-ir` `Module`
    kmodule: &'m KModule,
    /// Map from function name to the `FunctionAnalysis` for that function
    fn_analyses: HashMap<&'m str, FunctionAnalysis<'m>>,
}

impl<'m> ModuleAnalysis<'m> {
    /// Create a new `ModuleAnalysis` for the given `Module`.
    ///
    /// This method itself is cheap; individual analyses will be computed lazily
    /// on demand.
    pub fn new(module: &'m KModule) -> Self {
        Self {
            kmodule: module,
            fn_analyses: module
                .kfuncs
                .values()
                .map(|f| (f.name.as_str(), FunctionAnalysis::new(f)))
                .collect(),
        }
    }

    /// Get a reference to the `Module` which the `ModuleAnalysis` was created
    /// with.
    pub fn module(&self) -> &'m KModule {
        self.kmodule
    }

    /// Get the `FunctionAnalysis` for the function with the given name.
    ///
    /// Panics if no function of that name exists in the `Module` which the
    /// `ModuleAnalysis` was created with.
    pub fn fn_analysis<'s>(&'s self, func_name: &str) -> &'s FunctionAnalysis<'m> {
        self.fn_analyses
            .get(func_name)
            .unwrap_or_else(|| panic!("Function named {:?} not found in the Module", func_name))
    }
}

/// Computes (and caches the results of) various analyses on a given `Function`
pub struct FunctionAnalysis<'m> {
    /// Reference to the `llvm-ir` `Function`
    function: &'m KFunction,
    /// Control flow graph for the function
    control_flow_graph: SimpleCache<ControlFlowGraph<'m>>,
}

impl<'m> FunctionAnalysis<'m> {
    /// Create a new `FunctionAnalysis` for the given `Function`.
    ///
    /// This method itself is cheap; individual analyses will be computed lazily
    /// on demand.
    pub fn new(function: &'m KFunction) -> Self {
        Self {
            function,
            control_flow_graph: SimpleCache::new(),
        }
    }

    pub fn function(&self) -> &'m KFunction {
        self.function
    }

    /// Get the `ControlFlowGraph` for the function.
    pub fn control_flow_graph(&self) -> Ref<ControlFlowGraph<'m>> {
        self.control_flow_graph
            .get_or_insert_with(|| ControlFlowGraph::new(self.function))
    }
}

struct SimpleCache<T> {
    /// `None` if not computed yet
    data: RefCell<Option<T>>,
}

impl<T> SimpleCache<T> {
    fn new() -> Self {
        Self {
            data: RefCell::new(None),
        }
    }

    /// Get the cached value, or if no value is cached, compute the value using
    /// the given closure, then cache that result and return it
    fn get_or_insert_with(&self, f: impl FnOnce() -> T) -> Ref<T> {
        // borrow mutably only if it's empty. else don't even try to borrow mutably
        let need_mutable_borrow = self.data.borrow().is_none();
        if need_mutable_borrow {
            let _ = self.data.borrow_mut().replace(f());
        }
        // now, either way, it's populated, so we borrow immutably and return.
        // future users can also borrow immutably using this function (even
        // while this borrow is still outstanding), since it won't try to borrow
        // mutably in the future.
        Ref::map(self.data.borrow(), |o| {
            o.as_ref().expect("should be populated now")
        })
    }
}
