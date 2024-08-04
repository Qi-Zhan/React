use std::collections::HashMap;

use llvm_ir::Module;

use super::kfunction::KFunction;

pub struct KModule {
    pub moudle: Module,
    pub kfuncs: HashMap<String, KFunction>,
}

impl KModule {
    pub fn from_bc_path(path: &str) -> Result<Self, String> {
        let module = Module::from_bc_path(path)?;
        let kfuncs = module
            .functions
            .iter()
            .map(|func| (func.name.clone(), KFunction::from(func)))
            .collect();
        Ok(Self {
            moudle: module,
            kfuncs,
        })
    }

    pub fn get_function(&self, name: &str) -> Option<&KFunction> {
        self.kfuncs.get(name)
    }
}
