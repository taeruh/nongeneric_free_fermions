pub mod paulis;

use pyo3::{
    Bound, PyResult, Python,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};

pub struct Module<'py> {
    pub pymodule: Bound<'py, PyModule>,
    pub path: String,
}

impl<'py> Module<'py> {
    pub fn new(py: Python<'py>, name: &str, mut path: String) -> PyResult<Self> {
        path.push_str(format!(".{name}").as_str());
        Ok(Self {
            pymodule: PyModule::new(py, name)?,
            path,
        })
    }

    pub fn add_submodule(&self, py: Python<'_>, submodule: Self) -> PyResult<()> {
        self.pymodule.add_submodule(&submodule.pymodule)?;
        py.import("sys")?
            .getattr("modules")?
            .set_item(submodule.path, submodule.pymodule)?;
        Ok(())
    }
}

pub fn create_full_module(
    py: Python,
    module: Bound<'_, PyModule>,
    path: String,
) -> PyResult<()> {
    let module = Module { pymodule: module, path };
    paulis::add_this_module(py, &module)?;
    Ok(())
}
