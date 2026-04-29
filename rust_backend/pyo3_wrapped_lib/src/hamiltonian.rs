use core_lib::hamiltonian::Hamiltonian as LibHamiltonian;
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModuleMethods, PyType},
};

use crate::{Module, paulis::Pauli};

#[pyo3::pyclass(subclass, from_py_object)]
#[derive(Clone)]
pub struct Hamiltonian(pub LibHamiltonian);

#[pyo3::pymethods]
impl Hamiltonian {
    #[new]
    fn __new__(weights: Vec<f64>, ops: Vec<Pauli>) -> Self {
        Self(LibHamiltonian::new(weights, ops.into_iter().map(|p| p.0).collect()))
    }
}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "hamiltonian", parent_module.path.clone())?;
    module.pymodule.add_class::<Hamiltonian>()?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
