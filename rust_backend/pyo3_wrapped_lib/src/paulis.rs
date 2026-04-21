use lib::paulis::{Pauli as LibPauli, PauliSum as LibPauliSum};
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModuleMethods, PyType},
};

use crate::Module;

#[pyo3::pyclass(subclass)]
pub struct Pauli(LibPauli);

#[pyo3::pymethods]
impl Pauli {
    #[new]
    fn __new__(n: usize, u: Vec<bool>, l: Vec<bool>, phase: u8) -> Self {
        Self(LibPauli::new(n, u.as_slice(), l.as_slice(), phase))
    }

    #[classmethod]
    fn identity(_: &Bound<'_, PyType>, n: usize) -> Self {
        Self(LibPauli::identity(n))
    }

    #[classmethod]
    fn from_indices(
        _: &Bound<'_, PyType>,
        n: usize,
        u_trues: Vec<usize>,
        l_trues: Vec<usize>,
        phase: u8,
    ) -> Self {
        Self(LibPauli::from_indices(n, u_trues.as_slice(), l_trues.as_slice(), phase))
    }

    fn get_hermitian_phase(&self) -> u8 {
        self.0.get_hermitian_phase()
    }

    #[pyo3(signature = (flip=False, with_phase=True))]
    fn to_string(&self, flip: Option<bool>, with_phase: Option<bool>) -> String {
        self.0.to_pretty_string_opt(flip.unwrap_or(false), with_phase.unwrap_or(true))
    }

}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "paulis", parent_module.path.clone())?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
