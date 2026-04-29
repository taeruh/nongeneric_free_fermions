use core_lib::krylov_without_gram_schmidt::GeneratorsWithoutGramSchmidt as LibGeneratorsWithoutGramSchmidt;
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModuleMethods, PyType},
};

use crate::{Module, hamiltonian::Hamiltonian, paulis::Pauli};

#[pyo3::pyclass(subclass, from_py_object)]
#[derive(Clone)]
pub struct GeneratorsWithoutGramSchmidt(pub LibGeneratorsWithoutGramSchmidt);

#[pyo3::pymethods]
impl GeneratorsWithoutGramSchmidt {
    #[new]
    pub fn __new__(
        simplicial_mode: (f64, Pauli),
        hamiltonian: Hamiltonian,
        max_eta: usize,
        renormalise: bool,
        eta_normalisation_factor: f64,
    ) -> Self {
        Self(LibGeneratorsWithoutGramSchmidt::new(
            (simplicial_mode.0, simplicial_mode.1.0),
            hamiltonian.0,
            max_eta,
            renormalise,
            eta_normalisation_factor,
        ))
    }

    fn init_eta_currents(&mut self) {
        self.0.init_eta_currents();
    }
}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module =
        Module::new(py, "krylov_without_gram_schmidt", parent_module.path.clone())?;
    module.pymodule.add_class::<GeneratorsWithoutGramSchmidt>()?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
