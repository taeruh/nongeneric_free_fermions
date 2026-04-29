use core_lib::fendley::Fendley as LibFendley;
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModuleMethods, PyType},
};

use crate::{
    Module, krylov_without_gram_schmidt::GeneratorsWithoutGramSchmidt, paulis::PauliSum,
};

#[pyo3::pyclass(subclass, from_py_object)]
#[derive(Clone)]
pub struct Fendley(pub LibFendley);

#[pyo3::pymethods]
impl Fendley {
    #[new]
    fn __new__(num_triangles: usize, alpha: f64, beta: f64, gamma: f64) -> Self {
        Self(LibFendley::new(num_triangles, alpha, beta, gamma))
    }

    fn extend_with_currents(
        &mut self,
        generator: &GeneratorsWithoutGramSchmidt,
        current_alpha: Vec<f64>,
    ) {
        let currents = &generator.0.eta_currents;
        println!("extending with {} currents", currents.len());
        self.0.extend_with_currents(currents, &current_alpha);
    }

    fn num_operators(&self) -> usize {
        self.0.ops.len()
    }
}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "fendley", parent_module.path.clone())?;
    module.pymodule.add_class::<Fendley>()?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
