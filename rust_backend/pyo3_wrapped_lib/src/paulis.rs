use lib::paulis::{Pauli as LibPauli, PauliSum as LibPauliSum};
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModuleMethods, PyType},
};

use crate::Module;

#[pyo3::pyclass(subclass, from_py_object)]
#[derive(Clone)]
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

    #[pyo3(signature = (flip=false, with_phase=true))]
    fn to_string(&self, flip: Option<bool>, with_phase: Option<bool>) -> String {
        self.0
            .to_pretty_string_opt(flip.unwrap_or(false), with_phase.unwrap_or(true))
    }

    fn is_proportional_to(&self, other: &Self) -> bool {
        self.0.is_proportional_to(&other.0)
    }

    /// assuming self and other are proportional, returns the phase difference between
    /// self and other, more precisely: self = (i^phase_difference) * other
    fn phase_difference(&self, other: &Self) -> u8 {
        self.0.phase_difference(&other.0)
    }

    fn symplectic_inner_product(&self, other: &Self) -> bool {
        self.0.symplectic_inner_product(&other.0)
    }

    fn multiply_as_paulis(&self, other: &Self) -> Self {
        Self(self.0.multiply_as_paulis(&other.0))
    }

    fn __repr__(&self) -> String {
        self.0.representation()
    }
}

#[pyo3::pyclass(subclass)]
pub struct PauliSum(LibPauliSum);

#[pyo3::pymethods]
impl PauliSum {
    #[new]
    fn __new__(ops: Vec<(f64, Pauli)>) -> Self {
        Self(LibPauliSum::new(ops.into_iter().map(|(c, p)| (c, p.0)).collect()))
    }

    fn multiply(&self, other: &Self) -> Self {
        Self(self.0.multiply(&other.0))
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    fn to_py_list(&self) -> Vec<(f64, Pauli)> {
        self.0.0.iter().map(|(c, p)| (*c, Pauli(p.clone()))).collect()
    }

    fn single_hilbert_schmidt_inner_product(&self, other: &Pauli) -> f64 {
        self.0.single_hilbert_schmidt_inner_product(&other.0)
    }
}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "paulis", parent_module.path.clone())?;
    module.pymodule.add_class::<Pauli>()?;
    module.pymodule.add_class::<PauliSum>()?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
