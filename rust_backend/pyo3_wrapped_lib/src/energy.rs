use core_lib::energy::{self as lib_energy, Float};
use pyo3::{PyResult, Python, types::PyModuleMethods};

use crate::Module;

#[pyo3::pyfunction]
pub fn calculate_mus(
    num_currents: usize,
    num_triangles: usize,
    effective_norm: Vec<f64>,
    eps: Vec<Float>,
) -> Vec<Vec<Float>> {
    lib_energy::calculate_mus(
        num_currents,
        num_triangles,
        effective_norm.as_slice(),
        eps.as_slice(),
    )
}

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "energy", parent_module.path.clone())?;
    module
        .pymodule
        .add_function(pyo3::wrap_pyfunction!(calculate_mus, &module.pymodule)?)?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
