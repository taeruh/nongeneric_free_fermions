use pyo3::{Bound, PyResult, Python, types::PyModule};

#[pyo3::pymodule]
fn rs(py: Python, module: Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_wrapped_lib::create_full_module(py, module, "rust_backend.rs".to_string())
}
