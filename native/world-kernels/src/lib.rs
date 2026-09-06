use pyo3::prelude::*;

mod contacts;
mod growth;
mod metabolism;
mod transport;

#[pymodule]
fn _world_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<contacts::ContactBatch>()?;
    module.add_class::<growth::GrowthKernel>()?;
    module.add_class::<metabolism::MetabolicCohort>()?;
    module.add_class::<transport::TransportSolver>()?;
    Ok(())
}
