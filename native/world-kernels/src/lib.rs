use pyo3::prelude::*;

mod biosphere_tissue;
mod contacts;
mod growth;
mod metabolism;
mod motor_runtime;
mod transport;

#[pymodule]
fn _world_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<contacts::ContactBatch>()?;
    module.add_class::<biosphere_tissue::BiosphereTissue>()?;
    module.add_class::<growth::GrowthKernel>()?;
    module.add_class::<metabolism::MetabolicCohort>()?;
    module.add_class::<motor_runtime::MotorRuntime>()?;
    module.add_class::<transport::TransportSolver>()?;
    Ok(())
}
