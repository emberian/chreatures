use pyo3::prelude::*;

mod actuation;
mod biosphere_tissue;
mod contacts;
mod environment;
mod growth;
mod metabolism;
mod motor_runtime;
mod sensorium;
mod transport;

#[pymodule]
fn _world_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<actuation::ActuationCohort>()?;
    module.add_class::<contacts::ContactBatch>()?;
    module.add_class::<biosphere_tissue::BiosphereTissue>()?;
    module.add_class::<environment::LightEnvironment>()?;
    module.add_class::<growth::GrowthKernel>()?;
    module.add_class::<metabolism::MetabolicCohort>()?;
    module.add_class::<motor_runtime::MotorRuntime>()?;
    module.add_class::<transport::TransportSolver>()?;
    module.add_function(wrap_pyfunction!(sensorium::transduce_retina, module)?)?;
    Ok(())
}
