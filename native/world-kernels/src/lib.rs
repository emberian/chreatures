use pyo3::prelude::*;

mod acoustics;
mod actuation;
mod biosphere_tissue;
mod contacts;
mod environment;
mod growth;
mod illumination;
mod metabolism;
mod motor_runtime;
mod sensorium;
mod transport;

#[pymodule]
fn _world_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<actuation::ActuationCohort>()?;
    module.add_class::<acoustics::AcousticEngine>()?;
    module.add_class::<contacts::ContactBatch>()?;
    module.add_class::<biosphere_tissue::BiosphereTissue>()?;
    module.add_class::<environment::LightEnvironment>()?;
    module.add_class::<growth::GrowthKernel>()?;
    module.add_class::<illumination::SolarCycle>()?;
    module.add_class::<metabolism::MetabolicCohort>()?;
    module.add_class::<motor_runtime::MotorRuntime>()?;
    module.add_class::<sensorium::RetinaCohort>()?;
    module.add_class::<transport::TransportSolver>()?;
    Ok(())
}
