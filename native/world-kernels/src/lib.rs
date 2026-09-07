use pyo3::prelude::*;

mod acoustics;
mod actuation;
mod biosphere_tissue;
mod contacts;
mod developmental_cues;
mod ecological_exchange;
mod environment;
mod growth;
mod habitat_family;
mod illumination;
mod lifecycle;
mod material_overlap;
mod metabolism;
mod population_trajectory;
mod regional_matter;
mod sensorium;
mod somatic;
mod transport;

#[pymodule]
fn _world_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<actuation::ActuationCohort>()?;
    module.add_class::<acoustics::AcousticEngine>()?;
    module.add_class::<contacts::ContactBatch>()?;
    module.add_function(wrap_pyfunction!(
        developmental_cues::developmental_surface_cues,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        ecological_exchange::mobile_release_candidates,
        module
    )?)?;
    module.add_class::<biosphere_tissue::BiosphereTissue>()?;
    module.add_class::<environment::LightEnvironment>()?;
    module.add_class::<growth::GrowthKernel>()?;
    module.add_class::<habitat_family::HabitatFamily>()?;
    module.add_class::<illumination::SolarCycle>()?;
    module.add_class::<lifecycle::LifecycleCohort>()?;
    module.add_function(wrap_pyfunction!(
        material_overlap::guaranteed_sphere_overlap_batch,
        module
    )?)?;
    module.add_class::<metabolism::MetabolicCohort>()?;
    module.add_class::<population_trajectory::PopulationTrajectory>()?;
    module.add_class::<regional_matter::RegionalMatter>()?;
    module.add_class::<somatic::SomaticCohort>()?;
    module.add_class::<sensorium::RetinaCohort>()?;
    module.add_class::<transport::TransportSolver>()?;
    Ok(())
}
