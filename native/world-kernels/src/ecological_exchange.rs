//! Packed finite-material release proposals for ecological exchange.

use numpy::{
    ndarray::{Array1, Array3},
    IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mobile_release_candidates<'py>(
    py: Python<'py>,
    dt: f64,
    elapsed: PyReadonlyArray1<'_, f64>,
    credit: PyReadonlyArray1<'_, f64>,
    budgets: PyReadonlyArray1<'_, f64>,
    pools: PyReadonlyArray3<'_, f64>,
    rates: PyReadonlyArray3<'_, f64>,
    intervals: PyReadonlyArray1<'_, f64>,
    minimum_mass: PyReadonlyArray1<'_, f64>,
    maximum_mass: PyReadonlyArray1<'_, f64>,
    mass_weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let shape = pools.shape();
    if shape.len() != 3 || shape[1] != 2 || !(1..=32).contains(&shape[0]) {
        return Err(PyValueError::new_err("mobile pool cohort must be [B,2,K]"));
    }
    let (b, k) = (shape[0], shape[2]);
    if rates.shape() != shape
        || elapsed.shape() != [b]
        || credit.shape() != [b]
        || budgets.shape() != [b]
        || intervals.shape() != [b]
        || minimum_mass.shape() != [b]
        || maximum_mass.shape() != [b]
        || mass_weights.shape() != [k]
        || !dt.is_finite()
        || dt <= 0.0
        || dt > 1.0
    {
        return Err(PyValueError::new_err("mobile release dimensions differ"));
    }
    let elapsed = elapsed.as_slice()?;
    let credit = credit.as_slice()?;
    let budgets = budgets.as_slice()?;
    let pools = pools.as_slice()?;
    let rates = rates.as_slice()?;
    let intervals = intervals.as_slice()?;
    let minimum_mass = minimum_mass.as_slice()?;
    let maximum_mass = maximum_mass.as_slice()?;
    let weights = mass_weights.as_slice()?;
    if pools
        .iter()
        .chain(rates)
        .chain(elapsed)
        .chain(credit)
        .chain(budgets)
        .chain(intervals)
        .chain(minimum_mass)
        .chain(maximum_mass)
        .chain(weights)
        .any(|value| !value.is_finite() || *value < 0.0)
        || intervals.iter().any(|value| *value < 0.05)
        || minimum_mass
            .iter()
            .zip(maximum_mass)
            .any(|(low, high)| low > high || *high <= 0.0)
    {
        return Err(PyValueError::new_err(
            "mobile release values must be finite, nonnegative, and bounded",
        ));
    }

    let mut candidates = vec![0.0; b * 2 * k];
    let mut next_elapsed = vec![0.0; b];
    let mut next_credit = vec![0.0; b];
    let mut masses = vec![0.0; b];
    py.detach(|| {
        for resident in 0..b {
            let duration = if budgets[resident] > 0.0 {
                (elapsed[resident] + dt).min(intervals[resident])
            } else {
                elapsed[resident]
            };
            let funded = (credit[resident] + budgets[resident]).min(maximum_mass[resident]);
            next_elapsed[resident] = duration;
            next_credit[resident] = funded;
            let mut mass = 0.0;
            for compartment in 0..2 {
                for pool in 0..k {
                    let offset = (resident * 2 + compartment) * k + pool;
                    let amount = pools[offset] * (-(-duration * rates[offset]).exp_m1());
                    candidates[offset] = amount;
                    mass += amount * weights[pool];
                }
            }
            if mass < minimum_mass[resident] || funded < minimum_mass[resident] || mass <= 0.0 {
                candidates[resident * 2 * k..(resident + 1) * 2 * k].fill(0.0);
                continue;
            }
            let factor = (funded / mass).min(1.0);
            for value in &mut candidates[resident * 2 * k..(resident + 1) * 2 * k] {
                *value *= factor;
            }
            masses[resident] = mass * factor;
        }
    });
    Ok((
        Array3::from_shape_vec((b, 2, k), candidates)
            .unwrap()
            .into_pyarray(py),
        Array1::from_vec(next_elapsed).into_pyarray(py),
        Array1::from_vec(next_credit).into_pyarray(py),
        Array1::from_vec(masses).into_pyarray(py),
    ))
}
