use pyo3::{exceptions::PyValueError, prelude::*};

mod cns_adapter;
mod contextual_episodic;
mod developmental;
pub mod gam_law;
mod learned_sequence_control;
mod motor_suffix;
pub mod personal_consequences;
pub mod personal_goals;
pub mod population_response;
mod sequence_memory;

#[derive(Clone)]
pub(crate) struct Linear {
    pub(crate) out: usize,
    pub(crate) input: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}
#[derive(Clone)]
pub(crate) struct Gru {
    pub(crate) hidden: usize,
    pub(crate) input: usize,
    pub(crate) w_ih: Vec<f32>,
    pub(crate) w_hh: Vec<f32>,
    pub(crate) b_ih: Vec<f32>,
    pub(crate) b_hh: Vec<f32>,
}

#[cfg(target_os = "macos")]
fn dense_sgemm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    out: usize,
    output: &mut [f32],
) {
    use std::ffi::c_int;

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            transpose_a: c_int,
            transpose_b: c_int,
            rows: c_int,
            out: c_int,
            cols: c_int,
            alpha: f32,
            input: *const f32,
            input_stride: c_int,
            weight: *const f32,
            weight_stride: c_int,
            beta: f32,
            output: *mut f32,
            output_stride: c_int,
        );
    }

    let rows = c_int::try_from(rows).expect("dense GEMM row count exceeds c_int");
    let cols = c_int::try_from(cols).expect("dense GEMM column count exceeds c_int");
    let out = c_int::try_from(out).expect("dense GEMM output count exceeds c_int");
    unsafe {
        // input is [rows, cols], weight is stored [out, cols], and output is
        // [rows, out], all contiguous row-major. Transposing weight therefore
        // computes input * weight^T without copying either operand.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows,
            out,
            cols,
            1.0,
            input.as_ptr(),
            cols,
            weight.as_ptr(),
            cols,
            0.0,
            output.as_mut_ptr(),
            out,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn dense_sgemm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    out: usize,
    output: &mut [f32],
) {
    unsafe {
        matrixmultiply::sgemm(
            rows,
            cols,
            out,
            1.0,
            input.as_ptr(),
            cols as isize,
            1,
            weight.as_ptr(),
            1,
            cols as isize,
            0.0,
            output.as_mut_ptr(),
            out as isize,
            1,
        );
    }
}

fn gemm_bias_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    out: usize,
    weight: &[f32],
    bias: &[f32],
    output: &mut Vec<f32>,
) {
    assert_eq!(
        input.len(),
        rows.checked_mul(cols)
            .expect("dense GEMM input size overflow")
    );
    assert_eq!(
        weight.len(),
        out.checked_mul(cols)
            .expect("dense GEMM weight size overflow")
    );
    assert_eq!(bias.len(), out);
    let output_len = rows
        .checked_mul(out)
        .expect("dense GEMM output size overflow");
    output.resize(output_len, 0.0);
    if out == 0 {
        return;
    }
    if rows != 0 && cols != 0 {
        dense_sgemm(input, rows, cols, weight, out, output);
    } else {
        output.fill(0.0);
    }
    for row in output.chunks_exact_mut(out) {
        for (x, b) in row.iter_mut().zip(bias) {
            *x += *b;
        }
    }
}

pub(crate) fn gemm_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    layer: &Linear,
    output: &mut Vec<f32>,
) {
    assert_eq!(cols, layer.input);
    gemm_bias_into(
        input,
        rows,
        cols,
        layer.out,
        &layer.weight,
        &layer.bias,
        output,
    );
}
fn gemm_parts_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    out: usize,
    weight: &[f32],
    bias: &[f32],
    output: &mut Vec<f32>,
) {
    gemm_bias_into(input, rows, cols, out, weight, bias, output);
}
pub(crate) fn tanh_all(value: &mut [f32]) {
    for x in value {
        *x = x.tanh();
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Gru {
    pub(crate) fn step_into(
        &self,
        input: &[f32],
        state: &[f32],
        rows: usize,
        gx: &mut Vec<f32>,
        gh: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        gemm_parts_into(
            input,
            rows,
            self.input,
            3 * self.hidden,
            &self.w_ih,
            &self.b_ih,
            gx,
        );
        gemm_parts_into(
            state,
            rows,
            self.hidden,
            3 * self.hidden,
            &self.w_hh,
            &self.b_hh,
            gh,
        );
        out.resize(rows * self.hidden, 0.0);
        let h = self.hidden;
        for b in 0..rows {
            for j in 0..h {
                let r = sigmoid(gx[b * 3 * h + j] + gh[b * 3 * h + j]);
                let z = sigmoid(gx[b * 3 * h + h + j] + gh[b * 3 * h + h + j]);
                let n = (gx[b * 3 * h + 2 * h + j] + r * gh[b * 3 * h + 2 * h + j]).tanh();
                out[b * h + j] = (1.0 - z) * n + z * state[b * h + j];
            }
        }
    }
}

pub(crate) fn take(flat: &[f32], cursor: &mut usize, count: usize) -> PyResult<Vec<f32>> {
    if *cursor + count > flat.len() {
        return Err(PyValueError::new_err("packed weights are truncated"));
    }
    let v = flat[*cursor..*cursor + count].to_vec();
    *cursor += count;
    Ok(v)
}
pub(crate) fn linear(flat: &[f32], c: &mut usize, out: usize, input: usize) -> PyResult<Linear> {
    Ok(Linear {
        out,
        input,
        weight: take(flat, c, out * input)?,
        bias: take(flat, c, out)?,
    })
}
pub(crate) fn gru(flat: &[f32], c: &mut usize, h: usize, input: usize) -> PyResult<Gru> {
    Ok(Gru {
        hidden: h,
        input,
        w_ih: take(flat, c, 3 * h * input)?,
        w_hh: take(flat, c, 3 * h * h)?,
        b_ih: take(flat, c, 3 * h)?,
        b_hh: take(flat, c, 3 * h)?,
    })
}

#[pymodule]
fn _cognitive_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<cns_adapter::CnsAdapter>()?;
    module.add_class::<developmental::DevelopmentalResidentCohort>()?;
    Ok(())
}
