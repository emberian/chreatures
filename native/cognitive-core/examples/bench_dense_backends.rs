// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare the cognitive core's generic and macOS dense matrix backends.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this backend comparison requires macOS Accelerate");
}

#[cfg(target_os = "macos")]
mod macos {
    use std::{ffi::c_int, hint::black_box, time::Instant};

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

    #[derive(Clone, Copy)]
    enum Backend {
        Matrixmultiply,
        Accelerate,
    }

    fn multiply(
        backend: Backend,
        input: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        out: usize,
        output: &mut [f32],
    ) {
        unsafe {
            match backend {
                Backend::Matrixmultiply => matrixmultiply::sgemm(
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
                ),
                Backend::Accelerate => cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_TRANS,
                    rows.try_into().unwrap(),
                    out.try_into().unwrap(),
                    cols.try_into().unwrap(),
                    1.0,
                    input.as_ptr(),
                    cols.try_into().unwrap(),
                    weight.as_ptr(),
                    cols.try_into().unwrap(),
                    0.0,
                    output.as_mut_ptr(),
                    out.try_into().unwrap(),
                ),
            }
        }
    }

    fn values(count: usize, state: &mut u64) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (((*state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn add_bias(output: &mut [f32], bias: &[f32]) {
        for row in output.chunks_exact_mut(bias.len()) {
            for (value, offset) in row.iter_mut().zip(bias) {
                *value += *offset;
            }
        }
    }

    fn compare(rows: usize, cols: usize, out: usize) {
        let mut state = 7;
        let input = values(rows * cols, &mut state);
        let weight = values(out * cols, &mut state);
        let bias = values(out, &mut state);
        let mut generic = vec![0.0; rows * out];
        let mut accelerate = generic.clone();
        multiply(
            Backend::Matrixmultiply,
            &input,
            rows,
            cols,
            &weight,
            out,
            &mut generic,
        );
        multiply(
            Backend::Accelerate,
            &input,
            rows,
            cols,
            &weight,
            out,
            &mut accelerate,
        );
        add_bias(&mut generic, &bias);
        add_bias(&mut accelerate, &bias);
        let max_abs = generic
            .iter()
            .zip(&accelerate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let rms = (generic
            .iter()
            .zip(&accelerate)
            .map(|(a, b)| f64::from(*a - *b).powi(2))
            .sum::<f64>()
            / (rows * out) as f64)
            .sqrt();
        println!("compare {rows}x{cols}->{out}: max_abs={max_abs:.8e} rms={rms:.8e}");
    }

    fn elapsed_micros(
        backend: Backend,
        repetitions: usize,
        input: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        out: usize,
        bias: &[f32],
        output: &mut [f32],
    ) -> f64 {
        for _ in 0..3 {
            multiply(backend, input, rows, cols, weight, out, output);
        }
        let started = Instant::now();
        for _ in 0..repetitions {
            multiply(
                backend,
                black_box(input),
                rows,
                cols,
                black_box(weight),
                out,
                black_box(output),
            );
            add_bias(output, bias);
        }
        started.elapsed().as_secs_f64() * 1e6 / repetitions as f64
    }

    fn benchmark(label: &str, rows: usize, cols: usize, out: usize) {
        let repetitions = (400_000_000usize / (rows * cols * out).max(1)).clamp(8, 1_000);
        let mut state = 11;
        let input = values(rows * cols, &mut state);
        let weight = values(out * cols, &mut state);
        let bias = values(out, &mut state);
        let mut output = vec![0.0; rows * out];
        let generic = elapsed_micros(
            Backend::Matrixmultiply,
            repetitions,
            &input,
            rows,
            cols,
            &weight,
            out,
            &bias,
            &mut output,
        );
        let accelerate = elapsed_micros(
            Backend::Accelerate,
            repetitions,
            &input,
            rows,
            cols,
            &weight,
            out,
            &bias,
            &mut output,
        );
        println!(
            "{label:18} {rows:6}x{cols:4}->{out:3} reps={repetitions:4} \
             matrix={generic:9.2}us accelerate={accelerate:9.2}us speedup={:.2}x",
            generic / accelerate
        );
    }

    pub fn run() {
        compare(48 * 4, 1_560, 256);
        for batch in [8, 48] {
            benchmark("latent128->256", batch, 128, 256);
            benchmark("latent256->256", batch, 256, 256);
            benchmark("predictor1560->256", batch * 4, 1_560, 256);
            benchmark("peripheral proj", batch, 768, 64);
            benchmark("foveal proj", batch, 2_304, 64);
            benchmark("peripheral conv1", batch * 8 * 16, 36, 16);
            benchmark("peripheral conv2", batch * 4 * 8, 144, 24);
            benchmark("foveal conv1", batch * 24 * 16, 36, 16);
            benchmark("foveal conv2", batch * 12 * 8, 144, 24);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}
