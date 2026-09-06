use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, BufRead, Read, Write},
    mem::size_of,
    path::Path,
    time::Instant,
};
const SHADER: &str = include_str!("../brain.metal");
const INPUTS: usize = 351;
const OUTPUTS: usize = 387;
const MAX_CAPACITY: usize = 32;
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    alpha: f32,
    gain: f32,
    dt: f32,
    recovery: f32,
    final_step: u32,
    active_mask: u32,
    capacity: u32,
    tiles: u32,
}
fn read_vec<T: Copy + Default>(r: &mut File, n: usize) -> Vec<T> {
    let mut v = vec![T::default(); n];
    r.read_exact(unsafe {
        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(&*v))
    })
    .unwrap();
    v
}
fn buf<T: Copy>(d: &Device, v: &[T]) -> Buffer {
    d.new_buffer_with_data(
        v.as_ptr() as *const _,
        std::mem::size_of_val(v) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn zeros(d: &Device, n: usize) -> Buffer {
    d.new_buffer(
        (n * size_of::<[f32; 4]>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn bind(enc: &metal::ComputeCommandEncoderRef, p: &ComputePipelineState, values: &[&Buffer]) {
    enc.set_compute_pipeline_state(p);
    for (i, b) in values.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0)
    }
}
fn copy<T: Copy>(b: &Buffer, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const T, n).to_vec() }
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Step {
        dt: f32,
        active_mask: u32,
        channels: Vec<f32>,
        #[serde(default)]
        selected_neuron_indices: Vec<u32>,
        #[serde(default)]
        target_neuron_indices: Vec<u32>,
        #[serde(default)]
        target_recurrent_correction: Vec<f32>,
    },
    Reset {
        mask: u32,
    },
    Snapshot {
        path: String,
        metadata: String,
    },
    Restore {
        path: String,
    },
    Metadata,
    Shutdown,
}
#[derive(Serialize)]
struct Reply {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    combined: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_rates: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}
struct Engine {
    device: Device,
    n: usize,
    capacity: usize,
    tiles: usize,
    queue: metal::CommandQueue,
    rec: [Buffer; 3],
    input: [Buffer; 3],
    readout: [Buffer; 3],
    rate: [Buffer; 2],
    adapt: Buffer,
    support: Buffer,
    drive: Buffer,
    channels: Buffer,
    output: Buffer,
    partial: Buffer,
    k_in: ComputePipelineState,
    k_rec: ComputePipelineState,
    k_out: ComputePipelineState,
    k_phys: ComputePipelineState,
    k_phys_final: ComputePipelineState,
    k_rec_corrected: ComputePipelineState,
    k_gather: ComputePipelineState,
    simd_rows: bool,
}
impl Engine {
    fn load(path: &Path, simd_rows: bool, capacity: usize) -> Self {
        assert!((1..=MAX_CAPACITY).contains(&capacity));
        let tiles = capacity.div_ceil(4);
        let mut f = File::open(path).unwrap();
        let h = read_vec::<u32>(&mut f, 6);
        assert_eq!(h[0], 0x4d424732);
        let (n, e, ie, re) = (h[1] as usize, h[2] as usize, h[3] as usize, h[4] as usize);
        let d = Device::system_default().unwrap();
        let rec = [
            buf(&d, &read_vec::<u32>(&mut f, n + 1)),
            buf(&d, &read_vec::<u32>(&mut f, e)),
            buf(&d, &read_vec::<f32>(&mut f, e)),
        ];
        let input = [
            buf(&d, &read_vec::<u32>(&mut f, n + 1)),
            buf(&d, &read_vec::<u32>(&mut f, ie)),
            buf(&d, &read_vec::<f32>(&mut f, ie)),
        ];
        let readout = [
            buf(&d, &read_vec::<u32>(&mut f, 385)),
            buf(&d, &read_vec::<u32>(&mut f, re)),
            buf(&d, &read_vec::<f32>(&mut f, re)),
        ];
        let lib = d
            .new_library_with_source(SHADER, &CompileOptions::new())
            .unwrap();
        let pipeline = |name| {
            d.new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
                .unwrap()
        };
        let z = vec![[0f32; 4]; n * tiles];
        let mut ones = vec![[1f32; 4]; n * tiles];
        if capacity % 4 != 0 {
            for row in 0..n {
                for lane in capacity % 4..4 {
                    ones[row * tiles + tiles - 1][lane] = 0.0;
                }
            }
        }
        Self {
            queue: d.new_command_queue(),
            rate: [buf(&d, &z), buf(&d, &z)],
            adapt: buf(&d, &z),
            support: buf(&d, &ones),
            drive: zeros(&d, n * tiles),
            channels: zeros(&d, INPUTS * tiles),
            output: zeros(&d, OUTPUTS * tiles),
            partial: zeros(&d, n.div_ceil(256) * tiles * 3),
            k_in: pipeline("project_inputs"),
            k_rec: pipeline(if simd_rows {
                "csr_rate_simd"
            } else {
                "csr_rate"
            }),
            k_out: pipeline("project_readouts"),
            k_phys: pipeline("physiology_partials"),
            k_phys_final: pipeline("physiology_final"),
            k_rec_corrected: pipeline(if simd_rows {
                "csr_rate_simd_corrected"
            } else {
                "csr_rate_corrected"
            }),
            k_gather: pipeline("gather_rates"),
            device: d,
            n,
            capacity,
            tiles,
            rec,
            input,
            readout,
            simd_rows,
        }
    }
    fn grid(enc: &metal::ComputeCommandEncoderRef, p: &ComputePipelineState, n: usize) {
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(p.thread_execution_width(), 1, 1),
        )
    }
    fn step(
        &mut self,
        dt: f32,
        mask: u32,
        ch: &[f32],
        selected_indices: &[u32],
        target_indices: &[u32],
        corrections: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
        if !dt.is_finite() || !(0.0..=0.2).contains(&dt) || dt == 0.0 {
            return Err("dt must be finite and in (0, 0.2]".into());
        }
        let valid_mask = if self.capacity == 32 {
            u32::MAX
        } else {
            (1_u32 << self.capacity) - 1
        };
        if mask & !valid_mask != 0 {
            return Err("active mask exceeds configured capacity".into());
        }
        if ch.len() != INPUTS * self.capacity {
            return Err(format!("channels must have shape [351,{}]", self.capacity));
        }
        if selected_indices.len() > 8192 || selected_indices.iter().any(|&x| x as usize >= self.n) {
            return Err("selected neuron indices are invalid".into());
        }
        let mut unique = selected_indices.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != selected_indices.len() {
            return Err("selected neuron indices must be unique".into());
        }
        let corrected = !target_indices.is_empty() || !corrections.is_empty();
        if corrected
            && (target_indices.len() != 2
                || corrections.len() != 2 * self.capacity
                || target_indices[0] == target_indices[1]
                || target_indices.iter().any(|&x| x as usize >= self.n))
        {
            return Err(format!(
                "target correction requires two unique indices and shape [2,{}]",
                self.capacity
            ));
        }
        let target_buffer = if corrected {
            Some(buf(&self.device, target_indices))
        } else {
            None
        };
        let correction_buffer = if corrected {
            let mut packed = vec![[0f32; 4]; 2 * self.tiles];
            for i in 0..2 {
                for resident in 0..self.capacity {
                    packed[i * self.tiles + resident / 4][resident % 4] =
                        corrections[i * self.capacity + resident];
                }
            }
            Some(buf(&self.device, &packed))
        } else {
            None
        };
        let selected_index_buffer = if selected_indices.is_empty() {
            None
        } else {
            Some(buf(&self.device, selected_indices))
        };
        let selected_buffer = if selected_indices.is_empty() {
            None
        } else {
            Some(zeros(&self.device, selected_indices.len() * self.tiles))
        };
        let mut packed = vec![[0f32; 4]; INPUTS * self.tiles];
        for i in 0..INPUTS {
            for resident in 0..self.capacity {
                packed[i * self.tiles + resident / 4][resident % 4] =
                    ch[i * self.capacity + resident];
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                self.channels.contents() as *mut [f32; 4],
                INPUTS * self.tiles,
            )
        }
        let p0 = Params {
            n: self.n as u32,
            alpha: (dt / 2.0 / 0.16).min(1.0),
            gain: 0.92,
            dt,
            recovery: 0.024,
            final_step: 0,
            active_mask: mask,
            capacity: self.capacity as u32,
            tiles: self.tiles as u32,
        };
        let p1 = Params {
            final_step: 1,
            ..p0
        };
        let pb0 = buf(&self.device, &[p0]);
        let pb1 = buf(&self.device, &[p1]);
        let cb = self.queue.new_command_buffer();
        {
            let e = cb.new_compute_command_encoder();
            bind(
                e,
                &self.k_in,
                &[
                    &self.input[0],
                    &self.input[1],
                    &self.input[2],
                    &self.channels,
                    &self.drive,
                ],
            );
            e.set_buffer(8, Some(&pb0), 0);
            Self::grid(e, &self.k_in, self.n * self.tiles);
            e.end_encoding()
        }
        for (final_step, pb, rin, rout) in [
            (false, &pb0, &self.rate[0], &self.rate[1]),
            (true, &pb1, &self.rate[1], &self.rate[0]),
        ] {
            let e = cb.new_compute_command_encoder();
            let recurrent_pipeline = if corrected {
                &self.k_rec_corrected
            } else {
                &self.k_rec
            };
            bind(
                e,
                recurrent_pipeline,
                &[
                    &self.rec[0],
                    &self.rec[1],
                    &self.rec[2],
                    rin,
                    rout,
                    &self.adapt,
                    &self.support,
                    &self.drive,
                ],
            );
            e.set_buffer(8, Some(pb), 0);
            if corrected {
                e.set_buffer(9, Some(target_buffer.as_ref().unwrap()), 0);
                e.set_buffer(10, Some(correction_buffer.as_ref().unwrap()), 0);
            }
            if self.simd_rows {
                e.dispatch_threads(
                    MTLSize::new((self.n * self.tiles) as u64 * 32, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            } else {
                Self::grid(e, recurrent_pipeline, self.n * self.tiles);
            }
            e.end_encoding();
            let _ = final_step;
        }
        {
            let e = cb.new_compute_command_encoder();
            bind(
                e,
                &self.k_out,
                &[
                    &self.readout[0],
                    &self.readout[1],
                    &self.readout[2],
                    &self.rate[0],
                    &self.output,
                ],
            );
            e.set_buffer(8, Some(&pb1), 0);
            Self::grid(e, &self.k_out, 384 * self.tiles);
            e.end_encoding()
        }
        if let (Some(indices), Some(selected)) = (&selected_index_buffer, &selected_buffer) {
            let e = cb.new_compute_command_encoder();
            bind(e, &self.k_gather, &[&self.rate[0], indices, selected]);
            e.set_buffer(8, Some(&pb1), 0);
            Self::grid(e, &self.k_gather, selected_indices.len() * self.tiles);
            e.end_encoding()
        }
        {
            let e = cb.new_compute_command_encoder();
            bind(
                e,
                &self.k_phys,
                &[&self.rate[0], &self.support, &self.partial],
            );
            e.set_buffer(8, Some(&pb1), 0);
            e.dispatch_threads(
                MTLSize::new((self.n.div_ceil(256) * self.tiles * 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            e.end_encoding()
        }
        {
            let e = cb.new_compute_command_encoder();
            bind(e, &self.k_phys_final, &[&self.partial, &self.output]);
            e.set_buffer(8, Some(&pb1), 0);
            Self::grid(e, &self.k_phys_final, self.tiles);
            e.end_encoding()
        }
        let t = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let raw = copy::<[f32; 4]>(&self.output, OUTPUTS * self.tiles);
        let mut out = Vec::with_capacity(OUTPUTS * self.capacity);
        for row in 0..OUTPUTS {
            for resident in 0..self.capacity {
                out.push(raw[row * self.tiles + resident / 4][resident % 4]);
            }
        }
        let mut selected_out = Vec::with_capacity(selected_indices.len() * self.capacity);
        if let Some(selected) = selected_buffer {
            let raw_selected = copy::<[f32; 4]>(&selected, selected_indices.len() * self.tiles);
            for row in 0..selected_indices.len() {
                for resident in 0..self.capacity {
                    selected_out.push(raw_selected[row * self.tiles + resident / 4][resident % 4]);
                }
            }
        }
        Ok((out, selected_out, ms))
    }
    fn reset(&mut self, mask: u32) {
        unsafe {
            let r = self.rate[0].contents() as *mut [f32; 4];
            let a = self.adapt.contents() as *mut [f32; 4];
            let s = self.support.contents() as *mut [f32; 4];
            for i in 0..self.n {
                for j in 0..self.capacity {
                    if mask & (1 << j) != 0 {
                        let index = i * self.tiles + j / 4;
                        (*r.add(index))[j % 4] = 0.;
                        (*a.add(index))[j % 4] = 0.;
                        (*s.add(index))[j % 4] = 1.;
                    }
                }
            }
        }
    }
    fn snapshot(&self, path: &str, metadata: &str) {
        let tmp = format!("{path}.tmp");
        let mut f = File::create(&tmp).unwrap();
        f.write_all(b"MBST1\0\0\0").unwrap();
        f.write_all(&(metadata.len() as u64).to_le_bytes()).unwrap();
        f.write_all(metadata.as_bytes()).unwrap();
        for b in [&self.rate[0], &self.adapt, &self.support] {
            let v = copy::<[f32; 4]>(b, self.n * self.tiles);
            f.write_all(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(&*v))
            })
            .unwrap()
        }
        f.sync_all().unwrap();
        std::fs::rename(tmp, path).unwrap()
    }
    fn restore(&mut self, path: &str) -> String {
        let mut f = File::open(path).unwrap();
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"MBST1\0\0\0");
        let mut l = [0u8; 8];
        f.read_exact(&mut l).unwrap();
        let mut m = vec![0u8; u64::from_le_bytes(l) as usize];
        f.read_exact(&mut m).unwrap();
        for b in [&self.rate[0], &self.adapt, &self.support] {
            let v = read_vec::<[f32; 4]>(&mut f, self.n * self.tiles);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    v.as_ptr(),
                    b.contents() as *mut [f32; 4],
                    self.n * self.tiles,
                )
            }
        }
        String::from_utf8(m).unwrap()
    }
}
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("artifact path");
    let simd_rows = matches!(args.next().as_deref(), Some("simd"));
    let capacity = args
        .next()
        .expect("capacity")
        .parse::<usize>()
        .expect("capacity must be an integer");
    assert!(
        (1..=MAX_CAPACITY).contains(&capacity),
        "capacity must be in 1..={MAX_CAPACITY}"
    );
    assert!(args.next().is_none(), "unexpected command-line argument");
    let mut x = Engine::load(Path::new(&path), simd_rows, capacity);
    println!(
        "{}",
        serde_json::json!({"ok":true,"device":x.device.name(),"neurons":x.n,"inputs":INPUTS,"readouts":384,"kernel":if simd_rows{"simd"}else{"row"},"capacity":x.capacity,"storage_tiles":x.tiles})
    );
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let response = match serde_json::from_str::<Request>(&line.unwrap()) {
            Ok(Request::Step {
                dt,
                active_mask,
                channels,
                selected_neuron_indices,
                target_neuron_indices,
                target_recurrent_correction,
            }) => {
                match x.step(
                    dt,
                    active_mask,
                    &channels,
                    &selected_neuron_indices,
                    &target_neuron_indices,
                    &target_recurrent_correction,
                ) {
                    Ok((o, s, m)) => Reply {
                        ok: true,
                        combined: Some(o),
                        gpu_ms: Some(m),
                        selected_rates: if selected_neuron_indices.is_empty() {
                            None
                        } else {
                            Some(s)
                        },
                        metadata: None,
                        error: None,
                    },
                    Err(e) => Reply {
                        ok: false,
                        combined: None,
                        gpu_ms: None,
                        selected_rates: None,
                        metadata: None,
                        error: Some(e),
                    },
                }
            }
            Ok(Request::Reset { mask }) => {
                x.reset(mask);
                Reply {
                    ok: true,
                    combined: None,
                    gpu_ms: None,
                    selected_rates: None,
                    metadata: None,
                    error: None,
                }
            }
            Ok(Request::Snapshot { path, metadata }) => {
                x.snapshot(&path, &metadata);
                Reply {
                    ok: true,
                    combined: None,
                    gpu_ms: None,
                    selected_rates: None,
                    metadata: None,
                    error: None,
                }
            }
            Ok(Request::Restore { path }) => Reply {
                ok: true,
                combined: None,
                gpu_ms: None,
                selected_rates: None,
                metadata: Some(x.restore(&path)),
                error: None,
            },
            Ok(Request::Metadata) => Reply {
                ok: true,
                combined: None,
                gpu_ms: None,
                selected_rates: None,
                metadata: Some(x.device.name().to_string()),
                error: None,
            },
            Ok(Request::Shutdown) => {
                println!(
                    "{}",
                    serde_json::to_string(&Reply {
                        ok: true,
                        combined: None,
                        gpu_ms: None,
                        selected_rates: None,
                        metadata: None,
                        error: None
                    })
                    .unwrap()
                );
                break;
            }
            Err(e) => Reply {
                ok: false,
                combined: None,
                gpu_ms: None,
                selected_rates: None,
                metadata: None,
                error: Some(e.to_string()),
            },
        };
        println!("{}", serde_json::to_string(&response).unwrap());
    }
}
