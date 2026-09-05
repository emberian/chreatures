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
const B: usize = 3;
const INPUTS: usize = 351;
const OUTPUTS: usize = 387;
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
    metadata: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}
struct Engine {
    device: Device,
    n: usize,
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
    simd_rows: bool,
}
impl Engine {
    fn load(path: &Path, simd_rows: bool) -> Self {
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
        let z = vec![[0f32; 4]; n];
        let ones = vec![[1f32; 4]; n];
        Self {
            queue: d.new_command_queue(),
            rate: [buf(&d, &z), buf(&d, &z)],
            adapt: buf(&d, &z),
            support: buf(&d, &ones),
            drive: zeros(&d, n),
            channels: zeros(&d, INPUTS),
            output: zeros(&d, OUTPUTS),
            partial: zeros(&d, n.div_ceil(256) * 3),
            k_in: pipeline("project_inputs"),
            k_rec: pipeline(if simd_rows {
                "csr_rate_simd"
            } else {
                "csr_rate"
            }),
            k_out: pipeline("project_readouts"),
            k_phys: pipeline("physiology_partials"),
            k_phys_final: pipeline("physiology_final"),
            device: d,
            n,
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
    fn step(&mut self, dt: f32, mask: u32, ch: &[f32]) -> (Vec<f32>, f64) {
        assert_eq!(ch.len(), INPUTS * B);
        let mut packed = vec![[0f32; 4]; INPUTS];
        for i in 0..INPUTS {
            packed[i][..B].copy_from_slice(&ch[i * B..i * B + B])
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                self.channels.contents() as *mut [f32; 4],
                INPUTS,
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
            Self::grid(e, &self.k_in, self.n);
            e.end_encoding()
        }
        for (final_step, pb, rin, rout) in [
            (false, &pb0, &self.rate[0], &self.rate[1]),
            (true, &pb1, &self.rate[1], &self.rate[0]),
        ] {
            let e = cb.new_compute_command_encoder();
            bind(
                e,
                &self.k_rec,
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
            if self.simd_rows {
                e.dispatch_threads(
                    MTLSize::new(self.n as u64 * 32, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            } else {
                Self::grid(e, &self.k_rec, self.n);
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
            Self::grid(e, &self.k_out, 384);
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
                MTLSize::new(self.n.div_ceil(256) as u64 * 256, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            e.end_encoding()
        }
        {
            let e = cb.new_compute_command_encoder();
            bind(e, &self.k_phys_final, &[&self.partial, &self.output]);
            e.set_buffer(8, Some(&pb1), 0);
            Self::grid(e, &self.k_phys_final, 1);
            e.end_encoding()
        }
        let t = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let raw = copy::<[f32; 4]>(&self.output, OUTPUTS);
        let mut out = Vec::with_capacity(OUTPUTS * B);
        for x in raw {
            out.extend_from_slice(&x[..B])
        }
        (out, ms)
    }
    fn reset(&mut self, mask: u32) {
        unsafe {
            let r = self.rate[0].contents() as *mut [f32; 4];
            let a = self.adapt.contents() as *mut [f32; 4];
            let s = self.support.contents() as *mut [f32; 4];
            for i in 0..self.n {
                for j in 0..B {
                    if mask & (1 << j) != 0 {
                        (*r.add(i))[j] = 0.;
                        (*a.add(i))[j] = 0.;
                        (*s.add(i))[j] = 1.;
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
            let v = copy::<[f32; 4]>(b, self.n);
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
            let v = read_vec::<[f32; 4]>(&mut f, self.n);
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut [f32; 4], self.n)
            }
        }
        String::from_utf8(m).unwrap()
    }
}
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("artifact path");
    let simd_rows = matches!(args.next().as_deref(), Some("simd"));
    let mut x = Engine::load(Path::new(&path), simd_rows);
    println!(
        "{}",
        serde_json::json!({"ok":true,"device":x.device.name(),"neurons":x.n,"inputs":INPUTS,"readouts":384,"kernel":if simd_rows{"simd"}else{"row"}})
    );
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let response = match serde_json::from_str::<Request>(&line.unwrap()) {
            Ok(Request::Step {
                dt,
                active_mask,
                channels,
            }) => {
                let (o, m) = x.step(dt, active_mask, &channels);
                Reply {
                    ok: true,
                    combined: Some(o),
                    gpu_ms: Some(m),
                    metadata: None,
                    error: None,
                }
            }
            Ok(Request::Reset { mask }) => {
                x.reset(mask);
                Reply {
                    ok: true,
                    combined: None,
                    gpu_ms: None,
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
                    metadata: None,
                    error: None,
                }
            }
            Ok(Request::Restore { path }) => Reply {
                ok: true,
                combined: None,
                gpu_ms: None,
                metadata: Some(x.restore(&path)),
                error: None,
            },
            Ok(Request::Metadata) => Reply {
                ok: true,
                combined: None,
                gpu_ms: None,
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
                metadata: None,
                error: Some(e.to_string()),
            },
        };
        println!("{}", serde_json::to_string(&response).unwrap());
    }
}
