use metal::{Buffer, CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::{env, fs::File, io::Read, mem::size_of, path::Path, time::Instant};

const SHADER: &str = include_str!("brain.metal");

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

fn read_vec<T: Copy + Default>(r: &mut File, len: usize) -> Vec<T> {
    let mut v = vec![T::default(); len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * size_of::<T>()) };
    r.read_exact(bytes).expect("truncated graph file");
    v
}

fn buffer<T: Copy>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr() as *const _,
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn copy_out<T: Copy>(b: &Buffer, len: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const T, len).to_vec() }
}

fn dispatch(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    bufs: &[&Buffer],
    p: Params,
) -> f64 {
    let pb = queue.device().new_buffer_with_data(
        &p as *const _ as *const _,
        size_of::<Params>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(8, Some(&pb), 0);
    let w = pipeline.thread_execution_width();
    enc.dispatch_threads(MTLSize::new(p.n as u64, 1, 1), MTLSize::new(w, 1, 1));
    enc.end_encoding();
    let start = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    start.elapsed().as_secs_f64() * 1000.0
}

fn max_delta(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |j| (x[j] - y[j]).abs()))
        .fold(0.0, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn cpu_step(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    rate: &mut Vec<[f32; 4]>,
    adapt: &mut [[f32; 4]],
    support: &mut [[f32; 4]],
    drive: &[[f32; 4]],
    dt: f32,
) {
    let alpha = (dt / 2.0 / 0.16).min(1.0);
    let mut out = rate.clone();
    for sub in 0..2 {
        for row in 0..rate.len() {
            let mut rec = [0f32; 4];
            for e in indptr[row] as usize..indptr[row + 1] as usize {
                let s = rate[indices[e] as usize];
                for j in 0..3 {
                    rec[j] += weights[e] * s[j];
                }
            }
            for j in 0..3 {
                let t = (0.005 + drive[row][j] + 0.92 * rec[j] - 0.1 * adapt[row][j])
                    .tanh()
                    .max(0.0);
                out[row][j] = rate[row][j] + alpha * (t * support[row][j] - rate[row][j]);
            }
        }
        std::mem::swap(rate, &mut out);
        if sub == 1 {
            for i in 0..rate.len() {
                for j in 0..3 {
                    adapt[i][j] += dt / 5.0 * (rate[i][j] - adapt[i][j]);
                    support[i][j] = (support[i][j]
                        + dt * (0.024 * (1.0 - support[i][j]) - 0.003 * rate[i][j]))
                        .clamp(0.65, 1.0);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: metal-brain GRAPH.bin [iterations]");
        std::process::exit(2);
    }
    let iterations: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(20);
    let mut f = File::open(Path::new(&args[1])).expect("open graph");
    let mut h = [0u32; 6];
    f.read_exact(unsafe { std::slice::from_raw_parts_mut(h.as_mut_ptr() as *mut u8, 24) })
        .unwrap();
    assert_eq!(h[0], 0x4d424732, "bad graph magic");
    let n = h[1] as usize;
    let e = h[2] as usize;
    let indptr = read_vec::<u32>(&mut f, n + 1);
    let indices = read_vec::<u32>(&mut f, e);
    let weights = read_vec::<f32>(&mut f, e);
    let mut seed = 7301u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32) / (1u32 << 24) as f32
    };
    let initial: Vec<[f32; 4]> = (0..n)
        .map(|_| [rand() * 0.02, rand() * 0.02, rand() * 0.02, 0.0])
        .collect();
    let adaptations = vec![[0.0; 4]; n];
    let supports = vec![[1.0; 4]; n];
    let drive: Vec<[f32; 4]> = (0..n)
        .map(|_| [rand() * 0.03, rand() * 0.03, rand() * 0.03, 0.0])
        .collect();
    let device = Device::system_default().expect("Metal device");
    let lib = device
        .new_library_with_source(SHADER, &CompileOptions::new())
        .expect("compile Metal");
    let fun = lib.get_function("csr_rate", None).unwrap();
    let pipe = device
        .new_compute_pipeline_state_with_function(&fun)
        .unwrap();
    let q = device.new_command_queue();
    let rp = buffer(&device, &indptr);
    let ci = buffer(&device, &indices);
    let wt = buffer(&device, &weights);
    let r0 = buffer(&device, &initial);
    let r1 = buffer(&device, &initial);
    let ad = buffer(&device, &adaptations);
    let su = buffer(&device, &supports);
    let dr = buffer(&device, &drive);
    let all = [&rp, &ci, &wt, &r0, &r1, &ad, &su, &dr];
    let dt = 0.05;
    let alpha = dt / 2.0 / 0.16;
    let one = |fin| Params {
        n: n as u32,
        alpha,
        gain: 0.92,
        dt,
        recovery: 0.024,
        final_step: fin,
        active_mask: 7,
    };
    dispatch(&q, &pipe, &all, one(0));
    let all2 = [&rp, &ci, &wt, &r1, &r0, &ad, &su, &dr];
    dispatch(&q, &pipe, &all2, one(1));
    let gpu = copy_out::<[f32; 4]>(&r0, n);
    let mut cr = initial.clone();
    let mut ca = adaptations.clone();
    let mut cs = supports.clone();
    cpu_step(
        &indptr, &indices, &weights, &mut cr, &mut ca, &mut cs, &drive, dt,
    );
    let parity = max_delta(&gpu, &cr);
    let snap_r = copy_out::<[f32; 4]>(&r0, n);
    let snap_a = copy_out::<[f32; 4]>(&ad, n);
    let snap_s = copy_out::<[f32; 4]>(&su, n);
    dispatch(&q, &pipe, &all, one(0));
    dispatch(&q, &pipe, &all2, one(1));
    let replay_expected = copy_out::<[f32; 4]>(&r0, n);
    unsafe {
        std::ptr::copy_nonoverlapping(snap_r.as_ptr(), r0.contents() as *mut [f32; 4], n);
        std::ptr::copy_nonoverlapping(snap_a.as_ptr(), ad.contents() as *mut [f32; 4], n);
        std::ptr::copy_nonoverlapping(snap_s.as_ptr(), su.contents() as *mut [f32; 4], n);
    }
    dispatch(&q, &pipe, &all, one(0));
    dispatch(&q, &pipe, &all2, one(1));
    let replay = max_delta(&replay_expected, &copy_out(&r0, n));
    let mut samples = Vec::new();
    for _ in 0..iterations {
        let t = Instant::now();
        dispatch(&q, &pipe, &all, one(0));
        dispatch(&q, &pipe, &all2, one(1));
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    println!("{{\"device\":\"{}\",\"neurons\":{},\"edges\":{},\"batch_size\":3,\"iterations\":{},\"median_step_ms\":{:.6},\"min_step_ms\":{:.6},\"cpu_parity_max_abs\":{:.9},\"snapshot_replay_max_abs\":{:.9}}}",device.name(),n,e,iterations,samples[samples.len()/2],samples[0],parity,replay);
}
