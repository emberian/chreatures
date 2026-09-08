// AGPL-3.0-or-later -- compact selected-resident V5 observer readback.
const N:u32=165122u;
struct Config{capacity:u32,active_mask:u32,reset_mask:u32,selected_resident:u32,dt:f32,neuron_count:u32,edge_count:u32,selected_field:u32};
struct State{rate:vec4<f32>,adapt:vec4<f32>,support:vec4<f32>,release:vec4<f32>,da:vec4<f32>,oa:vec4<f32>,ht:vec4<f32>};
@group(0)@binding(0)var<uniform>cfg:Config;
@group(0)@binding(1)var<storage,read>state:array<State>;
@group(0)@binding(2)var<storage,read_write>output:array<vec2<f32>>;
fn lane(v:vec4<f32>)->f32{switch cfg.selected_resident{case 1u:{return v.y;}case 2u:{return v.z;}case 3u:{return v.w;}default:{return v.x;}}}
@compute @workgroup_size(128)fn gather_observe(@builtin(global_invocation_id)i:vec3<u32>){let n=i.x;if(n>=N){return;}let s=state[n];var v=lane(s.rate);switch cfg.selected_field{case 1u:{v=lane(s.adapt);}case 2u:{v=lane(s.support);}case 3u:{v=lane(s.release);}case 4u:{v=lane(s.da);}case 5u:{v=lane(s.oa);}case 6u:{v=lane(s.ht);}default:{}}output[n]=vec2<f32>(lane(s.rate),v);}
