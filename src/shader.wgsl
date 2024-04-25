struct ShaderState {
//    pos:array<f32,3>, up:array<f32,3>, forward:array<f32,3>, left:array<f32,3>,
    pos: vec3f,
    up: vec3f,
    forward: vec3f,
    left: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc_up: f32,
    @location(1) ndc_left: f32,
}

@group(0)
@binding(0)
var<uniform> shader_state: ShaderState;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    let quad_ind = in_vertex_index + in_instance_index;
    let x = f32(i32(quad_ind & 2) - 1);
    let y = f32(i32(quad_ind << 1 & 2) - 1);
    return VertexOutput(vec4(x, y, 0, 1), y, -x);
}

const sphere1 = vec3f(0,2,0);
const sphere2 = vec3(100.0,100.0,100.0);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pos = shader_state.pos;
    let dir = normalize(shader_state.forward + in.ndc_up * shader_state.up + in.ndc_left * shader_state.left);
    var t = 0.0;
    var hit = false;
    for (var i = 0; i < 32; i = i + 1) {
        let cur_pos = pos + dir * t;
        let dist = min(length(cur_pos - sphere1) - 1 , length(cur_pos - sphere2) - 1);
        if dist < 1e-5 {
            hit = true;
            break;
        }
        if dist > 1e5 {
            break;
        }
        t += dist;
    }
    return vec4(f32(hit), 0, 0, 1);
}