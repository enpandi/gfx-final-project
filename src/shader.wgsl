struct ShaderState {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc_xy: vec2f,
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
    return VertexOutput(vec4(x, y, 0, 1), vec2(x, y));
}

const sphere1 = vec3f(-1, 1, -4);
const sphere2 = vec3f(-1, -1, -4);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let ro = vec3(shader_state.pos_x, shader_state.pos_y, shader_state.pos_z);
    let rd = vec3(in.ndc_xy.x, in.ndc_xy.y, -1.0);
    var t = 0.0;
    var hit = false;
    for (var i = 0; i < 32; i = i + 1) {
        let cur_pos = ro + rd * t;
        let dist = min(length(cur_pos - sphere1) - 1 , length(cur_pos - sphere2) - 1);
        if dist < 1e-3 {
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