const MAX_SPHERES = 100;

struct Sphere {
	color: vec4f,
	position: vec3f,
	radius: f32,
}

struct ShaderState {
    pos: vec3f,
    up: vec3f,
    forward: vec3f,
    left: vec3f,
    spheres: array<Sphere, MAX_SPHERES>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc_up: f32,
    @location(1) ndc_left: f32,
}

@group(0) @binding(0) var<uniform> state: ShaderState;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pos = state.pos;
    let dir = normalize(state.forward + in.ndc_up * state.up + in.ndc_left * state.left);
    var t = 0.0;
    var hit = false;
    var hit_color = vec4f(0,0,0,1);
    for (var i = 0; i < 32; i = i + 1) {
        let cur_pos = pos + dir * t;
        var dist = 1e5;
        var march_color = vec4f(0,0,0,1);
        for (var s = 0; s < MAX_SPHERES; s = s + 1) {
            if state.spheres[s].radius == 0.0 { break; }
            let sdf = length(cur_pos - state.spheres[s].position) - state.spheres[s].radius;
            if sdf < dist {
                dist = sdf;
                march_color = state.spheres[s].color;
            }
        }
        if dist < 1e-5 {
            hit = true;
            hit_color = march_color;
            break;
        }
        if dist > 1e5 {
            break;
        }
        t += dist;
    }
    return hit_color;
}