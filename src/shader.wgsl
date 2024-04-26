const MAX_SPHERES = 16;

struct Sphere {
	color: vec4f,
	position: vec3f,
	radius: f32,
}

struct ShaderState {
    cam_pos: vec3f,
    cam_up: vec3f,
    cam_forward: vec3f,
    cam_left: vec3f,
    light_global_color: vec3f,
    light_source_pos: vec3f,
    light_source_color: vec3f,
    floor_x: f32,
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

fn shadow_ray(pos: vec3f) -> bool {
    var dist_to_light = length(state.light_source_pos - pos);
    var dir = 1.0 / dist_to_light * (state.light_source_pos - pos);
    var t = 0.1;
    for (var i = 0; i < 64; i = i + 1) {
        let cur_pos = pos + dir * t;
        var dist = 1e5;
        for (var s = 0; s < MAX_SPHERES; s = s + 1) {
            if state.spheres[s].radius == 0.0 { break; }
            let sdf = length(cur_pos - state.spheres[s].position) - state.spheres[s].radius;
            dist = min(dist, sdf);
        }
        t += dist;
        if dist < 1e-6 {
            return false;
        }
        if t > dist_to_light { return true; }
    }
    return t > dist_to_light;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let dir = normalize(state.cam_forward + in.ndc_up * state.cam_up + in.ndc_left * state.cam_left);
    var t = 0.0;
    var hit = false;
    var hit_color = vec4f(0,0,0,1);
    for (var i = 0; i < 32; i = i + 1) {
        let cur_pos = state.cam_pos + dir * t;
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
        t += dist;
        if dist < 1e-6 {
            hit = true;
            if shadow_ray(state.cam_pos + dir * t) {
                hit_color = march_color * vec4(state.light_source_color, 1);;
            }
            break;
        }
        if dist > 1e3 {
            break;
        }
    }
    if hit {
        return vec4(state.light_global_color, 1) + hit_color;
    } else {
        return vec4f(0);
    }
}