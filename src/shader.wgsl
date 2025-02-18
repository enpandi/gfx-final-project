const MARCH_MAX_ITERS = 128;
const MARCH_EPSILON = 1e-4;
const MARCH_INF = 1e5;

const SMOOTH_MARCH = false;
const SMOOTH_K = 0.5;
const SMOOTH_K_INV = 1.0 / SMOOTH_K;
// direction vectors ("ideal points") square to 0
// so they have 0 norm and cannot be normalized via point_signum
// see section 7.1.1 "the ideal norm": https://arxiv.org/abs/2002.04509
fn squared_length_point(dir: Point) -> f32 {
    return dot(dir.g0, dir.g0);
}
fn length_point(dir: Point) -> f32 {
    return length(dir.g0);
}
fn normalize_point(dir: Point) -> Point {
    return Point(normalize(dir.g0), dir.g1);
}
// just gonna use .g0 .g1 etc because WGSL type aliasing is hard (no methods)

struct Camera {
    motion: Motor,
    up: Point,
    forward: Point,
    left: Point,
}

struct PointLight {
    position: Point,
    color: vec3f,
}

struct Shape {
    params: vecN,
}
// https://iquilezles.org/articles/distfunctions/
struct Body {
    motion: Motor,
    shape: Shape,
    material: vec3f
}

struct State {
    camera: Camera,
    global_light_color: vec3f,
    floor_color: vec3f,
    point_lights: array<PointLight, MAX_POINT_LIGHTS>,
    bodies: array<Body, MAX_BODIES>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc_up: f32,
    @location(1) ndc_left: f32,
}

@group(0) @binding(0) var<uniform> state: State;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    let quad_ind = in_vertex_index;
    let x = f32(i32(quad_ind << 1 & 4) - 1);
    let y = f32(i32(quad_ind << 2 & 4) - 1);
    return VertexOutput(vec4(x, y, 0, 1), y, -x);
}

struct MarchResult {
    t: f32,
    idx: i32, // non-negatives are body indices. -1 is the floor. everything else is the void.
}
fn march(pos: Point, dir: Point) -> MarchResult {
    var t = MARCH_EPSILON;
    var best_idx = -2;
    for (var iter = 0; iter < MARCH_MAX_ITERS; iter++) {
        var all_sdf = MARCH_INF;
        // exponential smoothmin from https://iquilezles.org/articles/smin/
        var bodies_sdf = 0.0;
        let cur_pos = point_point_add(pos, point_scalar_mul(dir, t));
        for (var idx = 0; idx < MAX_BODIES; idx++) {
            if state.bodies[idx].shape.params[0] == 0.0 { break; }
            let local_cur_pos = motor_point_transformation(motor_reversal(state.bodies[idx].motion), cur_pos);
            var sdf: f32;
            if state.bodies[idx].shape.params[1] == 0.0 {
                // sphere
                let radius = state.bodies[idx].shape.params[0];
                sdf = length_point(point_point_sub(local_cur_pos, ORIGIN)) - radius;
            } else {
                // box from https://iquilezles.org/articles/distfunctions/
                var half_widths = state.bodies[idx].shape.params;
                let q = abs(local_cur_pos.g0) - half_widths;
                var max_q = q[0];
                for (var i = 1; i < N; i++) { max_q = max(max_q, q[i]); }
                sdf = length(max(q, vecN())) + min(max_q, 0.0);
            }
            if sdf < all_sdf { best_idx = idx; all_sdf = sdf; }
            if SMOOTH_MARCH { bodies_sdf += exp2(-sdf * SMOOTH_K_INV); }
        }
        if SMOOTH_MARCH { bodies_sdf = -SMOOTH_K * log2(bodies_sdf); all_sdf = min(all_sdf, bodies_sdf); }
        let floor_sdf = abs(cur_pos.g0[0]);
        if floor_sdf < all_sdf { best_idx = -1; all_sdf = floor_sdf; }
        t += all_sdf;
        if all_sdf < MARCH_EPSILON { return MarchResult(t, best_idx); }
        best_idx = -2;
        if t > MARCH_INF { break; }
    }
    return MarchResult(t, best_idx);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // perform the marching in world frame
    let pos = motor_point_transformation(state.camera.motion, ORIGIN);
    let dir = normalize_point(
        motor_point_transformation(
            state.camera.motion,
            point_point_add(
                state.camera.forward,
                point_point_add(
                    point_scalar_mul(state.camera.up, in.ndc_up),
                    point_scalar_mul(state.camera.left, in.ndc_left)
                ),
            ),
        )
    );
    let primary = march(pos, dir);
    if primary.idx < -1 { return vec4f(0, 0, 0, 1); }
    var light = state.global_light_color;
    let isect = point_point_add(pos, point_scalar_mul(dir, primary.t));
    for (var pl_idx = 0; pl_idx < MAX_POINT_LIGHTS; pl_idx++) {
        let bruh = state.point_lights[pl_idx].position;
        let dir = normalize_point(point_point_sub(isect, state.point_lights[pl_idx].position));
        let shadow = march(state.point_lights[pl_idx].position, dir);
        if shadow.idx != primary.idx { continue; }
        let shadow_isect = point_point_add(
            state.point_lights[pl_idx].position,
            point_scalar_mul(dir, shadow.t)
        );
        let delta = point_point_sub(shadow_isect, isect);
        if squared_length_point(delta) > MARCH_EPSILON { continue; }
        light += state.point_lights[pl_idx].color;
    }
    if primary.idx == -1 { return vec4(light * state.floor_color, 1); }
    else { return vec4(light * state.bodies[primary.idx].material, 1); }
}
