// direction vectors ("ideal points") square to 0
// so they have 0 norm and cannot be normalized via point_signum
// see section 7.1.1 "the ideal norm": https://arxiv.org/abs/2002.04509
const OPTIMIZE_IDEAL_POINTS = true;
fn ideal_point_squared_magnitude(dir: Point) -> f32 {
    if OPTIMIZE_IDEAL_POINTS { return dot(dir.g0, dir.g0); }
    return line_squared_magnitude(point_point_regressive_product(dir, ORIGIN));
}
fn ideal_point_magnitude(dir: Point) -> f32 {
    if OPTIMIZE_IDEAL_POINTS { return length(dir.g0); }
    return sqrt(ideal_point_squared_magnitude(dir));
}
fn ideal_point_signum(dir: Point) -> Point {
    if OPTIMIZE_IDEAL_POINTS { return Point(normalize(dir.g0), 0.0); }
    return point_scalar_geometric_product(dir, 1.0 / ideal_point_magnitude(dir));
}

struct Camera {
    motion: Motor,
    up: Point,
    forward: Point,
    left: Point,
}

struct PointLight {
    color: vec3f,
    position: Point,
}

struct Material {
    ambient: vec3f,
    diffuse: vec3f,
}
struct Shape {
    params: vecN,
}
// https://iquilezles.org/articles/distfunctions/
struct Body {
    motion: Motor,
    shape: Shape,
    material: Material,
}

struct State {
    camera: Camera,
    global_light_color: vec3f,
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
    let quad_ind = in_vertex_index + in_instance_index;
    let x = f32(i32(quad_ind & 2) - 1);
    let y = f32(i32(quad_ind << 1 & 2) - 1);
    return VertexOutput(vec4(x, y, 0, 1), y, -x);
}

const MARCH_MAX_ITERS = 32;
const MARCH_EPSILON = 1e-4;
const MARCH_INF = 1e4;
struct MarchResult {
    t: f32,
    idx: i32,
}
fn march(pos: Point, dir: Point) -> MarchResult {
    var t = 0.0;
    for (var iter = 0; iter < MARCH_MAX_ITERS; iter++) {
//        if iter>0{return MarchResult(1,-3);}
        var best_idx = -1;
        var best_sdf = MARCH_INF;
        let cur_pos = point_point_add(pos, point_scalar_mul(dir, t));
//        if length(cur_pos.g0)<1e-5{return MarchResult(0.0,-2);}
//        else {return MarchResult(0.0,-3);}
        for (var idx = 0; idx < MAX_BODIES; idx++) {
            if state.bodies[idx].shape.params[0] == 0.0 { break; }
            var sdf = best_sdf;
            let local_cur_pos = motor_point_transformation(
                motor_reversal(state.bodies[idx].motion),
                cur_pos);
            // https://iquilezles.org/articles/distfunctions/
            if state.bodies[idx].shape.params[1] == 0.0 {
                // sphere
                let radius = state.bodies[idx].shape.params[0];
//                if true{return MarchResult(radius,-2);}
//                let center = motor_point_transformation(
//                    state.bodies[idx].motion,
//                    ORIGIN,
//                );
//                let diff = point_point_sub(cur_pos, center);
//                sdf = ideal_point_magnitude(diff) - radius;
//                sdf = line_magnitude(point_point_regressive_product(ORIGIN, local)) - radius;
                if OPTIMIZE_IDEAL_POINTS {
                    sdf = length(local_cur_pos.g0) - radius;
                } else {
                    sdf = ideal_point_magnitude(point_point_sub(local_cur_pos, ORIGIN)) - radius;
                }
//                return MarchResult(abs(point_magnitude(local)-ideal_point_magnitude(diff)), -2);
            } else {
                // box
                /*
                let q = abs(cur_pos) - state.boxes[idx].half_widths;
                let sdf = length(max(q, vecN())) + min(0.0, maxN(q));
                */
                var half_widths = state.bodies[idx].shape.params;
                let q = abs(local_cur_pos.g0) - half_widths;
                var inner_sdf = 0.0;
                for (var i = 0; i < N; i++) {
                    inner_sdf = max(inner_sdf, q[i]);
                }
                inner_sdf = min(inner_sdf, 0.0);
                sdf = length(max(q, vecN())) + inner_sdf;
            }
            if sdf < best_sdf {
                best_sdf = sdf;
                best_idx = idx;
            }
        }
        t += best_sdf;
//        if iter>1{return MarchResult(0.0,-2);}
//        if iter==0&&best_sdf<MARCH_EPSILON&&best_idx==0{return MarchResult(0.0,-2);}
        if best_idx != -1 && best_sdf < MARCH_EPSILON {
            return MarchResult(t, best_idx);
        }
        if t > MARCH_INF { break; }
    }
    return MarchResult(t, -1);
}
fn maxN(x: vecN) -> f32 {
    var res = x[0];
    for (var i = 1; i < N; i++) {
        res = max(res, x[i]);
    }
    return res;
}
/*
fn march_boxes(pos: vecN, dir: vecN) -> MarchResult {
    var t = 0.0;
    for (var iter = 0; iter < MARCH_MAX_ITERS; iter++) {
        var best_idx = -1;
        var best_sdf = MARCH_INF;
        let cur_pos = pos + dir * t;
        for (var idx = 0; idx < MAX_SPHERES; idx++) {
            if state.boxes[idx].half_widths.x == 0.0 { break; }
            // https://iquilezles.org/articles/distfunctions/
            let q = abs(cur_pos) - state.boxes[idx].half_widths;
            let sdf = length(max(q, vecN())) + min(0.0, maxN(q));
            if sdf < best_sdf {
                best_sdf = sdf;
                best_idx = idx;
            }
        }
        t += best_sdf;
        if best_idx != -1 && best_sdf < MARCH_EPSILON {
            return MarchResult(t, best_idx);
        }
        if t > MARCH_INF { break; }
    }
    return MarchResult(t, -1);
}
*/

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pos = motor_point_transformation(state.camera.motion, ORIGIN);
    let dir = ideal_point_signum(
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
//    let raw_dir = motor_point_transformation(
//        state.camera.motion,
//        point_point_add(
//            state.camera.forward,
//            point_point_add(
//                point_scalar_mul(state.camera.up, in.ndc_up),
//                point_scalar_mul(state.camera.left, in.ndc_left)
//            ),
//        ),
//    );
//    if true{return vec4(dir.g1,0,0,1.0);}
//    if true{return vec4(dir.g0,1.0);}
    let primary = march(pos, dir);
    if primary.idx == -1 {
        return vec4f(0, 0, 0, 1);
    } else {
//        if primary.idx==-2{return vec4f(primary.t,0,0,1);}
//        if primary.idx==-3{return vec4f(0,primary.t,0,1);}
//        if primary.idx==-4{return vec4f(0,0,primary.t,1);}
//        if true {
//            // ignore shadow rays for now
//            return vec4(state.bodies[primary.idx].material.diffuse, 1.0);
//        }
        var color = state.global_light_color * state.bodies[primary.idx].material.ambient;
        let isect = point_point_add(pos, point_scalar_mul(dir, primary.t));
        for (var pl_idx = 0; pl_idx < MAX_POINT_LIGHTS; pl_idx++) {
            let dir = ideal_point_signum(point_point_sub(isect, state.point_lights[pl_idx].position));
            let shadow = march(state.point_lights[pl_idx].position, dir);
//            if dir.g1==0.0{return vec4f(1,0,0,1);}
//            if state.point_lights[pl_idx].position.g1==1.0{return vec4f(1,0,0,1);}
            if shadow.idx != primary.idx { continue; }
            let shadow_isect = point_point_add(
                state.point_lights[pl_idx].position,
                point_scalar_mul(dir, shadow.t)
            );
            let delta = point_point_sub(shadow_isect, isect);
            if ideal_point_squared_magnitude(delta) > MARCH_EPSILON { continue; }
//            let normal = normalize(isect - state.bodies[primary.idx].center);
//            color += state.point_lights[pl_idx].color * state.bodies[primary.idx].material.diffuse * max(0.0, -dot(dir, normal));
            color += state.point_lights[pl_idx].color * state.bodies[primary.idx].material.diffuse;
        }
        return vec4(color, 1);
    }
}