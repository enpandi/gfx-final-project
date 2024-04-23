struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc_xy: vec2f,
}

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4(in.ndc_xy.x, in.ndc_xy.y, 0.0, 1.0);
}
