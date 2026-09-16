// Per-sample resolve preserves antialiased silhouette coverage and opaque occlusion.
// Renderer substitutes regular textures and sample 0 when MSAA is disabled.
@group(0) @binding(0) var accumulation:texture_multisampled_2d<f32>;
@group(0) @binding(1) var revealage:texture_multisampled_2d<f32>;

@vertex fn vertex(@builtin(vertex_index) i:u32) -> @builtin(position) vec4<f32> {
    let xy=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    return vec4<f32>(xy[i],0.,1.);
}

@fragment fn composite(@builtin(position) pixel:vec4<f32>, @builtin(sample_index) sample:u32)
    -> @location(0) vec4<f32> {
    let xy=vec2<i32>(pixel.xy);
    let accum=textureLoad(accumulation,xy,i32(sample));
    let reveal=textureLoad(revealage,xy,i32(sample)).r;
    let color=accum.rgb/max(accum.a,0.00000001);
    return vec4<f32>(color,clamp(1.0-reveal,0.0,1.0));
}
