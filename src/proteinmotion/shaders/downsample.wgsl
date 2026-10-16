// Average each FACTOR x FACTOR block of a supersampled frame in linear light.
// Renderer._create_downsample replaces FACTOR with the supersampling factor before compiling.
@group(0) @binding(0) var source:texture_2d<f32>;
// Vertices 0, 1, 2 map to corners (0, 0), (2, 0), (0, 2) and clip positions (-1, -1), (3, -1),
// (-1, 3): one triangle that covers the whole output.
@vertex fn vertex(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
    let corner=vec2f(f32((i<<1u)&2u),f32(i&2u));
    return vec4f(corner*2.0-1.0,0.0,1.0);
}
// sRGB transfer functions. Averaging the display-encoded values would darken edges.
fn to_linear(c:vec3f) -> vec3f {
    return select(pow((c+0.055)/1.055,vec3f(2.4)),c/12.92,c<=vec3f(0.04045));
}
fn to_display(c:vec3f) -> vec3f {
    return select(1.055*pow(c,vec3f(1.0/2.4))-0.055,c*12.92,c<=vec3f(0.0031308));
}
@fragment fn fragment(@builtin(position) p:vec4f) -> @location(0) vec4f {
    // Output pixel p covers source pixels base to base + FACTOR - 1 in x and y.
    let base=vec2i(p.xy)*FACTOR;
    var sum=vec4f(0.0);
    for (var y=0; y<FACTOR; y++) {
        for (var x=0; x<FACTOR; x++) {
            let c=textureLoad(source,base+vec2i(x,y),0);
            sum+=vec4f(to_linear(c.rgb),c.a);
        }
    }
    sum/=f32(FACTOR*FACTOR);
    return vec4f(to_display(sum.rgb),sum.a);
}
