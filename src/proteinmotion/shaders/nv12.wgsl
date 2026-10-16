// RGBA -> limited-range BT.709 NV12. Each invocation packs a 4x2 pixel tile.
// Renderer.enqueue copies output into a readback buffer: a full-height luma (Y) plane, then a
// half-height plane of interleaved U and V, each row (width + 3) / 4 words long.
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
// Clamp x to the last column, so a width that is not a multiple of 4 repeats the edge pixel.
fn rgb(x:u32,y:u32) -> vec3<f32> {
    let size=textureDimensions(source);
    return textureLoad(source,vec2<i32>(i32(min(x,size.x-1u)),i32(y)),0).rgb;
}
// BT.709 luma weights; limited range puts black at 16 and white at 235.
fn luma(c:vec3<f32>) -> u32 {
    return u32(clamp(round(16.+219.*dot(c,vec3<f32>(0.2126,0.7152,0.0722))),16.,235.));
}
// BT.709 Cb and Cr, scaled to 224 levels around 128 and clamped to [16, 240].
fn chroma(c:vec3<f32>) -> vec2<u32> {
    let u=128.+224.*dot(c,vec3<f32>(-0.114572,-0.385428,0.5));
    let v=128.+224.*dot(c,vec3<f32>(0.5,-0.454153,-0.045847));
    return vec2<u32>(clamp(round(vec2<f32>(u,v)),vec2<f32>(16.),vec2<f32>(240.)));
}
// Four bytes into one little-endian word: v.x is the first byte in memory.
fn pack(v:vec4<u32>) -> u32 { return v.x | (v.y<<8u) | (v.z<<16u) | (v.w<<24u); }
@compute @workgroup_size(8,8)
fn convert(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(source);
    let x=id.x*4u; let y=id.y*2u;
    if x>=size.x || y>=size.y { return; }
    let a=rgb(x,y); let b=rgb(x+1u,y); let c=rgb(x+2u,y); let d=rgb(x+3u,y);
    let e=rgb(x,y+1u); let f=rgb(x+1u,y+1u); let g=rgb(x+2u,y+1u); let h=rgb(x+3u,y+1u);
    // Two luma rows of four pixels each, then one chroma row with the (U, V) pairs of the two
    // 2 × 2 blocks, each the average of its four pixels.
    let stride=(size.x+3u)/4u;
    output[y*stride+id.x]=pack(vec4<u32>(luma(a),luma(b),luma(c),luma(d)));
    output[(y+1u)*stride+id.x]=pack(vec4<u32>(luma(e),luma(f),luma(g),luma(h)));
    let uv0=chroma((a+b+e+f)*0.25);
    let uv1=chroma((c+d+g+h)*0.25);
    output[size.y*stride+id.y*stride+id.x]=pack(vec4<u32>(uv0,uv1));
}
