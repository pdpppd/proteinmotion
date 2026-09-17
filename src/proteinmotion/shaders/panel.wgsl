struct Uniform { size:vec2f, opacity:f32, pad:f32 };
@group(0) @binding(0) var<uniform> panel:Uniform;
struct Out { @builtin(position) position:vec4f, @location(0) color:vec4f };
@vertex fn vertex(@location(0) point:vec2f, @location(1) color:vec4f) -> Out {
    var out:Out;
    out.position=vec4f(2.0*point.x/panel.size.x-1.0,1.0-2.0*point.y/panel.size.y,0.0,1.0);
    out.color=vec4f(color.rgb,color.a*panel.opacity);
    return out;
}
@fragment fn fragment(in:Out) -> @location(0) vec4f { return in.color; }
