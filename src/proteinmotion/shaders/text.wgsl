// Cached vector glyphs. Write draws arc-length contours, then introduces the fill.
struct Uniforms {
    viewport: vec2f, origin: vec2f,
    scale: f32, opacity: f32, progress: f32, lag: f32,
    glyph_count: f32, glyph_offset: f32, reverse: f32, stroke_width: f32,
    color: vec4f,
};
struct Segment { ends: vec4f, path: vec4f };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> segments: array<Segment>;

fn clip_position(pixel: vec2f) -> vec4f {
    return vec4f(pixel.x / u.viewport.x * 2.0 - 1.0,
                 1.0 - pixel.y / u.viewport.y * 2.0, 0.0, 1.0);
}
fn glyph_progress(index: f32) -> f32 {
    var rank = u.glyph_offset + index;
    if u.reverse > 0.5 { rank = u.glyph_count - 1.0 - rank; }
    return clamp(u.progress * (1.0 + max(0.0, u.glyph_count - 1.0) * u.lag) - rank * u.lag, 0.0, 1.0);
}
struct FillVertex { @builtin(position) position: vec4f, @location(0) opacity: f32 };
@vertex fn fill_vertex(@location(0) vertex: vec3f) -> FillVertex {
    var out: FillVertex;
    out.position = clip_position(u.origin + vertex.xy * u.scale);
    out.opacity = u.opacity * clamp(glyph_progress(vertex.z) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
@fragment fn fill_fragment(in: FillVertex) -> @location(0) vec4f {
    return vec4f(u.color.rgb, in.opacity);
}
struct StrokeVertex {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) ends: vec4f,
    @location(1) @interpolate(flat) radius: f32,
    @location(2) @interpolate(flat) visible: f32,
};
@vertex fn stroke_vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> StrokeVertex {
    let s = segments[instance];
    let progress = glyph_progress(s.path.z);
    let prefix = min(1.0, progress * 2.0);
    let fraction = clamp((prefix - s.path.x) / max(1e-8, s.path.y - s.path.x), 0.0, 1.0);
    let a = u.origin + s.ends.xy * u.scale;
    let b = u.origin + mix(s.ends.xy, s.ends.zw, fraction) * u.scale;
    let radius = 0.5 * u.stroke_width * u.scale * (1.0 - clamp(progress * 2.0 - 1.0, 0.0, 1.0));
    let corners = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
    let lower = min(a, b) - radius - 0.75;
    let upper = max(a, b) + radius + 0.75;
    var out: StrokeVertex;
    out.position = clip_position(mix(lower, upper, corners[vertex]));
    out.ends = vec4f(a, b);
    out.radius = radius;
    out.visible = select(0.0, 1.0, fraction > 0.0 && radius > 0.0);
    return out;
}
@fragment fn stroke_fragment(in: StrokeVertex) -> @location(0) vec4f {
    if in.visible < 0.5 { discard; }
    let a = in.ends.xy;
    let ab = in.ends.zw - a;
    let fraction = clamp(dot(in.position.xy - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    let distance = length(in.position.xy - a - ab * fraction);
    let coverage = 1.0 - smoothstep(max(0.0, in.radius - 0.6), in.radius + 0.6, distance);
    return vec4f(u.color.rgb, coverage * u.opacity);
}
