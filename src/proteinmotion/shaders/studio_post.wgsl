// Studio post-processing: ambient occlusion, bloom, halation, roll-off and film grain.
// The scene buffer holds display-referred colors that may exceed 1 (HDR highlights).
struct Post {
    vp: mat4x4f,
    inv_vp: mat4x4f,
    eye: vec4f,        // xyz eye, w ambient-occlusion radius (world units)
    settings: vec4f,   // ao strength, bloom strength, bloom threshold, vignette
    size: vec4f,       // width, height, 1/width, 1/height
    background: vec4f, // rgb background, w highlight knee
    flags: vec2f,      // front layer present, layer weight
    grain_frame: vec2u, // grain tick, seed (integers preserve all 32 seed bits)
    effects: vec4f,    // grain strength, grain size (1080p px), bloom radius, halation strength
    halation: vec4f,   // threshold, blur sigma (1080p px), reserved, reserved
    halation_color: vec4f,
};
@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var aux_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
// Front layer: depth and opacity of the nearest fragment of any opacity, so fading
// geometry receives ambient occlusion in proportion to its opacity.
@group(0) @binding(5) var front_depth_tex: texture_depth_multisampled_2d;
@group(0) @binding(6) var front_alpha_tex: texture_multisampled_2d<f32>;

struct Screen { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex fn fullscreen(@builtin(vertex_index) i: u32) -> Screen {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Screen;
    out.clip = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(xy.x, 1.0 - xy.y);
    return out;
}

fn depth_at(px: vec2i) -> f32 {
    let size = vec2i(textureDimensions(depth_tex));
    return textureLoad(depth_tex, clamp(px, vec2i(0), size - 1), 0);
}
fn front_depth_at(px: vec2i) -> f32 {
    let size = vec2i(textureDimensions(front_depth_tex));
    return textureLoad(front_depth_tex, clamp(px, vec2i(0), size - 1), 0);
}
fn layer_depth(px: vec2i, front: bool) -> f32 {
    if front { return front_depth_at(px); }
    return depth_at(px);
}
fn world_at(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = post.inv_vp * ndc;
    return w.xyz / w.w;
}
fn bayer4(px: vec2i) -> f32 {
    // A fixed 4 × 4 pattern of sample rotations. The composite blur averages exactly one
    // 4 × 4 tile, so every pixel sees all 16 rotations and the result does not shimmer.
    let m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    let i = (px.x & 3) + 4 * (px.y & 3);
    return (m[i] + 0.5) / 16.0;
}

// Screen-space ambient occlusion in world space, with normals reconstructed from depth.
fn occlusion(px: vec2i, front: bool) -> f32 {
    let d = layer_depth(px, front);
    if d <= 0.0 { return 1.0; }
    let uv = (vec2f(px) + 0.5) * post.size.zw;
    let p = world_at(uv, d);
    // Choose the neighbor on the flatter side, so silhouettes keep sensible normals.
    let dl = layer_depth(px - vec2i(1, 0), front); let dr = layer_depth(px + vec2i(1, 0), front);
    let du = layer_depth(px - vec2i(0, 1), front); let dd = layer_depth(px + vec2i(0, 1), front);
    var hx = world_at(uv + vec2f(post.size.z, 0.0), dr) - p;
    if abs(dl - d) < abs(dr - d) { hx = p - world_at(uv - vec2f(post.size.z, 0.0), dl); }
    var hy = world_at(uv + vec2f(0.0, post.size.w), dd) - p;
    if abs(du - d) < abs(dd - d) { hy = p - world_at(uv - vec2f(0.0, post.size.w), du); }
    var n = normalize(cross(hy, hx));
    let to_eye = normalize(post.eye.xyz - p);
    if dot(n, to_eye) < 0.0 { n = -n; }
    var t = normalize(cross(n, vec3f(0.0, 1.0, 0.0)));
    if abs(n.y) > 0.95 { t = normalize(cross(n, vec3f(1.0, 0.0, 0.0))); }
    let b = cross(n, t);
    let radius = post.eye.w;
    let spin = bayer4(px) * 6.2831853;
    var occlusion = 0.0;
    let count = 20;
    for (var k = 0; k < count; k++) {
        let fk = f32(k);
        // Cosine-weighted spiral over the hemisphere, scaled toward the center.
        let a = fk * 2.3999632 + spin;
        let r = sqrt((fk + 0.5) / f32(count));
        let h = sqrt(max(1.0 - r * r, 0.0));
        let scale = mix(0.25, 1.0, ((fk + 0.5) / f32(count)) * ((fk + 0.5) / f32(count)));
        let s = p + (t * cos(a) * r + b * sin(a) * r + n * h) * radius * scale;
        let clip = post.vp * vec4f(s, 1.0);
        let ndc = clip.xyz / clip.w;
        let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if any(suv < vec2f(0.0)) || any(suv > vec2f(1.0)) { continue; }
        let sd = layer_depth(vec2i(suv * post.size.xy), front);
        if sd <= 0.0 { continue; }
        let q = world_at(suv, sd);
        // Continuous occlusion: how far the visible surface lies in front of the sample,
        // with a smooth falloff. A binary in-front test would switch samples on and off as
        // geometry slides by a fraction of a pixel, which shimmers during motion.
        let lead = distance(post.eye.xyz, s) - distance(post.eye.xyz, q);
        let covered = smoothstep(0.03 * radius, 0.3 * radius, lead);
        let range = smoothstep(0.0, 1.0, radius / max(distance(p, q), 1e-4));
        occlusion += covered * range;
    }
    return clamp(1.0 - occlusion / f32(count), 0.0, 1.0);
}

@fragment fn ambient_occlusion(in: Screen) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy);
    let front = select(1.0, occlusion(px, true), post.flags.x > 0.5);
    return vec4f(occlusion(px, false), front, 0.0, 1.0);
}

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Bloom prefilter: keep the part of each pixel above the threshold (soft knee).
@fragment fn bloom_prefilter(in: Screen) -> @location(0) vec4f {
    // Karis average: weight each tap by 1 / (1 + luminance), so isolated sparkles on
    // glossy highlights cannot flash in and out of the bloom between frames.
    let o = post.size.zw;
    var c = vec3f(0.0); var total = 0.0;
    for (var k = 0; k < 4; k++) {
        let offset = vec2f(select(-o.x, o.x, (k & 1) == 1), select(-o.y, o.y, k >= 2));
        let tap = textureSampleLevel(color_tex, linear_sampler, in.uv + offset, 0.0).rgb;
        let w = 1.0 / (1.0 + luminance(tap));
        c += tap * w; total += w;
    }
    c /= total;
    let threshold = post.settings.z;
    let l = luminance(c);
    let knee = 0.25 * threshold;
    let soft = clamp(l - threshold + knee, 0.0, 2.0 * knee);
    let weight = max(soft * soft / (4.0 * knee + 1e-4), l - threshold) / max(l, 1e-4);
    return vec4f(c * max(weight, 0.0), 1.0);
}
// 13-tap downsample (Jimenez 2014), from the next larger level in aux_tex.
@fragment fn bloom_down(in: Screen) -> @location(0) vec4f {
    let o = 1.0 / vec2f(textureDimensions(aux_tex));
    let s = in.uv;
    let a = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-2.0, 0.0), 0.0).rgb;
    let e = textureSampleLevel(aux_tex, linear_sampler, s, 0.0).rgb;
    let f = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(2.0, 0.0), 0.0).rgb;
    let g = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-2.0, 2.0), 0.0).rgb;
    let h = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(0.0, 2.0), 0.0).rgb;
    let i = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(2.0, 2.0), 0.0).rgb;
    let j = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-1.0, 1.0), 0.0).rgb;
    let m = textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(1.0, 1.0), 0.0).rgb;
    var c0 = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    return vec4f(c0, 1.0);
}
// 9-tap tent upsample from the next smaller level, added onto this level.
@fragment fn bloom_up(in: Screen) -> @location(0) vec4f {
    let o = post.effects.z * (post.size.y / 1080.0) / vec2f(textureDimensions(aux_tex));
    let s = in.uv;
    var c = textureSampleLevel(aux_tex, linear_sampler, s, 0.0).rgb * 4.0;
    c += (textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-1.0, 0.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(1.0, 0.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(0.0, -1.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(0.0, 1.0), 0.0).rgb) * 2.0;
    c += textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-1.0, -1.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(1.0, -1.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(-1.0, 1.0), 0.0).rgb
        + textureSampleLevel(aux_tex, linear_sampler, s + o * vec2f(1.0, 1.0), 0.0).rgb;
    return vec4f(c / 16.0, 1.0);
}

// Final image: blurred occlusion, bloom, and a hue-preserving highlight roll-off.
// Colors below the knee pass unchanged, so the background and fog match exactly.
// Depth-aware blur over 2 × 2 rotation tiles (8 × 8 pixels), so each pixel averages every
// rotation four times and sub-pixel motion of fine features stays smooth. Returns the
// blurred occlusion of one layer and whether that layer covers this pixel.
fn blurred_occlusion(px: vec2i, front: bool) -> vec2f {
    let d = layer_depth(px, front);
    if d <= 0.0 { return vec2f(1.0, 0.0); }
    var sum = 0.0; var weight = 0.0;
    let center = world_at((vec2f(px) + 0.5) * post.size.zw, d);
    for (var y = -4; y <= 3; y++) {
        for (var x = -4; x <= 3; x++) {
            let q = px + vec2i(x, y);
            let qd = layer_depth(q, front);
            let qw = world_at((vec2f(q) + 0.5) * post.size.zw, qd);
            let gap = distance(center, qw) / max(post.eye.w * 0.5, 1e-3);
            let w = select(0.0, exp(-gap * gap), qd > 0.0);
            let ao = textureLoad(aux_tex, clamp(q, vec2i(0), vec2i(post.size.xy) - 1), 0);
            sum += select(ao.r, ao.g, front) * w;
            weight += w;
        }
    }
    return vec2f(select(1.0, sum / weight, weight > 1e-4), 1.0);
}

// Ambient-occlusion shading factor for a pixel.
fn ao_shade(px: vec2i) -> f32 {
    if post.settings.x <= 0.0 { return 1.0; }
    let solid = blurred_occlusion(px, false);
    var shade = solid.x * solid.x;
    if post.flags.x > 0.5 {
        // Front layer over the solid layer: w·AO_front + (1 − w)·AO_solid. The weight is
        // zero at opacity 0.5, where a layer joins or leaves the front layer, and one for
        // opaque layers, which are also the solid layer; fades therefore stay smooth.
        let front = blurred_occlusion(px, true);
        var a = select(0.0, smoothstep(0.5, 1.0, textureLoad(front_alpha_tex, px, 0).r), front.y > 0.0);
        // Where the front layer is the solid surface itself, keep the solid layer's occlusion
        // (the front pass samples depth slightly differently, without multisampling).
        let uv = (vec2f(px) + 0.5) * post.size.zw;
        let solid_d = depth_at(px);
        if solid_d > 0.0 && front.y > 0.0 {
            let gap = distance(world_at(uv, solid_d), world_at(uv, front_depth_at(px)));
            a *= smoothstep(0.3, 1.0, gap);
        }
        shade = a * front.x * front.x + (1.0 - a) * shade;
    }
    return mix(1.0, shade, post.settings.x);
}

// One render layer, shaded by its ambient occlusion and weighted, added into the accumulator.
// A frame is one layer of weight 1, or several when whole objects fade (see StudioRenderer).
@fragment fn accumulate(in: Screen) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy);
    let color = textureLoad(color_tex, px, 0).rgb * ao_shade(px);
    return vec4f(color * post.flags.y, 0.0);
}

@fragment fn copy_color(in: Screen) -> @location(0) vec4f {
    return vec4f(textureLoad(aux_tex, vec2i(in.clip.xy), 0).rgb, 1.0);
}

// Film halation: warm light spills just outside bright boundaries. Subtract the
// local source mask from its blur so flat highlights do not acquire an orange wash.
// This is an artistic screen-space approximation, not a film-stock simulation.
@fragment fn halation_prefilter(in: Screen) -> @location(0) vec4f {
    let c = textureSampleLevel(color_tex, linear_sampler, in.uv, 0.0).rgb;
    let bright = max(luminance(c) - post.halation.x, 0.0);
    return vec4f(vec3f(bright), 1.0);
}
fn halation_blur(uv: vec2f, direction: vec2f) -> vec4f {
    let sigma = post.halation.y * (post.size.y / 1080.0);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var i = -8; i <= 8; i++) {
        let x = f32(i) / 4.0;
        let w = exp(-0.5 * x * x);
        let offset = direction * post.size.zw * sigma * x;
        sum += textureSampleLevel(aux_tex, linear_sampler, uv + offset, 0.0).rgb * w;
        weight += w;
    }
    return vec4f(sum / weight, 1.0);
}
@fragment fn halation_horizontal(in: Screen) -> @location(0) vec4f {
    return halation_blur(in.uv, vec2f(1.0, 0.0));
}
@fragment fn halation_vertical(in: Screen) -> @location(0) vec4f {
    return halation_blur(in.uv, vec2f(0.0, 1.0));
}
@fragment fn add_halation(in: Screen) -> @location(0) vec4f {
    let blurred = textureSampleLevel(aux_tex, linear_sampler, in.uv, 0.0).r;
    let source = textureSampleLevel(color_tex, linear_sampler, in.uv, 0.0).r;
    let fringe = max(blurred - source, 0.0);
    return vec4f(post.halation_color.rgb * fringe * post.effects.w * 3.0, 0.0);
}

// Integer hashing avoids sine-based noise patterns and is deterministic when seeking.
fn grain_hash(value: u32) -> f32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return f32(x) / 4294967295.0;
}
fn grain_sample(p: vec2u) -> f32 {
    let seed = post.grain_frame.y;
    let tick = post.grain_frame.x;
    let h = p.x * 0x9e3779b9u + p.y * 0x85ebca6bu + tick * 0xc2b2ae35u + seed;
    // Triangular, centered noise is softer than a uniform distribution.
    return (grain_hash(h) + grain_hash(h ^ 0x68bc21ebu) - 1.0) * 2.4494897;
}
fn film_grain(px: vec2f) -> f32 {
    let cell = max(post.effects.y * post.size.y / 1080.0, 0.25);
    let p = px / cell;
    let ij = vec2u(floor(p));
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    return mix(mix(grain_sample(ij), grain_sample(ij + vec2u(1u, 0u)), w.x),
               mix(grain_sample(ij + vec2u(0u, 1u)), grain_sample(ij + vec2u(1u, 1u)), w.x), w.y);
}

@fragment fn composite(in: Screen) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy);
    var color = textureLoad(color_tex, px, 0).rgb;
    let peak = max(max(color.r, color.g), color.b);
    let knee = post.background.w;
    if peak > knee {
        let rolled = knee + (1.0 - knee) * (1.0 - exp(-(peak - knee) / (1.0 - knee)));
        color *= rolled / peak;
    }
    let vignette = 1.0 - post.settings.w * pow(length(in.uv - 0.5) * 1.4, 2.5);
    color = mix(post.background.rgb, color, vignette);
    if post.effects.x > 0.0 {
        let l = clamp(luminance(color), 0.0, 1.0);
        // Luminance-based grain: most visible in midtones, vanishing in black/white.
        let response = 2.0 * sqrt(max(l * (1.0 - l), 0.0));
        color += vec3f(film_grain(in.clip.xy) * post.effects.x * response);
    }
    return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}

@fragment fn add_bloom(in: Screen) -> @location(0) vec4f {
    return vec4f(textureSampleLevel(aux_tex, linear_sampler, in.uv, 0.0).rgb * post.settings.y, 0.0);
}
