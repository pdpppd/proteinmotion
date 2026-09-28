
// GPU molecular surface, step 2: ray-march the density grid inside its bounding box.
struct GridSurface {
    model: mat4x4f,
    inv_model: mat4x4f,
    inv_vp: mat4x4f,
    grid_min: vec4f,   // xyz local origin, w voxel spacing
    grid_size: vec4f,  // nx, ny, nz, isovalue
    params: vec4f,     // opacity, tan(fov / 2), aspect, cutaway keep radius
};
@group(3) @binding(0) var<uniform> grid: GridSurface;
@group(3) @binding(1) var grid_density: texture_3d<f32>;
@group(3) @binding(2) var grid_color: texture_3d<f32>;
@group(3) @binding(3) var grid_sampler: sampler;

struct GridVertex {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};
@vertex fn grid_vertex(@builtin(vertex_index) i: u32) -> GridVertex {
    // A full-screen triangle; each pixel intersects its ray with the grid box.
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: GridVertex;
    out.clip = vec4f(xy, 0.0, 1.0);
    out.ndc = xy;
    return out;
}

fn grid_uvw(local: vec3f) -> vec3f {
    return ((local - grid.grid_min.xyz) / grid.grid_min.w + 0.5) / grid.grid_size.xyz;
}
fn grid_value(local: vec3f) -> f32 {
    return textureSampleLevel(grid_density, grid_sampler, grid_uvw(local), 0.0).r;
}
fn grid_world(local: vec3f) -> vec3f { return (grid.model * vec4f(local, 1.0)).xyz; }
// The cutaway carves the density field itself, so cut walls are smooth isosurfaces
// with correct normals rather than sharp slices through the interior.
fn grid_field(local: vec3f) -> f32 {
    var value = grid_value(local);
    if camera.cutaway_shape.x > 0.0 {
        value -= 24.0 * cut_to(grid_world(local), grid.params.w);
    }
    return value;
}

struct GridHit { found: bool, local: vec3f };
fn grid_march(ndc: vec2f) -> GridHit {
    var hit: GridHit;
    // Build the ray from the camera basis: exact at any near/far plane ratio.
    let right = camera.view_right.xyz;
    let up = camera.view_up.xyz;
    let forward = cross(up, right);
    let world_dir = forward + ndc.x * grid.params.y * grid.params.z * right + ndc.y * grid.params.y * up;
    let eye = (grid.inv_model * vec4f(camera.eye_fog_start.xyz, 1.0)).xyz;
    let dir = normalize((grid.inv_model * vec4f(world_dir, 0.0)).xyz);
    let extent = (grid.grid_size.xyz - 1.0) * grid.grid_min.w;
    let lo = grid.grid_min.xyz; let hi = lo + extent;
    let inv = 1.0 / dir;
    let t0 = (lo - eye) * inv; let t1 = (hi - eye) * inv;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    var t = max(near, 0.0);
    let step = grid.grid_min.w * 0.55;
    let iso = grid.grid_size.w;
    var prev = t;
    loop {
        if t > far { break; }
        let p = eye + dir * t;
        if grid_field(p) >= iso {
            // Refine the crossing between the last empty sample and this one.
            var a = prev; var b = t;
            for (var k = 0; k < 6; k++) {
                let m = 0.5 * (a + b);
                let q = eye + dir * m;
                if grid_field(q) >= iso { b = m; } else { a = m; }
            }
            hit.found = true;
            hit.local = eye + dir * b;
            return hit;
        }
        prev = t;
        t += step;
    }
    hit.found = false;
    return hit;
}
fn grid_normal(local: vec3f) -> vec3f {
    let e = grid.grid_min.w * 0.75;
    let g = vec3f(
        grid_field(local + vec3f(e, 0.0, 0.0)) - grid_field(local - vec3f(e, 0.0, 0.0)),
        grid_field(local + vec3f(0.0, e, 0.0)) - grid_field(local - vec3f(0.0, e, 0.0)),
        grid_field(local + vec3f(0.0, 0.0, e)) - grid_field(local - vec3f(0.0, 0.0, e)));
    return -g;
}

fn grid_rgba(local: vec3f) -> vec4f {
    let uvw = grid_uvw(local);
    let d = textureSampleLevel(grid_density, grid_sampler, uvw, 0.0);
    let c = textureSampleLevel(grid_color, grid_sampler, uvw, 0.0).rgb;
    let w = max(d.r, 1e-4);
    // Opacity is a ratio of separately quantized sums; snap fully opaque surfaces to 1.
    let alpha = d.g / w;
    return vec4f(c / w, select(alpha, 1.0, alpha > 0.985));
}

struct GridPixel { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };
@fragment fn grid_fragment(in: GridVertex) -> GridPixel {
    let hit = grid_march(in.ndc);
    if !hit.found { discard; }
    let rgba = grid_rgba(hit.local);
    if grid.params.x * rgba.a < 0.999 { discard; }
    let world = grid_world(hit.local);
    let normal = safe_normal((grid.model * vec4f(grid_normal(hit.local), 0.0)).xyz);
    let clip = camera.vp * vec4f(world, 1.0);
    var out: GridPixel;
    out.color = shade(world, normal, rgba.rgb);
    out.depth = clip.z / clip.w;
    return out;
}

struct GridTransparent {
    @location(0) accumulation: vec4f,
    @location(1) revealage: f32,
    @builtin(frag_depth) depth: f32,
};
@fragment fn grid_transparent(in: GridVertex) -> GridTransparent {
    let hit = grid_march(in.ndc);
    if !hit.found { discard; }
    let rgba = grid_rgba(hit.local);
    let alpha = clamp(grid.params.x * rgba.a, 0.0, 1.0);
    if alpha <= 0.0 || alpha >= 0.999 { discard; }
    let world = grid_world(hit.local);
    let normal = safe_normal((grid.model * vec4f(grid_normal(hit.local), 0.0)).xyz);
    let clip = camera.vp * vec4f(world, 1.0);
    let pixel = transparent(shade(world, normal, rgba.rgb).rgb, alpha, world);
    var out: GridTransparent;
    out.accumulation = pixel.accumulation;
    out.revealage = pixel.revealage;
    out.depth = clip.z / clip.w;
    return out;
}
