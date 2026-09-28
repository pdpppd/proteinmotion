// GPU molecular surface, step 1: splat each atom's Gaussian density and color into a grid.
struct Splat {
    grid_min: vec4f,   // xyz local grid origin, w voxel spacing
    dims: vec4u,       // nx, ny, nz, atom count
    params: vec4f,     // radius inflation (Å), blobbiness, 0, 0
};
@group(0) @binding(0) var<uniform> splat: Splat;
@group(0) @binding(1) var<storage, read> atoms: array<vec4f>;    // xyz local, w radius
@group(0) @binding(2) var<storage, read> colors: array<vec4f>;   // rgb, opacity
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(4) var density_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var color_out: texture_storage_3d<rgba16float, write>;

const DENSITY_SCALE = 4096.0;
const COLOR_SCALE = 1024.0;

@compute @workgroup_size(64)
fn splat_atoms(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= splat.dims.w { return; }
    let atom = atoms[i];
    let color = colors[i];
    if color.a <= 0.0 { return; }
    let r = atom.w + splat.params.x;
    let k = splat.params.y;
    let h = splat.grid_min.w;
    let cutoff = 2.0 * r;
    let lo = vec3i(floor((atom.xyz - cutoff - splat.grid_min.xyz) / h));
    let hi = vec3i(ceil((atom.xyz + cutoff - splat.grid_min.xyz) / h));
    let size = vec3i(splat.dims.xyz);
    let a = max(lo, vec3i(0));
    let b = min(hi, size - 1);
    for (var z = a.z; z <= b.z; z++) {
        for (var y = a.y; y <= b.y; y++) {
            for (var x = a.x; x <= b.x; x++) {
                let p = splat.grid_min.xyz + vec3f(f32(x), f32(y), f32(z)) * h;
                let d2 = dot(p - atom.xyz, p - atom.xyz);
                // Blobby (Blinn) density: exactly 1 at the inflated radius of a lone atom.
                let w = exp(-k * (d2 / (r * r) - 1.0));
                if w < 0.004 { continue; }
                let v = u32(x) + splat.dims.x * (u32(y) + splat.dims.y * u32(z));
                atomicAdd(&accum[v * 5u], u32(w * DENSITY_SCALE));
                atomicAdd(&accum[v * 5u + 1u], u32(w * color.r * COLOR_SCALE));
                atomicAdd(&accum[v * 5u + 2u], u32(w * color.g * COLOR_SCALE));
                atomicAdd(&accum[v * 5u + 3u], u32(w * color.b * COLOR_SCALE));
                atomicAdd(&accum[v * 5u + 4u], u32(w * color.a * COLOR_SCALE));
            }
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn resolve_grid(@builtin(global_invocation_id) id: vec3u) {
    if any(id >= splat.dims.xyz) { return; }
    let v = id.x + splat.dims.x * (id.y + splat.dims.y * id.z);
    // Colors and opacity stay weighted by density; the ray-marcher divides by the
    // interpolated density, so empty neighbors do not darken the surface.
    let density = f32(atomicLoad(&accum[v * 5u])) / DENSITY_SCALE;
    let weighted = vec4f(
        f32(atomicLoad(&accum[v * 5u + 1u])),
        f32(atomicLoad(&accum[v * 5u + 2u])),
        f32(atomicLoad(&accum[v * 5u + 3u])),
        f32(atomicLoad(&accum[v * 5u + 4u])),
    ) / COLOR_SCALE;
    textureStore(density_out, id, vec4f(density, weighted.a, 0.0, 0.0));
    textureStore(color_out, id, vec4f(weighted.rgb, 1.0));
}
