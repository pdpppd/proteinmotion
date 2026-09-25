struct Camera {
    vp: mat4x4<f32>,
    eye_fog_start: vec4<f32>,
    background_fog_end: vec4<f32>,
    lighting: vec4<f32>,  // depth cue, cutaway rim color
    cutaway: vec4<f32>,   // target center, kept radius around it
    cutaway_shape: vec4<f32>,  // opening in [0, 1], window radius at the target, edge softness, depth band
    cutaway_surface: vec4<f32>,  // kept radius for cartoons, ribbons, bases and surfaces
    cutaway_tunnel: vec4<f32>,   // tunnel mode, outer-surface distance from the target, wall opacity, ring spacing
    cutaway_axis: vec4<f32>,     // tunnel direction from the target, fixed to the molecule
};
struct Object {
    model: mat4x4<f32>,
    params: vec4<f32>, // tween, opacity, atom scale, bond radius
    style: vec4<f32>,  // cartoon proportion, ribbon width, object scale, draw atoms
    appearance: vec4<f32>, // color clock, opacity clock, detail clock, ball-and-stick fraction
};
struct AtomState { position: vec4<f32>, guide: vec4<f32> };
struct Appearance { before:vec4f, after:vec4f, timing:vec4f, alpha:vec4f,
                    detail:vec4f, detail_timing:vec4f };
struct Control { motion: vec4<f32>, visibility: vec4<f32> };
struct Segment {
    atoms: vec4<u32>,
    shape: vec4<f32>,
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    caps: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var<storage, read> state_a: array<AtomState>;
@group(1) @binding(2) var<storage, read> state_b: array<AtomState>;
@group(1) @binding(3) var<storage, read> atoms: array<vec4<f32>>; // packed colors, radius
@group(1) @binding(4) var<storage, read> bonds: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> segments: array<Segment>;
@group(1) @binding(6) var<storage, read> controls: array<Control>;
@group(1) @binding(7) var<storage, read> appearance:array<Appearance>;

fn safe_normal(v: vec3<f32>) -> vec3<f32> {
    return v * inverseSqrt(max(dot(v,v), 0.00000001));
}
fn ease(t:f32) -> f32 {
    // Mirror the upper half, as on the CPU: avoid cancellation near 1 that
    // misclassifies fully visible fragments as transparent at a fade endpoint.
    let u=select(t,1.0-t,t>0.5);
    let value=u*u*u*(10.0+u*(-15.0+6.0*u));
    return select(value,1.0-value,t>0.5);
}
fn atom_progress(i:u32) -> f32 {
    let c=controls[i].motion;
    if c.z<0.5 { return object.params.x; }
    let t=clamp((object.params.x-c.x)/max(c.y,0.00000001),0.0,1.0);
    if c.z>1.5 { return ease(t); }
    return t;
}
fn appearance_progress(clock:f32, start:f32, span:f32, eased:f32) -> f32 {
    let t=clamp((clock-start)/max(span,1e-8),0.0,1.0);
    return select(t,ease(t),eased>0.5);
}
fn tint(i:u32, base:vec3f) -> vec3f {
    let s=appearance[i];
    let t=appearance_progress(object.appearance.x,s.timing.x,s.timing.y,s.timing.z);
    let c=mix(s.before,s.after,t);
    return base*(1.0-c.w)+c.xyz;
}
fn atom_opacity(i:u32) -> f32 {
    let c=controls[i].visibility;
    let t=ease(clamp((object.params.x-c.z)/max(c.w,0.00000001),0.0,1.0));
    let s=appearance[i];
    let a=appearance_progress(object.appearance.y,s.alpha.z,s.alpha.w,s.timing.w);
    return mix(c.x,c.y,t)*mix(s.alpha.x,s.alpha.y,a);
}
// Atoms and bonds draw with the ball-and-stick representation, or per atom as
// detail (ligands, ions, side chains) over a cartoon, ribbon or surface.
fn stick_opacity(i:u32) -> f32 {
    let s=appearance[i];
    let t=appearance_progress(object.appearance.z,s.detail.z,s.detail.w,s.detail_timing.x);
    let ball=object.appearance.w;
    return atom_opacity(i)*(ball+(1.0-ball)*mix(s.detail.x,s.detail.y,t));
}
// Ball-and-stick and detail palettes are packed as RGBA8 bits; blend by representation.
fn atom_color(i:u32) -> vec3f {
    let rgb=unpack4x8unorm(bitcast<u32>(atoms[i].x)).xyz;
    let detail=unpack4x8unorm(bitcast<u32>(atoms[i].y)).xyz;
    return mix(detail,rgb,object.appearance.w);
}
fn position(i: u32) -> vec3<f32> {
    let t=atom_progress(i);
    // Preserve exact coordinates at endpoints even when a driver contracts mix
    // into a fused multiply-add. Analytic sphere depth amplifies tiny errors.
    if t<=0.0 { return state_a[i].position.xyz; }
    if t>=1.0 { return state_b[i].position.xyz; }
    return mix(state_a[i].position.xyz, state_b[i].position.xyz, t);
}
fn guide(i: u32) -> vec3<f32> {
    var b = state_b[i].guide.xyz;
    let a = state_a[i].guide.xyz;
    if dot(a,b) < 0.0 { b = -b; }
    return mix(a,b,atom_progress(i));
}
fn world(p: vec3<f32>) -> vec3<f32> { return (object.model*vec4<f32>(p,1.0)).xyz; }
fn world_normal(n: vec3<f32>) -> vec3<f32> { return safe_normal((object.model*vec4<f32>(n,0.0)).xyz); }

// View-dependent cutaway: fade geometry in front of the target inside a cone from the
// eye, which is a fixed disc on screen. The target sphere and everything behind stay.
fn cut_amount(p:vec3f) -> f32 { return cut_to(p,camera.cutaway.w); }
fn cut_to(p:vec3f, keep:f32) -> f32 {
    let shape=camera.cutaway_shape;
    if shape.x<=0.0 { return 0.0; }
    let eye=camera.eye_fog_start.xyz;
    if camera.cutaway_tunnel.x>0.5 {
        // Tunnel: a straight cylinder toward the camera. Perspective shows its walls.
        let axis=tunnel_axis();
        let d=p-camera.cutaway.xyz;
        let height=dot(d,axis);
        let r=length(d-axis*height);
        let radius=shape.x*shape.y;
        let radial=1.0-smoothstep(radius*(1.0-shape.z),radius,r);
        let depth=smoothstep(keep,keep+shape.w,height);
        return radial*depth;
    }
    let to_target=camera.cutaway.xyz-eye;
    let dist=max(length(to_target),1e-4);
    let axis=to_target/dist;
    let d=p-eye;
    let t=dot(d,axis);
    let front=dist-keep;
    let aperture=shape.x*shape.y*max(t,0.0)/dist;
    let r=length(d-axis*t);
    let radial=1.0-smoothstep(aperture*(1.0-shape.z),aperture,r);
    let depth=1.0-smoothstep(front-shape.w,front,t);
    return radial*depth;
}
fn tunnel_axis() -> vec3f {
    return safe_normal(camera.cutaway_axis.xyz);
}
fn cutaway_rim(c:vec3f, p:vec3f) -> vec3f {
    // A faint rim marks where geometry is fading, so the window reads as a lens.
    let amount=cut_amount(p);
    return mix(c,camera.lighting.yzw,clamp(amount*(1.0-amount)*1.6,0.0,0.4));
}
fn visible(value:f32, p:vec3f) -> f32 {
    let alpha=clamp(object.params.y*value,0.0,1.0)*(1.0-cut_amount(p));
    return select(alpha,1.0,alpha>=0.999999);
}
// Atoms keep the whole target sphere; cartoons and surfaces, which pass in front of a
// ligand or enclose it, are carved closer to the target's center.
fn surface_visible(value:f32, p:vec3f) -> f32 {
    let alpha=clamp(object.params.y*value,0.0,1.0)*(1.0-cut_to(p,camera.cutaway_surface.x));
    return select(alpha,1.0,alpha>=0.999999);
}

fn opacity(value:f32) -> f32 {
    let alpha=clamp(object.params.y*value,0.0,1.0);
    // Perspective interpolation can turn constant 1.0 into 0.99999994 on
    // discrete GPUs. Snap that rounding error before splitting opaque/OIT
    // passes, otherwise fully opaque triangles acquire holes or draw twice.
    return select(alpha,1.0,alpha>=0.999999);
}

// Weighted blended OIT: accumulate premultiplied colors and background transmittance.
// Opaque fragments have their own depth-writing pass. This pass never writes depth.
struct TransparentPixel {
    @location(0) accumulation:vec4<f32>,
    @location(1) revealage:f32,
};
fn transparent(color:vec3<f32>, alpha:f32, p:vec3<f32>) -> TransparentPixel {
    let relative_depth=clamp((distance(camera.eye_fog_start.xyz,p)-camera.eye_fog_start.w)/
                            max(camera.background_fog_end.w-camera.eye_fog_start.w,0.001),0.0,1.0);
    let weight=(0.1+8.0*pow(1.0-relative_depth,3.0))*(0.1+0.9*alpha*alpha*alpha);
    var out:TransparentPixel;
    out.accumulation=vec4<f32>(color*alpha,alpha)*weight;
    out.revealage=alpha;
    return out;
}

fn shade(p: vec3<f32>, norm: vec3<f32>, base: vec3<f32>) -> vec4<f32> {
    let v = safe_normal(camera.eye_fog_start.xyz-p);
    var n = safe_normal(norm);
    if dot(n,v)<0.0 { n = -n; }
    let key = safe_normal(vec3<f32>(-0.45,0.8,0.85));
    let fill = safe_normal(vec3<f32>(0.8,0.2,0.3));
    let rim = pow(1.0-max(dot(n,v),0.0),3.0);
    let diffuse = 0.31+0.59*max(dot(n,key),0.0)+0.17*max(dot(n,fill),0.0);
    let spec = pow(max(dot(n,safe_normal(key+v)),0.0),48.0)*0.24;
    var c = base*diffuse + vec3<f32>(spec)+base*rim*0.18;
    let d = distance(camera.eye_fog_start.xyz,p);
    let fog = smoothstep(camera.eye_fog_start.w,camera.background_fog_end.w,d)*camera.lighting.x;
    c = cutaway_rim(c,p);
    c = mix(c,camera.background_fog_end.xyz,fog);
    return vec4<f32>(c,1.0);
}

struct Surface {
    @builtin(position) clip: vec4<f32>,
    @location(0) p: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) opacity: f32,
};
fn catmull(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, t:f32) -> vec3<f32> {
    return 0.5*((2.0*b)+(-a+c)*t+(2.0*a-5.0*b+4.0*c-d)*t*t+(-a+3.0*b-3.0*c+d)*t*t*t);
}
fn tangent(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>,t:f32) -> vec3<f32> {
    return safe_normal(0.5*((-a+c)+2.0*(2.0*a-5.0*b+4.0*c-d)*t+3.0*(-a+3.0*b-3.0*c+d)*t*t));
}

@vertex fn ribbon_vertex(@location(0) uv: vec2<f32>, @builtin(instance_index) instance:u32) -> Surface {
    let s = segments[instance];
    let a=position(s.atoms.x); let b=position(s.atoms.y);
    let c=position(s.atoms.z); let d=position(s.atoms.w);
    let t=uv.x;
    let center=catmull(a,b,c,d,t);
    let direction=tangent(a,b,c,d,t);
    let g=mix(guide(s.atoms.y),guide(s.atoms.z),t);
    let width_axis=safe_normal(g-direction*dot(g,direction));
    let height_axis=safe_normal(cross(direction,width_axis));
    let angle=uv.y*6.28318530718;
    let width=mix(object.style.y,mix(s.shape.x,s.shape.y,t),object.style.x);
    let height=mix(0.13,mix(s.shape.z,s.shape.w,t),object.style.x);
    var cap=1.0;
    if s.caps.x>0.5 { cap*=smoothstep(0.0,0.12,t); }
    if s.caps.y>0.5 { cap*=smoothstep(0.0,0.12,1.0-t); }
    let offset=width_axis*cos(angle)*width + height_axis*sin(angle)*height;
    let normal=safe_normal(width_axis*cos(angle)/width+height_axis*sin(angle)/height);
    var out:Surface;
    out.p=world(center+offset*cap);
    out.clip=camera.vp*vec4<f32>(out.p,1.0);
    out.normal=world_normal(normal);
    out.color=mix(tint(s.atoms.y,s.color_a.xyz),tint(s.atoms.z,s.color_b.xyz),t);
    out.opacity=mix(atom_opacity(s.atoms.y),atom_opacity(s.atoms.z),t);
    return out;
}

struct Bond {
    @builtin(position) clip: vec4<f32>,
    @location(0) p: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) opacity: f32,
    @location(4) @interpolate(flat) sphere_a: vec4<f32>,
    @location(5) @interpolate(flat) sphere_b: vec4<f32>,
};

@vertex fn bond_vertex(@builtin(vertex_index) vertex:u32, @builtin(instance_index) instance:u32) -> Bond {
    let pair=bonds[instance];
    let a=position(pair.x); let b=position(pair.y);
    let dir=safe_normal(b-a);
    var basis_ref=vec3<f32>(0.0,1.0,0.0);
    if abs(dir.y)>0.9 { basis_ref=vec3<f32>(1.0,0.0,0.0); }
    let u=safe_normal(cross(dir,basis_ref)); let v=cross(dir,u);
    let corner=vertex%6u;
    let ti=array<f32,6>(0.,1.,0.,0.,1.,1.);
    let si=array<u32,6>(0u,0u,1u,1u,0u,1u);
    let t=ti[corner];
    let angle=f32(vertex/6u+si[corner])*6.28318530718/12.0;
    let normal=u*cos(angle)+v*sin(angle);
    // Remove the cylinder buried in each atom, even when the atoms are transparent.
    // Rulers without atom spheres retain their full center-to-center length.
    let radii=vec2<f32>(atoms[pair.x].w,atoms[pair.y].w)*object.params.z*object.style.w;
    let offsets=sqrt(max(radii*radii-vec2<f32>(object.params.w*object.params.w),vec2<f32>(0.0)));
    let length=distance(a,b);
    let exposed=max(length-offsets.x-offsets.y,0.0);
    let along=offsets.x+t*exposed;
    let color_t=clamp(along/max(length,1e-8),0.0,1.0);
    var out:Bond;
    out.p=world(a+dir*along+normal*object.params.w);
    out.clip=camera.vp*vec4<f32>(out.p,1.0);
    out.normal=world_normal(normal);
    out.color=mix(tint(pair.x,atom_color(pair.x)),tint(pair.y,atom_color(pair.y)),color_t);
    out.sphere_a=vec4<f32>(world(a),radii.x*object.style.z);
    out.sphere_b=vec4<f32>(world(b),radii.y*object.style.z);
    // A bond disappears with its less-visible endpoint; no dangling half-bonds.
    out.opacity=select(0.0,min(stick_opacity(pair.x),stick_opacity(pair.y)),exposed>1e-8);
    return out;
}

fn inside_bond_atom(in:Bond) -> bool {
    // A polygonal ring's chords dip inside the analytic sphere. Clip those last
    // fragments exactly, so both opaque and faded junctions share the same surface.
    let a=in.p-in.sphere_a.xyz; let b=in.p-in.sphere_b.xyz;
    return dot(a,a)<in.sphere_a.w*in.sphere_a.w || dot(b,b)<in.sphere_b.w*in.sphere_b.w;
}

@fragment fn bond_fragment(in:Bond) -> @location(0) vec4<f32> {
    if visible(in.opacity,in.p)<1.0 || inside_bond_atom(in) { discard; }
    return shade(in.p,in.normal,in.color);
}

@fragment fn bond_transparent(in:Bond) -> TransparentPixel {
    let alpha=visible(in.opacity,in.p);
    if alpha<=0.0 || alpha>=1.0 || inside_bond_atom(in) { discard; }
    return transparent(shade(in.p,in.normal,in.color).rgb,alpha,in.p);
}

@fragment fn surface_fragment(in:Surface) -> @location(0) vec4<f32> {
    if surface_visible(in.opacity,in.p)<1.0 { discard; }
    return shade(in.p,in.normal,in.color);
}

@fragment fn surface_transparent(in:Surface) -> TransparentPixel {
    let alpha=surface_visible(in.opacity,in.p);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    return transparent(shade(in.p,in.normal,in.color).rgb,alpha,in.p);
}

@fragment fn mesh_fragment(in:Surface) -> @location(0) vec4<f32> {
    if surface_visible(in.opacity,in.p)<1.0 { discard; }
    return shade(in.p,in.normal,in.color);
}

@fragment fn mesh_transparent(in:Surface) -> TransparentPixel {
    let alpha=surface_visible(in.opacity,in.p);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    return transparent(shade(in.p,in.normal,in.color).rgb,alpha,in.p);
}

// Nucleotide planes and rods read the same interpolated atoms and appearance tracks.
// Shape buffers stay resident through rotation, deformation, and trajectory playback.
@vertex fn base_vertex(@location(0) local:vec3f, @location(1) normal:vec3f,
                       @location(2) base:vec3f, @location(3) owner:u32,
                       @location(4) frame:vec3u, @location(5) mode:u32) -> Surface {
    let origin=position(frame.x);
    var x=safe_normal(position(frame.y)-origin);
    var z=safe_normal(cross(x,position(frame.z)-origin));
    var y=cross(z,x);
    if mode>0u {
        let axis=x;
        var basis=vec3f(0,1,0);
        if abs(axis.y)>0.9 { basis=vec3f(1,0,0); }
        x=safe_normal(cross(axis,basis));
        y=cross(axis,x);
        z=position(frame.y)-origin;
    }
    var out:Surface;
    out.p=world(origin+x*local.x+y*local.y+z*local.z);
    out.clip=camera.vp*vec4f(out.p,1.0);
    out.normal=world_normal(x*normal.x+y*normal.y+safe_normal(z)*normal.z);
    out.color=tint(owner,base);
    out.opacity=atom_opacity(owner);
    return out;
}

struct Sphere {
    @builtin(position) clip:vec4<f32>,
    @location(0) plane:vec3<f32>,
    @location(1) @interpolate(flat) center:vec3<f32>,
    @location(2) @interpolate(flat) color:vec3<f32>,
    @location(3) @interpolate(flat) radius:f32,
    @location(4) @interpolate(flat) opacity:f32,
};
@vertex fn sphere_vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32) -> Sphere {
    let corners=array<vec2<f32>,6>(vec2<f32>(-1.,-1.),vec2<f32>(1.,-1.),vec2<f32>(-1.,1.),
                                  vec2<f32>(-1.,1.),vec2<f32>(1.,-1.),vec2<f32>(1.,1.));
    let center=world(position(instance));
    let radius=atoms[instance].w*object.params.z*object.style.z;
    let delta=camera.eye_fog_start.xyz-center;
    let distance2=dot(delta,delta);
    let facing=safe_normal(delta);
    var basis_ref=vec3<f32>(0.,1.,0.);
    if abs(facing.y)>0.95 { basis_ref=vec3<f32>(1.,0.,0.); }
    let u=safe_normal(cross(basis_ref,facing)); let v=cross(facing,u);
    let bound=radius*sqrt(distance2/max(distance2-radius*radius,0.001));
    var out:Sphere;
    out.plane=center+(u*corners[vertex].x+v*corners[vertex].y)*bound;
    out.clip=camera.vp*vec4<f32>(out.plane,1.0);
    out.center=center; out.radius=radius; out.color=tint(instance,atom_color(instance));
    out.opacity=stick_opacity(instance);
    return out;
}
struct SpherePixel { @location(0) color:vec4<f32>, @builtin(frag_depth) depth:f32 };
struct SphereHit { p:vec3<f32>, depth:f32 };
fn sphere_hit(in:Sphere) -> SphereHit {
    let ray=safe_normal(in.plane-camera.eye_fog_start.xyz);
    let origin=camera.eye_fog_start.xyz-in.center;
    let b=dot(origin,ray);
    let discriminant=b*b-dot(origin,origin)+in.radius*in.radius;
    if discriminant<0.0 { discard; }
    let t=-b-sqrt(discriminant);
    if t<0.0 { discard; }
    let p=camera.eye_fog_start.xyz+ray*t;
    let clip=camera.vp*vec4<f32>(p,1.0);
    var out:SphereHit;
    out.depth=clip.z/clip.w;
    out.p=p;
    return out;
}

@fragment fn sphere_fragment(in:Sphere) -> SpherePixel {
    if opacity(in.opacity)<1.0 { discard; }
    let hit=sphere_hit(in);
    if visible(in.opacity,hit.p)<1.0 { discard; }
    var out:SpherePixel;
    out.depth=hit.depth;
    out.color=shade(hit.p,(hit.p-in.center)/in.radius,in.color);
    return out;
}

struct TransparentSphere {
    @location(0) accumulation:vec4<f32>,
    @location(1) revealage:f32,
    @builtin(frag_depth) depth:f32,
};
@fragment fn sphere_transparent(in:Sphere) -> TransparentSphere {
    let hit=sphere_hit(in);
    let alpha=visible(in.opacity,hit.p);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    let color=shade(hit.p,(hit.p-in.center)/in.radius,in.color).rgb;
    let blended=transparent(color,alpha,hit.p);
    var out:TransparentSphere;
    out.depth=hit.depth;
    out.accumulation=blended.accumulation;
    out.revealage=blended.revealage;
    return out;
}

@group(2) @binding(0) var<storage, read> mesh_reference:array<vec4f>;
@vertex fn mesh_vertex(@location(0) point:vec3f, @location(1) normal:vec3f,
                      @location(2) neighbors:vec4u, @location(3) weights:vec4f,
                      @location(4) base:vec3f, @location(5) owner:u32) -> Surface {
    var offset=vec3f(0.0);
    var gradient=vec3f(0.0);
    var g:array<vec3f,4>;
    var delta:array<vec3f,4>;
    for(var j=0u;j<4u;j++) {
        let reference_point=mesh_reference[neighbors[j]].xyz;
        let v=point-reference_point;
        g[j]=-2.0*v/(dot(v,v)+4.0);
        delta[j]=position(neighbors[j])-reference_point;
        offset+=weights[j]*delta[j];
        gradient+=weights[j]*g[j];
    }
    var jac=mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1));
    for(var j=0u;j<4u;j++) {
        let grad=weights[j]*(g[j]-gradient);
        jac+=mat3x3f(delta[j]*grad.x,delta[j]*grad.y,delta[j]*grad.z);
    }
    let cofactor=mat3x3f(cross(jac[1],jac[2]),cross(jac[2],jac[0]),cross(jac[0],jac[1]));
    var out:Surface;
    out.p=world(point+offset);
    out.clip=camera.vp*vec4f(out.p,1.0);
    out.normal=world_normal(cofactor*normal);
    out.color=tint(owner,base);
    out.opacity=atom_opacity(owner);
    return out;
}

// General triangle meshes use only the camera and object uniform bindings.
@vertex fn density_vertex(@location(0) p:vec3f, @location(1) n:vec3f,
                          @location(2) c:vec3f) -> Surface {
    var out:Surface;
    out.p=world(p); out.clip=camera.vp*vec4f(out.p,1.0);
    out.normal=world_normal(n); out.color=c; out.opacity=1.0;
    return out;
}
fn density_color(in:Surface) -> vec3f {
    if object.appearance.z>0.5 { return in.color; }
    return shade(in.p,in.normal,in.color).rgb;
}
@fragment fn density_fragment(in:Surface) -> @location(0) vec4f {
    if visible(in.opacity,in.p)<1.0 { discard; }
    return vec4f(density_color(in),1.0);
}
@fragment fn density_transparent(in:Surface) -> TransparentPixel {
    let alpha=visible(in.opacity,in.p);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    return transparent(density_color(in),alpha,in.p);
}

// Tunnel wall: a procedural open cylinder from the target to the outer surface of the
// molecule, with a ring every `spacing` Å measured inward from that surface.
struct TunnelWall {
    @builtin(position) clip:vec4f,
    @location(0) p:vec3f,
    @location(1) normal:vec3f,
    @location(2) below:f32,   // Å below the outer surface
};
const TUNNEL_SIDES:u32=72u;
const TUNNEL_SEGMENTS:u32=64u;
@vertex fn tunnel_vertex(@builtin(vertex_index) vertex:u32) -> TunnelWall {
    let quad=vertex/6u;
    let corner=vertex%6u;
    let cx=array<f32,6>(0.,1.,0.,0.,1.,1.);
    let cy=array<f32,6>(0.,0.,1.,1.,0.,1.);
    let segment=f32(quad/TUNNEL_SIDES)+cy[corner];
    let side=f32(quad%TUNNEL_SIDES)+cx[corner];
    let axis=tunnel_axis();
    var reference=vec3f(0.,1.,0.);
    if abs(axis.y)>0.9 { reference=vec3f(1.,0.,0.); }
    let u=safe_normal(cross(axis,reference));
    let v=cross(axis,u);
    // The wall reaches down to where cartoons and surfaces stop being carved.
    let keep=camera.cutaway_surface.x;
    let outer=camera.cutaway_tunnel.y;
    let height=mix(keep,outer,segment/f32(TUNNEL_SEGMENTS));
    let angle=side/f32(TUNNEL_SIDES)*6.28318530718;
    let radial=u*cos(angle)+v*sin(angle);
    let radius=camera.cutaway_shape.x*camera.cutaway_shape.y;
    var out:TunnelWall;
    out.p=camera.cutaway.xyz+axis*height+radial*radius;
    out.clip=camera.vp*vec4f(out.p,1.0);
    out.normal=-radial;
    out.below=outer-height;
    return out;
}
@fragment fn tunnel_transparent(in:TunnelWall) -> TransparentPixel {
    let spacing=camera.cutaway_tunnel.w;
    let f=in.below/spacing;
    // Antialiased rings, a fixed width in pixels at any distance.
    let minor=1.0-clamp(abs(fract(f+0.5)-0.5)/max(fwidth(f),1e-4)-0.6,0.0,1.0);
    // Every second ring is a major tick, drawn wider and brighter.
    let g=f*0.5;
    let major=1.0-clamp(abs(fract(g+0.5)-0.5)/max(fwidth(g),1e-4)-1.2,0.0,1.0);
    let line=max(minor*0.55,major);
    let length_total=max(camera.cutaway_tunnel.y-camera.cutaway_surface.x,1e-3);
    let deep=clamp(in.below/length_total,0.0,1.0);
    let base=mix(vec3f(0.42,0.62,0.78),vec3f(0.95,0.73,0.40),deep);
    let v=safe_normal(camera.eye_fog_start.xyz-in.p);
    let facing=abs(dot(safe_normal(in.normal),v));
    let key=safe_normal(vec3f(-0.45,0.8,0.85));
    let light=0.45+0.4*abs(dot(safe_normal(in.normal),key))+0.15*facing;
    var color=mix(base*light,mix(base,vec3f(1.0),0.55),line*0.8);
    let d=distance(camera.eye_fog_start.xyz,in.p);
    let fog=smoothstep(camera.eye_fog_start.w,camera.background_fog_end.w,d)*camera.lighting.x;
    color=mix(color,camera.background_fog_end.xyz,fog);
    // The wall fades in at its ends so it does not end in a hard edge.
    let ends=smoothstep(0.0,2.0,in.below)*smoothstep(0.0,2.0,length_total-in.below);
    // Like glass, the wall is most visible where it is seen at a grazing angle.
    let wall=camera.cutaway_tunnel.z*(0.35+0.65*(1.0-facing));
    // Fade the wall near the camera so flying down the tunnel does not veil the view.
    let near=smoothstep(4.0,22.0,distance(camera.eye_fog_start.xyz,in.p));
    let alpha=clamp((wall+line*0.45)*ends*near,0.0,0.9);
    if alpha<=0.001 { discard; }
    return transparent(color,alpha,in.p);
}
