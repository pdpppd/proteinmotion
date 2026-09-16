struct Camera {
    vp: mat4x4<f32>,
    eye_fog_start: vec4<f32>,
    background_fog_end: vec4<f32>,
    lighting: vec4<f32>,
};
struct Object {
    model: mat4x4<f32>,
    params: vec4<f32>, // tween, opacity, atom scale, bond radius
    style: vec4<f32>,  // cartoon proportion, ribbon width, object scale, reserved
};
struct AtomState { position: vec4<f32>, guide: vec4<f32> };
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
@group(1) @binding(3) var<storage, read> atoms: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> bonds: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> segments: array<Segment>;
@group(1) @binding(6) var<storage, read> controls: array<Control>;

fn safe_normal(v: vec3<f32>) -> vec3<f32> {
    return v * inverseSqrt(max(dot(v,v), 0.00000001));
}
fn ease(t:f32) -> f32 { return t*t*t*(10.0+t*(-15.0+6.0*t)); }
fn atom_progress(i:u32) -> f32 {
    let c=controls[i].motion;
    if c.z<0.5 { return object.params.x; }
    let t=clamp((object.params.x-c.x)/max(c.y,0.00000001),0.0,1.0);
    if c.z>1.5 { return ease(t); }
    return t;
}
fn atom_opacity(i:u32) -> f32 {
    let c=controls[i].visibility;
    let t=ease(clamp((object.params.x-c.z)/max(c.w,0.00000001),0.0,1.0));
    return mix(c.x,c.y,t);
}
fn position(i: u32) -> vec3<f32> {
    return mix(state_a[i].position.xyz, state_b[i].position.xyz, atom_progress(i));
}
fn guide(i: u32) -> vec3<f32> {
    var b = state_b[i].guide.xyz;
    let a = state_a[i].guide.xyz;
    if dot(a,b) < 0.0 { b = -b; }
    return mix(a,b,atom_progress(i));
}
fn world(p: vec3<f32>) -> vec3<f32> { return (object.model*vec4<f32>(p,1.0)).xyz; }
fn world_normal(n: vec3<f32>) -> vec3<f32> { return safe_normal((object.model*vec4<f32>(n,0.0)).xyz); }

fn opacity(value:f32) -> f32 {
    return clamp(object.params.y*value,0.0,1.0);
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
    out.color=mix(s.color_a.xyz,s.color_b.xyz,t);
    out.opacity=mix(atom_opacity(s.atoms.y),atom_opacity(s.atoms.z),t);
    return out;
}

@vertex fn bond_vertex(@builtin(vertex_index) vertex:u32, @builtin(instance_index) instance:u32) -> Surface {
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
    var out:Surface;
    out.p=world(mix(a,b,t)+normal*object.params.w);
    out.clip=camera.vp*vec4<f32>(out.p,1.0);
    out.normal=world_normal(normal);
    out.color=mix(atoms[pair.x].xyz,atoms[pair.y].xyz,t);
    // A bond disappears with its less-visible endpoint; no dangling half-bonds.
    out.opacity=min(atom_opacity(pair.x),atom_opacity(pair.y));
    return out;
}

@fragment fn surface_fragment(in:Surface) -> @location(0) vec4<f32> {
    if opacity(in.opacity)<1.0 { discard; }
    return shade(in.p,in.normal,in.color);
}

@fragment fn surface_transparent(in:Surface) -> TransparentPixel {
    let alpha=opacity(in.opacity);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    return transparent(shade(in.p,in.normal,in.color).rgb,alpha,in.p);
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
    out.center=center; out.radius=radius; out.color=atoms[instance].xyz;
    out.opacity=atom_opacity(instance);
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
    let alpha=opacity(in.opacity);
    if alpha<=0.0 || alpha>=1.0 { discard; }
    let hit=sphere_hit(in);
    let color=shade(hit.p,(hit.p-in.center)/in.radius,in.color).rgb;
    let blended=transparent(color,alpha,hit.p);
    var out:TransparentSphere;
    out.depth=hit.depth;
    out.accumulation=blended.accumulation;
    out.revealage=blended.revealage;
    return out;
}
