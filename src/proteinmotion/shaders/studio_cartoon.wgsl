// Classic protein cartoons. Atom coordinates are untouched: the display spline
// is evaluated from the current interpolated CA positions on the GPU.
fn cartoon_curve(a:vec3f,b:vec3f,c:vec3f,d:vec3f,t:f32) -> vec3f {
    let u=1.0-t;
    return (u*u*u*a+(3.0*t*t*t-6.0*t*t+4.0)*b+
            (-3.0*t*t*t+3.0*t*t+3.0*t+1.0)*c+t*t*t*d)/6.0;
}
fn cartoon_tangent(a:vec3f,b:vec3f,c:vec3f,d:vec3f,t:f32) -> vec3f {
    let u=1.0-t;
    return safe_normal(-u*u*a+(3.0*t*t-4.0*t)*b+(-3.0*t*t+2.0*t+1.0)*c+t*t*d);
}
fn cartoon_exponent(ss:f32) -> f32 {
    // Superellipses: circles for loops (0), rounded flat faces for helices (1) and
    // sheets (2). Codes are continuous so the profile blends as structure changes.
    if ss>1.0 { return mix(0.5,0.33,clamp(ss-1.0,0.0,1.0)); }
    return mix(1.0,0.5,clamp(ss,0.0,1.0));
}
struct CartoonFrame {
    center:vec3f,
    width_axis:vec3f,
    height_axis:vec3f,
    size:vec2f,
    exponent:f32,
};
fn cartoon_frame(s:Segment,t:f32) -> CartoonFrame {
    let a=position(s.atoms.x); let b=position(s.atoms.y);
    let c=position(s.atoms.z); let d=position(s.atoms.w);
    let center=cartoon_curve(a,b,c,d,t);
    let direction=cartoon_tangent(a,b,c,d,t);
    let frame=segment_guide(s,t,1.0);
    let width_axis=safe_normal(frame.xyz-direction*dot(frame.xyz,direction));
    let height_axis=safe_normal(cross(direction,width_axis));
    let blend=smoothstep(0.0,1.0,t);
    var size=mix(s.shape.xz,s.shape.yw,blend);
    var exponent=mix(cartoon_exponent(s.color_a.w),cartoon_exponent(s.color_b.w),blend);
    let arrow=clamp(s.caps.z,0.0,1.0);
    if arrow>0.0 {
        // A short shoulder followed by a long taper makes a real sheet arrow,
        // rather than widening the last residue of an elliptical ribbon. The arrow
        // weight grows and shrinks with the strand during secondary-structure changes.
        let body=mix(s.shape.y,s.shape.x,clamp(s.color_a.w-1.0,0.0,1.0));
        let ordered=smoothstep(0.25,0.75,s.color_b.w);
        let tip=mix(s.shape.y,body*0.15,ordered);
        let shoulder=mix(body,body*1.75,smoothstep(0.08,0.16,t));
        var head=size;
        head.x=mix(shoulder,tip,clamp((t-0.16)/0.84,0.0,1.0));
        head.y=mix(s.shape.z,s.shape.w,smoothstep(0.85,1.0,t));
        let head_exponent=mix(0.33,cartoon_exponent(s.color_b.w),smoothstep(0.85,1.0,t));
        // Direct sheet-to-helix boundaries connect through a short neck.
        if s.caps.y<0.5 {
            head.x=mix(head.x,mix(head.x,s.shape.y,smoothstep(0.85,1.0,t)),ordered);
        }
        size=mix(size,head,arrow);
        exponent=mix(exponent,head_exponent,arrow);
    }
    // Round, rather than flip, where the twist to the next residue is ambiguous.
    size.x=mix(size.y,size.x,frame.w);
    exponent=mix(1.0,exponent,frame.w);
    var cap=1.0;
    if s.caps.x>0.5 { cap*=smoothstep(0.0,0.1,t); }
    if s.caps.y>0.5 { cap*=smoothstep(0.0,0.1,1.0-t); }
    return CartoonFrame(center,width_axis,height_axis,size*cap,exponent);
}
fn cartoon_offset(f:CartoonFrame,angle:f32) -> vec3f {
    let circle=vec2f(cos(angle),sin(angle));
    let profile=sign(circle)*pow(abs(circle),vec2f(f.exponent))*f.size;
    return f.width_axis*profile.x+f.height_axis*profile.y;
}

@vertex fn ribbon_vertex(@location(0) uv:vec2f,@builtin(instance_index) instance:u32) -> Surface {
    let s=segments[instance];
    // Uniform ribbons and nucleotide tubes keep their original geometry.
    if object.style.x<=0.0 || s.color_a.w<0.0 {
        return ribbon_sweep(uv,instance,object.style.x);
    }
    let t=uv.x; let angle=uv.y*6.28318530718;
    let f=cartoon_frame(s,t);
    let point=f.center+cartoon_offset(f,angle);
    let before=cartoon_frame(s,max(0.0,t-0.001));
    let after=cartoon_frame(s,min(1.0,t+0.001));
    let along=(after.center+cartoon_offset(after,angle)-before.center-cartoon_offset(before,angle))/0.002;
    let around=(cartoon_offset(f,angle+0.001)-cartoon_offset(f,angle-0.001))/0.002;
    var normal=cross(around,along);
    if dot(normal,normal)<1e-14 {
        normal=select(-along,along,t>0.5);
    }
    var out:Surface;
    out.p=world(point);
    out.normal=world_normal(normal);
    if object.style.x<1.0 {
        let ribbon=ribbon_sweep(uv,instance,0.0);
        out.p=mix(ribbon.p,out.p,object.style.x);
        out.normal=safe_normal(mix(ribbon.normal,out.normal,object.style.x));
    }
    out.clip=camera.vp*vec4f(out.p,1.0);
    out.color=mix(tint(s.atoms.y,s.color_a.xyz),tint(s.atoms.z,s.color_b.xyz),t);
    out.opacity=mix(atom_opacity(s.atoms.y),atom_opacity(s.atoms.z),t);
    return out;
}
