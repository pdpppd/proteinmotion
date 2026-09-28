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
    // Superellipses: circles for loops, rounded flat faces for helices/sheets.
    if ss>1.5 { return 0.33; }
    if ss>0.5 { return 0.5; }
    return 1.0;
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
    let g=cartoon_curve(guide(s.atoms.x),guide(s.atoms.y),guide(s.atoms.z),guide(s.atoms.w),t);
    let width_axis=safe_normal(g-direction*dot(g,direction));
    let height_axis=safe_normal(cross(direction,width_axis));
    let blend=smoothstep(0.0,1.0,t);
    var size=mix(s.shape.xz,s.shape.yw,blend);
    var exponent=mix(cartoon_exponent(s.color_a.w),cartoon_exponent(s.color_b.w),blend);
    if s.caps.z>0.5 {
        // A short shoulder followed by a long taper makes a real sheet arrow,
        // rather than widening the last residue of an elliptical ribbon.
        let body=select(s.shape.x,s.shape.y,s.color_a.w<1.5);
        let tip=select(s.shape.y,body*0.15,s.color_b.w>0.5);
        let shoulder=mix(body,body*1.75,smoothstep(0.08,0.16,t));
        size.x=mix(shoulder,tip,clamp((t-0.16)/0.84,0.0,1.0));
        size.y=mix(s.shape.z,s.shape.w,smoothstep(0.85,1.0,t));
        exponent=mix(0.33,cartoon_exponent(s.color_b.w),smoothstep(0.85,1.0,t));
        // Direct sheet-to-helix boundaries connect through a short neck.
        if s.color_b.w>0.5 && s.caps.y<0.5 {
            size.x=mix(size.x,s.shape.y,smoothstep(0.85,1.0,t));
        }
    }
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
