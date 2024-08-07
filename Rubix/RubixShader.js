export const frag_vert_shader = /*wgsl*/ `
    override ratio: f32 = 1.;
    override scale: f32 = 1.;
    override origin: f32 = 1.; 

    override x: f32 = 0;
    override y: f32 = 0;
    override z: f32 = 0;
    override xRot: f32 = radians(2 * (x - 0.5) * 360);
    override yRot: f32 = radians(2 * (y - 0.5) * 360);
    override zRot: f32 = radians(2 * (z - 0.5) * 360);

    const colors = array<vec3f, 6>(vec3(1., 1., 1.), vec3(0., 0.5, 0.), vec3(1., 0.55, 0.), vec3(1., 0., 0.), vec3(0., 0., 1.), vec3(1., 1., 0.));

    struct VertexOutput {
        @builtin(position) position : vec4<f32>,
        @location(0) @interpolate(flat) uv: vec2<f32>,
        @location(1) @interpolate(flat) i: vec2<u32>,
        @location(2) @interpolate(flat) face012: vec3<u32>,
        @location(3) @interpolate(flat) face345: vec3<u32>,

    }

    @vertex
    fn vert_main(@builtin(instance_index) i: u32, 
                    @location(0) vertex: vec3<f32>,
                    @location(1) triangle_index: f32,
                    @location(2) pos: vec3<f32>,
                    @location(3) faceColors012: vec3<u32>,
                    @location(4) faceColors345: vec3<u32>) -> VertexOutput {
        var output : VertexOutput;

        
        
        if (faceColors012.x == 7) {
            output.face012 = vec3(0, 1, 2);
            output.face345 = vec3(3, 4, 5);
        } else {
            output.face012 = faceColors012;
            output.face345 = faceColors345;
        }

        
        var rotated = 
        mat3x3(vec3(cos(zRot), sin(zRot), 0), vec3(-sin(zRot), cos(zRot), 0), vec3(0., 0., 1.))
        *
        mat3x3(vec3(1., 0., 0.), vec3(0., cos(xRot), sin(xRot)), vec3(0., -sin(xRot), cos(xRot)))
        *
        mat3x3(vec3(cos(yRot), 0, -sin(yRot)), vec3(0, 1, 0), vec3(sin(yRot), 0., cos(yRot)))
        * 
        (vertex + 1.05 * (pos - vec3(origin)));

        output.uv = vec2(triangle_index);
        output.i.x = i;
        output.position = vec4(scale * fixPoint(rotated) + vec3(0, 0, 0.8), 2.);
        return output;
    }

    fn fixPoint(vector: vec3<f32>) -> vec3<f32> {
        return vec3((vector.x) / ratio, vector.y, vector.z + 1);
    }
    
    @fragment
    fn frag_main(frag: VertexOutput) -> @location(0) vec4f {
        return vec4(getColor(frag), 1.);
    }

    fn getColor(frag: VertexOutput) -> vec3<f32> {
        if (frag.uv.y < 2 && frag.face012.x != 6) {
            return colors[frag.face012.x];
        } else if (frag.uv.y < 4 && frag.uv.y > 1 && frag.face012.y != 6) {
            return colors[frag.face012.y];
        } else if (frag.uv.y < 6 && frag.uv.y > 3 &&  frag.face012.z != 6) {
            return colors[frag.face012.z];
        } else if (frag.uv.y < 8 && frag.uv.y > 5 && frag.face345.x != 6) {
            return colors[frag.face345.x];
        } else if (frag.uv.y < 10 && frag.uv.y > 7 && frag.face345.y != 6) {
            return colors[frag.face345.y];
        } else if (frag.uv.y < 12 && frag.uv.y > 9 && frag.face345.z != 6) {
            return colors[frag.face345.z];
        }
        return vec3(0);
    }
`;
