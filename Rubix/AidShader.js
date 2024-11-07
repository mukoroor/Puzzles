export const frag_vert_aid_shader = /*wgsl*/ `
    diagnostic(off,derivative_uniformity);

    struct RubixRenderState {
        ratio: f32,
        scale: f32,
        origin: f32,
        cursorX: f32,
        cursorY: f32,
        cursorZ: f32,
    }

    @binding(0) @group(0) var<storage, read> renderState: RubixRenderState;
    @binding(1) @group(0) var mySampler: sampler;
    @binding(2) @group(0) var face0Texture: texture_2d<f32>;
    @binding(3) @group(0) var face1Texture: texture_2d<f32>;
    @binding(4) @group(0) var face2Texture: texture_2d<f32>;
    @binding(5) @group(0) var face3Texture: texture_2d<f32>;
    @binding(6) @group(0) var face4Texture: texture_2d<f32>;
    @binding(7) @group(0) var face5Texture: texture_2d<f32>;

    struct VertexOutput {
        @builtin(position) position : vec4<f32>,
        @location(1) @interpolate(linear) uv: vec2<f32>,
        @location(2) @interpolate(flat) faceId: u32
    }
    
    @vertex
    fn vert_main(@location(0) vertex: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) id: f32) -> VertexOutput {
        var xRot: f32 = radians(2 * (renderState.cursorX - 0.5) * 360);
        var yRot: f32 = radians(2 * (renderState.cursorY - 0.5) * 360);
        var zRot: f32 = radians(2 * (renderState.cursorZ - 0.5) * 360);
        var output : VertexOutput;
        var rotated: vec3f = 
        rotateZ(zRot, rotateX(xRot, rotateY(yRot, vertex, vec3(0)), vec3(0)), vec3(0));

        output.position = vec4(1.1 * 0.8 * fixPoint(rotated) + vec3(0, 0, 0), 2);
        output.uv = uv;
        output.faceId = u32(id);
        return output;
    }

    fn rotateX(angle: f32, vector:  vec3f, origin: vec3f) -> vec3<f32> {
        return origin + mat3x3(vec3(1., 0., 0.), vec3(0., cos(angle), sin(angle)), vec3(0., -sin(angle), cos(angle))) * (vector - origin);
    }
    
    fn rotateY(angle: f32, vector:  vec3f, origin: vec3f) -> vec3<f32> {
        return origin + mat3x3(vec3(cos(angle), 0, -sin(angle)), vec3(0, 1, 0), vec3(sin(angle), 0., cos(angle))) * (vector - origin);
    }

    fn rotateZ(angle: f32, vector:  vec3f, origin: vec3f) -> vec3<f32> {
        return origin + mat3x3(vec3(cos(angle), sin(angle), 0), vec3(-sin(angle), cos(angle), 0), vec3(0., 0., 1.)) * (vector - origin);
    }

    fn fixPoint(vector: vec3<f32>) -> vec3<f32> {
        return vec3((vector.x) / renderState.ratio, vector.y, vector.z + 1);
    }
    
    @fragment
    fn frag_main(frag: VertexOutput) -> @location(0) vec4f {
        // return vec4(0.5 * frag.position.z, 0.5 * frag.position.z, 0.5 * frag.position.z, 1.);
        var col: vec4f;
        if (frag.faceId == 0) { col = textureSample(face0Texture, mySampler, frag.uv); }
        else if (frag.faceId == 1) { col = textureSample(face1Texture, mySampler, vec2(1-frag.uv.x, frag.uv.y)); }
        else if (frag.faceId == 2) { col = textureSample(face2Texture, mySampler,vec2(frag.uv.y, frag.uv.x)); }
        else if (frag.faceId == 3) { col = textureSample(face3Texture, mySampler, vec2(1-frag.uv.y, 1-frag.uv.x)); }
        else if (frag.faceId == 4) { col = textureSample(face4Texture, mySampler, vec2(frag.uv.x, 1-frag.uv.y)); }
        else { col = textureSample(face5Texture, mySampler, vec2(1-frag.uv.x, frag.uv.y)); }
        if (length(col - vec4(0.)) == 0) {discard;}
        // return 0.9 * vec4(0.97, 0.008, 0.49, 1.);
        return 0.9 * vec4(0., 0.76, 0.98, 1.);
        // return vec4(1., 0., 0., 1.);
    }
`;
