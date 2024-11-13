import { UTIL_SHADER } from "./RubixShader.js";

export const frag_vert_aid_shader = /*wgsl*/ `
    diagnostic(off,derivative_uniformity);

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

    const BOUNDS_ARR = array(
        vec2(-1., 0),
        vec2(1., 0),
        vec2(0, 1.),
        vec2(0, -1.),
        vec2(-1, -1.),
        vec2(-1, 1.),
        vec2(1, -1.),
        vec2(1, 1.),
      );
        
    @vertex
    fn vert_main(@location(0) vertex: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) id: f32) -> VertexOutput {
        var xRot: f32 = radians(2 * (renderState.cursorY - 0.5) * 60);
        var yRot: f32 = radians(2 * (renderState.cursorX - 0.5) * 360);
        var output : VertexOutput;
        var rotated: vec3f = 
        rotateX(xRot, rotateY(yRot, vertex, vec3(0)), vec3(0));

        output.position = vec4(fixPoint(1.1 * rotated), 1);
        output.uv = uv;
        output.faceId = u32(id);
        return output;
    }
    
    @fragment
    fn frag_main(frag: VertexOutput) -> @location(0) vec4f {
        var bias = 0.007;
        var alpha = 0.8;
        if (frag.faceId == 0) { return getTextCol(frag.uv, face0Texture, bias, alpha); }
        else if (frag.faceId == 1) { return getTextCol(vec2(1-frag.uv.x, frag.uv.y), face1Texture, bias, alpha); }
        else if (frag.faceId == 2) { return getTextCol(vec2(frag.uv.y, frag.uv.x), face2Texture, bias, alpha); }
        else if (frag.faceId == 3) { return getTextCol(vec2(1-frag.uv.y, 1-frag.uv.x), face3Texture, bias, alpha); }
        else if (frag.faceId == 4) { return getTextCol(vec2(frag.uv.x, 1-frag.uv.y), face4Texture, bias, alpha); }
        else { return getTextCol(vec2(1-frag.uv.x, frag.uv.y), face5Texture, bias, alpha); }
    }

    fn getTextCol(uv: vec2f, texture: texture_2d<f32>, borderBias: f32, alpha: f32) -> vec4f {
        var edge = 0.4;
        var alphCop = alpha;
        var col = textureSample(texture, mySampler, uv);

        if (length(col - vec4(0.)) == 0) {discard;}
        col = col + vec4(0.6, 0.8, 1, 0);
        // for (var i: u32 = 0; i < 9; i++) {
        //     var samp = textureSample(texture, mySampler, uv + borderBias * BOUNDS_ARR[i]);
        //     if (length(samp - vec4(0.)) == 0) {
        //         col = vec4(0.);
        //         alphCop = 1;
        //         break;
        //     }
        // }

        return  vec4(col.xyz, alpha);
    }
` + UTIL_SHADER;
