export const frag_vert_shader = /*wgsl*/ `

    struct RubixRenderState {
        ratio: f32,
        scale: f32,
        origin: f32,
        cursorX: f32,
        cursorY: f32,
        cursorZ: f32,
    }

    @binding(0) @group(0) var<storage, read> pieceColoring: array<array<u32, 6>>;
    @binding(1) @group(0) var<storage, read> piecePositions: array<vec4f>;
    @binding(2) @group(0) var<storage, read> pieceIsRotating: array<u32>;
    @binding(3) @group(0) var<storage, read> rotatingCenterIds: array<i32>;
    @binding(4) @group(0) var<storage, read> rotation_interpolation: vec2<f32>;

    @binding(0) @group(1) var<storage, read> renderState: RubixRenderState;
    @binding(1) @group(1) var<storage, read> faceColors: array<vec4f>;

    
    struct VertexOutput {
        @builtin(position) position : vec4<f32>,
        @location(1) @interpolate(flat) indices: vec2<u32>,
    }
    
    @vertex
    fn vert_main(@builtin(instance_index) i: u32, 
    @location(0) vertex: vec3<f32>,
    @location(1) triangle_index: f32,
) -> VertexOutput {
        var xRot: f32 = radians(2 * (renderState.cursorX - 0.5) * 360);
        var yRot: f32 = radians(2 * (renderState.cursorY - 0.5) * 360);
        var zRot: f32 = radians(2 * (renderState.cursorZ - 0.5) * 360);
        var output : VertexOutput;
        var og = vec3(renderState.origin);
        var rotated: vec3f = vertex + 1.05 * (piecePositions[i].xyz - og);

        if (pieceIsRotating[i] == 1) {
            var center = vec3(0.);
            var len = 0;
            for (var i: u32 = 0; i < arrayLength(&rotatingCenterIds); i++) {
                if (rotatingCenterIds[i] == -1) {
                    break;
                }
                len++;
                center += piecePositions[rotatingCenterIds[i]].xyz;
            }
            center /= f32(len);

            switch u32(rotation_interpolation.y) {
                case 0: {
                    rotated = rotateY(radians(rotation_interpolation.x), rotated, 1.05 * (center - og));
                }
                case 1: {
                    rotated = rotateZ(radians(rotation_interpolation.x), rotated, 1.05 * (center - og));
                }
                case 2: {
                    rotated = rotateX(radians(rotation_interpolation.x), rotated, 1.05 * (center - og));
                }
                default: {}
            }
        }
        
        rotated = 
        rotateZ(zRot, rotateX(xRot, rotateY(yRot, rotated, vec3(0)), vec3(0)), vec3(0));

        output.indices = vec2(i, u32(triangle_index));
        output.position = vec4(renderState.scale * fixPoint(rotated) + vec3(0, 0, 0.8), 2.);
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
        return getColor(frag);
    }

    fn getColor(frag: VertexOutput) -> vec4<f32> {
        var coloring = pieceColoring[frag.indices.x];
        var triangleIndex = frag.indices.y;

        for (var i: u32 = 0; i < 6; i++) {
            if (triangleIndex >= i * 2 && triangleIndex < (i + 1) * 2 && coloring[i] != 6) {
                return faceColors[coloring[i]];
            }
        }

        return vec4(0, 0, 0, 1);
    }
`;
