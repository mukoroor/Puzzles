const typeConv =  (originalType, finalType) => /*wgsl*/`
    @group(0) @binding(0) var<storage> data: array<vec4<${originalType}>>;
    @group(0) @binding(1) var<storage, read_write> dataDest: array<vec4<${originalType}>>;
    
    @compute @workgroup_size(16 * 16)
    fn convert(
        @builtin(global_invocation_id) id : vec3<u32>
    ) {
        if (id.x < arrayLength(&dataDest)) {
            dataDest[id.x] = bitcast<vec4<${originalType}>>(vec4<${finalType}>(data[id.x])); 
        }
    }
`

export const F32_to_I32 = typeConv('f32', 'i32');

export const F32_to_U32 = typeConv('f32', 'u32');

export const I32_to_U32 = typeConv('i32', 'u32');

export const I32_to_F32 = typeConv('i32', 'f32');

export const U32_to_I32 = typeConv('u32', 'i32');

export const U32_to_F32 = typeConv('u32', 'f32');




