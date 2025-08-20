export const TYPE_CONVERSION_ENTRY_POINT = 'convert'

const typeConv =  (originalType, finalType) => /*wgsl*/`
    @group(0) @binding(0) var<storage> data: array<vec4<${originalType}>>;
    @group(0) @binding(1) var<storage, read_write> dataDest: array<vec4<${originalType}>>;
    
    @compute @workgroup_size(16 * 16)
    fn ${TYPE_CONVERSION_ENTRY_POINT}(
        @builtin(global_invocation_id) id : vec3<u32>
    ) {
        if (id.x < arrayLength(&dataDest)) {
            dataDest[id.x] = bitcast<vec4<${originalType}>>(vec4<${finalType}>(data[id.x])); 
        }
    }
`
const typeMap = {
    Float32Array: 'f32',
    Int32Array: 'i32',
    Uint32Array: 'u32'
};

export function getConverisonShader(currentType, targetType) {
    const originalType = typeMap[currentType.name];
    const finalType = typeMap[targetType.name];

    if (!originalType || !finalType) {
        throw new Error('Unsupported type conversion');
    }

    const cacheKey = `${originalType}->${finalType}`;
    if (shaderCache.has(cacheKey)) {
        return shaderCache.get(cacheKey);
    }

    const shader = typeConv(originalType, finalType);
    shaderCache.set(cacheKey, shader);
    return shader;
}


