const VEC_TYPES = {
    VEC2F: {
        wgsl: /*wgsl */`vec2f`,
        size: 2
    },
    VEC4F: {
        wgsl: /*wgsl */`vec4f`,
        size: 4
    },
    VEC2H: {
        wgsl: /*wgsl */`vec2h`,
        size: 2
    },
    VEC4H: {
        wgsl: /*wgsl */`vec4h`,
        size: 4
    },
    // VEC2I: {
    //     wgsl: /*wgsl */`vec2i`,
    //     size: 2
    // },
    // VEC4I: {
    //     wgsl: /*wgsl */`vec4i`,
    //     size: 4
    // },
    // VEC2U: {
    //     wgsl: /*wgsl */`vec2u`,
    //     size: 2
    // },
    // VEC4U: {
    //     wgsl: /*wgsl */`vec4u`,
    //     size: 4
    // },
}

const VEC_TYPE = VEC_TYPES.VEC4H

const A_NAME = 'A'
const B_NAME = 'B'
const C_NAME = 'C'

const IDX_A = 'unraveled_idx_a'
const IDX_B = 'unraveled_idx_b'
const IDX_C = 'unraveled_idx_c'

const IDX_MACRO = (baseName, i=0) => `${baseName}${i}`;
const IDX_INITIALIZE_MACRO = (baseName, stepName) => Array.from({length: VEC_TYPE.size - 1}, (_, i) => /*wgsl*/`var ${IDX_MACRO(baseName, i + 1)} = ${IDX_MACRO(baseName, i)} + ${stepName};`).join('\n\t\t')
const IDX_INCREMENT_MACRO = (baseName, stepName, op) => Array.from({length: VEC_TYPE.size}, (_, i) => /*wgsl*/`${IDX_MACRO(baseName, i)} += ${stepName};`).join('\n\t\t\t')

const BLOCK_ACCESS_MACRO = (baseName, sourceName) => Array.from({length: VEC_TYPE.size}, (_, i) => /*wgsl*/`${sourceName}[${IDX_MACRO(baseName, i)}],`).join('\n\t\t\t\t')
const BLOCK_ASSIGN_MACRO = (baseName, sourceName, destinationName) => Array.from({length: VEC_TYPE.size}, (_, i) => /*wgsl*/`${destinationName}[${IDX_MACRO(baseName, i)}] = ${sourceName}[${i}];`).join('\n\t\t\t')

export const SHADER_ENTRY_POINT = 'tiledMatMul'
export const TILE_SIZE = VEC_TYPE.size;
export const TILE_BLOCK_DIM = 16;

export const SHADER = /*wgsl*/`
    enable f16;

    struct Dims {
        aDims: vec2u,
        bDims: vec2u,
        cDims: vec2u
    }

    @group(0) @binding(0) var<storage> ${A_NAME}: array<${VEC_TYPE.wgsl}>;
    @group(0) @binding(1) var<storage> ${B_NAME}: array<${VEC_TYPE.wgsl}>;
    @group(0) @binding(2) var<storage, read_write> ${C_NAME}: array<${VEC_TYPE.wgsl}>;
    @group(0) @binding(3) var<uniform> productDims: Dims;

    
    const VEC_SIZE = ${VEC_TYPE.size};

    const TILE_SIZE = ${TILE_SIZE};
    const TILE_SIZE_VEC = TILE_SIZE / VEC_SIZE;
    
    const TILE_BLOCK_DIM = ${TILE_BLOCK_DIM};

    const STEP = TILE_SIZE * TILE_BLOCK_DIM;
    const STEP_VEC = STEP / VEC_SIZE;

    alias TileBlock = mat${VEC_TYPE.size}x${VEC_TYPE.size}h;

    @compute @workgroup_size(TILE_BLOCK_DIM, TILE_BLOCK_DIM)
    fn ${SHADER_ENTRY_POINT}(
        @builtin(global_invocation_id) id : vec3<u32>
    ) {
        let tot_row = id.y * TILE_SIZE;
        let tot_col = id.x;
        
        var aMat: TileBlock;
        var bMat: TileBlock;
        var cMat: TileBlock;
        
        let aD4 = productDims.aDims.y / VEC_SIZE;
        let bD4 = productDims.bDims.y / VEC_SIZE;
        let B_ROW_STEP = productDims.bDims.y * TILE_SIZE_VEC;

        var ${IDX_MACRO(IDX_A)} = tot_row * aD4;
        ${IDX_INITIALIZE_MACRO(IDX_A, 'aD4')}
        var ${IDX_MACRO(IDX_B)} = tot_col;
        ${IDX_INITIALIZE_MACRO(IDX_B, 'bD4')}
        var ${IDX_MACRO(IDX_C)} = tot_row * bD4 + tot_col;
        ${IDX_INITIALIZE_MACRO(IDX_C, 'bD4')}

        for (var i: u32 = 0; i < productDims.aDims.y / TILE_SIZE; i++) {   
            aMat = TileBlock(
                ${BLOCK_ACCESS_MACRO(IDX_A, A_NAME)}
            );

            bMat = TileBlock(
                ${BLOCK_ACCESS_MACRO(IDX_B, B_NAME)}
            );

            cMat += bMat * aMat;

            ${IDX_INCREMENT_MACRO(IDX_A, 'TILE_SIZE_VEC')}
            ${IDX_INCREMENT_MACRO(IDX_B, 'B_ROW_STEP')}
        }

        if (tot_row < productDims.aDims.x && tot_col < productDims.bDims.y / VEC_SIZE) {
            ${BLOCK_ASSIGN_MACRO(IDX_C, 'cMat', C_NAME)}
        }
    }
`
console.log(SHADER)
