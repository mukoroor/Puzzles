import WGSLActivations , { X_VAR } from "./ActivationFunctions.js";
import { NETWORK_MODE } from "./NetworkConsts.js";

export const neural_net_shader = (network) => {
  const maxLayer = network.maxLayer;
  const neuronCountCumSum = [0, ...network.cumulativeSum];
  const weightStarts = [0, ...network.weightCounts];
  const { activations, derivatives, activationSwitch, derivativeSwitch } = WGSLActivations.createActivationShaderCode();

  return /*wgsl*/ `
        alias InputDataPoint = array<f32, inputDims>;
        alias OutputDataPoint = array<f32, outputDims>;

        alias SlopeOutput = array<f32, maxLayer>;
        alias WeightsArray = array<f32, totalWeights>;

        alias SingletonValsF32 = array<f32, totalNeurons>;
        alias SingletonValsU32 = array<u32, totalNeurons>;

        struct SingletonNeuronData {
            inputs: SingletonValsF32,
            biases: SingletonValsF32,
        }

        struct MaxContainer {
            val: f32,
            index: u32,
        }

        struct TrainParams {
            learningRate: f32,
            dataOffset: f32,
            lossFunc: f32,
            batchSize: f32,
            mode: f32,
        }

        const inputDims = ${network.layers[0].size};
        const outputDims = ${network.layers.at(-1).size};

        const layerSizes = array(${network.layerSizes.join("u, ")});
        const layerWeightStarts = array(${weightStarts.join("u, ")});
        const layerCountCum = array(${neuronCountCumSum.join("u, ")});
        
        const maxLayer = ${maxLayer};
        const workGroupSize = min(256,  maxLayer);
        const signalsPerWorkgroup = maxLayer / workGroupSize;
        const layerCount: u32 = ${network.layers.length};
        const totalWeights: u32 = layerWeightStarts[layerCount];
        const totalNeurons: u32 = layerCountCum[layerCount];

        var<workgroup> singletonInputs: SingletonValsF32;
        var<workgroup> slopeOutputCurr: SlopeOutput;
        var<workgroup> slopeOutputNext: SlopeOutput;

        @group(0) @binding(0)
        var<storage, read_write> params: TrainParams;
        
        @group(1) @binding(0)
        var<storage, read_write> neuronWeights: WeightsArray;
        @group(1) @binding(1)
        var<storage, read> neuronActivationFuncIds: SingletonValsU32;
        
        @group(2) @binding(0)
        var<storage, read> inputData: array<InputDataPoint>;
        @group(2) @binding(1)
        var<storage, read_write> outputData: array<OutputDataPoint>;
        @group(2) @binding(2)
        var<storage, read> expectedOutputData: array<OutputDataPoint>;
        
        @group(3) @binding(0)
        var<storage, read_write> batchDerivatives: array<WeightsArray>;
        @group(3) @binding(1)
        var<storage, read_write> neuronInput: array<SingletonValsF32>;

        @compute @workgroup_size(workGroupSize)
        fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) id : vec3<u32>) {
            let dataIdx = (workgroup_id.x + u32(params.dataOffset)) % arrayLength(&inputData);
            let relNeuronIdx = id.x * signalsPerWorkgroup;

            for (var i: u32 = 0; i < layerCount; i++) {
                for (var signal: u32 = 0; signal < signalsPerWorkgroup; signal++) {
                    var sigIdx = signal + relNeuronIdx;

                    if (sigIdx > layerSizes[i]) { continue; }
                    
                    if (i == 0) {
                        singletonInputs[sigIdx] = inputData[dataIdx][sigIdx];
                        continue;
                    } else {
                        feedFoward(i, sigIdx);
                    }

                    if (i == layerCount - 1) { calculateOutput(dataIdx, sigIdx); }
                }
                workgroupBarrier();
            }

            if (u32(params.mode) == ${NETWORK_MODE.TRAIN}) { neuronInput[dataIdx] = singletonInputs; }
        }

        @compute @workgroup_size(workGroupSize)
        fn backward(@builtin(workgroup_id) workgroup_id : vec3<u32>,  @builtin(local_invocation_id) id : vec3<u32>) {
            let batchIdx = workgroup_id.x;
            let dataIdx = (batchIdx + u32(params.dataOffset)) % arrayLength(&inputData);
            let relNeuronIdx = id.x * signalsPerWorkgroup;

            singletonInputs = neuronInput[dataIdx];

            for (var i: u32 = 0; i < signalsPerWorkgroup; i++) {
                var sigIdx = i + relNeuronIdx;

                if (sigIdx > outputDims) { continue; }

                slopeOutputCurr[sigIdx] = calculateLossGradient(dataIdx, sigIdx) * getDerivative(sigIdx + layerCountCum[layerCount - 1]);
            }
            workgroupBarrier();

            for (var i: u32 = layerCount - 1; i > 0; i--) {
                for (var signal: u32 = 0; signal < signalsPerWorkgroup; signal++) {
                    var sigIdx = signal + relNeuronIdx;
                    
                    if (sigIdx > layerSizes[i - 1] + 1) { continue; }

                    var activation = getActivation(sigIdx + layerCountCum[i - 1]);
                    
                    for (var source: u32 = 0; source < layerSizes[i]; source++) {
                        var weightIndex = getWeightsStartIndex(source, i) + sigIdx;
                        slopeOutputNext[sigIdx] += slopeOutputCurr[source] * neuronWeights[weightIndex];

                        if (params.dataOffset == 0) {
                            batchDerivatives[dataIdx][weightIndex] = slopeOutputCurr[source] * activation;
                        } else {
                            batchDerivatives[dataIdx][weightIndex] += slopeOutputCurr[source] * activation;
                        }
                    }
                }
                workgroupBarrier();

                for (var signal: u32 = 0; signal < signalsPerWorkgroup; signal++) {
                    var sigIdx = signal + relNeuronIdx;
                    slopeOutputCurr[sigIdx] = slopeOutputNext[sigIdx] * getDerivative(sigIdx + layerCountCum[i - 1]);
                    slopeOutputNext[sigIdx] = 0;
                }
            }

            if (workgroup_id.x == 0 && id.x == 0) { 
                params.dataOffset += params.batchSize;
                if (u32(params.dataOffset) >= arrayLength(&inputData))  { params.dataOffset = 0; }
            }
        }

        @compute @workgroup_size(layerCount - 1)
        fn descent(@builtin(local_invocation_id) local_invocation_id : vec3<u32>) {
            let layerIdx = local_invocation_id.x + 1;
            descendWeights(layerIdx);
        }

        fn calculateOutput(dataIdx: u32, relNeuronIdx: u32) {
            outputData[dataIdx][relNeuronIdx] = getActivation(relNeuronIdx + layerCountCum[layerCount - 1]);
        }

        fn calculateLossGradient(dataIdx: u32, relNeuronIdx: u32) -> f32 {
            // mse grad
            return 2 * (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
            // categorical cross entropy grad
            // return (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
        }

        fn feedFoward(layerIndex: u32, relNeuronIdx: u32) {
            var neuronID = relNeuronIdx + layerCountCum[layerIndex];
            var newInput: f32;
            var weightIndex: u32 = getWeightsStartIndex(relNeuronIdx, layerIndex);

            for (var i: u32 = 0; i < layerSizes[layerIndex - 1] + 1; i++) {
                newInput += neuronWeights[weightIndex] * getActivation(i + layerCountCum[layerIndex - 1]);
                weightIndex++;
            }
            singletonInputs[neuronID] = newInput;
        }

        fn backPropagate(layerIndex: u32, batchIdx: u32, prevSlope: ptr<function, SlopeOutput>, iter: u32) -> SlopeOutput {
            var nextSlope = SlopeOutput();

            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                var dl_du_i = (*prevSlope)[i] * getDerivative(i + layerCountCum[layerIndex]);
                var weightIndex: u32 = getWeightsStartIndex(i, layerIndex);
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1] + 1; j++) {
                    if (iter == 0) {
                        batchDerivatives[batchIdx][weightIndex] = dl_du_i * getActivation(j + layerCountCum[layerIndex - 1]);
                    } else {
                        batchDerivatives[batchIdx][weightIndex] += dl_du_i * getActivation(j + layerCountCum[layerIndex - 1]);
                    }
                    nextSlope[j] += dl_du_i * neuronWeights[weightIndex];
                    weightIndex++;
                }
            }

            return nextSlope;
        }

        fn descendWeights(layerIndex: u32) {
            for (var i: u32 = layerWeightStarts[layerIndex]; i < layerWeightStarts[layerIndex + 1]; i++) {
                var tot: f32;
                for (var j: u32 = 0; j < arrayLength(&batchDerivatives); j++) {
                    tot += batchDerivatives[j][i];
                }
                neuronWeights[i] -=  tot * params.learningRate / params.batchSize;
            }
        }

        fn getWeightsStartIndex(relNeuronIdx: u32, layerIndex: u32) -> u32 {
            return relNeuronIdx * (layerSizes[layerIndex - 1] + 1) + layerWeightStarts[layerIndex];
        }

        fn getActivation(neuronID: u32) -> f32 {
            var ${X_VAR} = singletonInputs[neuronID];
            switch neuronActivationFuncIds[neuronID] {
                ${activationSwitch}
                default: {
                    return 0;
                }
            }
        }

        fn getDerivative(neuronID: u32) -> f32 {
            var ${X_VAR} = singletonInputs[neuronID];
            switch neuronActivationFuncIds[neuronID] {
                ${derivativeSwitch}
                default: {
                    return 0;
                }
            }
        }

        ${activations}

        ${derivatives}

        fn mse(dataIdx: u32) -> f32 {
            var error = 0f;
            for (var i: u32 = 0; i < outputDims; i++) {
                error += pow(outputData[dataIdx][i] - expectedOutputData[dataIdx][i], 2);
            }
            return error / f32(outputDims);
        }

        fn categoricalCrossEntropy(dataIdx: u32) -> f32 {
            var error = 0f;
            for (var i: u32 = 0; i < outputDims; i++) {
                error += expectedOutputData[dataIdx][i] * log2(outputData[dataIdx][i]);
            }
            return -error;
        }

        fn maxOutput(dataIdx: u32) -> MaxContainer {
            var out: MaxContainer;
            for (var i: u32 = 0; i < outputDims; i++) {
                if (i == 0 || out.val < outputData[dataIdx][i]){
                    out.val = outputData[dataIdx][i];
                    out.index = i;
                }
            }
            return out;
        }

        fn softmax(dataIdx: u32) { 
            var max = maxOutput(dataIdx).val;
            var divisor = 0f;
            for (var i: u32 = 0; i < outputDims; i++) {
                outputData[dataIdx][i] = exp(outputData[dataIdx][i] - max);
                divisor += outputData[dataIdx][i];
            }
            for (var i: u32 = 0; i < outputDims; i++) {
                outputData[dataIdx][i] /= divisor;
            }
        }
    `;
};

const MATRIX_MULTIPLY_SHADER_NAME = 'matrix_multiply';

const MATRIX_MULTIPLY_GROUP_LAYOUTS = [
    [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]
    ,
    [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]
];

const MATRIX_MULTIPLY_TILE_SIZE = 16;

const MATRIX_MULTIPLY_TILE_DOT_PROD = Array.from({length: MATRIX_MULTIPLY_TILE_SIZE}, (_, i) => `TILE_DATA_A[id.y][${i}] * TILE_DATA_B[${i}][id.x]`).join('+ \n');
const MATRIX_MULTIPLY_TILE_DOT_PROD_A = (aName, bName) => Array.from({length: MATRIX_MULTIPLY_TILE_SIZE * 2}, (_, i) => `${aName}[${i}] * ${bName}[${i}]`).join('+ \n');
const MATRIX_MULTIPLY_TILE_DOT_PROD_V = (idA, idB) => Array.from({length: MATRIX_MULTIPLY_TILE_SIZE}, (_, i) => `TILE_DATA_A[${idA}][${i}] * TILE_DATA_B[${i}][${idB}]`).join('+ \n');

export const MATRIX_MULTIPLY_SHADER2 = /*wgsl*/`
    // enable subgroups;

    @group(0) @binding(0) var<storage> A: array<f32>;
    @group(0) @binding(1) var<storage> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;

    const TILE_SIZE = ${MATRIX_MULTIPLY_TILE_SIZE};
    const DB_TILE_SIZE = 2 * TILE_SIZE;

    var<workgroup> TILE_DATA_A: array<array<f32, DB_TILE_SIZE >, DB_TILE_SIZE >;
    var<workgroup> TILE_DATA_B: array<array<f32, DB_TILE_SIZE >, DB_TILE_SIZE >;

    @compute @workgroup_size(TILE_SIZE, TILE_SIZE)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) id : vec3<u32>,
    ) {
        let row = workgroup_id.y * DB_TILE_SIZE;
        let col = workgroup_id.x * DB_TILE_SIZE;
        let b_step = B_dims[1] * DB_TILE_SIZE;

        let tot_row = row + id.y;
        let tot_col = col + id.x;
        
        var unraveled_idx_a1 = tot_row * A_dims[1] + id.x;
        var unraveled_idx_a2 = unraveled_idx_a1 + A_dims[1];
        var unraveled_idx_b1 = id.y * B_dims[1] + tot_col;
        var unraveled_idx_b2 = unraveled_idx_b1 + B_dims[1];
        let unraveled_idx_c1 = tot_row * C_dims[1] + tot_col;
        let unraveled_idx_c2 = unraveled_idx_c1 + C_dims[1];

        var t1 = 0.;
        var t2 = 0.;
        var t3 = 0.;
        var t4 = 0.;

        var pA1 = TILE_DATA_A[id.y];
        var pA2 = TILE_DATA_A[id.y + 1];
        var pB1 = TILE_DATA_B[id.x];
        var pB2 = TILE_DATA_B[id.x + 1];

        for (var i: u32 = 0; i < u32(ceil(f32(A_dims[1]) / f32(DB_TILE_SIZE))); i++) { 
            pA1[id.x] = A[unraveled_idx_a1];
            pA1[id.x + 1] = A[unraveled_idx_a1 + 1];
            pA2[id.x] = A[unraveled_idx_a2];
            pA2[id.x + 1] = A[unraveled_idx_a2 + 1];
            pB1[id.y] = B[unraveled_idx_b1];
            pB2[id.y] = B[unraveled_idx_b1 + 1];
            pB1[id.y + 1] = B[unraveled_idx_b2];
            pB2[id.y + 1] = B[unraveled_idx_b2 + 1];
            workgroupBarrier();

            unraveled_idx_a1 += DB_TILE_SIZE;
            unraveled_idx_a2 += DB_TILE_SIZE;
            unraveled_idx_b1 += b_step;
            unraveled_idx_b2 += b_step;
            

            // for (var k: u32 = 0; k < DB_TILE_SIZE; k+=2) {
            t1 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_A('pA1', 'pB1')};
            t2 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_A('pA1', 'pB2')};
            t3 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_A('pA2', 'pB1')};
            t4 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_A('pA2', 'pB2')};
            // }
        }


        if (tot_row < C_dims[0] && tot_col < C_dims[1]) {
            C[unraveled_idx_c1] += t1;
            C[unraveled_idx_c1 + 1] += t2;
            C[unraveled_idx_c2] += t3;
            C[unraveled_idx_c2 + 1] += t4;
        }
        // C[unraveled_idx_c1] = f32(id.x);
    }
`

export const MATRIX_MULTIPLY_SHADER3 = /*wgsl*/`
    // enable subgroups;

    @group(0) @binding(0) var<storage> A: array<vec4f>;
    @group(0) @binding(1) var<storage> B: array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> C: array<vec4f>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;

    const TILE_SIZE = ${MATRIX_MULTIPLY_TILE_SIZE};
    const DB_TILE_SIZE = 2 * TILE_SIZE;
    const D4_TILE_SIZE = u32(ceil(f32(TILE_SIZE) / 4.));

    var<workgroup> TILE_DATA_A: array<vec4f, D4_TILE_SIZE * TILE_SIZE>;
    var<workgroup> TILE_DATA_B: array<vec4f, D4_TILE_SIZE * TILE_SIZE>;

    @compute @workgroup_size(4, 16)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) id : vec3<u32>,
    ) {
        let row = workgroup_id.y * TILE_SIZE;
        let col = workgroup_id.x * TILE_SIZE;
        let b_step = B_dims[1] * TILE_SIZE / 4;

        let tot_row = row + id.y;
        let tot_col = col + id.x;
        
        var unraveled_idx_a = tot_row * A_dims[1] / 4 + id.x;
        var unraveled_idx_b = id.y * B_dims[1] / 4 + tot_col;
        let unraveled_idx_c = tot_row * C_dims[1] / 4 + tot_col;

        let baseA = D4_TILE_SIZE * id.y;
        let baseB = TILE_SIZE * id.x;

        // var vecA: ptr<workgroup, vec4f> = &(TILE_DATA_A[baseA + id.x]);
        // var vecB = &(TILE_DATA_B[baseB + id.y]);

        // var matB: ptr< = 

        var t = vec4(0.);

        for (var i: u32 = 0; i < u32(ceil(f32(A_dims[1]) / f32(TILE_SIZE))); i++) {
            TILE_DATA_A[baseA + id.x] = A[unraveled_idx_a];
            TILE_DATA_B[baseB + id.y] = B[unraveled_idx_b];

            workgroupBarrier();

            unraveled_idx_a += TILE_SIZE / 4;
            unraveled_idx_b += b_step;

            var matB0 = transpose(mat4x4f(
                TILE_DATA_B[baseB],
                TILE_DATA_B[baseB + 1],
                TILE_DATA_B[baseB + 2],
                TILE_DATA_B[baseB + 3],
            ));

            var matB1 = transpose(mat4x4f(
                TILE_DATA_B[baseB + 4],
                TILE_DATA_B[baseB + 5],
                TILE_DATA_B[baseB + 6],
                TILE_DATA_B[baseB + 7],
            ));

            var matB2 = transpose(mat4x4f(
                TILE_DATA_B[baseB + 8],
                TILE_DATA_B[baseB + 9],
                TILE_DATA_B[baseB + 10],
                TILE_DATA_B[baseB + 11],
            ));

            var matB3 = transpose(mat4x4f(
                TILE_DATA_B[baseB + 12],
                TILE_DATA_B[baseB + 13],
                TILE_DATA_B[baseB + 14],
                TILE_DATA_B[baseB + 15],
            ));
            

            // for (var k: u32 = 0; k < DB_TILE_SIZE; k+=2) {
            t = 
            TILE_DATA_A[baseA] * matB0 +
            TILE_DATA_A[baseA + 1] * matB1 +
            TILE_DATA_A[baseA + 2] * matB2 +
            TILE_DATA_A[baseA + 3] * matB3
            ;



            // t += 
            //     TILE_DATA_A[baseA + id.x][0] * TILE_DATA_B[baseB] +
            //     TILE_DATA_A[baseA + id.x][1] * TILE_DATA_B[baseB+1] +
            //     TILE_DATA_A[baseA + id.x][2] * TILE_DATA_B[baseB+2] +
            //     TILE_DATA_A[baseA + id.x][3] * TILE_DATA_B[baseB+3];
        }


        if (tot_row < C_dims[0] && tot_col < C_dims[1]) {
            C[unraveled_idx_c] += t;
        }
        // C[unraveled_idx_c1] = f32(id.x);
    }
`

export const MATRIX_MULTIPLY_SHADER = /*wgsl*/`
    @group(0) @binding(0) var<storage> A: array<f32>;
    @group(0) @binding(1) var<storage> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;

    const TILE_SIZE = ${MATRIX_MULTIPLY_TILE_SIZE};

    var<workgroup> TILE_DATA_A: array<array<f32, TILE_SIZE>, TILE_SIZE>;
    var<workgroup> TILE_DATA_B: array<array<f32, TILE_SIZE>, TILE_SIZE>;

    @compute @workgroup_size(TILE_SIZE, TILE_SIZE)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) id : vec3<u32>
    ) {
        let row = workgroup_id.y * TILE_SIZE;
        let col = workgroup_id.x * TILE_SIZE;
        let b_step = B_dims[1] * TILE_SIZE;

        let tot_row = row + id.y;
        let tot_col = col + id.x;

        var unraveled_idx_a = tot_row * A_dims[1] + id.x;
        var unraveled_idx_b = id.y * B_dims[1] + tot_col;
        let unraveled_idx_c = tot_row * C_dims[1] + tot_col;

        var t = 0.;

        let pA = &(TILE_DATA_A[id.y][id.x]);
        let pB = &(TILE_DATA_B[id.y][id.x]);

        for (var i: u32 = 0; i < u32(ceil(f32(A_dims[1]) / f32(TILE_SIZE))); i++) {   
            *pA = select(0, A[unraveled_idx_a], unraveled_idx_a < A_dims[3]);
            *pB = select(0, B[unraveled_idx_b], unraveled_idx_b < B_dims[3]);

            unraveled_idx_a += TILE_SIZE;
            unraveled_idx_b += b_step;
            
            workgroupBarrier();

            t += ${MATRIX_MULTIPLY_TILE_DOT_PROD};
        }

        if (tot_row < C_dims[0] && tot_col < C_dims[1]) {
            C[unraveled_idx_c] = t;
        }
    }
`

export const MATRIX_MULTIPLY_SHADER_V = /*wgsl*/`
    @group(0) @binding(0) var<storage> A: array<vec4f>;
    @group(0) @binding(1) var<storage> B: array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> C: array<vec4f>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;
    
    const VEC_SIZE = 4;

    const TILE_SIZE = 4;
    const TILE_SIZE_VEC = TILE_SIZE / VEC_SIZE;
    
    const BLOCK_DIM = 16;

    const STEP = TILE_SIZE * BLOCK_DIM;
    const STEP_VEC = STEP / VEC_SIZE;


    @compute @workgroup_size(BLOCK_DIM, BLOCK_DIM)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) id : vec3<u32>
    ) {
        
        let tot_row = workgroup_id.y * STEP + id.y * TILE_SIZE;
        let tot_col = workgroup_id.x * STEP_VEC + id.x;
        
        var aMat: mat4x4f;
        var bMat: mat4x4f;
        var cMat: mat4x4f;
        
        let aD4 = A_dims[1] / 4;
        let bD4 = B_dims[1] / 4;
        let B_ROW_STEP = B_dims[1] * TILE_SIZE_VEC;

        var unraveled_idx_a0 = tot_row * aD4;
        var unraveled_idx_a1 = unraveled_idx_a0 + aD4;
        var unraveled_idx_a2 = unraveled_idx_a1 + aD4;
        var unraveled_idx_a3 = unraveled_idx_a2 + aD4;
        var unraveled_idx_b0 = tot_col;
        var unraveled_idx_b1 = unraveled_idx_b0 + bD4;
        var unraveled_idx_b2 = unraveled_idx_b1 + bD4;
        var unraveled_idx_b3 = unraveled_idx_b2 + bD4;
        var unraveled_idx_c0 = tot_row * bD4 + tot_col;
        var unraveled_idx_c1 = unraveled_idx_c0 + bD4;
        var unraveled_idx_c2 = unraveled_idx_c1 + bD4;
        var unraveled_idx_c3 = unraveled_idx_c2 + bD4;

        for (var i: u32 = 0; i < u32(ceil(f32(A_dims[1]) / f32(TILE_SIZE))); i++) {   
            aMat = mat4x4f(
                A[unraveled_idx_a0],
                A[unraveled_idx_a1],
                A[unraveled_idx_a2],
                A[unraveled_idx_a3],
            );

            bMat = mat4x4f(
                B[unraveled_idx_b0],
                B[unraveled_idx_b1],
                B[unraveled_idx_b2],
                B[unraveled_idx_b3],
            );

            cMat += bMat * aMat;

            unraveled_idx_a0 += TILE_SIZE_VEC;
            unraveled_idx_a1 += TILE_SIZE_VEC;
            unraveled_idx_a2 += TILE_SIZE_VEC;
            unraveled_idx_a3 += TILE_SIZE_VEC;
            unraveled_idx_b0 += B_ROW_STEP;
            unraveled_idx_b1 += B_ROW_STEP;
            unraveled_idx_b2 += B_ROW_STEP;
            unraveled_idx_b3 += B_ROW_STEP;

        }

        if (tot_row < A_dims[0] && tot_col < B_dims[1]) {
            C[unraveled_idx_c0] = cMat[0];
            C[unraveled_idx_c1] = cMat[1];
            C[unraveled_idx_c2] = cMat[2];
            C[unraveled_idx_c3] = cMat[3];
        }
    }
`

export const MATRIX_MULTIPLY_SHADER_V2 = /*wgsl*/`
    @group(0) @binding(0) var<storage> A: array<f32>;
    @group(0) @binding(1) var<storage> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;

    const TILE_SIZE = ${MATRIX_MULTIPLY_TILE_SIZE};

    var<workgroup> TILE_DATA_A: array<array<f32, TILE_SIZE * 2>, TILE_SIZE * 2>;
    var<workgroup> TILE_DATA_B: array<array<f32, TILE_SIZE * 2>, TILE_SIZE * 2>;

    @compute @workgroup_size(TILE_SIZE, TILE_SIZE)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) id : vec3<u32>
    ) {
        let row = workgroup_id.y * TILE_SIZE * 2;
        let col = workgroup_id.x * TILE_SIZE * 2;
        let b_step = B_dims[1] * TILE_SIZE * 2;

        let idY = id.y * 2;
        let idX = id.x * 2;

        let tot_row = row + idY;
        let tot_col = col + idX;

        var unraveled_idx_a0 = tot_row * A_dims[1] + idX;
        var unraveled_idx_a2 = unraveled_idx_a0 + tot_row;
        var unraveled_idx_b0 = idY * B_dims[1] + tot_col;
        var unraveled_idx_b2 = unraveled_idx_b0 + idY * B_dims[1];
        let unraveled_idx_c0 = tot_row * C_dims[1] + tot_col;
        let unraveled_idx_c2 = unraveled_idx_c0 + tot_row;

        var t0 = 0.;
        var t1 = 0.;
        var t2 = 0.;
        var t3 = 0.;

        let pA0 = &(TILE_DATA_A[idY][idX]);
        let pA1 = &(TILE_DATA_A[idY][idX + 1]);
        let pA2 = &(TILE_DATA_A[idY + 1][idX]);
        let pA3 = &(TILE_DATA_A[idY + 1][idX + 1]);
        let pB0 = &(TILE_DATA_B[idY][idX]);
        let pB1 = &(TILE_DATA_B[idY][idX + 1]);
        let pB2 = &(TILE_DATA_B[idY + 1][idX]);
        let pB3 = &(TILE_DATA_B[idY + 1][idX + 1]);

        for (var i: u32 = 0; i < u32(ceil(f32(A_dims[1]) / f32(TILE_SIZE))); i++) {   
            *pA0 = select(0, A[unraveled_idx_a0], unraveled_idx_a0 < A_dims[3]);
            *pA1 = select(0, A[unraveled_idx_a0 + 1], unraveled_idx_a0 + 1 < A_dims[3]);
            *pA2 = select(0, A[unraveled_idx_a2], unraveled_idx_a2 < A_dims[3]);
            *pA3 = select(0, A[unraveled_idx_a2 + 1], unraveled_idx_a2 + 1< A_dims[3]);
            *pB0 = select(0, B[unraveled_idx_b0], unraveled_idx_b0 < B_dims[3]);
            *pB1 = select(0, B[unraveled_idx_b0 + 1], unraveled_idx_b0 + 1 < B_dims[3]);
            *pB2 = select(0, B[unraveled_idx_b2], unraveled_idx_b2 < B_dims[3]);
            *pB3 = select(0, B[unraveled_idx_b2 + 1], unraveled_idx_b2 + 1 < B_dims[3]);

            unraveled_idx_a0 += TILE_SIZE * 2;
            unraveled_idx_a2 += TILE_SIZE * 2;
            unraveled_idx_b0 += b_step;
            unraveled_idx_b2 += b_step;
            
            workgroupBarrier();

            t0 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_V('idY', 'idX')};
            t1 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_V('idY', 'idX + 1')};
            t2 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_V('idY + 1', 'idX')};
            t3 += ${MATRIX_MULTIPLY_TILE_DOT_PROD_V('idY + 1', 'idX + 1')};
        }

        if (tot_row < C_dims[0] && tot_col < C_dims[1]) {
            C[unraveled_idx_c0] = t0;
            C[unraveled_idx_c0 + 1] = t1;
            C[unraveled_idx_c2] = t2;
            C[unraveled_idx_c2 + 1] = t3;
        }
    }
`

export const matmulShader = /*wgsl */`
    @group(0) @binding(0) var<storage> a: array<f32>;
    @group(0) @binding(1) var<storage> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;

    const BLOCKSIZE: u32 = 16;
    const TILE_M: u32 = 8;  // Tile size in M dimension
    const TILE_N: u32 = 8;  // Tile size in N dimension

    @compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
    fn ${MATRIX_MULTIPLY_SHADER_NAME}(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.y * TILE_M;
        let col = global_id.x * TILE_N;

        var sums: array<array<f32, TILE_N>, TILE_M>;
        for (var i = 0u; i < TILE_M; i++) {
            for (var j = 0u; j < TILE_N; j++) {
                sums[i][j] = 0.0;
            }
        }

        // Compute the 2D tile
        for (var k = 0u; k < A_dims[1]; k++) {
        let a_00 = a[row * A_dims[1] + k];
        let a01 = a[(row + 1) * A_dims[1] + k];
        let a02 = a[(row + 2) * A_dims[1] + k];
        let a03 = a[(row + 3) * A_dims[1] + k];
        let a04 = a[(row + 4) * A_dims[1] + k];
        let a05 = a[(row + 5) * A_dims[1] + k];
        let a06 = a[(row + 6) * A_dims[1] + k];
        let a07 = a[(row + 7) * A_dims[1] + k];
        let b_00 = b[k * B_dims[1] + col];
        let b01 = b[k * B_dims[1] + (col + 1)];
        let b02 = b[k * B_dims[1] + (col + 2)];
        let b03 = b[k * B_dims[1] + (col + 3)];
        let b04 = b[k * B_dims[1] + (col + 4)];
        let b05 = b[k * B_dims[1] + (col + 5)];
        let b06 = b[k * B_dims[1] + (col + 6)];
        let b07 = b[k * B_dims[1] + (col + 7)];
        sums[0][0] += a_00 * b_00;
        sums[0][1] += a_00 * b01;
        sums[0][2] += a_00 * b02;
        sums[0][3] += a_00 * b03;
        sums[0][4] += a_00 * b04;
        sums[0][5] += a_00 * b05;
        sums[0][6] += a_00 * b06;
        sums[0][7] += a_00 * b07;
        sums[1][0] += a01 * b_00;
        sums[1][1] += a01 * b01;
        sums[1][2] += a01 * b02;
        sums[1][3] += a01 * b03;
        sums[1][4] += a01 * b04;
        sums[1][5] += a01 * b05;
        sums[1][6] += a01 * b06;
        sums[1][7] += a01 * b07;
        sums[2][0] += a02 * b_00;
        sums[2][1] += a02 * b01;
        sums[2][2] += a02 * b02;
        sums[2][3] += a02 * b03;
        sums[2][4] += a02 * b04;
        sums[2][5] += a02 * b05;
        sums[2][6] += a02 * b06;
        sums[2][7] += a02 * b07;
        sums[3][0] += a03 * b_00;
        sums[3][1] += a03 * b01;
        sums[3][2] += a03 * b02;
        sums[3][3] += a03 * b03;
        sums[3][4] += a03 * b04;
        sums[3][5] += a03 * b05;
        sums[3][6] += a03 * b06;
        sums[3][7] += a03 * b07;
        sums[4][0] += a04 * b_00;
        sums[4][1] += a04 * b01;
        sums[4][2] += a04 * b02;
        sums[4][3] += a04 * b03;
        sums[4][4] += a04 * b04;
        sums[4][5] += a04 * b05;
        sums[4][6] += a04 * b06;
        sums[4][7] += a04 * b07;
        sums[5][0] += a05 * b_00;
        sums[5][1] += a05 * b01;
        sums[5][2] += a05 * b02;
        sums[5][3] += a05 * b03;
        sums[5][4] += a05 * b04;
        sums[5][5] += a05 * b05;
        sums[5][6] += a05 * b06;
        sums[5][7] += a05 * b07;
        sums[6][0] += a06 * b_00;
        sums[6][1] += a06 * b01;
        sums[6][2] += a06 * b02;
        sums[6][3] += a06 * b03;
        sums[6][4] += a06 * b04;
        sums[6][5] += a06 * b05;
        sums[6][6] += a06 * b06;
        sums[6][7] += a06 * b07;
        sums[7][0] += a07 * b_00;
        sums[7][1] += a07 * b01;
        sums[7][2] += a07 * b02;
        sums[7][3] += a07 * b03;
        sums[7][4] += a07 * b04;
        sums[7][5] += a07 * b05;
        sums[7][6] += a07 * b06;
        sums[7][7] += a07 * b07;
        }

        // Row 0
        if (row < A_dims[0]) {
            if (col < B_dims[1]) {
                result[row * B_dims[1] + col] = sums[0][0];
            }
            if (col + 1 < B_dims[1]) {
                result[row * B_dims[1] + (col + 1)] = sums[0][1];
            }
            if (col + 2 < B_dims[1]) {
                result[row * B_dims[1] + (col + 2)] = sums[0][2];
            }
            if (col + 3 < B_dims[1]) {
                result[row * B_dims[1] + (col + 3)] = sums[0][3];
            }
            if (col + 4 < B_dims[1]) {
                result[row * B_dims[1] + (col + 4)] = sums[0][4];
            }
            if (col + 5 < B_dims[1]) {
                result[row * B_dims[1] + (col + 5)] = sums[0][5];
            }
            if (col + 6 < B_dims[1]) {
                result[row * B_dims[1] + (col + 6)] = sums[0][6];
            }
            if (col + 7 < B_dims[1]) {
                result[row * B_dims[1] + (col + 7)] = sums[0][7];
            }
        }

        // Row 1
        if (row + 1 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 1) * B_dims[1] + col] = sums[1][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 1)] = sums[1][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 2)] = sums[1][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 3)] = sums[1][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 4)] = sums[1][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 5)] = sums[1][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 6)] = sums[1][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 1) * B_dims[1] + (col + 7)] = sums[1][7];
            }
        }

        // Row 2
        if (row + 2 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 2) * B_dims[1] + col] = sums[2][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 1)] = sums[2][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 2)] = sums[2][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 3)] = sums[2][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 4)] = sums[2][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 5)] = sums[2][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 6)] = sums[2][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 2) * B_dims[1] + (col + 7)] = sums[2][7];
            }
        }

        // Row 3
        if (row + 3 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 3) * B_dims[1] + col] = sums[3][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 1)] = sums[3][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 2)] = sums[3][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 3)] = sums[3][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 4)] = sums[3][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 5)] = sums[3][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 6)] = sums[3][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 3) * B_dims[1] + (col + 7)] = sums[3][7];
            }
        }
        if (row + 4 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 4) * B_dims[1] + col] = sums[4][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 1)] = sums[4][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 2)] = sums[4][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 3)] = sums[4][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 4)] = sums[4][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 5)] = sums[4][5];
            }
            if (col + 6 < B_dims[1]) { 
                result[(row + 4) * B_dims[1] + (col + 6)] = sums[4][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 4) * B_dims[1] + (col + 7)] = sums[4][7];
            }
        }
        if (row + 5 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 5) * B_dims[1] + col] = sums[5][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 1)] = sums[5][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 2)] = sums[5][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 3)] = sums[5][3]; 
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 4)] = sums[5][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 5)] = sums[5][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 6)] = sums[5][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 5) * B_dims[1] + (col + 7)] = sums[5][7];
            }
        }
        if (row + 6 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 6) * B_dims[1] + col] = sums[6][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 1)] = sums[6][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 2)] = sums[6][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 3)] = sums[6][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 4)] = sums[6][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 5)] = sums[6][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 6)] = sums[6][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 6) * B_dims[1] + (col + 7)] = sums[6][7];
            }
        }
        if (row + 7 < A_dims[0]) {
            if (col < B_dims[1]) {
                result[(row + 7) * B_dims[1] + col] = sums[7][0];
            }
            if (col + 1 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 1)] = sums[7][1];
            }
            if (col + 2 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 2)] = sums[7][2];
            }
            if (col + 3 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 3)] = sums[7][3];
            }
            if (col + 4 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 4)] = sums[7][4];
            }
            if (col + 5 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 5)] = sums[7][5];
            }
            if (col + 6 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 6)] = sums[7][6];
            }
            if (col + 7 < B_dims[1]) {
                result[(row + 7) * B_dims[1] + (col + 7)] = sums[7][7];
            }
        }
    }
`;


function matrixMultiplyShaderPassCallback(device_connector, A, B, C) {
    const bindGroupLayouts = MATRIX_MULTIPLY_GROUP_LAYOUTS.map((e) => device_connector.device.createBindGroupLayout({ entries: e }));

    const bindGroup1 = device_connector.device.createBindGroup({
      layout: bindGroupLayouts[0],
      entries: [
          { binding: 0, resource: A.data },
          { binding: 1, resource: B.data },
          { binding: 2, resource: C.data },
      ],
    });

    const bindGroup2 = device_connector.device.createBindGroup({
      layout: bindGroupLayouts[1],
      entries: [
          { binding: 0, resource: A.dims },
          { binding: 1, resource: B.dims },
          { binding: 2, resource: C.dims },
      ],
    });

    const pipelineLayout = device_connector.device.createPipelineLayout({ bindGroupLayouts });

    if (!device_connector.getShader(MATRIX_MULTIPLY_SHADER_NAME)) device_connector.createShader(MATRIX_MULTIPLY_SHADER_NAME, MATRIX_MULTIPLY_SHADER);

    const computePipeline = device_connector.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device_connector.getShader(MATRIX_MULTIPLY_SHADER_NAME),
            entryPoint: MATRIX_MULTIPLY_SHADER_NAME,
        },
    });
    
    return (encoder, encoder_type = 0, end = false) => {
        const passEncoder = coerceEncoder(encoder, encoder_type);
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup1);
        passEncoder.setBindGroup(1, bindGroup2);
      
        const workgroupCountX = Math.ceil((B.data.buffer.size / (layer.size * Uint32Array.BYTES_PER_ELEMENT)) / MATRIX_MULTIPLY.tileSize);
        const workgroupCountY = Math.ceil(layer.size / MATRIX_MULTIPLY.tileSize);
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        
        terminateShaderPass(device_connector, passEncoder, encoder, encoder_type, end);
    }
}

function coerceEncoder(encoder, encoder_type) {
    return encoder_type == 0 ? encoder.beginComputePass(): encoder
}

function terminateShaderPass(device_connector, passEncoder, commandEncoder, encoder_type, end) {
    if (end) {
        passEncoder.end();
        if (encoder_type == 0) {
            device_connector.submitCommandEncoder(commandEncoder);
        }
    }
}

export const MATRIX_MULTIPLY = {
    shader: MATRIX_MULTIPLY_SHADER,
    name: MATRIX_MULTIPLY_SHADER_NAME,
    bindGroupLayouts: MATRIX_MULTIPLY_GROUP_LAYOUTS,
    tileSize: MATRIX_MULTIPLY_TILE_SIZE,
    createCallback: matrixMultiplyShaderPassCallback,
}


const MAX_FILTER_DIMS = [32, 32, -1, -1];

const CORRELATION_FILTER_NAME = 'correlation_filter'

const CORRELATION_FILTER_GROUP_LAYOUTS = [
    [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]
    ,
    [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]
];


export const CORRELATION_FILTER_SHADER = /*wgsl*/`
    @group(0) @binding(0) var<storage> A: array<f32>;
    @group(0) @binding(1) var<storage> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;
    @group(1) @binding(3) var<uniform> params: Params;

    struct Params {
        stride: u32,
        dialation: u32,
    }

    const MAX_B_DIMS = vec4(${MAX_FILTER_DIMS.join(', ')});

    @compute @workgroup_size(64)
    fn main2(@builtin(global_invocation_id) id : vec3<u32>) {
        let row = id.x;

        if (row >= C_dims[0]) {
            return;
        }

        let col_step = params.stride;
        let col_spacing = 1 + params.dialation;
        let row_step_A = row * A_dims[1];
        let row_step_C = row * C_dims[1];

        var window = array<array<f32, MAX_B_DIMS.y>, MAX_B_DIMS.x>();
        let slide_cols = B_dims[1] - col_step;

        var t = 0.;
        var A_col: u32 = col_step;
        var C_col: u32 = 0;

        for (var i: u32 = 0; i < B_dims[1]; i++) {
            for (var j: u32 = 0; j < B_dims[0]; j++) {
                window[i][j] = A[row_step_A + j * A_dims[1] + i * col_spacing];
                t += window[i][j] * B[j * B_dims[1] + i];
            }
        }

        C[row_step_C] = t;
        for (;A_col <= A_dims[1] - (B_dims[1] + (B_dims[1] - 1) * params.dialation); A_col += col_step) {
            t = 0;
            C_col++;

            for (var j: u32 = 0; j < slide_cols; j++) {
                let curr_col = (j + A_col) % B_dims[1];
                for (var k: u32 = 0; k < B_dims[0]; k++) {
                    t += window[curr_col][k] * B[k * B_dims[1] + j];
                }
            }
            
            for (var j: u32 = slide_cols; j < B_dims[1]; j++) {
                let curr_col = (j + A_col) % B_dims[1];
                for (var k: u32 = 0; k < B_dims[0]; k++) {
                    window[curr_col][k] = A[row_step_A + k * A_dims[1] + A_col + j * col_spacing];
                    t += window[curr_col][k] * B[k * B_dims[1] + j];
                    
                }
            }
            C[row_step_C + C_col] = t;
        }
    }
`

export const CORRELATION_FILTER = {
    name: CORRELATION_FILTER_NAME,
    shader: CORRELATION_FILTER_SHADER,
    bindGroupLayouts: CORRELATION_FILTER_GROUP_LAYOUTS,
}


export const POOL = /*wgsl*/`
    @group(0) @binding(0) var<storage> A: array<f32>;
    @group(0) @binding(1) var<storage> B: array<f32>; // not needed
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: vec4u;
    @group(1) @binding(1) var<uniform> B_dims: vec4u;
    @group(1) @binding(2) var<uniform> C_dims: vec4u;
    @group(1) @binding(3) var<uniform> params: Params;

    struct Params {
        stride: u32,
        dialation: u32,
        // poolFunc: u32,
        // reset_val: f32,
    }

    const RESET_VAL = 0.;
    const MAX_B_DIMS = vec4(${MAX_FILTER_DIMS.join(', ')});

    @compute @workgroup_size(64)
    fn main2(@builtin(global_invocation_id) id : vec3<u32>) {
        let row = id.x;

        if (row >= C_dims[0]) {
            return;
        }

        let col_step = params.stride;
        let col_spacing = 1 + params.dialation;
        let row_step_A = row * A_dims[1];
        let row_step_C = row * C_dims[1];

        var window = array<array<f32, MAX_B_DIMS.y>, MAX_B_DIMS.x>();
        let slide_cols = B_dims[1] - col_step;

        var t = RESET_VAL;
        var A_col: u32 = col_step;
        var C_col: u32 = 0;

        for (var i: u32 = 0; i < B_dims[1]; i++) {
            for (var j: u32 = 0; j < B_dims[0]; j++) {
                window[i][j] = A[row_step_A + j * A_dims[1] + i * col_spacing];
                t = max(window[i][j], t);
            }
        }

        C[row_step_C] = t;
        for (;A_col <= A_dims[1] - (B_dims[1] + (B_dims[1] - 1) * params.dialation); A_col += col_step) {
            t = RESET_VAL;
            C_col++;

            for (var j: u32 = 0; j < slide_cols; j++) {
                let curr_col = (j + A_col) % B_dims[1];
                for (var k: u32 = 0; k < B_dims[0]; k++) {
                    t = max(window[curr_col][k], t);
                }
            }
            
            for (var j: u32 = slide_cols; j < B_dims[1]; j++) {
                let curr_col = (j + A_col) % B_dims[1];
                for (var k: u32 = 0; k < B_dims[0]; k++) {
                    window[curr_col][k] = A[row_step_A + k * A_dims[1] + A_col + j * col_spacing];
                    t = max(window[curr_col][k], t);
                    
                }
            }
            C[row_step_C + C_col] = t;
        }
    }
`
