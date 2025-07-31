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


const MATRIX_MULTIPLY = /*wgsl*/`
    @group(0) @binding(0) var<uniform> A: array<f32>;
    @group(0) @binding(1) var<uniform> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @group(1) @binding(0) var<uniform> A_dims: array<u32>;
    @group(1) @binding(1) var<uniform> B_dims: array<u32>;
    @group(1) @binding(2) var<uniform> C_dims: array<u32>;

    override TILE_SIZE = vec4u();

    @compute @workgroup_size(16)
    fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(subgroup_invocation_id) id : u32) {
        let row = workgroup_id.y * TILE_SIZE.y;
        let col = workgroup_id.x * TILE_SIZE.x;
        
        let unraveled_idx_a = row * A_dims[1] + col + id; 
        let unraveled_idx_b = row * B_dims[1] + col + id;
        let unraveled_idx_c = row * C_dims[1] + col + id;

        let val = A[unraveled_idx_a] * B[unraveled_idx_b];

        C[unraveled_idx_c] = subgroupAdd(val);
    }

    @compute @workgroup_size(32)
    fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(subgroup_invocation_id) id : u32, @builtin(num_workgroups) s : vec3<u32>) {
        let col = workgroup_id.x;
        let row = workgroup_id.y;
        
        var unraveled_idx_a = row * A_dims[1] + id; 
        var unraveled_idx_b = id * B_dims[1] + col;
        let unraveled_idx_c = row * C_dims[1] + col;

        var val = A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        unraveled_idx_a += 32; 
        unraveled_idx_b += 32 * B_dims[1];

        val += A[unraveled_idx_a] * B[unraveled_idx_b];

        C[unraveled_idx_c] = subgroupAdd(val);
    }
`