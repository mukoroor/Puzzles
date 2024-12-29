import WGSLActivations , { X_VAR } from "./ActivationFunctions.js";
import { NETWORK_MODE } from "./NetworkConsts.js";

export const neural_net_shader = (network) => {
  const maxLayer = network.maxLayer;
  const neuronCountCumSum = [0, ...network.cumSum];
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
            propagateIterations: f32,
            batchSize: f32,
            mode: f32,
        }

        const inputDims = ${network.layers[0].size};
        const outputDims = ${network.layers.at(-1).size};

        const layerSizes = array(${network.layerSizes.join("u, ")});
        const layerWeightStarts = array(${weightStarts.join("u, ")});
        const layerCountCum = array(${neuronCountCumSum.join("u, ")});
        
        const maxLayer = ${maxLayer};
        const layerCount: u32 = ${network.layers.length};
        const totalWeights: u32 = layerWeightStarts[layerCount];
        const totalNeurons: u32 = layerCountCum[layerCount];

        var<workgroup> singletonInputs: SingletonValsF32;

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

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {

            let batchIdx = id.x;
            let trainCount = arrayLength(&inputData);
            let isPredict = params.mode == ${NETWORK_MODE.PREDICT};
            
            if (batchIdx >= trainCount) { return; }
            
            let dispatchBatchSize = u32(ceil(f32(trainCount) / params.propagateIterations));
            
            for (var propIter: u32 = 0; propIter < u32(select(ceil(params.propagateIterations), 1, isPredict)); propIter++) {

                let dataIdx = select((batchIdx + u32(params.dataOffset) + dispatchBatchSize * propIter) % trainCount, batchIdx, isPredict);

                if (dataIdx < dispatchBatchSize && propIter > 0) { break; }

                //copy data features into inputs arr
                for (var i: u32 = 0; i < inputDims; i++) {
                    singletonInputs[i] = inputData[dataIdx][i];
                }
                
                //feed
                for (var i: u32 = 1; i < layerCount; i++) {
                    for (var j: u32 = 0; j < layerSizes[i]; j++) {
                        feedFoward(i, j);
                    }
                }
                
                var slopeOutput = SlopeOutput();
                for (var i: u32 = 0; i < outputDims; i++) {
                    calculateOutput(dataIdx, i);
                    // slopeOutput[i] = calculateLossGradient(dataIdx, i);
                }
                softmax(dataIdx);
                for (var i: u32 = 0; i < outputDims; i++) {
                    // calculateOutput(dataIdx, i);
                    slopeOutput[i] = calculateLossGradient(dataIdx, i);
                    // slopeOutput[i] = 1;
                    // outputData[dataIdx][i] = expectedOutputData[dataIdx][i];
                    // outputData[dataIdx][i] = 12345.1;
                }
    
                // for (var i: u32 = 0; i < totalNeurons; i++) {
                //     singletonNeuronData.inputs[i] = singletonInputs[i];
                // }
    
                if (isPredict) { return; }
    
                for (var i: u32 = layerCount - 1; i > 0; i--) {
                    slopeOutput = backPropagate(i, batchIdx, &slopeOutput, propIter);
                }

            }
        }

        @compute @workgroup_size(layerCount - 1)
        fn backward(@builtin(local_invocation_id) local_invocation_id : vec3<u32>) {
            let layerIdx = local_invocation_id.x + 1;

        }

        @compute @workgroup_size(layerCount - 1)
        fn descent(@builtin(local_invocation_id) local_invocation_id : vec3<u32>) {
            let layerIdx = local_invocation_id.x + 1;
            descendWeights(layerIdx);

            if (layerIdx == 1){
                params.dataOffset += params.batchSize;
                if (u32(params.dataOffset) >= arrayLength(&inputData)) {
                    params.dataOffset = 0;
                }
            }
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

        fn calculateOutput(dataIdx: u32, relNeuronIdx: u32) {
            outputData[dataIdx][relNeuronIdx] = getActivation(relNeuronIdx + layerCountCum[layerCount - 1]);
        }

        fn calculateLossGradient(dataIdx: u32, relNeuronIdx: u32) -> f32 {
            // mse grad
            // return 2 * (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
            // categorical cross entropy grad
            return (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
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
    `;
};
