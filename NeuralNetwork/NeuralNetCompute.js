export const neural_net_shader = (network) => {
    const maxLayer = network.maxLayer;
    // const batchSize = network.batchSize;
    const neuronCountCumSum = [0, ...network.cumSum];
    const weightStarts = [0, ...network.weightCounts];

    return /*wgsl*/`
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

        struct TrainParams {
            learningRate: f32,
            // dataOffset: f32,
            // batchIterations: f32,
            batchSize: f32,
            mode: f32,
        }

        const inputDims = ${network.layers[0]};
        const outputDims = ${network.layers.at(-1)};

        const layerSizes = array(${network.layers.join('u, ')});
        const layerWeightStarts = array(${weightStarts.join('u, ')});
        const layerCountCum = array(${neuronCountCumSum.join('u, ')});
        
        const maxLayer = ${maxLayer};
        const layerCount: u32 = ${network.layers.length};
        const totalWeights: u32 = layerWeightStarts[layerCount];
        const totalNeurons: u32 = layerCountCum[layerCount];

        var<workgroup> singletonInputs: SingletonValsF32;

        @group(0) @binding(0)
        var<storage, read_write> neuronWeights: WeightsArray;
        @group(0) @binding(1)
        var<storage, read> neuronActivationFuncIds: SingletonValsU32;

        @group(1) @binding(0)
        var<storage, read_write> batchDerivatives: array<WeightsArray>;
        @group(1) @binding(1)
        var<uniform> params: TrainParams;

        @group(2) @binding(0)
        var<storage, read> inputData: array<InputDataPoint>;
        @group(2) @binding(1)
        var<storage, read_write> outputData: array<OutputDataPoint>;
        @group(2) @binding(2)
        var<storage, read> expectedOutputData: array<OutputDataPoint>;
        
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {

            let batchIdx =  id.x;

            if (batchIdx >= u32(params.batchSize)) {
                return;
            }
            
            //copy data features into inputs arr
            for (var i: u32 = 0; i < inputDims; i++) {
                singletonInputs[i] = inputData[batchIdx][i];
            }
            
            //feed
            for (var i: u32 = 1; i < layerCount; i++) {
                for (var j: u32 = 0; j < layerSizes[i]; j++) {
                    feedFoward(i, j);
                }
            }
            
            var slopeOutput = SlopeOutput();
            for (var i: u32 = 0; i < outputDims; i++) {
                calculateOutput(batchIdx, i);
                slopeOutput[i] = calculateLossGradient(batchIdx, i);
            }

            // for (var i: u32 = 0; i < totalNeurons; i++) {
            //     singletonNeuronData.inputs[i] = singletonInputs[i];
            // }

            if (params.mode == 1) { return; }

            for (var i: u32 = layerCount - 1; i > 0; i--) {
                slopeOutput = backPropagate(i, batchIdx, &slopeOutput);
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
            return 2 * (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
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

        fn backPropagate(layerIndex: u32, batchIdx: u32, prevSlope: ptr<function, SlopeOutput>) -> SlopeOutput {
            var nextSlope = SlopeOutput();

            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                var dl_du_i = (*prevSlope)[i] * getDerivative(i + layerCountCum[layerIndex]);
                var weightIndex: u32 = getWeightsStartIndex(i, layerIndex);
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1] + 1; j++) {
                    batchDerivatives[batchIdx][weightIndex] = dl_du_i * getActivation(j + layerCountCum[layerIndex - 1]);
                    nextSlope[j] += dl_du_i * neuronWeights[weightIndex];
                    weightIndex++;
                }
            }

            return nextSlope;
        }

        fn descendWeights(layerIndex: u32) {
            for (var i: u32 = layerWeightStarts[layerIndex]; i < layerWeightStarts[layerIndex + 1]; i++) {
                var tot: f32;
                for (var j: u32 = 0; j < u32(params.batchSize); j++) {
                    tot += batchDerivatives[j][i];
                }
                neuronWeights[i] -=  tot * params.learningRate / params.batchSize;
            }
        }

        fn getWeightsStartIndex(relNeuronIdx: u32, layerIndex: u32) -> u32 {
            return relNeuronIdx * (layerSizes[layerIndex - 1] + 1) + layerWeightStarts[layerIndex];
        }

        fn getActivation(neuronID: u32) -> f32 {
            var u = singletonInputs[neuronID];
            switch neuronActivationFuncIds[neuronID] {
                case 0: {
                    return 1;
                }
                case 1: {
                    return u;
                }
                case 2: {
                    return sigmoid(u);                                                
                }
                case 3: {
                    return relU(u);                                                 
                }
                default: {
                    return 0;
                }
            }
        }

        fn getDerivative(neuronID: u32) -> f32 {
            var u = singletonInputs[neuronID];
            switch neuronActivationFuncIds[neuronID] {
                case 2: {
                    return d_sigmoid(u);                                               
                }
                case 3: {
                    return d_relU(u);                                                 
                }
                default: {
                    return 1;
                }
            }
        }

        fn sigmoid(x: f32) -> f32 {
            return 1 / (1 + exp(-x));
        }

        fn d_sigmoid(x: f32) -> f32 {
            var sig = sigmoid(x);
            return sig * (1 - sig);
        }

        fn relU(x: f32) -> f32 {
            return max(0, x);
        }

        fn d_relU(x: f32) -> f32 {
            if (x > 0) {
                return 1f;
            } else {
                return 0f;
            }
        }
    `;
}
