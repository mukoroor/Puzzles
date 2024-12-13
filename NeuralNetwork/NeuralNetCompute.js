export const neural_net_shader = (layerSizes, batchSize=4) => {
    const maxLayer = layerSizes.reduce((a, c) => Math.max(a, c));
    const layerWeightStarts = layerSizes.reduce((a, c, i) => {
        if (!i) return [0, 0]
        let val = a[i] + c * layerSizes[i - 1];
        a.push(val);
        return a;
    }, [])
    const layerNeuronCountCumSum = layerSizes.reduce((a, c) => {
        a.push((a.at(-1) || 0) + c)
        return a;
    }, [0])

    return /*wgsl*/`
        alias InputDataPoint = array<f32, inputDims>;
        alias OutputDataPoint = array<f32, outputDims>;

        alias SlopeOutput = array<f32, maxLayer>;
        alias SlopeOutputAtomic = array<f32, maxLayer>;
        alias SlopeOutputWeight = array<SlopeOutputAtomic, maxLayer>;

        alias SingletonValsF32 = array<f32, totalNeurons>;
        alias SingletonValsU32 = array<u32, totalNeurons>;

        struct DV {
            dl_dw: SlopeOutputWeight,
            dl_du: SlopeOutputAtomic
        }
        alias LayerDerivatives = array<DV, layerCount - 1>;

        struct Output {
            desired: OutputDataPoint,
            network: OutputDataPoint
        }

        struct SingletonNeuronData {
            inputs: SingletonValsF32,
            biases: SingletonValsF32,
        }

        struct TrainParams {
            learningRate: f32,
            // dataOffset: f32,
            // batchIterations: f32,
            // batchSize: f32,
            mode: f32,
        }

        const inputDims = ${layerSizes[0]};
        const outputDims = ${layerSizes.at(-1)};

        const layerSizes = array(${layerSizes.join('u, ')});
        const layerWeightStarts = array(${layerWeightStarts.join('u, ')});
        const layerCountCum = array(${layerNeuronCountCumSum.join('u, ')});
        
        const maxLayer = ${maxLayer};
        const batchSize = ${batchSize};
        const layerCount: u32 = ${layerSizes.length};
        const totalNeurons: u32 = layerCountCum[layerCount];

        const pointWeight = 1f / f32(batchSize);

        var<workgroup> slopeOutput1: SlopeOutput;
        var<workgroup> slopeOutput2: SlopeOutput;
        var<workgroup> singletonInputs: SingletonValsF32;

        @group(0) @binding(0)
        var<storage, read_write> singletonNeuronData: SingletonNeuronData;
        @group(0) @binding(1)
        var<storage, read_write> neuronWeights: array<f32, ${layerWeightStarts.at(-1)}>;
        @group(0) @binding(2)
        var<storage, read> neuronActivationFuncIds: SingletonValsU32;

        @group(1) @binding(0)
        var<storage, read_write> batchDerivatives: array<LayerDerivatives, batchSize>;
        @group(1) @binding(1)
        var<storage, read_write> outputs: Output;
        @group(1) @binding(2)
        var<uniform> networkStep: vec4u;

        @group(2) @binding(0)
        var<storage, read> inputData: array<InputDataPoint>;
        @group(2) @binding(1)
        var<storage, read_write> outputData: array<OutputDataPoint>;
        @group(2) @binding(2)
        var<storage, read> expectedOutputData: array<OutputDataPoint>;
        @group(2) @binding(3)
        var<uniform> params: TrainParams;
        // need to add output buff
        
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {

            let batchIdx =  id.x;
            
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
            
            for (var i: u32 = 0; i < outputDims; i++) {
                calculateOutput(batchIdx, i);
                calculateLossGradient(batchIdx, i);
            }
            if (params.mode == 1) { return; }

            var iterCounter: u32 = 0;
            for (var i: u32 = layerCount - 1; i > 0; i--) {
                if ((iterCounter & 1) == 0) {
                    backPropagate(i, batchIdx, &slopeOutput1, &slopeOutput2);
                } else {
                    backPropagate(i, batchIdx, &slopeOutput2, &slopeOutput1);
                }
                iterCounter++;
            }

        }

        @compute @workgroup_size(layerCount - 1)
        fn descent(@builtin(local_invocation_id) local_invocation_id : vec3<u32>) {
            let layerIdx = local_invocation_id.x + 1;

            for (var i: u32 = 0; i < layerSizes[layerIdx]; i++) {
                descend(i, layerIdx);
            }
        }

        fn calculateOutput(dataIdx: u32, relNeuronIdx: u32) {
            outputData[dataIdx][relNeuronIdx] = getActivation(relNeuronIdx + layerCountCum[layerCount - 1]);
        }

        fn calculateLossGradient(dataIdx: u32, relNeuronIdx: u32) {
            slopeOutput1[relNeuronIdx] = (outputData[dataIdx][relNeuronIdx] - expectedOutputData[dataIdx][relNeuronIdx]);
        }

        fn feedFoward(layerIndex: u32, relNeuronIdx: u32) {
            var neuronID = relNeuronIdx + layerCountCum[layerIndex];
            var newInput: f32 = 0;
            for (var i: u32 = 0; i < layerSizes[layerIndex - 1]; i++) {
                newInput += getWeight(relNeuronIdx, i, layerIndex) * getActivation(i + layerCountCum[layerIndex - 1]);
            }
            singletonInputs[neuronID] = newInput;
        }

        fn backPropagate(layerIndex: u32, batchIdx: u32, dOutputPtr: ptr<workgroup, SlopeOutput>, dOutputPtrNext: ptr<workgroup, SlopeOutput>) {
            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                var dl_du_i = (*dOutputPtr)[i] * getDerivative(i + layerCountCum[layerIndex]);
                batchDerivatives[batchIdx][layerIndex - 1].dl_du[i] = dl_du_i;
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                    if (i == 0) { (*dOutputPtrNext)[j] = 0; }
                    batchDerivatives[batchIdx][layerIndex - 1].dl_dw[i][j] = dl_du_i * getActivation(j + layerCountCum[layerIndex - 1]);
                    (*dOutputPtrNext)[j] += dl_du_i * getWeight(i, j, layerIndex);
                }
            }
        }

        fn descend(relNeuronIdx: u32, layerIndex: u32) {
            var neuronID = relNeuronIdx + layerCountCum[layerIndex];

            for (var i: u32 = 0; i < batchSize; i++) {
                singletonNeuronData.biases[neuronID] -= pointWeight * batchDerivatives[i][layerIndex - 1].dl_du[relNeuronIdx] * params.learningRate;
            }

            for (var i: u32 = 0; i < layerSizes[layerIndex - 1]; i++) {
                var currWeight = getWeight(relNeuronIdx, i, layerIndex);
                for (var j: u32 = 0; j < batchSize; j++) {
                    currWeight -= pointWeight * batchDerivatives[i][layerIndex - 1].dl_dw[relNeuronIdx][i] * params.learningRate;
                }
                setWeight(relNeuronIdx, i, layerIndex, currWeight);
            }
        }

        fn getWeight(relNeuronIdx: u32, weightIndex: u32, layerIndex: u32) -> f32 {
            return neuronWeights[relNeuronIdx * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]];
        }

        fn setWeight(relNeuronIdx: u32, weightIndex: u32, layerIndex: u32, val: f32) {
            neuronWeights[relNeuronIdx * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]] = val;
        }

        fn getLinearVal(neuronID: u32) -> f32 {
            return singletonInputs[neuronID] + singletonNeuronData.biases[neuronID];
        }
 
        fn getActivation(neuronID: u32) -> f32 {
            var u = getLinearVal(neuronID);
            switch neuronActivationFuncIds[neuronID] {
                case 1: {
                    return sigmoid(u);                                                
                }
                case 2: {
                    return relU(u);                                                 
                }
                default: {
                    return u;
                }
            }
        }

        fn getDerivative(neuronID: u32) -> f32 {
            var u = getLinearVal(neuronID);
            switch neuronActivationFuncIds[neuronID] {
                case 1: {
                    return d_sigmoid(u);                                               
                }
                case 2: {
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
            return exp(-x) / pow((1 + exp(-x)), 2);
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

        fn mse() -> f32 {
            var error = 0f;
            for (var i: u32 = 0; i < layerSizes[layerCount - 1]; i++) {
                error += pow(outputs.network[i] - outputs.desired[i], 2);
            }
            return error / f32(outputDims);
        }
    `;
}
