export const neural_net_shader = (layerSizes) => {
    const maxLayer = layerSizes.reduce((a, c) => Math.max(a, c));
    const totalNeurons = layerSizes.reduce((a, c) => a + c);
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
        alias InputDataPoint = array<f32, ${layerSizes[0]}>;
        alias OutputDataPoint = array<f32, ${layerSizes.at(-1)}>;

        struct DV {
            dl_dw: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_dw_final: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_du: array<f32, ${maxLayer}>,
            dl_du_final: array<f32, ${maxLayer}>,
            dl_do_prev: array<f32, ${maxLayer}>,
        }

        struct Output {
            desired: OutputDataPoint,
            network: OutputDataPoint
        }

        struct SingletonNeuronData {
            inputs: array<f32, ${totalNeurons}>,
            biases: array<f32, ${totalNeurons}>,
        }

        struct TrainParams {
            learningRate: f32,
            dataOffset: f32,
            batchIterations: f32,
            batchSize: f32,
            mode: f32,
        }

        const layerCount: u32 = ${layerSizes.length};
        const layerSizes = array(${layerSizes.join('u, ')});
        const layerWeightStarts = array(${layerWeightStarts.join('u, ')});
        const layerCountCum = array(${layerNeuronCountCumSum.join('u, ')});

        @group(0) @binding(0)
        var<storage, read_write> singletonNeuronData: SingletonNeuronData;
        @group(0) @binding(1)
        var<storage, read_write> neuronWeights: array<f32, ${layerWeightStarts.at(-1)}>;
        @group(0) @binding(2)
        var<storage, read> neuronActivationFuncIds: array<u32, ${totalNeurons}>;

        @group(1) @binding(0)
        var<storage, read_write> layerDerivatives: array<DV>;
        @group(1) @binding(1)
        var<storage, read_write> outputs: Output;
        @group(1) @binding(2)
        var<uniform> networkStep: vec4u;

        @group(2) @binding(0)
        var<storage, read> inputData: array<InputDataPoint>;
        @group(2) @binding(1)
        var<storage, read_write> outputData: array<OutputDataPoint>;
        @group(2) @binding(2)
        var<uniform> params: TrainParams;
        
        @compute @workgroup_size(${maxLayer})
        fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
                @builtin(num_workgroups) num_workgroups: vec3<u32>,
                @builtin(local_invocation_id) local_invocation_id : vec3<u32>) {

            let local_index =  
            local_invocation_id.x;


            for (var iter: u32 = 0; iter < u32(params.batchIterations); iter++) {
                var iterOffset = (iter * u32(params.batchSize)) % arrayLength(&inputData);
                for (var batchIndex: u32 = 0; batchIndex < u32(params.batchSize); batchIndex++) {
                    var dataIndex = (iterOffset + batchIndex + u32(params.dataOffset)) % arrayLength(&inputData);
                    if (local_index < layerSizes[0]) { stageDataPoint(dataIndex, local_index); }
                    
                    for (var i: u32 = 1; i < layerCount; i++) {
                        if (local_index < layerSizes[i]) {
                            feedFoward(local_index, i);
                        }
                        storageBarrier();
                    }
                    
                    if (local_index < layerSizes[layerCount - 1]) {
                        calculateOutput(local_index);
                        d_mse(local_index);
                    }

                    if (u32(params.mode) == 1) {                                                                     
                        outputData[dataIndex][local_index] = outputs.network[local_index];
                        continue;
                    }

                    for (var i: u32 = layerCount - 1; i > 0; i--) {
                        if (local_index == 0) {
                            backPropagate(i);
                        }
                        storageBarrier();
                    }
                }

                if (u32(params.mode) == 1) { return; }

                for (var i: u32 = 1; i < layerCount; i++) {
                    if (local_index < layerSizes[i]) {
                        descend(local_index,  i);
                    }
                }
            }
        }

        fn stageDataPoint(dataIndex: u32, featureIndex: u32) {
            outputs.desired[featureIndex] = outputData[dataIndex][featureIndex];
            singletonNeuronData.inputs[featureIndex] = inputData[dataIndex][featureIndex];
        }

        fn calculateOutput(outputIndex: u32) {
            outputs.network[outputIndex] = getActivation(outputIndex +  layerCountCum[layerCount - 1]);
        }

        fn feedFoward(outputIndex: u32, layerIndex: u32) {
            var neuronIndex = outputIndex + layerCountCum[layerIndex];
            var newInput: f32 = 0;
            for (var i: u32 = 0; i < layerSizes[layerIndex - 1]; i++) {
                newInput += getWeight(outputIndex, i, layerIndex) * getActivation(i + layerCountCum[layerIndex - 1]);
            }
            singletonNeuronData.inputs[neuronIndex] = newInput;
        }

        fn backPropagate(layerIndex: u32) {
            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                layerDerivatives[layerIndex - 1].dl_du[i] = layerDerivatives[layerIndex].dl_do_prev[i] * getDerivative(i + layerCountCum[layerIndex]);
                layerDerivatives[layerIndex - 1].dl_du_final[i] += layerDerivatives[layerIndex - 1].dl_du[i];
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                    if (i == 0) { layerDerivatives[layerIndex - 1].dl_do_prev[j] = 0; }
                    layerDerivatives[layerIndex - 1].dl_dw[i][j] = layerDerivatives[layerIndex - 1].dl_du[i] * getActivation(j + layerCountCum[layerIndex - 1]);
                    layerDerivatives[layerIndex - 1].dl_dw_final[i][j] += layerDerivatives[layerIndex - 1].dl_dw[i][j];
                    layerDerivatives[layerIndex - 1].dl_do_prev[j] += layerDerivatives[layerIndex - 1].dl_du[i] * getWeight(i, j, layerIndex);
                }
            }
        }

        fn descend(outputIndex: u32, layerIndex: u32) {
            var pointWeight = 1f / f32(arrayLength(&inputData));
            var neuronIndex = outputIndex + layerCountCum[layerIndex];
            singletonNeuronData.biases[neuronIndex] -= pointWeight * layerDerivatives[layerIndex - 1].dl_du_final[outputIndex] * params.learningRate;
            layerDerivatives[layerIndex - 1].dl_du_final[outputIndex] = 0;


            for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                setWeight(outputIndex, j, layerIndex, getWeight(outputIndex, j, layerIndex) - pointWeight * layerDerivatives[layerIndex - 1].dl_dw_final[outputIndex][j] * params.learningRate);
                layerDerivatives[layerIndex - 1].dl_dw_final[outputIndex][j] = 0;
            }
        }

        fn getWeight(outputIndex: u32, weightIndex: u32, layerIndex: u32) -> f32 {
            return neuronWeights[outputIndex * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]];
        }

        fn setWeight(outputIndex: u32, weightIndex: u32, layerIndex: u32, val: f32) {
            neuronWeights[outputIndex * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]] = val;
        }
 
        fn getActivation(neuronIndex: u32) -> f32 {
            var u = singletonNeuronData.inputs[neuronIndex] + singletonNeuronData.biases[neuronIndex];
            switch neuronActivationFuncIds[neuronIndex] {
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

        fn getDerivative(neuronIndex: u32) -> f32 {
            var u = singletonNeuronData.inputs[neuronIndex] + singletonNeuronData.biases[neuronIndex];
            switch neuronActivationFuncIds[neuronIndex] {
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
            return error / f32(layerSizes[layerCount - 1]);
        }

        fn d_mse(outputIndex: u32) {
            layerDerivatives[layerCount].dl_do_prev[outputIndex] = 2 * (outputs.network[outputIndex] - outputs.desired[outputIndex]) / f32(layerSizes[layerCount - 1]);
        }
    `;
}
