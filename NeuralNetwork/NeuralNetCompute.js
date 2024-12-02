export const neural_net_shader = (layerSizes, batchSize = 1, learningRate=0.01) => {
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
        struct DV {
            dl_dw: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_dw_final: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_du: array<f32, ${maxLayer}>,
            dl_du_final: array<f32, ${maxLayer}>,
            dl_do_prev: array<f32, ${maxLayer}>,
        }

        struct Ouput {
            desired: array<f32, ${layerSizes.at(-1)}>,
            network: array<f32, ${layerSizes.at(-1)}>
        }

        struct SingletonNeuronData {
            inputs: array<f32, ${totalNeurons}>,
            biases: array<f32, ${totalNeurons}>,
        }

        const layerCount: u32 = ${layerSizes.length};
        const layerSizes = array(${layerSizes.join('u, ')});
        const layerWeightStarts = array(${layerWeightStarts.join('u, ')});
        const layerCountCum = array(${layerNeuronCountCumSum.join('u, ')});
        const weight = 1f / ${batchSize}f;
        const learnRate = ${learningRate};

        @group(0) @binding(0)
        var<storage, read_write> singletonNeuronData: SingletonNeuronData;
        @group(0) @binding(1)
        var<storage, read_write> neuronWeights: array<f32, ${layerWeightStarts.at(-1)}>;
        @group(0) @binding(2)
        var<storage, read> neuronActivationFuncIds: array<u32, ${totalNeurons}>;

        @group(1) @binding(0)
        var<storage, read_write> layerDerivatives: array<DV>;
        @group(1) @binding(1)
        var<storage, read_write> outputs: Ouput;
        @group(1) @binding(2)
        var<uniform> networkStep: u32;

        @compute @workgroup_size(${maxLayer})
        fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
                @builtin(num_workgroups) num_workgroups: vec3<u32>,
                @builtin(local_invocation_id) local_invocation_id : vec3<u32>) {

                let local_index =  
                local_invocation_id.x;

            switch networkStep {
                case 0: {

                    for (var i: u32 = 1; i < layerCount; i++) {
                        if (local_index < layerSizes[i]) {
                            feedFoward(local_invocation_id.x, i);
                        }
                        storageBarrier();
                    }

                    if (local_index == 0) {
                        for (var i: u32 = 0; i < layerSizes[layerCount - 1]; i++) {
                            outputs.network[i] = getActivation(i +  layerCountCum[layerCount - 1]);
                        }
                        d_mse();
                    }
                }
                case 1: {
                    for (var i: u32 = layerCount - 1; i > 0; i--) {
                        if (local_index == 0) {
                            backPropagate(i);
                        }
                        storageBarrier();
                    }
                }
                case 2: {

                    let workgroup_index =  
                    workgroup_id.x +
                    workgroup_id.y * num_workgroups.x +
                    workgroup_id.z * num_workgroups.x * num_workgroups.y;

                    if (local_index < layerSizes[workgroup_index + 1]) {
                        descend(local_index, workgroup_index + 1);
                    }
                } 
                default {}
            }
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
            var neuronIndex = outputIndex + layerCountCum[layerIndex];
            singletonNeuronData.biases[neuronIndex] -= weight * layerDerivatives[layerIndex - 1].dl_du_final[outputIndex] * learnRate;
            layerDerivatives[layerIndex - 1].dl_du_final[outputIndex] = 0;


            for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                setWeight(outputIndex, j, layerIndex, getWeight(outputIndex, j, layerIndex) - weight * layerDerivatives[layerIndex - 1].dl_dw_final[outputIndex][j] * learnRate);
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

        fn d_mse() {
            var len = layerSizes[layerCount - 1];
            for (var i: u32 = 0; i < len; i++) {
                layerDerivatives[layerCount].dl_do_prev[i] = 2 * (outputs.network[i] - outputs.desired[i]) / f32(len);
            }
        }
    `;
}