export const neural_net_shader = (layerSizes, batchSize = 1, learningRate=0.01) => {
    const maxLayer = layerSizes.reduce((a, c) => Math.max(a, c));
    const totalNeurons = layerSizes.reduce((a, c) => a + c);
    const layerWeightStarts = layerSizes.reduce((a, c, i) => {
        if (!i) return [0, 0]
        let val = a[i] + c * layerSizes[i - 1];
        a.push(val);
        return a;
    }, [])
    return /*wgsl*/`
        struct DV {
            dl_dw: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_dw_final: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_du: array<f32, ${maxLayer}>,
            dl_do_prev: array<f32, ${maxLayer}>,
            dl_du_final: array<f32, ${maxLayer}>,
        }

        struct Ouput {
            desired: array<f32, ${layerSizes.at(-1)}>,
            network: array<f32, ${layerSizes.at(-1)}>
        }

        struct SingletonNeuronData {
            inputs: array<f32, ${totalNeurons}>,
            biases: array<f32, ${totalNeurons}>
        }

        const layerCount: u32 = ${layerSizes.length};
        const layerSizes = array(${layerSizes.join('u, ')});
        const layerWeightStarts = array(${layerWeightStarts.join('u, ')});
        const weight = 1f / ${batchSize}f;
        const learnRate = ${learningRate};

        @group(0) @binding(0)
        var<storage, read_write> singletonNeuronData: SingletonNeuronData;
        @group(0) @binding(1)
        var<storage, read_write> neuronWeights: array<f32, ${layerWeightStarts.at(-1)}>;
        @group(0) @binding(2)
        var<storage, read> neuronActivationFuncIds: array<u32, ${totalNeurons}>;

        @group(1) @binding(0)
        var<storage, read> inputLayerNeurons: array<u32>;
        @group(1) @binding(1)
        var<storage, read> outputLayerNeurons: array<u32>;
        @group(1) @binding(2)
        var<storage, read_write> derivativesInputLayer: DV;
        @group(1) @binding(3)
        var<storage, read_write> derivativesOutputLayer: DV;
        @group(1) @binding(4)
        var<storage, read_write> outputs: Ouput;
        @group(1) @binding(5)
        var<uniform> networkStep: u32;
        @group(1) @binding(6)
        var<uniform> layerIndex: u32;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id : vec3u) {

            switch networkStep {
                case 0: {
                    feedFoward(global_id.x);

                    if (layerIndex == layerCount - 1 && global_id.x == 0) {
                        for (var i: u32 = 0; i < arrayLength(&outputLayerNeurons); i++) {
                            outputs.network[i] = getActivation(outputLayerNeurons[i]);
                        }
                        d_mse();
                    }
                }
                case 1: {
                    backPropagate();
                }
                case 2: {
                    descend(global_id.x);
                } 
                default {}
            }
        }

        fn feedFoward(outputIndex: u32) {
            var newInput: f32 = 0;
            for (var i: u32 = 0; i < layerSizes[layerIndex - 1]; i++) {
                newInput += getWeight(outputIndex, i) * getActivation(inputLayerNeurons[i]);
            }
            singletonNeuronData.inputs[outputLayerNeurons[outputIndex]] = newInput;
        }

        fn backPropagate() {
            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                derivativesInputLayer.dl_du[i] = derivativesOutputLayer.dl_do_prev[i] * getDerivative(outputLayerNeurons[i]);
                derivativesInputLayer.dl_du_final[i] += derivativesInputLayer.dl_du[i];
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                    if (i == 0) { derivativesInputLayer.dl_do_prev[j] = 0; }
                    derivativesInputLayer.dl_dw[i][j] = derivativesInputLayer.dl_du[i] * getActivation(inputLayerNeurons[j]);
                    derivativesInputLayer.dl_dw_final[i][j] += derivativesInputLayer.dl_dw[i][j];
                    derivativesInputLayer.dl_do_prev[j] += derivativesInputLayer.dl_du[i] * getWeight(i, j);
                }
            }
        }

        fn descend(outputIndex: u32) {
            singletonNeuronData.biases[outputLayerNeurons[outputIndex]] -= weight * derivativesInputLayer.dl_du_final[outputIndex] * learnRate;
            derivativesInputLayer.dl_du_final[outputIndex] = 0;

            for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                setWeight(outputIndex, j, getWeight(outputIndex, j) - weight * derivativesInputLayer.dl_dw_final[outputIndex][j] * learnRate);
                derivativesInputLayer.dl_dw_final[outputIndex][j] = 0;
            }
        }

        fn getWeight(outputIndex: u32, weightIndex: u32) -> f32 {
            return neuronWeights[(outputLayerNeurons[outputIndex] - outputLayerNeurons[0]) * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]];
        }

        fn setWeight(outputIndex: u32, weightIndex: u32, val: f32) {
            neuronWeights[(outputIndex - outputLayerNeurons[0]) * layerSizes[layerIndex - 1] + weightIndex + layerWeightStarts[layerIndex]] = val;
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
            for (var i: u32 = 0; i < arrayLength(&outputLayerNeurons); i++) {
                error += pow(outputs.network[i] - outputs.desired[i], 2);
            }
            return error / f32(arrayLength(&outputLayerNeurons));
        }

        fn d_mse() {
            var len = arrayLength(&outputLayerNeurons);
            for (var i: u32 = 0; i < len; i++) {
                derivativesOutputLayer.dl_do_prev[i] = 2 * (outputs.network[i] - outputs.desired[i]) / f32(len);
            }
        }
    `;
}