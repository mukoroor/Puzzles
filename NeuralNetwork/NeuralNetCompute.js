export const neural_net_shader = (layerSizes, batchSize = 1, learningRate=0.01) => {
    const maxLayer = layerSizes.reduce((a, c) => Math.max(a, c));
    return /*wgsl*/`
        struct Neuron {
            weightedInput: f32,
            funcId: f32, // essentialy an id mapping to which actibation fucntion to use for combination of inputs
            bias: f32,
            weights: array<f32, ${maxLayer}>,
            // ids are [0: linear, 1: sigmoid, 2: relu]
        }

        struct DV {
            dl_dw: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_dw_final: array<array<f32, ${maxLayer}>, ${maxLayer}>,
            dl_du: array<f32, ${maxLayer}>,
            dl_do_prev: array<f32, ${maxLayer}>,
            dl_du_final: array<f32, ${maxLayer}>,
        }

        const layerCount: u32 = ${layerSizes.length};
        const layerSizes = array(${layerSizes.join('u, ')});
        const weight = 1f / ${batchSize}f;
        const learnRate = ${learningRate};

        @group(0) @binding(0)
        var<storage, read> layerIndex: u32;
        @group(0) @binding(1)
        var<storage, read_write> inputNeurons: array<Neuron>;
        @group(0) @binding(2)
        var<storage, read_write> outputNeurons: array<Neuron>;
        @group(0) @binding(3)
        var<storage, read_write> derivatives: DV;
        @group(0) @binding(4)
        var<storage, read_write> derivativesNextLayer: DV;
        @group(0) @binding(5)
        var<storage, read_write> mode: u32;
        @group(0) @binding(6)
        var<storage, read> targetD: array<f32>;
        @group(0) @binding(7)
        var<storage, read_write> outputRes: array<f32>;


        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id : vec3u) {

            switch mode {
                case 0: {
                    feedFoward(global_id.x);
                    // mode += 2;
                    // outputNeurons[global_id.x].funcId = f32(mode);
                    if (layerIndex == layerCount - 1 && global_id.x == 0) {
                        for (var i: u32 = 0; i < arrayLength(&outputNeurons); i++) {
                            outputRes[i] = getActivation(&outputNeurons[i]);
                        }
                        d_mse();
                    }
                }
                case 1: {
                    backPropagate();
                    // outputNeurons[0].weights[2] = 4;
                }
                case 2: {
                    applyChanges(global_id.x);
                    // outputNeurons[0].weightedInput = 10.;
                    // outputNeurons[0].funcId = derivativesNextLayer.dl_do_prev[0];
                    // outputNeurons[0].funcId = outputRes[0];
                    // if (layerIndex == layerCount - 1) {
                    //     mode = 0;
                    // }
                } 
                default {}
            }
        }

        fn feedFoward(outputIndex: u32) {
            var newInput: f32 = 0;
            for (var i: u32 = 0; i < layerSizes[layerIndex - 1]; i++) {
                newInput += outputNeurons[outputIndex].weights[i] * getActivation(&inputNeurons[i]);
            }
            outputNeurons[outputIndex].weightedInput = newInput;
        }

        fn backPropagate() {
            for (var i: u32 = 0; i < layerSizes[layerIndex]; i++) {
                derivatives.dl_du[i] = derivativesNextLayer.dl_do_prev[i] * getDerivative(&outputNeurons[i]);
                derivatives.dl_du_final[i] += derivatives.dl_du[i];
                
                for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                    if (i == 0) { derivatives.dl_do_prev[j] = 0; }
                    derivatives.dl_dw[i][j] = derivatives.dl_du[i] * getActivation(&inputNeurons[j]);
                    derivatives.dl_dw_final[i][j] += derivatives.dl_dw[i][j];
                    derivatives.dl_do_prev[j] += derivatives.dl_du[i] * outputNeurons[i].weights[j];
                }
            }
        }

        fn applyChanges(outputIndex: u32) {
            outputNeurons[outputIndex].bias -= weight * derivatives.dl_du_final[outputIndex] * learnRate;
            derivatives.dl_du_final[outputIndex] = 0;

            for (var j: u32 = 0; j < layerSizes[layerIndex - 1]; j++) {
                outputNeurons[outputIndex].weights[j] -= weight * derivatives.dl_dw_final[outputIndex][j] * learnRate;
                derivatives.dl_dw_final[outputIndex][j] = 0;
            }
        }

        fn getActivation(neuronPtr: ptr<storage, Neuron, read_write>) -> f32 {
            switch u32((*neuronPtr).funcId){
                case 1: {
                    return sigmoid((*neuronPtr).weightedInput + (*neuronPtr).bias);                                                
                }
                case 2: {
                    return relU((*neuronPtr).weightedInput + (*neuronPtr).bias);                                                 
                }
                default: {
                    return (*neuronPtr).weightedInput + (*neuronPtr).bias;
                }
            }
        }

        fn getDerivative(neuronPtr: ptr<storage, Neuron, read_write>) -> f32 {
            switch u32((*neuronPtr).funcId) {
                case 1: {
                    return d_sigmoid((*neuronPtr).weightedInput + (*neuronPtr).bias);                                               
                }
                case 2: {
                    return d_relU((*neuronPtr).weightedInput + (*neuronPtr).bias);                                                 
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
            for (var i: u32 = 0; i < arrayLength(&outputNeurons); i++) {
                error += pow(outputRes[i] - targetD[i], 2);
            }
            return error / f32(arrayLength(&outputNeurons));
        }

        fn d_mse() {
            var len = arrayLength(&outputNeurons);
            for (var i: u32 = 0; i < len; i++) {
                derivativesNextLayer.dl_do_prev[i] = 2 * (outputRes[i] - targetD[i]) / f32(len);
            }
        }
    `;
}