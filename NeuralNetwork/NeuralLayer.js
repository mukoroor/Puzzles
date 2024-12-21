import { ACTIVATIONS } from "./NetworkConsts.js";

export default class NeuralLayer {
    weightsGenerator = () => Math.random();
    activationGenerator = () => ACTIVATIONS.BIAS;


    constructor(size, { activationGenerator, weightsGenerator, biasWeight } = {}) {
        this.size = size;
        if (activationGenerator) this.activationGenerator = activationGenerator;
        if (weightsGenerator) this.weightsGenerator = weightsGenerator;
        this.biasWeight = biasWeight;
    }

    getNeuronData(withBias) {
        const weights = generateVals(this.size, this.weightsGenerator);
        const activationFunctions = generateVals(this.size, this.activationGenerator);

        if (withBias) {
            activationFunctions.push(ACTIVATIONS.BIAS);
            weights.push(this.biasWeight || this.weightsGenerator());
        }

        // const dropoutMask
        return [weights, activationFunctions];
    }

    static toLayers(layerSizes) {
        return layerSizes.map(e => new NeuralLayer(e));
    }
}

export class SigmoidLayer extends NeuralLayer {

    constructor(size, { weightsGenerator, biasWeight } = {}) {
        super(size, { activationGenerator: () => ACTIVATIONS.SIGMOID, weightsGenerator, biasWeight });
    }
}

export class ReluLayer extends NeuralLayer {

    constructor(size, { weightsGenerator, biasWeight } = {}) {
        super(size, { activationGenerator: () => ACTIVATIONS.RELU, weightsGenerator, biasWeight });
    }
}

export class LinearLayer extends NeuralLayer {

    constructor(size, { weightsGenerator, biasWeight } = {}) {
        super(size, { activationGenerator: () => ACTIVATIONS.LINEAR, weightsGenerator, biasWeight });
    }
}

function generateVals(length, generator) {
    return Array.from({ length }, generator);
}