import { ACTIVATIONS } from "./NetworkConsts.js";

export default class NeuralLayer {
    weightsGenerator = () => Math.random();
    activationGenerator = () => ACTIVATIONS.SIGMOID;


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
    #defAct = () => ACTIVATIONS.SIGMOID;

    constructor(size, options) {
        super(size, options);
    }

    get activationGenerator() {
        return this.#defAct;
    }
}

export class ReluLayer extends NeuralLayer {
    #defAct = () => ACTIVATIONS.RELU;

    constructor(size, options) {
        super(size, options);
    }
    
    get activationGenerator() {
        return this.#defAct;
    }
}

export class LinearLayer extends NeuralLayer {
    #defAct = () => ACTIVATIONS.LINEAR;

    constructor(size, options) {
        super(size, options);
    }
    
    get activationGenerator() {
        return this.#defAct;
    }
}

function generateVals(length, generator) {
    return Array.from({ length }, generator);
}