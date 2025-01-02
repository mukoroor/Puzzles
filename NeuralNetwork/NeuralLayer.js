import WGSLActivationContainer from "./ActivationFunctions.js";

class NeuralLayer {
  weightsGenerator = () => Math.random();

  constructor() {}

  static toLayers(layerSizes) {
    return layerSizes.map((e, i, arr) => {
      if (!i) {
        return new LinearLayer(e);
      } else {
        return new SigmoidLayer(e, { isTerminal: (i == arr.length - 1) })
      }
    })
  }
}


export default class ActivationLayer extends NeuralLayer {
  weightsGenerator = () => Math.random();
  activationGenerator = () => WGSLActivationContainer.getActivationId("DROPOUT");
  hasBias = true;
  isTerminal = false;

  constructor(size, options = {}) {
    super();
    const {
      activationGenerator,
      weightsGenerator,
      biasWeight,
      hasBias,
      isTerminal,
    } = options;
    this.size = size;
    this.biasWeight = biasWeight;
    if (activationGenerator) this.activationGenerator = activationGenerator;
    if (weightsGenerator) this.weightsGenerator = weightsGenerator;
    if (hasBias) this.hasBias = hasBias;
    if (isTerminal !== undefined) this.isTerminal = isTerminal;
  }

  getNeuronData(outputSize) {
    const weights = Array.from({ length: outputSize }, () => {
      const n_weights = generateVals(this.size, (_, i) => this.weightsGenerator(this.size, outputSize, i));
      if (!this.isTerminal)
        n_weights.push(
          this.hasBias ? this.biasWeight || this.weightsGenerator(this.size, outputSize, n_weights.length) : 0
        );
      return n_weights;
    });

    const activationFunctions = generateVals(
      this.size,
      this.activationGenerator
    );
    if (!this.isTerminal)
      activationFunctions.push(
        this.hasBias ? WGSLActivationContainer.getActivationId("CONSTANT") : WGSLActivationContainer.getActivationId("DROPOUT")
      );

    return [weights, activationFunctions];
  }
}

export class SigmoidLayer extends ActivationLayer {
  constructor(size, options = {}) {
    super(size, { weightsGenerator: (inSize) => (Math.random() * 2 - 1) / Math.sqrt(inSize), ...options, activationGenerator: () => WGSLActivationContainer.getActivationId("SIGMOID") });
  }
}

export class ReluLayer extends ActivationLayer {
  constructor(size, options = {}) {
    super(size, { ...options, activationGenerator: () => WGSLActivationContainer.getActivationId("ReLU") });
  }
}

export class LinearLayer extends ActivationLayer {
  constructor(size, options = {}) {
    super(size, { ...options, activationGenerator: () => WGSLActivationContainer.getActivationId("LINEAR") });
  }
}

export class ConvolutionLayer extends NeuralLayer {
  constructor ([kernelSizeX, kernelSizeY], kernel) {
    super();
  }
}

function generateVals(length, generator) {
  return Array.from({ length }, generator);
}
