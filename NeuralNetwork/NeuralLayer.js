import WGSLActivationContainer from "./ActivationFunctions.js";
import { MATRIX_MULTIPLY } from "./NeuralNetCompute.js";

class NeuralLayer {
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
  static activationGenerator = () => WGSLActivationContainer.getActivationId("DROPOUT");
  shaderPipeline = []
  forwardCallback;

  constructor(options = {}) {
    super();
    const {
      activationGenerator,
    } = options;
    this.activationGenerator = activationGenerator || ActivationLayer.activationGenerator;
  }

  initLayerData(device_connector, inputSize) {

    const dims = [inputSize, this.size]
    const [weights, funcIds] = this.generateNeuronData(inputSize);

    const bufferCreationParams = [
      [dims, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, Uint32Array],
      [weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, Float32Array],
      [funcIds, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, Uint32Array],
    ]
    const networkParamBuffers = bufferCreationParams.map(e => device_connector.createBuffer(...e));

    this.params = networkParamBuffers;
  }


  setupForwardCallback(device_connector, inputsObj, resultsObj) {
    const callbacks = this.shaderPipeline.map(e => e.createCallback(device_connector, workgroups, inputs, outputs, perShaderArgs))

    this.forwardCallback = (commandEncoder) => {
      callbacks.forEach((e, i) => e(commandEncoder, 0, i === callbacks.length - 1));
    }
  }

  forward(commandEncoder) {
    this.forwardCallback(commandEncoder);
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

export class LinearLayer extends NeuralLayer {
  static weightsGenerator = () => Math.random();
  hasBias = true;
  isTerminal = false;
  shaderPipeline = [MATRIX_MULTIPLY];

  constructor(size, options = {}) {
    super();
    const {
      hasBias,
      biasGenerator,
      weightsGenerator,
    } = options;
    if (hasBias) this.hasBias = hasBias;
    this.biasGenerator = biasGenerator || NeuralLayer.weightsGenerator;
    this.weightsGenerator = weightsGenerator || NeuralLayer.weightsGenerator;
    this.size = size;
  }

  generateWeights(inputSize) {
    const weights = Array.from({ length: inputSize + (this.hasBias && 1) }, (_, neuronIdx) => {
      let n_weights;
      if (this.hasBias && neuronIdx == inputSize) {
        n_weights = generateVals(this.size, (_, i) => this.biasGenerator(inputSize, this.size, i));
      } else {
        n_weights = generateVals(this.size, (_, i) => this.weightsGenerator(inputSize, this.size, i));
      }
      return n_weights;
    });
    return weights;
  }

  initLayerData(device_connector, inputSize) {

    const dims = [inputSize, this.size]
    const weights = this.generateWeights(inputSize);

    const bufferCreationParams = [
      [dims, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, Uint32Array],
      [weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, Float32Array],
    ]
    const [dimsBuff, weightBuff] = bufferCreationParams.map(e => device_connector.createBuffer(...e));

    this.params = {
      dimsCPU: dims,
      dims: dimsBuff,
      weightBuff
    };
  }


  setupForwardCallback(device_connector, inputsObj, resultsObj) {
    const callbacks = this.shaderPipeline.map(e => e.createCallback(device_connector, inputs, outputs, perShaderArgs))

    this.forwardCallback = (commandEncoder) => {
      callbacks.forEach((e, i) => e(commandEncoder, 0, i === callbacks.length - 1));
    }
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
