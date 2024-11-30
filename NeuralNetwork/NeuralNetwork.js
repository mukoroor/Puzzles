import GPUConnector from "../GPUConnector.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #neuronSize;
  #cumSum;
  #training;

  constructor(layers) {
    super();
    this.layers = [...layers];
    this.maxLayer = this.layers.reduce((a,c) => Math.max(a, c))
  }

  async init() {
    await super.initGPU();

    this.setUpBuffers();
    this.createBindGroupLayouts();
    this.createBindGroups();
    this.fillLayers();
  }

  createComputeShader(batchSize, learningRate) {
    this.createShader(
      "neural_compute",
      neural_net_shader(this.layers, batchSize, learningRate)
    );
  }

  createBindGroupLayouts() {
    const bindGroupLayout0 = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        },
      ],
    });
    const bindGroupLayout1 = this.device.createBindGroupLayout({
      label: '2',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
          },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });

    this.gpuData.bindGroupLayouts.push(bindGroupLayout0);
    this.gpuData.bindGroupLayouts.push(bindGroupLayout1);
  }

  createComputePipeline() {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [...this.gpuData.bindGroupLayouts],
      }),
      compute: {
        module: this.getShader("neural_compute"),
        entryPoint: "main",
      },
    });
  }

  createLayerBindGroup(layerIndex) {
    const bindGroup0 = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[0],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer('Neuron_Inputs&Biases'),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer('Neuron_Weights'),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer('Neuron_Activation_Func_Ids'),
          },
        },
      ],
    });
    const bindGroup1 = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[1],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer(this.generateLayerNeuronsName(layerIndex - 1)),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(this.generateLayerNeuronsName(layerIndex)),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer(this.generateLayerDerivativesName(layerIndex)),
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.getBuffer(this.generateLayerDerivativesName(layerIndex + 1)),
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.getBuffer(`target&result`),
          },
        },
        {
          binding: 5,
          resource: {
            buffer: this.getBuffer(`Network_Mode`),
          },
        },
        {
          binding: 6,
          resource: {
            buffer: this.getBuffer(this.generateLayerIndexName(layerIndex)),
          },
        },
      ],
    });
    this.gpuData.bindGroups.push([bindGroup0, bindGroup1]);
  }

  setUpBuffers() {
    this.createBuffer(
      `Neuron_Inputs&Biases`,
      Float32Array.BYTES_PER_ELEMENT * this.neuronCumulativeSum(this.layers.length - 1) * 2,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Neuron_Weights`,
      Float32Array.BYTES_PER_ELEMENT * this.neuronCumulativeWeightsCount(this.layers.length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Neuron_Activation_Func_Ids`,
      Uint32Array.BYTES_PER_ELEMENT * this.neuronCumulativeSum(this.layers.length - 1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      `Network_Mode`,
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      `target&result`,
      Float32Array.BYTES_PER_ELEMENT * this.layers.at(-1) * 2,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      this.generateLayerDerivativesName(this.layers.length),
      Float32Array.BYTES_PER_ELEMENT * this.maxLayer * (2 * this.maxLayer + 3),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.layers.forEach((_, i) => this.createLayerBuffers(i));
  }

  fillLayers() {
    this.layers.forEach((_, i) => this.fillLayerBuffer(i))
  }

  createBindGroups() {
    this.layers.forEach((_, i) => {
      if (i) this.createLayerBindGroup(i);
    })
  }

  fillLayerBuffer(layerIndex, neuronData, init=true) {
    if (!neuronData) {
      neuronData = [
        Array(this.layers[layerIndex]).fill(0),
        Array.from({length: (this.layers[layerIndex - 1] || 0) * this.layers[layerIndex]}, () => Math.random()),
        Array.from({length: this.layers[layerIndex]}, () => Math.random()),
        Array(this.layers[layerIndex]).fill(layerIndex ? 1: 0),
      ]
    }
    let [inputs, weights, biases, funcIds] = neuronData;
    const offset = this.neuronCumulativeSum(layerIndex - 1) || 0;
    
    this.writeBuffer1to1(this.generateLayerIndexName(layerIndex), new Uint32Array([layerIndex]));
    if (init) {
      const NEURON_IDXS = new Uint32Array(Array.from({length: this.layers[layerIndex]}, (_, i) => i + (offset)))
      this.writeBuffer1to1(this.generateLayerNeuronsName(layerIndex), NEURON_IDXS);
    }
    if (inputs) {
      const INPUTS = new Float32Array(inputs);
      this.writeBuffer('Neuron_Inputs&Biases', offset * Float32Array.BYTES_PER_ELEMENT, INPUTS);
    }
    if (weights) {
      const WEIGHTS = new Float32Array(weights);
      this.writeBuffer('Neuron_Weights', this.neuronCumulativeWeightsCount(layerIndex - 1) * Float32Array.BYTES_PER_ELEMENT, WEIGHTS);
    }
    if (biases) {
      const BIASES = new Float32Array(biases);
      this.writeBuffer('Neuron_Inputs&Biases', Float32Array.BYTES_PER_ELEMENT * this.neuronCumulativeSum(this.layers.length - 1) + offset * Float32Array.BYTES_PER_ELEMENT, BIASES);
    }
    if (funcIds) {
      const FUNC_IDS = new Uint32Array(funcIds);
      this.writeBuffer('Neuron_Activation_Func_Ids', offset * Uint32Array.BYTES_PER_ELEMENT, FUNC_IDS);
    }
  }

  createLayerBuffers(layerIndex) {
    this.createBuffer(
      this.generateLayerIndexName(layerIndex),
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      this.generateLayerNeuronsName(layerIndex),
      Uint32Array.BYTES_PER_ELEMENT * this.neuronSize * this.layers[layerIndex],
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      this.generateLayerDerivativesName(layerIndex),
      Float32Array.BYTES_PER_ELEMENT * this.maxLayer * (2 * this.maxLayer + 3),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
  }

  get neuronSize() {
    if (this.#neuronSize) return this.#neuronSize;
    this.#neuronSize = 1;
    return this.#neuronSize;
  }

  neuronCumulativeWeightsCount(stoppingLayerIndex) {
    return this.layers.reduce((a, c, i) => i >= stoppingLayerIndex ? a : a + c * (this.layers[i + 1] || 0), 0)
  }

  neuronCumulativeSum(stoppingLayerIndex) {
    if (!this.#cumSum) {
      this.#cumSum = this.layers.reduce((a, c) => {
        a.push((a[a.length - 1] || 0) + c);
        return a;
      }, [])
    }
    return this.#cumSum[stoppingLayerIndex];
  }

  set neuronSize(newNeuronSize) {
    this.#neuronSize = newNeuronSize;
  }

  generateRandomNeuron() {
    return [0, 1, ...Array.from({length: this.neuronSize - 2}, () => Math.random())];
  }

  fillData(features, output) {
    this.fillLayerBuffer(0, [features], false);
    if (output) this.writeBuffer('target&result', 0, new Float32Array(output));
  } 

  async train(points, outputs, epochs=10000, learningRate=0.04, batchSize=points.length) {
    this.checkValidDimensions(points, outputs);
    console.time("train")

    if (!this.device) await this.init();

    this.createComputeShader(batchSize, learningRate);
    this.setPipeline("main", this.createComputePipeline());


    return new Promise((res, rej) => {
      const resolve = async() => {
        await this.device.queue.onSubmittedWorkDone();
        res();
        console.timeEnd('train'); 
      }

      requestAnimationFrame(() => this.#trainingLoop(0, epochs, points, outputs, batchSize, 0, resolve))
    })
  }

  async #trainingLoop(currEpoch, maxEpoch, points, outputs, batchSize, batchIndex, finish) {
    const start = performance.now();
    while(currEpoch < maxEpoch && (performance.now() - start) <= MAX_TRAINING_INTERRUPT) {
      let batchOffset = (currEpoch * batchSize) % points.length;
      this.fillData(points[(batchIndex + batchOffset) % points.length], outputs[(batchIndex + batchOffset) % points.length]);
      this.forward();
      this.backward();
      if (++batchIndex == batchSize) {
        this.descent();
        batchIndex = 0;
        currEpoch++;
      }
    }

    if (currEpoch != maxEpoch) requestAnimationFrame(() => this.#trainingLoop(currEpoch, maxEpoch, points, outputs, batchSize, batchIndex, finish));
    else finish();
  }


  async extractNetworkParameters() {
    const layers = [];
    const commandEncoder = this.device.createCommandEncoder();

    for (let i = 0; i < this.layers.length; i++) {
      this.copyBuffer(this.generateLayerNeuronsName(i), commandEncoder);
    }
    this.copyBuffer(`Neuron_Inputs&Biases`, commandEncoder);
    this.copyBuffer(`Neuron_Weights`, commandEncoder);

    this.device.queue.submit([commandEncoder.finish()]);

    for (let i = 0; i < this.layers.length; i++) {
      layers.push(await this.mapBufferToCPU(`${this.generateLayerNeuronsName(i)}_copy`, Uint32Array));
    }
    layers.push(await this.mapBufferToCPU(`Neuron_Inputs&Biases_copy`, Float32Array));
    layers.push(await this.mapBufferToCPU(`Neuron_Weights_copy`, Float32Array));

    return layers
  }

  async extractOutput() {
    const commandEncoder = this.device.createCommandEncoder();
    const offset = Float32Array.BYTES_PER_ELEMENT * this.layers.at(-1);
    this.copyBuffer(`target&result`, commandEncoder, offset, offset);
    this.device.queue.submit([commandEncoder.finish()]);

    const outputResult = await this.mapBufferToCPU(`target&result_copy`, Float32Array);

    return outputResult;
  }

  generateLayerName(layerIndex) {
    return `Layer_${layerIndex}`
  }

  generateLayerNeuronsName(layerIndex) {
    return `${this.generateLayerName(layerIndex)}_neurons`
  }

  generateLayerIndexName(layerIndex) {
    return `${this.generateLayerName(layerIndex)}_index`
  }

  generateLayerDerivativesName(layerIndex) {
    return `${this.generateLayerName(layerIndex)}_derivatives`
  }

  forward() {
    const commandEncoder = this.device.createCommandEncoder();
    this.writeBuffer1to1(`Network_Mode`, new Uint32Array([0]));
    this.#layerPassForward(commandEncoder)
    this.device.queue.submit([commandEncoder.finish()]);
  }

  backward() {
    const commandEncoder = this.device.createCommandEncoder();
    this.writeBuffer1to1(`Network_Mode`, new Uint32Array([1]));
    const passEncoder = commandEncoder.beginComputePass();
    for (let i = this.layers.length - 1; i > 0; i--) {
      passEncoder.setPipeline(this.getPipeline("main"));
      passEncoder.setBindGroup(0, this.gpuData.bindGroups[i - 1][0]);
      passEncoder.setBindGroup(1, this.gpuData.bindGroups[i - 1][1]);
      passEncoder.dispatchWorkgroups(1);
    }
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
  
  descent() {
    const commandEncoder = this.device.createCommandEncoder();
    this.writeBuffer1to1(`Network_Mode`, new Uint32Array([2]));
    this.#layerPassForward(commandEncoder)
    this.device.queue.submit([commandEncoder.finish()]);
  }

  #layerPassForward(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    for (let i = 1; i < this.layers.length; i++) {
      passEncoder.setPipeline(this.getPipeline("main"));
      passEncoder.setBindGroup(0, this.gpuData.bindGroups[i - 1][0]);
      passEncoder.setBindGroup(1, this.gpuData.bindGroups[i - 1][1]);
      passEncoder.dispatchWorkgroups(this.layers[i]);
    }
    passEncoder.end();
  }

  async predict(points) {
    this.checkValidDimensions(points);

    if (!this.device) await this.init();
    if (!this.getShader("neural_compute")) {
      this.createComputeShader(1, 1);
      this.setPipeline("main", this.createComputePipeline());
    }

    return new Promise((res, rej) => {
        requestAnimationFrame(() => this.#predictionLoop(points, 0, [], res))
    });
  }

  async #predictionLoop(points, pointIndex, predictions, res) {
    const start = performance.now();
    while(pointIndex < points.length && (performance.now() - start) <= MAX_TRAINING_INTERRUPT) {
      this.fillData(points[pointIndex++]);
      this.forward();
      predictions.push(await this.extractOutput());
    }

    if (pointIndex == points.length) res(predictions);
    else requestAnimationFrame(() => this.#predictionLoop(points, pointIndex, predictions, res));
    
  }

  static meanSquaredError(y, yPred) {
    let error = 0;
    for (let i = 0; i < Math.min(y.length, yPred.length); i++) {
      let currError = 0;
      for (let j = 0; j < Math.min(y[i].length, yPred[i].length); j++) {
        currError += ((y[i][j] || 0) - (yPred[i][j] || 0)) ** 2;
      }
      error += currError / Math.min(y[i].length, yPred[i].length)
    }
    return error / Math.min(y.length, yPred.length);
  }

  checkValidDimensions(points, outputs=undefined) {
    if (points[0]?.length != this.layers[0]
      || outputs && (outputs[0].length != this.layers.at(-1)
      || points?.length != outputs.length)) throw new Error("invalid data dimensions")
  }
}

const MAX_TRAINING_INTERRUPT = 8;
// function generateExpected(epochs) {
//   let [input, w1, b1, w2, b2, output] = [1, 1/3, 1, 0.5, 1, 1];

//   const trace = [];

//   for (let i = 0; i < epochs; i++) {
//     let u1 = 3 * w1 * input + b1;
//     let o1 = sigmoid(u1);
//     let u2 = 2 * w2 * o1 + b2;
//     let o2 = sigmoid(u2);
//     let loss = (o2 - output) ** 2;

//     let dl_do2 = 2 * (o2 - output);
//     let dl_du2 = dl_do2 * o2 * (1 - o2);
//     let dl_dw2 = dl_du2 * o1;
//     let dl_db2 = dl_du2;

//     let dl_do1 = dl_du2 * w2;
//     let dl_du1 = dl_do1 * o1 * (1 - o1);
//     let dl_dw1 = dl_du1 * input;
//     let dl_db1 = dl_du1;

//     w1 -= dl_dw1;
//     b1 -= dl_db1;
//     w2 -= dl_dw2;
//     b2 -= dl_db2;

//     if (i + 1 != epochs) continue;
//     console.log('epoch '  + i)
//     console.log([input, w1, b1, w2, b2, output])
//     console.log(loss)
//     console.log(u2)
//     console.log('2', [dl_do2, dl_du2, dl_dw2, dl_db2])
//     console.log('1', [dl_do1, dl_du1, dl_dw1, dl_db1])

//   }
//   return [w1, b1, w2, b2]
// }

// function pred(input, [w1, b1, w2, b2]) {
//   let u1 = w1 * (input.reduce((a, c) => a + c)) + b1;
//   let o1 = sigmoid(u1);
//   let u2 = 2 * w2 * o1 + b2;
//   let o2 = sigmoid(u2);
//   return o2;
// }
// let iter = 800;
// generateExpected(iter)

// function sigmoid(v) {
//   return 1 / (1 + Math.exp(-v))
// }

// let counter = 0


