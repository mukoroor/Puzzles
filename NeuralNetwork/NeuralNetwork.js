import GPUConnector from "../GPUConnector.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #cumSum;

  constructor(layers) {
    super();
    this.layers = [...layers];
    this.maxLayer = this.layers.reduce((a, c) => Math.max(a, c));
  }

  async init() {
    await super.initGPU();

    this.createBuffers();
    this.createBindGroupLayouts();
    this.createBindGroups();
    this.fillLayerData();
    this.createComputeShader();
  }

  createBuffers() {
    this.createBuffer(
      `Neuron_Inputs&Biases`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeSum(this.layers.length - 1) *
        2,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Neuron_Weights`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeWeightsCount(this.layers.length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Neuron_Activation_Func_Ids`,
      Uint32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeSum(this.layers.length - 1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "Layer_Derivatives",
      Float32Array.BYTES_PER_ELEMENT *
        this.maxLayer *
        (2 * this.maxLayer + 3) *
        this.layers.length,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Network_Step`,
      Uint32Array.BYTES_PER_ELEMENT * 4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      `Outputs`,
      Float32Array.BYTES_PER_ELEMENT * this.layers.at(-1) * 2,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Params`,
      Float32Array.BYTES_PER_ELEMENT * 5,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
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
            type: "uniform",
          },
        },
      ],
    });
    const bindGroupLayout2 = this.device.createBindGroupLayout({
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
            type: "storage",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });

    this.gpuData.bindGroupLayouts.push(
      bindGroupLayout0,
      bindGroupLayout1,
      bindGroupLayout2
    );
  }

  createBindGroups() {
    const bindGroup0 = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[0],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer("Neuron_Inputs&Biases"),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer("Neuron_Weights"),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer("Neuron_Activation_Func_Ids"),
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
            buffer: this.getBuffer(`Layer_Derivatives`),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(`Outputs`),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer(`Network_Step`),
          },
        },
      ],
    });
    this.gpuData.bindGroups.push(bindGroup0, bindGroup1, undefined);
  }

  createComputeShader() {
    this.createShader("neural_compute", neural_net_shader(this.layers));
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

  updateDynamicBindGroup() {
    const dynamicBindGroup = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[2],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer(`Input_Data`),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(`Output_Data`),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer(`Params`),
          },
        },
      ],
    });
    this.gpuData.bindGroups[2] = dynamicBindGroup;
  }

  fillLayerBuffer(layerIndex, neuronData) {
    if (!neuronData) {
      neuronData = [
        Array(this.layers[layerIndex]).fill(0),
        Array.from(
          { length: this.layers[layerIndex] },
          () => Math.random() * 2 - 1
        ),
        // Array(this.layers[layerIndex]).fill(layerIndex ? 1: 0),
        Array.from(
          {
            length:
              (this.layers[layerIndex - 1] || 0) * this.layers[layerIndex],
          },
          () => Math.random() * 2 - 1
        ),
        // Array.from({length: (this.layers[layerIndex - 1] || 0) * this.layers[layerIndex]}, () => (1 / this.layers[layerIndex - 1] || 0)),
        Array(this.layers[layerIndex]).fill(layerIndex ? 1 : 0),
      ];
    }
    let [inputs, biases, weights, funcIds] = neuronData;
    const offset = this.neuronCumulativeSum(layerIndex - 1) || 0;

    if (inputs) {
      const INPUTS = new Float32Array(inputs);
      this.writeBuffer(
        "Neuron_Inputs&Biases",
        offset * Float32Array.BYTES_PER_ELEMENT,
        INPUTS
      );
    }
    if (biases) {
      const BIASES = new Float32Array(biases);
      this.writeBuffer(
        "Neuron_Inputs&Biases",
        Float32Array.BYTES_PER_ELEMENT *
          this.neuronCumulativeSum(this.layers.length - 1) +
          offset * Float32Array.BYTES_PER_ELEMENT,
        BIASES
      );
    }
    if (weights) {
      const WEIGHTS = new Float32Array(weights);
      this.writeBuffer(
        "Neuron_Weights",
        this.neuronCumulativeWeightsCount(layerIndex - 1) *
          Float32Array.BYTES_PER_ELEMENT,
        WEIGHTS
      );
    }
    if (funcIds) {
      const FUNC_IDS = new Uint32Array(funcIds);
      this.writeBuffer(
        "Neuron_Activation_Func_Ids",
        offset * Uint32Array.BYTES_PER_ELEMENT,
        FUNC_IDS
      );
    }
  }

  fillTrainingPointData(features, output) {
    this.fillLayerBuffer(0, [features], false);
    if (output) this.writeBuffer("Outputs", 0, new Float32Array(output));
  }

  fillLayerData() {
    this.layers.forEach((_, i) => this.fillLayerBuffer(i));
  }

  fillParams(offset, params) {
    this.writeBuffer(
      "Params",
      offset * Float32Array.BYTES_PER_ELEMENT,
      new Float32Array(params)
    );
  }

  fillData(points, outputs = []) {
    this.createBuffer(
      `Input_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers[0],
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.writeBuffer1to1(`Input_Data`, new Float32Array(points.flat()));

    this.createBuffer(
      `Output_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers.at(-1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    if (outputs.length) this.writeBuffer1to1(`Output_Data`, new Float32Array(outputs.flat()));
  }

  neuronCumulativeWeightsCount(stoppingLayerIndex) {
    return this.layers.reduce(
      (a, c, i) =>
        i >= stoppingLayerIndex ? a : a + c * (this.layers[i + 1] || 0),
      0
    );
  }

  neuronCumulativeSum(stoppingLayerIndex) {
    if (!this.#cumSum) {
      this.#cumSum = this.layers.reduce((a, c) => {
        a.push((a[a.length - 1] || 0) + c);
        return a;
      }, []);
    }
    return this.#cumSum[stoppingLayerIndex];
  }

  async train(
    points,
    outputs,
    epochs = 10000,
    learningRate = 0.04,
    batchSize = points.length
  ) {
    this.checkValidDimensions(points, outputs);
    console.time("train");

    if (!this.device) await this.init();

    this.fillParams(0, [
      learningRate,
      1,
      Math.ceil(points.length / batchSize),
      batchSize,
      0,
    ]);
    this.fillData(points, outputs);
    this.updateDynamicBindGroup();
    this.createComputeShader();
    this.setPipeline("main", this.createComputePipeline());

    return new Promise((res) => {
      const resolve = async () => {
        await this.device.queue.onSubmittedWorkDone();
        res();
      };

      requestAnimationFrame(() =>
        this.#trainingLoop(0, epochs, points, outputs, batchSize, 0, resolve)
      );
    });
  }

  async #trainingLoop(
    currEpoch,
    maxEpoch,
    points,
    outputs,
    batchSize,
    batchIndex,
    finish
  ) {
    const start = performance.now();
    while(currEpoch < maxEpoch && (performance.now() - start) <= MAX_TRAINING_INTERRUPT) {
      // if (currEpoch % 100 == 0) console.log('epoch', currEpoch)
    // let batchOffset = (currEpoch * batchSize) % points.length;
    // this.fillTrainingPointData(points[(batchIndex + batchOffset) % points.length], outputs[(batchIndex + batchOffset) % points.length]);
      this.forward();
    // finish();
    // this.backward();
    // if (++batchIndex == batchSize) {
    //   this.descent();
    //   batchIndex = 0;
      currEpoch++;
    // }
    }

    if (currEpoch == maxEpoch) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        console.timeEnd("train");
      })
      finish();
    }
    else requestAnimationFrame(() => this.#trainingLoop(currEpoch, maxEpoch, points, outputs, batchSize, batchIndex, finish));
  }

  async extractNetworkParameters() {
    const params = {};
    const commandEncoder = this.device.createCommandEncoder();

    this.copyBuffer(`Neuron_Inputs&Biases`, commandEncoder);
    this.copyBuffer(`Neuron_Weights`, commandEncoder);
    this.copyBuffer(`Params`, commandEncoder);
    // this.copyBuffer(`Layer_Derivatives`, commandEncoder);

    this.device.queue.submit([commandEncoder.finish()]);

    const tempInpBias = await this.mapBufferToCPU(
      `Neuron_Inputs&Biases_copy`,
      Float32Array
    );
    params.inputs = tempInpBias.slice(0, tempInpBias.length / 2);
    params.biases = tempInpBias.slice(tempInpBias.length / 2);

    params.params = await this.mapBufferToCPU(`Params_copy`, Float32Array);
    // params.derv = await this.mapBufferToCPU(
      //   `Layer_Derivatives_copy`,
      //   Float32Array
      // );
      
    const allWeights = await this.mapBufferToCPU(
      `Neuron_Weights_copy`,
      Float32Array
    );
    const segmentedWeights = [];
    let pointer = 0;
    for (let i = 0; i < this.layers.length - 1; i++) {
      const weights_i = [];
      for (let j = 0; j < this.layers[i + 1]; j++) {
        weights_i.push(allWeights.slice(pointer, pointer + this.layers[i]));
        pointer += this.layers[i];
      }
      segmentedWeights.push(weights_i);
    }
    params.weights = segmentedWeights;

    return params;
  }

  async extractNetworkOutput() {
    const commandEncoder = this.device.createCommandEncoder();
    this.copyBuffer(`Output_Data`, commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);

    const outputData = await this.mapBufferToCPU(
      `Output_Data_copy`,
      Float32Array
    );

    const separatedOutputs = [];
    const outPutDim = this.layers.at(-1);
    console.log(outPutDim, outputData.length)

    for (let i = 0; i < outputData.length; i+=outPutDim) {
      separatedOutputs.push(Array.from(outputData.slice(i, i + outPutDim)));
    }

    return separatedOutputs;
  }

  forward() {
    const commandEncoder = this.device.createCommandEncoder();
    // this.writeBuffer1to1(`Network_Step`, new Uint32Array([0]));
    this.#layerPassForward(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
  }

  backward() {
    const commandEncoder = this.device.createCommandEncoder();
    // this.writeBuffer1to1(`Network_Step`, new Uint32Array([1]));
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline("main"));
    passEncoder.setBindGroup(0, this.gpuData.bindGroups[0]);
    passEncoder.setBindGroup(1, this.gpuData.bindGroups[1]);
    passEncoder.setBindGroup(2, this.gpuData.bindGroups[2]);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  descent() {
    const commandEncoder = this.device.createCommandEncoder();
    // this.writeBuffer1to1(`Network_Step`, new Uint32Array([2]));
    this.#layerPassForward(commandEncoder, [this.layers.length - 1]);
    this.device.queue.submit([commandEncoder.finish()]);
  }

  #layerPassForward(commandEncoder, workgroups = [1]) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline("main"));
    passEncoder.setBindGroup(0, this.gpuData.bindGroups[0]);
    passEncoder.setBindGroup(1, this.gpuData.bindGroups[1]);
    passEncoder.setBindGroup(2, this.gpuData.bindGroups[2]);
    passEncoder.dispatchWorkgroups(...workgroups);
    passEncoder.end();
  }

  async predict(points) {
    this.checkValidDimensions(points);

    if (!this.device) {
      await this.init();
      this.setPipeline("main", this.createComputePipeline());
    }

    return new Promise((res) => {
      requestAnimationFrame(() => this.#predictionLoop(points, res));
    });
  }

  async #predictionLoop(points, finish) {
    // const start = performance.now();
    // while(pointIndex < points.length && (performance.now() - start) <= MAX_TRAINING_INTERRUPT) {
    // this.fillTrainingPointData(points[pointIndex++]);
    this.fillParams(2, [1, points.length, 1]);
    this.fillData(points);
    this.updateDynamicBindGroup();
    this.forward();
    finish(await this.extractNetworkOutput());
    // }

    // if (pointIndex == points.length) res(predictions);
    // else requestAnimationFrame(() => this.#predictionLoop(points, pointIndex, predictions, res));
  }

  static meanSquaredError(y, yPred) {
    let error = 0;
    for (let i = 0; i < Math.min(y.length, yPred.length); i++) {
      let currError = 0;
      for (let j = 0; j < Math.min(y[i].length, yPred[i].length); j++) {
        currError += ((y[i][j] || 0) - (yPred[i][j] || 0)) ** 2;
      }
      error += currError / Math.min(y[i].length, yPred[i].length);
    }
    return error / Math.min(y.length, yPred.length);
  }

  checkValidDimensions(points, outputs = undefined) {
    if (
      points[0]?.length != this.layers[0] ||
      (outputs &&
        (outputs[0].length != this.layers.at(-1) ||
          points?.length != outputs.length))
    )
      throw new Error("invalid data dimensions");
  }
}

const MAX_TRAINING_INTERRUPT = 2;
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
