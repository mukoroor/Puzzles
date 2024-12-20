import GPUConnector from "../GPUConnector.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #cumSum;
  #weightCounts;
  batchSize = 1;

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
      `Neuron_Weights`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeWeightsCount(this.layers.length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Neuron_Activation_Func_Ids`,
      Uint32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeSum(this.layers.length - 1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      "Batch_Derivatives",
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeWeightsCount(this.layers.length)
        * this.batchSize,
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
            type: "read-only-storage",
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
            buffer: this.getBuffer("Neuron_Weights"),
          },
        },
        {
          binding: 1,
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
            buffer: this.getBuffer(`Batch_Derivatives`),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(`Params`),
          },
        },        
      ],
    });
    this.gpuData.bindGroups.push(bindGroup0, bindGroup1, undefined);
  }

  createComputeShader() {
    this.createShader("neural_compute", neural_net_shader(this));
  }

  createComputePipelines() {
    return [
      this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [...this.gpuData.bindGroupLayouts],
      }),
      compute: {
        module: this.getShader("neural_compute"),
        entryPoint: "main",
      },
    }),
    this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [...this.gpuData.bindGroupLayouts],
      }),
      compute: {
        module: this.getShader("neural_compute"),
        entryPoint: "descent",
      },
    }),
  ];
  }

  setAllPipelines() {
    const [fB, descent] = this.createComputePipelines();
    
    this.setPipeline('forward_backward', fB);
    this.setPipeline('descent', descent);
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
            buffer: this.getBuffer(`Expected_Output_Data`),
          },
        },
      ],
    });
    this.gpuData.bindGroups[2] = dynamicBindGroup;
  }

  fillLayerBuffer(layerIndex, neuronData) {
    if (!neuronData) {
      neuronData = [
        // Array(this.layers[layerIndex]).fill(0),
        // Array.from(
        //   { length: this.layers[layerIndex] },
        //   () => Math.random() * 2 - 1
        // ),
        // Array(this.layers[layerIndex]).fill(layerIndex ? 1: 0),
        Array.from(
          {
            length:
              (this.layers[layerIndex - 1] + 1 || 0) * this.layers[layerIndex],
          },
          () => Math.random() * 2 - 1
        ),
        // Array.from({length: (this.layers[layerIndex - 1] + 1 || 0) * this.layers[layerIndex]}, () => (1 / this.layers[layerIndex - 1] || 0)),
        // [
        //   [],
        //   [...Array(3).fill(1/3), 1, ...Array(3).fill(1/3), 1],
        //   [0.5, 0.5, 1]
        // ][layerIndex],
        Array.from({ length: this.layers[layerIndex] + (layerIndex == this.layers.length - 1 ? 0 : 1)}, (_, i) => (i != this.layers[layerIndex] ? layerIndex ? 2 : 1 : 0)),
      ];
    }
    let [weights, funcIds] = neuronData;

    if (weights && weights.length) {
      const WEIGHTS = new Float32Array(weights);
      this.writeBuffer(
        "Neuron_Weights",
        this.neuronCumulativeWeightsCount(layerIndex - 1) *
          Float32Array.BYTES_PER_ELEMENT,
        WEIGHTS
      );
    }
    if (funcIds && funcIds.length) {
      const FUNC_IDS = new Uint32Array(funcIds);
      this.writeBuffer(
        "Neuron_Activation_Func_Ids",
        (this.neuronCumulativeSum(layerIndex - 1) || 0) * Uint32Array.BYTES_PER_ELEMENT,
        FUNC_IDS
      );
    }
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

    this.createBuffer(
      `Expected_Output_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers.at(-1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    if (outputs.length) this.writeBuffer1to1(`Expected_Output_Data`, new Float32Array(outputs.flat()));
  }

  neuronCumulativeWeightsCount(stoppingLayerIndex) {
    return this.weightCounts[stoppingLayerIndex];
  }

  neuronCumulativeSum(stoppingLayerIndex) {
    return this.cumSum[stoppingLayerIndex];
  }

  get cumSum() {
    if (!this.#cumSum) {
      this.#cumSum = this.layers.reduce((a, c, i, arr) => {
        a.push((a.at(-1) || 0) + c + (arr.at(i + 1) ? 1 : 0))
        return a;
      }, []);
    }
    return this.#cumSum;
  }

  get weightCounts() {
    if (!this.#weightCounts) {
      this.#weightCounts = this.layers.reduce((a, c, i, arr) => {
        a.push(a.at(-1) + (c + 1) * (arr[i + 1] || 0));
        return a;
      }, [0]);
    }
    return this.#weightCounts;
  }

  async train(
    points,
    outputs,
    epochs = 1000,
    learningRate = 0.01,
    batchSize = points.length
  ) {
    this.checkValidDimensions(points, outputs);
    console.time("train");

    if (!this.device) await this.init();

    this.fillParams(0, [
      learningRate,
      // 0,
      // Math.ceil(points.length / batchSize),
      batchSize,
      0,
    ]);
    this.fillData(points, outputs);
    this.updateDynamicBindGroup();
    this.createComputeShader();
    this.setAllPipelines();

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
    finish,
    a = []
  ) {
    const start = performance.now();
    while(currEpoch < maxEpoch && (performance.now() - start) <= MAX_TRAINING_INTERRUPT) {
      // let batchOffset = (currEpoch * batchSize) % points.length;
      {     
        const commandEncoder = this.device.createCommandEncoder(); 
        this.forwardBackwardWave(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
      }
      {     
        const commandEncoder = this.device.createCommandEncoder(); 
        this.descent(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
      }
      
      if (currEpoch % 10 == 0) {
        await this.device.queue.onSubmittedWorkDone();
        let t = await this.extractNetworkOutput();
        a.push([currEpoch, NeuralNetwork.meanSquaredError(outputs, t)]);
      }
      currEpoch++;
    }

    if (currEpoch == maxEpoch) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        console.timeEnd("train");
        console.log(a);
      })
      finish();
    }
    else requestAnimationFrame(() => this.#trainingLoop(currEpoch, maxEpoch, points, outputs, batchSize, batchIndex, finish, a));
  }

  async extractNetworkParameters() {
    const params = {};
    const commandEncoder = this.device.createCommandEncoder();

    this.copyBuffer(`Params`, commandEncoder);
    this.copyBuffer(`Neuron_Activation_Func_Ids`, commandEncoder);
    this.copyBuffer(`Batch_Derivatives`, commandEncoder);
    this.copyBuffer(`Neuron_Weights`, commandEncoder);

    this.device.queue.submit([commandEncoder.finish()]);

    params.params = await this.mapBufferToCPU(`Params_copy`, Float32Array);
    params.functionIds = await this.mapBufferToCPU(`Neuron_Activation_Func_Ids_copy`, Uint32Array);

    const derv = await this.mapBufferToCPU(
        `Batch_Derivatives_copy`,
        Float32Array
    );

    const segDerv = [];
    const size = this.neuronCumulativeWeightsCount(this.layers.length);
    for (let i = 0; i < this.batchSize * (this.layers.length - 1); i++) {
      segDerv.push(derv.slice(i * size, (i + 1) * size))
    }
    params.derv = segDerv;
      
    const allWeights = await this.mapBufferToCPU(
      `Neuron_Weights_copy`,
      Float32Array
    );
    const segmentedWeights = [];

    let pointer = 0;
    for (let i = 0; i < this.layers.length - 1; i++) {
      const weights_i = [];
      for (let j = 0; j < this.layers[i + 1]; j++) {
        weights_i.push(allWeights.slice(pointer, pointer + this.layers[i] + 1));
        pointer += this.layers[i] + 1;
      }
      segmentedWeights.push(weights_i);
    }
    params.weights = segmentedWeights;

    return params;
  }

  // async expose(points, ex) {
  //   const params = await this.extractNetworkParameters();
  //   const res = await this.extractNetworkOutput();
  //   const p = []

  //   for (let z = 0; z < points.length; z++) {
  //     let inputs = points[z];
  //     let tot = inputs.length;
  //     // let sum;
  //     // for (let i = 1; i < this.layers.length; i++) {
  //     //   const outputs = Array(this.layers[i]);
  //     //   sum = [...outputs];
  //     //   for (let j = 0; j < outputs.length; j++) {
  //     //     outputs[j] = params.biases[tot + j];
  //     //     for (let k = 0; k < inputs.length; k++) {
  //     //       outputs[j] += params.weights[i - 1][j][k] * inputs[k];
  //     //     }
  //     //     sum[j] = outputs[j];
  //     //     outputs[j] = sigmoid(sum[j]);
  //     //   }
  //     //   tot += outputs.length;
  //     // }
  //     inputs = res[z];

  //     let loss = [...inputs];
  //     let loss_d = [...loss];

  //     for (let i = 0; i < loss.length; i++) {
  //       loss[i] -= ex[z][i];
  //       loss_d[i] = loss[i] * dsigmoid(Math.log(loss_d[i]) - Math.log(1 - loss_d[i]))
  //     }
  //     p.push({inputs, loss, loss_d});
  //   }
  //   return p
  // }

  async extractNetworkOutput() {
    await this.device.queue.onSubmittedWorkDone();
    const commandEncoder = this.device.createCommandEncoder();
    this.copyBuffer(`Output_Data`, commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);

    const outputData = await this.mapBufferToCPU(
      `Output_Data_copy`,
      Float32Array
    );

    const separatedOutputs = [];
    const outPutDim = this.layers.at(-1);

    for (let i = 0; i < outputData.length; i+=outPutDim) {
      separatedOutputs.push(Array.from(outputData.slice(i, i + outPutDim)));
    }

    return separatedOutputs;
  }

  forwardBackwardWave(commandEncoder) {
    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline("forward_backward"));
    passEncoder.setBindGroup(0, this.gpuData.bindGroups[0]);
    passEncoder.setBindGroup(1, this.gpuData.bindGroups[1]);
    passEncoder.setBindGroup(2, this.gpuData.bindGroups[2]);
    passEncoder.dispatchWorkgroups(this.batchSize);
    passEncoder.end();
  }

  descent(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline("descent"));
    passEncoder.setBindGroup(0, this.gpuData.bindGroups[0]);
    passEncoder.setBindGroup(1, this.gpuData.bindGroups[1]);
    passEncoder.setBindGroup(2, this.gpuData.bindGroups[2]);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
  }

  async predict(points) {
    this.batchSize = Math.min(points.length, 64);
    this.checkValidDimensions(points);

    if (!this.device) {
      await this.init();
      this.setAllPipelines();
    }

    return new Promise((res) => {
      requestAnimationFrame(() => this.#predictionLoop(points, res));
    });
  }

  async #predictionLoop(points, finish) {
    this.fillParams(1, [this.batchSize, 1]);
    this.fillData(points);
    this.updateDynamicBindGroup();

    const commandEncoder = this.device.createCommandEncoder();
    this.forwardBackwardWave(commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);
    
    const output = (await this.extractNetworkOutput()).slice(0, points.length);
    finish(output);
  }

  static meanSquaredError(y, yPred) {
    let error = 0;
    for (let i = 0; i < Math.min(y.length, yPred.length); i++) {
      for (let j = 0; j < Math.min(y[i].length, yPred[i].length); j++) {
        error += ((y[i][j] || 0) - (yPred[i][j] || 0)) ** 2;
      }
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
 
const MAX_TRAINING_INTERRUPT = 4;
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

// function dsigmoid(v) {
//   return Math.exp(-v) / ((1 + Math.exp(-v)) ** 2)
// }

