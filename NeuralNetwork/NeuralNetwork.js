import GPUConnector from "../GPUConnector.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #cumSum;
  #weightCounts;
  batchSize;

  constructor(layers) {
    super();
    this.layers = [...layers];
    this.maxLayer = this.layers.reduce((a, c) => Math.max(a, c.size), 0);
  }

  async init() {
    await super.initGPU();

    this.createComputeShader();
    
    this.createStaticBuffers();
    this.createBatchBuffer();

    this.createBindGroupLayouts();
    
    this.createStaticBindGroups();
    this.updateDerivativesBindGroup();
    
    this.fillLayers();
  }

  createStaticBuffers() {
    this.createBuffer(
      `Params`,
      Float32Array.BYTES_PER_ELEMENT * 5,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
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
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
  }

  createBatchBuffer() {
    this.createBuffer(
      `Batch_Derivatives`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeWeightsCount(this.layers.length)
        * this.batchSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
  }

  createBindGroupLayouts() {
    const bindGroupLayout0 = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: `uniform`,
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
            type: `storage`,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: `read-only-storage`,
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
            type: `read-only-storage`,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: `storage`,
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: `read-only-storage`,
          },
        },
      ],
    });
    const bindGroupLayout3 = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: `storage`,
          },
        },
      ],
    });

    this.gpuData.bindGroupLayouts.push(
      bindGroupLayout0,
      bindGroupLayout1,
      bindGroupLayout2,
      bindGroupLayout3
    );
  }

  createStaticBindGroups() {
    const bindGroup0 = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[0],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer(`Params`),
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
            buffer: this.getBuffer(`Neuron_Weights`),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(`Neuron_Activation_Func_Ids`),
          },
        },
      ],
    });
    this.gpuData.bindGroups.push(bindGroup0, bindGroup1, undefined, undefined);
  }

  createComputeShader() {
    this.createShader(`neural_compute`, neural_net_shader(this));
  }

  createComputePipelines() {
    return [
      this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [...this.gpuData.bindGroupLayouts],
      }),
      compute: {
        module: this.getShader(`neural_compute`),
        entryPoint: `main`,
      },
    }),
    this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [...this.gpuData.bindGroupLayouts],
      }),
      compute: {
        module: this.getShader(`neural_compute`),
        entryPoint: `descent`,
      },
    }),
  ];
  }

  setAllPipelines() {
    const [fB, descent] = this.createComputePipelines();
    
    this.setPipeline(`forward_backward`, fB);
    this.setPipeline(`descent`, descent);
  }

  updateDataBindGroup() {
    const dataBindGroup = this.device.createBindGroup({
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
    this.gpuData.bindGroups[2] = dataBindGroup;
  }

  updateDerivativesBindGroup() {
    const derivativesBindGroup = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[3],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer(`Batch_Derivatives`),
          },
        },
      ],
    });
    this.gpuData.bindGroups[3] = derivativesBindGroup;
  }

  fillLayerBuffer(layerIndex, neuronData) {
    if (!neuronData) neuronData = this.layers[layerIndex].getNeuronData(layerIndex != this.layers.length - 1);

    const [weights, funcIds] = neuronData;

    if (weights && weights.length) {
      const WEIGHTS = new Float32Array(weights);
      this.writeBuffer(
        `Neuron_Weights`,
        this.neuronCumulativeWeightsCount(layerIndex - 1) *
          Float32Array.BYTES_PER_ELEMENT,
        WEIGHTS
      );
    }
    if (funcIds && funcIds.length) {
      const FUNC_IDS = new Uint32Array(funcIds);
      this.writeBuffer(
        `Neuron_Activation_Func_Ids`,
        this.neuronCumulativeSum(layerIndex - 1) * Uint32Array.BYTES_PER_ELEMENT,
        FUNC_IDS
      );
    }
  }

  fillLayers() {
    this.layers.forEach((_, i) => this.fillLayerBuffer(i));
  }

  fillParams(params, offset = 0) {
    this.writeBuffer(
      `Params`,
      offset * Float32Array.BYTES_PER_ELEMENT,
      new Float32Array(params)
    );
  }

  fillData(points, outputs = []) {
    this.createBuffer(
      `Input_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers[0].size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.writeBuffer1to1(`Input_Data`, new Float32Array(points.flat()));
    
    this.createBuffer(
      `Output_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers.at(-1).size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );

    this.createBuffer(
      `Expected_Output_Data`,
      Float32Array.BYTES_PER_ELEMENT * points.length * this.layers.at(-1).size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    if (outputs.length) this.writeBuffer1to1(`Expected_Output_Data`, new Float32Array(outputs.flat()));
  }

  neuronCumulativeWeightsCount(stopLayerIndex) {
    return this.weightCounts[stopLayerIndex] || 0;
  }

  neuronCumulativeSum(stopLayerIndex) {
    return this.cumSum[stopLayerIndex] || 0;
  }

  get cumSum() {
    if (!this.#cumSum) {
      this.#cumSum = this.layers.reduce((a, c, i, arr) => {
        a.push((a.at(-1) || 0) + c.size + (arr.at(i + 1) ? 1 : 0))
        return a;
      }, []);
    }
    return this.#cumSum;
  }

  get weightCounts() {
    if (!this.#weightCounts) {
      this.#weightCounts = this.layers.reduce((a, c, i, arr) => {
        a.push(a.at(-1) + (c.size + 1) * (arr[i + 1]?.size || 0));
        return a;
      }, [0]);
    }
    return this.#weightCounts;
  }

  get layerSizes() {
    return this.layers.map(e => e.size);
  }

  async train(
    points,
    outputs,
    epochs = 1000,
    learningRate = 0.01,
    batchSize = points.length
  ) {
    this.checkValidDimensions(points, outputs);
    console.time(`train`);

    if (!this.device) await this.init();

    this.fillParams([
      learningRate,
      // 0,
      // Math.ceil(points.length / batchSize),
      batchSize,
      0,
    ]);

    this.fillData(points, outputs);
    this.updateDataBindGroup();

    if (batchSize != this.batchSize) {
      this.batchSize = batchSize;
      this.createBatchBuffer();
      this.updateDerivativesBindGroup();
    }

    this.setAllPipelines();

    return new Promise((resolve) => {
      const resolveOnComplete = async (val) => {
        await this.device.queue.onSubmittedWorkDone();
        resolve(val);
      };

      requestAnimationFrame(() =>
        this.#trainingLoop(0, epochs, points, outputs, batchSize, 0, resolveOnComplete)
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
    lossTicks = 100,
    lossHistory = []
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
      
      if (currEpoch % lossTicks == 0) {
        await this.device.queue.onSubmittedWorkDone();
        const prediction = await this.extractNetworkOutput();
        lossHistory.push([currEpoch, NeuralNetwork.meanSquaredError(outputs, prediction)]);
      }
      currEpoch++;
    }

    if (currEpoch == maxEpoch) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        console.timeEnd(`train`);
      })
      finish(lossHistory);
    } else requestAnimationFrame(() => this.#trainingLoop(currEpoch, maxEpoch, points, outputs, batchSize, batchIndex, finish, lossTicks, lossHistory));
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
      for (let j = 0; j < this.layers[i + 1].size; j++) {
        weights_i.push(allWeights.slice(pointer, pointer + this.layers[i].size + 1));
        pointer += this.layers[i].size + 1;
      }
      segmentedWeights.push(weights_i);
    }
    params.weights = segmentedWeights;

    return params;
  }

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
    const outPutDim = this.layers.at(-1).size;

    for (let i = 0; i < outputData.length; i+=outPutDim) {
      separatedOutputs.push(Array.from(outputData.slice(i, i + outPutDim)));
    }

    return separatedOutputs;
  }

  forwardBackwardWave(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline(`forward_backward`));
    this.gpuData.bindGroups.map((e, i) => passEncoder.setBindGroup(i, e));
    passEncoder.dispatchWorkgroups(this.batchSize);
    passEncoder.end();
  }

  descent(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline(`descent`));
    this.gpuData.bindGroups.map((e, i) => passEncoder.setBindGroup(i, e));
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
  }

  async predict(points) {
    this.checkValidDimensions(points);
    
    this.batchSize = points.length;
    if (!this.device) {
      await this.init();
      this.setAllPipelines();
    }

    return new Promise((res) => {
      requestAnimationFrame(() => this.#predictionLoop(points, res));
    });
  }

  async #predictionLoop(points, finish) {
    this.fillParams([this.batchSize, 1], 1);
    this.fillData(points);
    this.updateDataBindGroup();

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
      points[0]?.length != this.layers[0].size ||
      (outputs &&
        (outputs[0].length != this.layers.at(-1).size ||
          points?.length != outputs.length))
    )
      throw new Error(`invalid data dimensions`);
  }
}
 
const MAX_TRAINING_INTERRUPT = 4;

