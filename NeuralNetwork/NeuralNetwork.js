import GPUConnector from "../GPUConnector.js";
import { NETWORK_MODE, TRAIN_METHOD } from "./NetworkConsts.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #cumulativeSum;
  #weightCounts;
  batchSize = 1;

  constructor(layers) {
    super();
    this.layers = [...layers];
    this.maxLayer = this.layers.reduce((a, c) => Math.max(a, c.size), 0);
  }

  async init() {
    await super.initGPU();

    this.createComputeShader();

    this.createStaticBuffers();
    this.createBatchBuffers(1);

    this.createBindGroupLayouts();

    this.createStaticBindGroups();
    this.updateBatchBindGroups();

    this.fillLayers();
  }

  createStaticBuffers() {
    this.createBuffer(
      `Params`,
      Float32Array.BYTES_PER_ELEMENT * 5,
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
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
  }

  createBatchBuffers(size=this.batchSize) {
    this.createBuffer(
      `Batch_Derivatives`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeWeightsCount(this.layers.length) *
        size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `Batch_Neuron_Inputs`,
      Float32Array.BYTES_PER_ELEMENT *
        this.neuronCumulativeSum(this.layers.length - 1) *
        size,
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
            type: `storage`,
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
        {
          binding: 1,
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
    // console.log(neural_net_shader(this))
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
          entryPoint: `backward`,
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
    const [fB, backward, descent] = this.createComputePipelines();

    this.setPipeline(`forward`, fB);
    this.setPipeline(`backward`, backward);
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

  updateBatchBindGroups() {
    const derivativesBindGroup = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[3],
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
            buffer: this.getBuffer(`Batch_Neuron_Inputs`),
          },
        },
      ],
    });
    this.gpuData.bindGroups[3] = derivativesBindGroup;
  }

  fillLayerBuffer(layerIndex, neuronData) {
    if (!neuronData)
      neuronData = this.layers[layerIndex].getNeuronData(
        this.layers[layerIndex + 1]?.size || 0
      );

    const [weights, funcIds] = neuronData;

    if (weights && weights.length) {
      const WEIGHTS = new Float32Array(weights.flat());
      this.writeBuffer(
        `Neuron_Weights`,
        this.neuronCumulativeWeightsCount(layerIndex) *
          Float32Array.BYTES_PER_ELEMENT,
        WEIGHTS
      );
    }
    if (funcIds && funcIds.length) {
      const FUNC_IDS = new Uint32Array(funcIds.flat());
      this.writeBuffer(
        `Neuron_Activation_Func_Ids`,
        this.neuronCumulativeSum(layerIndex - 1) *
          Uint32Array.BYTES_PER_ELEMENT,
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
    if (outputs.length)
      this.writeBuffer1to1(
        `Expected_Output_Data`,
        new Float32Array(outputs.flat())
      );
    // console.log(new Float32Array(outputs?.flat()))
  }

  neuronCumulativeWeightsCount(stopLayerIndex) {
    return this.weightCounts[stopLayerIndex] || 0;
  }

  neuronCumulativeSum(stopLayerIndex) {
    return this.cumulativeSum[stopLayerIndex] || 0;
  }

  get maxBatchSize() {
    return Math.floor(
      this.device.limits.maxStorageBufferBindingSize /
        (this.neuronCumulativeWeightsCount(this.layers.length) *
          Float32Array.BYTES_PER_ELEMENT)
    );
  }

  get cumulativeSum() {
    if (!this.#cumulativeSum) {
      this.#cumulativeSum = this.layers.reduce((a, c, i, arr) => {
        a.push((a.at(-1) || 0) + c.size + (arr.at(i + 1) ? 1 : 0));
        return a;
      }, []);
    }
    return this.#cumulativeSum;
  }

  get weightCounts() {
    if (!this.#weightCounts) {
      this.#weightCounts = this.layers.reduce(
        (a, c, i, arr) => {
          a.push(a.at(-1) + (c.size + 1) * (arr[i + 1]?.size || 0));
          return a;
        },
        [0]
      );
    }
    return this.#weightCounts;
  }

  get layerSizes() {
    return this.layers.map((e) => e.size);
  }

  calculateUpdateIterations(trainMethod, batchSize, trainCount) {
    switch (trainMethod) {
      case TRAIN_METHOD.STOCHASTIC:
        return trainCount;
      case TRAIN_METHOD.MINI_BATCH:
        return Math.ceil(trainCount / batchSize);
      case TRAIN_METHOD.BATCH:
        return 1;
      default:
        return 0;
    }
  }

  calculatePropagateIterations(trainMethod, batchSize, trainCount) {
    switch (trainMethod) {
      case TRAIN_METHOD.STOCHASTIC:
      case TRAIN_METHOD.MINI_BATCH:
        return 1;
      case TRAIN_METHOD.BATCH:
        return trainCount / batchSize;
      default:
        return 0;
    }
  }

  async train(
    points,
    outputs,
    epochs = 1000,
    { learningRate, trainMethod, batchSize, traceHistory, historyTicks } = {
      learningRate: 0.001,
      trainMethod: TRAIN_METHOD.BATCH,
      batchSize: points.length,
      traceHistory: false,
      historyTicks: Math.min(100, Math.floor(epochs * 0.1)),
    }
  ) {
    this.checkValidDimensions(points, outputs);
    console.time(`train`);

    if (!this.device) await this.init();

    const dispatchBatchSize = this.#calculateBatchSize(batchSize, trainMethod, points.length);

    const updateIterations = this.calculateUpdateIterations(
      trainMethod,
      dispatchBatchSize,
      points.length
    );
    const propagateIterations = this.calculatePropagateIterations(
      trainMethod,
      dispatchBatchSize,
      points.length
    );

    this.fillParams([
      learningRate,
      0,
      propagateIterations,
      trainMethod == TRAIN_METHOD.BATCH ? batchSize : dispatchBatchSize,
      NETWORK_MODE.TRAIN,
    ]);

    this.fillData(points, outputs);
    this.updateDataBindGroup();

    if (dispatchBatchSize != this.batchSize) {
      this.batchSize = dispatchBatchSize;
      this.createBatchBuffers();
      this.updateBatchBindGroups();
    }

    this.setAllPipelines();

    return new Promise((resolve) => {
      const resolveOnComplete = async (val) => {
        await this.device.queue.onSubmittedWorkDone();
        resolve(val);
      };

      requestAnimationFrame(() =>
        this.#trainingLoop(
          [...points],
          [...outputs],
          0,
          epochs,
          0,
          updateIterations,
          propagateIterations,
          resolveOnComplete,
          trainMethod == TRAIN_METHOD.MINI_BATCH,
          traceHistory,
          historyTicks
        )
      );
    });
  }

  #calculateBatchSize(batchSize, trainMethod, trainCount) {
    switch (trainMethod) {
      case TRAIN_METHOD.BATCH:
      case TRAIN_METHOD.MINI_BATCH:
        const max = Math.min(this.maxBatchSize, batchSize);
        return Math.min(max, trainCount);
      case TRAIN_METHOD.STOCHASTIC:
        return 1;
      default:
        return 0;
    }
  }

  async #trainingLoop(
    points,
    outputs,
    currEpoch,
    maxEpoch,
    currUpdateIteration,
    updateIterations,
    propagateIterations,
    onTrainComplete,
    shuffle = false,
    traceHistory = false,
    lossTicks = 10,
    lossHistory = []
  ) {
    const start = performance.now();
    while (
      currEpoch < maxEpoch &&
      performance.now() - start <= MAX_TRAINING_INTERRUPT
    ) {
      const commandEncoder = this.device.createCommandEncoder();
      for (let i = 0; i < propagateIterations; i++) {
        this.forward(commandEncoder);
        this.backward(commandEncoder);
      }

      this.descent(commandEncoder);
      this.device.queue.submit([commandEncoder.finish()]);
      currUpdateIteration++;

      if (currUpdateIteration == updateIterations) {
        if (traceHistory && currEpoch % lossTicks == 0) {
          await this.device.queue.onSubmittedWorkDone();
          const prediction = await this.predict(points, false);
          lossHistory.push([
            currEpoch,
            NeuralNetwork.meanSquaredError(outputs, prediction),
          ]);
          this.fillParams([NETWORK_MODE.TRAIN], 4);
        }

        if (shuffle) NeuralNetwork.shuffleTrainData(points, outputs);
        currEpoch++;
        currUpdateIteration = 0;
      }
    }

    if (currEpoch == maxEpoch) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        console.timeEnd(`train`);
      });
      onTrainComplete(lossHistory);
    } else
      requestAnimationFrame(() =>
        this.#trainingLoop(
          points,
          outputs,
          currEpoch,
          maxEpoch,
          currUpdateIteration,
          updateIterations,
          propagateIterations,
          onTrainComplete,
          shuffle,
          traceHistory,
          lossTicks,
          lossHistory
        )
      );
  }

  async extractNetworkParameters() {
    const params = {};
    const commandEncoder = this.device.createCommandEncoder();

    this.copyBuffer(`Params`, commandEncoder);
    this.copyBuffer(`Neuron_Activation_Func_Ids`, commandEncoder);
    this.copyBuffer(`Batch_Derivatives`, commandEncoder);
    this.copyBuffer(`Neuron_Weights`, commandEncoder);
    this.copyBuffer(`Batch_Neuron_Inputs`, commandEncoder);

    this.device.queue.submit([commandEncoder.finish()]);

    params.params = await this.mapBufferToCPU(`Params_copy`, Float32Array);
    params.functionIds = await this.mapBufferToCPU(
      `Neuron_Activation_Func_Ids_copy`,
      Uint32Array
    );

    const derv = await this.mapBufferToCPU(
      `Batch_Derivatives_copy`,
      Float32Array
    );

    const segDerv = [];
    const size = this.neuronCumulativeWeightsCount(this.layers.length);
    for (let i = 0; i < this.batchSize; i++) {
      segDerv.push(derv.slice(i * size, (i + 1) * size));
    }
    params.derv = segDerv;

    const allWeights = await this.mapBufferToCPU(
      `Neuron_Weights_copy`,
      Float32Array
    );
    const allIn = await this.mapBufferToCPU(
      `Batch_Neuron_Inputs_copy`,
      Float32Array
    );
    params.allIn = allIn;
    const segmentedWeights = [];

    let pointer = 0;
    for (let i = 0; i < this.layers.length - 1; i++) {
      const weights_i = [];
      for (let j = 0; j < this.layers[i + 1].size; j++) {
        weights_i.push(
          allWeights.slice(pointer, pointer + this.layers[i].size + 1)
        );
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
    this.copyBuffer(`Expected_Output_Data`, commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);

    const outputData = await this.mapBufferToCPU(
      `Output_Data_copy`,
      Float32Array
    );
    // const eoutputData = await this.mapBufferToCPU(
    //   `Expected_Output_Data_copy`,
    //   Float32Array
    // );
    // console.log(eoutputData)

    const separatedOutputs = [];
    const outPutDim = this.layers.at(-1).size;

    for (let i = 0; i < outputData.length; i += outPutDim) {
      separatedOutputs.push(Array.from(outputData.slice(i, i + outPutDim)));
    }

    return separatedOutputs;
  }

  forward(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline(`forward`));
    this.gpuData.bindGroups.map((e, i) => passEncoder.setBindGroup(i, e));
    passEncoder.dispatchWorkgroups(this.batchSize);
    passEncoder.end();
  }

  backward(commandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.getPipeline(`backward`));
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

  async predict(points, load = true) {
    this.checkValidDimensions(points);

    const cachedBatchSize = this.batchSize || 1;
    this.batchSize = points.length;
    if (!this.device) {
      await this.init();
      this.setAllPipelines();
    }

    this.fillParams([NETWORK_MODE.PREDICT], 4);
    if (load) {
      await this.device.queue.onSubmittedWorkDone();
      this.fillData(points);
      this.updateDataBindGroup();
    }

    return new Promise((res) => {
      const resolve = (val) => {
        this.batchSize = cachedBatchSize;
        res(val);
      };

      requestAnimationFrame(() => this.#predictionLoop(points, resolve));
    });
  }

  async #predictionLoop(points, finish) {
    const commandEncoder = this.device.createCommandEncoder();
    this.forward(commandEncoder);
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

  static shuffleTrainData(X, y) {
    if (X.length !== y.length) {
      throw new Error("Arrays must have the same length");
    }

    for (let i = X.length - 1; i > 0; i--) {
      // Generate a random index
      const j = Math.floor(Math.random() * (i + 1));

      // Swap elements in the first array
      [X[i], X[j]] = [X[j], X[i]];

      // Swap corresponding elements in the second array
      [y[i], y[j]] = [y[j], y[i]];
    }
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
