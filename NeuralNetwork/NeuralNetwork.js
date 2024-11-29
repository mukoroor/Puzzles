import GPUConnector from "../GPUConnector.js";
import { neural_net_shader } from "./NeuralNetCompute.js";

export default class NeuralNetwork extends GPUConnector {
  #neuronSize;

  constructor(layers) {
    super();
    this.layers = [...layers];
    this.maxLayer = this.layers.reduce((a,c) => Math.max(a, c))
  }

  async init() {
    await super.initGPU();

    this.setUpBuffers();
    this.createBindGroupLayout();
    this.createBindGroups();
    this.fillLayers();
  }

  createComputeShader(batchSize, learningRate) {
    this.createShader(
      "neural_compute",
      neural_net_shader(this.layers, batchSize, learningRate)
    );
  }

  createBindGroupLayout() {
    const bindGroupLayout = this.device.createBindGroupLayout({
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
            type: "storage",
          },
        },

        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
      ],
    });

    this.gpuData.bindGroupLayouts.push(bindGroupLayout);
  }

  createComputePipeline() {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.gpuData.bindGroupLayouts[0]],
      }),
      compute: {
        module: this.getShader("neural_compute"),
        entryPoint: "main",
      },
    });
  }

  createLayerBindGroup(layerIndex) {
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[0],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer(this.generateLayerIndexName(layerIndex)),
          },
        },

        {
          binding: 1,
          resource: {
            buffer: this.getBuffer(this.generateLayerNeuronsName(layerIndex - 1)),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer(this.generateLayerNeuronsName(layerIndex)),
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.getBuffer(this.generateLayerDerivativesName(layerIndex)),
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.getBuffer(this.generateLayerDerivativesName(layerIndex + 1)),
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
            buffer: this.getBuffer(`target`),
          },
        },
        {
          binding: 7,
          resource: {
            buffer: this.getBuffer(`result`),
          },
        },
      ],
    });
    this.gpuData.bindGroups.push(bindGroup);
  }

  setUpBuffers() {
    this.createBuffer(
      `Network_Mode`,
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      this.generateLayerDerivativesName(this.layers.length),
      Float32Array.BYTES_PER_ELEMENT * this.maxLayer * (2 * this.maxLayer + 3),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `target`,
      Float32Array.BYTES_PER_ELEMENT * this.layers.at(-1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      `result`,
      Float32Array.BYTES_PER_ELEMENT * this.layers.at(-1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.layers.forEach((e, i) => this.createLayerBuffers(i, e));
  }

  fillLayers() {
    this.layers.forEach((_, i) => this.fillLayerBuffer(i))
  }

  createBindGroups() {
    this.layers.forEach((_, i) => {
      if (i) this.createLayerBindGroup(i);
    })
  }

  fillLayerBuffer(layerIndex, neurons) {
    if (!neurons) {
      neurons = Array.from({length: this.layers[layerIndex]},
          () => this.generateRandomNeuron(this.layers[layerIndex - 1] || 1))
    }
    const NEURONS = new Float32Array(neurons.flat());
    this.writeBuffer1to1(this.generateLayerNeuronsName(layerIndex), NEURONS);
    this.writeBuffer1to1(this.generateLayerIndexName(layerIndex), new Uint32Array([layerIndex]));
  }

  createLayerBuffers(layerIndex, neuronCount) {
    this.createBuffer(
      this.generateLayerIndexName(layerIndex),
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      this.generateLayerNeuronsName(layerIndex),
      Float32Array.BYTES_PER_ELEMENT * this.neuronSize * neuronCount,
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
    this.#neuronSize = 3 + this.maxLayer;
    return this.#neuronSize;
  }

  set neuronSize(newNeuronSize) {
    this.#neuronSize = newNeuronSize;
  }

  generateRandomNeuron() {
    return [0, 1, ...Array.from({length: this.neuronSize - 2}, () => Math.random())];
  }

  fillData(features, output) {
    this.fillLayerBuffer(0, features.map(e => [e, ...Array(this.neuronSize - 1).fill(0)]))
    if (output) this.writeBuffer1to1('target', new Float32Array(output));
  } 

  async train(points, outputs, epochs=10000, learningRate=0.04, batchSize=points.length) {
    console.time("train")

    if (!this.device) await this.init();

    this.createComputeShader(batchSize, learningRate);
    this.setPipeline("main", this.createComputePipeline());

    if (points[0]?.length != this.layers[0]
        || outputs[0].length != this.layers.at(-1)
        || points?.length != outputs.length) throw new Error("invalid data dimensions")

    for (let i = 0; i < epochs; i++) {
      this.#trainBatch(points, outputs, batchSize, (i * batchSize) % points.length);
    }
    console.timeEnd('train')
  }


  async extractNetworkParameters() {
    const layers = [];
    const commandEncoder = this.device.createCommandEncoder();

    for (let i = 0; i < this.layers.length; i++) {
      this.copyBuffer(this.generateLayerNeuronsName(i), commandEncoder);
    }
    this.device.queue.submit([commandEncoder.finish()]);

    for (let i = 0; i < this.layers.length; i++) {
      layers.push(await this.mapBufferToCPU(`${this.generateLayerNeuronsName(i)}_copy`, Float32Array));
    }

    return layers
  }

  async extractOutput() {
    const commandEncoder = this.device.createCommandEncoder();
    this.copyBuffer(`result`, commandEncoder);
    this.device.queue.submit([commandEncoder.finish()]);

    const outputResult = await this.mapBufferToCPU(`result_copy`, Float32Array);

    return outputResult;
  }

  async #trainBatch(points, outputs, batchSize, batchOffset) {
    for (let i = 0; i < batchSize; i++) {
      this.fillData(points[(i + batchOffset) % points.length], outputs[(i + batchOffset) % points.length]);
      this.forward();
      this.backward();
    }
    this.descent();
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
      passEncoder.setBindGroup(0, this.gpuData.bindGroups[i - 1]);
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
      passEncoder.setBindGroup(0, this.gpuData.bindGroups[i - 1]);
      passEncoder.dispatchWorkgroups(this.layers[i]);
    }
    passEncoder.end();
  }

  async predict(points, commandEncoder) {
    if (!this.device) await this.init();
    if (!this.getShader("neural_compute")) {
      this.createComputeShader(1, 1);
      this.setPipeline("main", this.createComputePipeline());
    }

    const predictions = [];
    for (const point of points) {
      this.fillData(point);
      this.forward(commandEncoder);
      predictions.push(await this.extractOutput());
    }
    return predictions;
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
}


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


