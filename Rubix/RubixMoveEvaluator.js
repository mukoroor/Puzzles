import GPUConnector from "../GPUConnector.js";
import { move_eval_shader } from "./MoveEvalShader.js";

export default class RubixMoveEvaluator extends GPUConnector {
  #moves;
  #faceDimension
  constructor() {
    super();
    // this.puzzle = puzzle;
  }

  async init() {
    await super.initGPU();
  }

  get moves() {
    if (this.#moves) return this.#moves;
    const moves = [
      [0, 0, 0], // do nothing
    ];

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < this.faceDimension; j++) {
        for (let k = 1; k < 4; k++) {
          moves.push([i, j, k]);
        }
      }
    }
    this.moves = moves;
    return moves;
  }

  set moves(newMoves) {
    this.#moves = newMoves;
  }

  get faceDimension() {
    return this.#faceDimension;
  }

  set faceDimension(newFaceDimension) {
    this.#faceDimension = newFaceDimension;
  }

  get facePieceCount() {
    return Math.pow(this.faceDimension, 2);
  }

  #createBindBuffers() {
    this.createBuffer(
      "face_colorings",
      Math.ceil((3 * 6 * this.facePieceCount) / (8 * Uint32Array.BYTES_PER_ELEMENT)) *
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.#createOutputBuffers(1);
  }

  #createOutputBuffers(faceColoringsInstanceCount) {
    this.createBuffer(
      "cube_colorings_output",
      faceColoringsInstanceCount *
      Math.ceil(
        (this.moves.length * 3 * 6 * this.facePieceCount) /
        (8 * Uint32Array.BYTES_PER_ELEMENT)
      ) *
      Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.createBuffer(
      "score_output",
      faceColoringsInstanceCount *
      this.moves.length *
      Float32Array.BYTES_PER_ELEMENT *
      6,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
  }

  #destroyBuffers() {
    this.getBuffer("face_colorings")?.destroy();
    this.getBuffer("score_output")?.destroy();
    // this.getBuffer("cube_colorings_output")?.destroy();
  }

  #updateFaceColoringsBuffer(colorings) {
    const COLORINGS = new Uint32Array(colorings);
    // let s = Array.from(colorings).map(num => num.toString(2).padStart(32, '0'))
    // let parsed = segmentString(s.reverse().join('')).map(e => parseInt(e, 2));
    // let array2D = [];
    // for (let i = 0; i < parsed.length / 9; i++) {
    //   array2D.push(Array.from(parsed.slice(i * 9, (i + 1) * 9))); // Slice each 9-element chunk
    // }
    // console.log(array2D)
    this.writeBuffer1to1("face_colorings", COLORINGS);
  }

  #createComputePipeline(faceDimension) {
    this.createShader(
      "eval_compute_shader",
      move_eval_shader(this.faceDimension, this.moves)
    );
    const bindGroupLayout = this.device.createBindGroupLayout({
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
            type: "storage",
          },
        },
      ],
    });

    this.gpuData.bindGroupLayouts.push(bindGroupLayout);

    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: this.getShader("eval_compute_shader"),
        entryPoint: "main",
      },
    });
  }

  #createBindGroup() {
    return this.device.createBindGroup({
      layout: this.gpuData.bindGroupLayouts[0],
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer("face_colorings"),
          },
        },

        {
          binding: 1,
          resource: {
            buffer: this.getBuffer("score_output"),
          },
        },

        {
          binding: 2,
          resource: {
            buffer: this.getBuffer("cube_colorings_output"),
          },
        },
      ],
    });
  }

  static toBinary(array) {
    return array
      .flat(2)
      .map((e) => e.toString(2).padStart(3, "0"))
      .reverse()
      .join("");
  }

  async evaluate(generations = 1, faceColorings) {
    const binaryRep = segmentString(
      faceColorings
        .map((e) => RubixMoveEvaluator.toBinary(e))
        .reverse()
        .join(""),
      32
    ).map((e) => parseInt(e, 2));
    this.faceDimension = faceColorings[0].length;
    this.#createBindBuffers();
    this.setPipeline("compute_scoring", this.#createComputePipeline());
    this.#updateFaceColoringsBuffer(binaryRep);

    let currGeneration = 0;
    let finish = false;
    const calcGeneration = async () => {
      currGeneration++;
      let commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(this.getPipeline("compute_scoring"));
      passEncoder.setBindGroup(0, this.#createBindGroup());
      // need to figure out how to split dispatches
      passEncoder.dispatchWorkgroups(
        Math.pow(this.faceDimension + 1, currGeneration),
        Math.pow(3, currGeneration),
        Math.pow(3, currGeneration)
      );
      // passEncoder.dispatchWorkgroups(this.moves.length);
      // passEncoder.dispatchWorkgroups(2);
      passEncoder.end();

      if (currGeneration != generations) {
        this.device.queue.submit([commandEncoder.finish()]);
        try {
          this.#swapColorings();
          await GPUConnector.waitForAnimationFrame(calcGeneration);
        } catch {
          finish = true;
          commandEncoder = this.device.createCommandEncoder();
          console.log(
            `max generation ${currGeneration} possible reached, buffer limit have been reached`
          );
        }
      }

      if (currGeneration == generations || finish) {
        this.copyBuffer("score_output", commandEncoder);
        this.copyBuffer("face_colorings", commandEncoder);
        this.copyBuffer("cube_colorings_output", commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        // console.timeEnd('gpcmp_score')
        const len2 = this.facePieceCount;
        const values = await Promise.all([
          this.mapBufferToCPU("score_output_copy", Float32Array).then((e) => {
            let array2D = [];
            for (let i = 0; i < e.length / 6; i++) {
              array2D.push(Array.from(e.slice(i * 6, (i + 1) * 6))); // Slice each 6-element chunk
            }
            return array2D;
          }),
          // this.mapBufferToCPU('face_colorings_copy', Uint32Array).then(e => {
          //   let s = Array.from(e).map(num => num.toString(2).padStart(32, '0'))
          //   let parsed = segmentString(s.reverse().join('')).map(e => parseInt(e, 2));
          //   let array2D = [];
          //   for (let i = 0; i < parsed.length / len2; i++) {
          //     array2D.push(Array.from(parsed.slice(i * len2, (i + 1) * len2))); // Slice each len2-element chunk
          //   }
          //   return array2D;
          //   // return parsed;
          // }),
          // this.mapBufferToCPU('cube_colorings_output_copy', Uint32Array).then(e => {
          //   let s = Array.from(e).map(num => num.toString(2).padStart(32, '0'))
          //   let parsed = segmentString(s.reverse().join('')).map(e => parseInt(e, 2));
          //   let array2D = [];
          //   for (let i = 0; i < parsed.length / len2; i++) {
          //     array2D.push(Array.from(parsed.slice(i * len2, (i + 1) * len2))); // Slice each len2-element chunk
          //   }
          //   return array2D;
          //   // return parsed;
          // }),
        ])

        console.timeEnd("gpcmp_score");
        // console.log("Scores", v[0]);
        // console.log('Og', v[1]);
        // console.log('Cubes', v[2]);
        function ev() {
          let max = 0,
            maxI = -1;
          for (let i = 0; i < values[0].length; i++) {
            let s = values[0][i].reduce((a, c) => a + c);
            if (s > max) {
              maxI = i;
              max = s;
            }
          }
          return { max, maxI };
        }
        // if (ev().maxI.length != 1) console.log(iee, ev().maxI)
        // if (ev().maxI.length != 1) console.log(iee, ev().maxI.map(index => [Math.floor(index / 28), index % 28]))
        // console.log(ev().maxI)
        // return ev().maxI.map((sI) => {
        return this.moves[(ev().maxI % Math.pow(this.moves.length, 2)) % this.moves.length];
          // console.log(Math.floor(maxI % Math.pow(this.moves.length, 2)) / this.moves.length))
          // return [
            // Math.floor(sI / Math.pow(this.moves.length, 2)),
            // Math.floor(
              // (sI % Math.pow(this.moves.length, 2)) / this.moves.length
            // ),
            // (sI % Math.pow(this.moves.length, 2)) % this.moves.length,
          // ];
        // })
      }
    };

    return await GPUConnector.waitForAnimationFrame(calcGeneration);
    // return "done";
  }

  #swapColorings(faceDimension) {
    // this.#destroyBuffers();
    const count =
      Math.ceil(
        (this.moves.length * 3 * 6 * this.facePieceCount) /
        (8 * Uint32Array.BYTES_PER_ELEMENT)
      ) * Uint32Array.BYTES_PER_ELEMENT;
    const nextFaceColorings = this.getBuffer("cube_colorings_output");
    this.#createOutputBuffers(
      (this.moves.length * nextFaceColorings.size) / count);
    this.setBuffer("face_colorings", nextFaceColorings);
  }
}

//util will remove
function segmentString(str, len = 3) {
  var chunks = [];

  for (var i = str.length; i > 0; i -= len) {
    chunks.push(str.substring(i - len, i));
  }

  return chunks;
}
