import GPUConnector from "../GPUConnector.js";
import { move_eval_shader } from "./MoveEvalShader.js";

export default class RubixMoveEvaluator extends GPUConnector {
  #moves;
  #faceDimension
  constructor() {
    super();
  }

  async init() {
    await super.initGPU();
  }

  get moves() {
    if (this.#moves && this.#moves.length == 12 * this.faceDimension) return this.#moves;
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

  get colorsPerFace() {
    return Math.pow(this.faceDimension, 2);
  }

  #createBindBuffers() {
    this.createBuffer(
      "face_colorings",
      Math.ceil((3 * 6 * this.colorsPerFace) / (8 * Uint32Array.BYTES_PER_ELEMENT)) *
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
        (this.moves.length * 3 * 6 * this.colorsPerFace) /
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
  }

  #updateFaceColoringsBuffer(colorings) {
    const COLORINGS = new Uint32Array(colorings);
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
    console.time("gpcmp_score");
    this.faceDimension = faceColorings[0].length;
    this.#createBindBuffers();
    this.setPipeline("compute_scoring", this.#createComputePipeline());
    this.#updateFaceColoringsBuffer(segmentString(
      faceColorings
        .map((e) => RubixMoveEvaluator.toBinary(e))
        .reverse()
        .join(""),
      32
    ).map((e) => parseInt(e, 2)));

    const [scores, generation] = await this.#calcPermutations(generations);
    // console.log(c)
    
    console.timeEnd("gpcmp_score");
    const {maxI, ..._} = getMaxSubArr(scores);
    // console.log(scores[maxI])
    const puzzleState = await this.extractPermutation(maxI)
    const moveIndices = this.#scoreIndexToMoveIndices(maxI, generations);
    const moves = moveIndices.map(i => RubixMoveEvaluator.convertMoveToRubixFormat(this.moves[i]));
    // console.log(maxI)
    // console.log(moves)
    return { maxI, ..._, moveIndices, moves, generation, puzzleState };
  }

  async extractPermutation(permutationIndex) {
    const faceSize = this.colorsPerFace;
    const U32_BITS = 8 * Uint32Array.BYTES_PER_ELEMENT;
    const sizeUint32 = 3 * 6 * faceSize / (U32_BITS);
    const strBitOff = permutationIndex * sizeUint32 ;
    const offsetBytes = Math.floor(strBitOff) * Uint32Array.BYTES_PER_ELEMENT;
    const offsetExtra = offsetBytes % 8 ? 4 : 0;

    const perm = await this.mapBufferToCPU("cube_colorings_output_copy", Uint32Array, offsetBytes - offsetExtra, 2 * Math.ceil(sizeUint32) * Uint32Array.BYTES_PER_ELEMENT + offsetExtra);

    let s = Array.from(perm).map(num => num.toString(2).padStart(U32_BITS, '0'))
    if (offsetExtra) s.shift();
    s[0] = s[0].slice(-U32_BITS, -(strBitOff * U32_BITS) % (U32_BITS) || undefined)

    let parsed = segmentString(s.reverse().join(''), 3).map(e => parseInt(e, 2));
    let twoD = [];
    for (let i = 0; i < 6; i++) {
      let face = [];
      for (let j = 0; j < this.faceDimension; j++) {
        let start = i * faceSize + j * this.faceDimension;
        face.push(parsed.slice(start, start + this.faceDimension));
      }
      twoD.push(face);
    }
    return twoD
  }

  async #calcPermutations(generations) {
    let currGeneration = 0;
    let finish = false;
    const calcGeneration = async () => {
      currGeneration++;
      let commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(this.getPipeline("compute_scoring"));
      passEncoder.setBindGroup(0, this.#createBindGroup());
      passEncoder.dispatchWorkgroups(
        Math.pow(this.faceDimension + 1, currGeneration),
        Math.pow(3, currGeneration),
        Math.pow(3, currGeneration)
      );
      passEncoder.end();

      if (currGeneration != generations) {
        this.device.queue.submit([commandEncoder.finish()]);
        try {
          this.#swapColorings();
          return await GPUConnector.waitForAnimationFrame(calcGeneration);
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
        // this.copyBuffer("face_colorings", commandEncoder);
        this.copyBuffer("cube_colorings_output", commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);

        return await Promise.all([
          this.mapBufferToCPU("score_output_copy", Float32Array).then((e) => {
            let array2D = [];
            for (let i = 0; i < e.length / 6; i++) {
              array2D.push(Array.from(e.slice(i * 6, (i + 1) * 6))); // Slice each 6-element chunk
            }
            return array2D;
          }),
          // this.mapBufferToCPU('cube_colorings_output_copy', Uint32Array).then(e => {
          //   let s = Array.from(e).map(num => num.toString(2).padStart(32, '0'))
          //   let parsed = segmentString(s.reverse().join('')).map(e => parseInt(e, 2));
          //   let array2D = [];
          //   for (let i = 0; i < parsed.length / this.colorsPerFace; i++) {
          //     // array2D.push(Array.from(parsed.slice(i * this.colorsPerFace, (i + 1) * this.colorsPerFace))); // Slice each len2-element chunk
          //     let face = [];
          //     for (let j = 0; j < this.faceDimension; j++) {
          //       let start = i * this.colorsPerFace + j * this.faceDimension;
          //       face.push(parsed.slice(start, start + this.faceDimension));
          //     }
          //     array2D.push(face);
          //   }
          //   return array2D;
          //   // return parsed;
          // }),
          currGeneration
        ])
      }
    }
    return await GPUConnector.waitForAnimationFrame(calcGeneration)
  }

  static convertMoveToRubixFormat(move) {
    return [move[0], move[1], ...(move[2] > 2 ? ['cw', 4 - move[2]] : ['ccw', move[2]])];
  }

  #scoreIndexToMoveIndices(index, generations) {
    const moveLen = this.moves.length, indices = [];
    let val = index, divisorCurr = moveLen, divisorPrev = 1;
    for (let _ = 0; _ < generations; _++) {
      indices.unshift((val % divisorCurr) / divisorPrev);
      val -= indices[0] * divisorPrev;
      divisorPrev = divisorCurr
      divisorCurr *= moveLen;
    }
    return indices;
  }

  #swapColorings() {
    const count =
      Math.ceil(
        (this.moves.length * 3 * 6 * this.colorsPerFace) /
        (8 * Uint32Array.BYTES_PER_ELEMENT)
      ) * Uint32Array.BYTES_PER_ELEMENT;
    const nextFaceColorings = this.getBuffer("cube_colorings_output");
    this.#createOutputBuffers(
      (this.moves.length * nextFaceColorings.size) / count);
    this.setBuffer("face_colorings", nextFaceColorings);
  }
}

//util will relocate
function segmentString(str, len = 3) {
  var chunks = [];
  
  for (var i = str.length; i > 0; i -= len) {
    chunks.push(str.substring(i - len, i));
  }
  
  return chunks;
}
//util will relocate
function getMaxSubArr(arr) {
  let max = 0, maxI = -1;
  for (let i = 0; i < arr.length; i++) {
    const currVal = arr[i].reduce((a, c) => a + c);
    if (currVal > max) {
      maxI = i;
      max = currVal;
    }
  }
  // console.log(max)
  return { max, maxI, val: [...arr[maxI]] };
}
