import GPUConnector from "../GPUConnector.js";
import { move_eval_shader } from "./MoveEvalShader.js";

export default class RubixMoveEvaluator extends GPUConnector {
    #moves;
    constructor(puzzle) {
        super();
        this.puzzle = puzzle;
    }

    async init() {
        await super.initGPU();
        this.#createBindBuffers();
        this.setPipeline("compute_scoring", this.#createComputePipeline());
    }

    get moves() {
      if (this.#moves) return this.#moves
      const moves = [
        [0, 0, 0] // do nothing
      ];
  
      for (let i = 0; i < 3; i++) {
          for (let j = 0; j < this.puzzle.length; j++) {
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
  

    #createBindBuffers() {
        this.createBuffer(
            "face_colorings",
            Math.ceil(3 * this.puzzle.pieceFaceCount / (8 * Uint32Array.BYTES_PER_ELEMENT)) * Uint32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.#createOutputBuffers(1);
    }
      
    #createOutputBuffers(faceColoringsInstanceCount) {
      this.createBuffer(
        "cube_colorings_output",
        faceColoringsInstanceCount * Math.ceil(this.moves.length * 3 * this.puzzle.pieceFaceCount / (8 * Uint32Array.BYTES_PER_ELEMENT)) * Uint32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      );
      this.createBuffer(
          "score_output",
          faceColoringsInstanceCount * this.moves.length * Float32Array.BYTES_PER_ELEMENT * 6,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
    }

    #destroyBuffers() {
        this.getBuffer("face_colorings")?.destroy();
        this.getBuffer("score_output")?.destroy();
        // this.getBuffer("cube_colorings_output")?.destroy();
    }

    #updateFaceColoringsBuffer(colorings) {
        const COLORINGS = new Uint32Array(colorings);
        this.writeBuffer1to1("face_colorings", COLORINGS);
    }


    #createComputePipeline() {
        this.createShader('eval_compute_shader', move_eval_shader(this.puzzle.length, this.moves));
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

        this.gpuData.bindGroupLayouts.push(bindGroupLayout)

        return this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
              module: this.getShader('eval_compute_shader'),
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
              buffer: this.getBuffer('face_colorings'),
            },
          },
    
          {
            binding: 1,
            resource: {
              buffer: this.getBuffer('score_output'),
            },
          },

          {
            binding: 2,
            resource: {
              buffer: this.getBuffer('cube_colorings_output'),
            },
          },
        ],
      });
    }

    async evaluate(generations = 4) {
        const binaryRep = segmentString(this.puzzle.faces.map(e => e.toBinary()).reverse().join(''), 32).map(e => parseInt(e, 2));
        this.#updateFaceColoringsBuffer(binaryRep);

        
        let currGeneration = 0;
        let finish = false;
        const calcGeneration = async () => {
          currGeneration++;
          let commandEncoder = this.device.createCommandEncoder();
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(this.getPipeline('compute_scoring'));
          passEncoder.setBindGroup(0, this.#createBindGroup());
          // need to figure out how to split dispatches
          passEncoder.dispatchWorkgroups(Math.pow(7, currGeneration), Math.pow(2, currGeneration), Math.pow(2, currGeneration));
          passEncoder.end();
          
          if (currGeneration != generations) {
            this.device.queue.submit([commandEncoder.finish()]);
            try {
              this.#swapColorings();
              window.requestAnimationFrame(calcGeneration);
            } catch {
              finish = true;
              commandEncoder = this.device.createCommandEncoder();
              console.log(`max generation ${currGeneration} possible reached, buffer limit have been reached`)
            }
          } 
          
          if (currGeneration == generations || finish) {
            this.copyBuffer('score_output', commandEncoder);
            this.copyBuffer('cube_colorings_output', commandEncoder);
            this.device.queue.submit([commandEncoder.finish()]);
            console.timeEnd('gpcmp_score')
            Promise.all([
              this.mapBufferToCPU('score_output_copy', Float32Array).then(e => {
                let array2D = [];
                for (let i = 0; i < e.length / 6; i++) {
                  array2D.push(Array.from(e.slice(i * 6, (i + 1) * 6))); // Slice each 6-element chunk
                }
                return array2D;
              }),
              this.mapBufferToCPU('cube_colorings_output_copy', Uint32Array).then(e => {
                let s = Array.from(e).map(num => num.toString(2).padStart(32, '0'))
                let parsed = segmentString(s.reverse().join('')).map(e => parseInt(e, 2));
                let array2D = [];
                for (let i = 0; i < parsed.length / 9; i++) {
                  array2D.push(Array.from(parsed.slice(i * 9, (i + 1) * 9))); // Slice each 9-element chunk
                }
                return array2D;
              }),
            ]).then(console.log)
          }
        }

        calcGeneration();
        return 'done'
    }

    #swapColorings() {
      // this.#destroyBuffers();
      const count = Math.ceil(this.moves.length * 3 * this.puzzle.pieceFaceCount / (8 * Uint32Array.BYTES_PER_ELEMENT)) * Uint32Array.BYTES_PER_ELEMENT
      const nextFaceColorings = this.getBuffer("cube_colorings_output");
      this.#createOutputBuffers(this.moves.length * nextFaceColorings.size / count);
      this.setBuffer('face_colorings', nextFaceColorings);
    }
}

//util will remove
function segmentString(str, len=3){

  var chunks = [];

  for (var i = str.length; i >0; i -= len) {
      chunks.push(str.substring(i-len, i));
  }
  
  return chunks
}