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
  

    #createBindBuffers(permutationCount = 1) {
        const colorCount = Math.pow(this.puzzle.length, 2) * 6;
        this.createBuffer(
            "face_colorings",
            permutationCount * Uint32Array.BYTES_PER_ELEMENT * colorCount,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.#createOutputBuffers(this.getBuffer("face_colorings").size);
    }
      
    #createOutputBuffers(faceColoringsSize) {
      const colorCount = Math.pow(this.puzzle.length, 2) * 6;
      console.log(this.moves.length * faceColoringsSize)
      this.createBuffer(
        "cube_colorings_output",
        this.moves.length * faceColoringsSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      );
      this.createBuffer(
          "score_output",
          this.moves.length * Float32Array.BYTES_PER_ELEMENT * 6 * faceColoringsSize / (Uint32Array.BYTES_PER_ELEMENT * colorCount),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
    }

    #destroyBuffers() {
        this.getBuffer("face_colorings")?.destroy();
        this.getBuffer("score_output")?.destroy();
        // this.getBuffer("cube_colorings_output")?.destroy();
    }

    #updateFaceColoringsBuffer(colorings) {
        const COLORINGS = new Uint32Array(colorings.flat(2));
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

    async evaluate(generations = 7) {
        this.#updateFaceColoringsBuffer(this.puzzle.faces.map(e => e.to2DArray().map(x => x.map(c => c.faceData[e.id]))));

        
        let currGeneration = 0;
        let finish = false;
        const calcGeneration = async () => {
          currGeneration++;
          let commandEncoder = this.device.createCommandEncoder();
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(this.getPipeline('compute_scoring'));
          passEncoder.setBindGroup(0, this.#createBindGroup());
          passEncoder.dispatchWorkgroups(this.moves.length);
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
                let array2D = [];
                for (let i = 0; i < e.length / 9; i++) {
                  array2D.push(Array.from(e.slice(i * 9, (i + 1) * 9))); // Slice each 9-element chunk
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
      const nextFaceColorings = this.getBuffer("cube_colorings_output");
      this.#createOutputBuffers(nextFaceColorings.size);
      this.setBuffer('face_colorings', nextFaceColorings);
    }
}